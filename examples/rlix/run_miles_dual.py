"""M11.2 dual-pipeline driver — `examples/rlix/run_miles_dual.py`.

Spawns two MilesCoordinator + MilesPipeline pairs in separate Ray
namespaces with disjoint ``cluster_device_mappings``. Each pipeline
runs its own ``rlix_train_loop`` concurrently via ``asyncio.gather``.

Topology (Codex-recommended Option A — disjoint pools, no cross-pipeline
GPU contention):
    pipeline 1: actor_train=[0,1], actor_infer=[0,1]
    pipeline 2: actor_train=[2,3], actor_infer=[2,3]

This is the minimum-viable M11.2 PASS — proves two pipelines can register
+ initialize + train + sync + generate + clean up concurrently without
namespace, actor-name, port, or scheduler-ledger collisions. It does NOT
exercise cross-pipeline preemption (Option B/C — overlap topology — needs
the deferred F22 shell-init contract per ``miles_pipeline.py:14-33``).

Per-pipeline isolation that this driver enforces:
- Distinct pipeline IDs from ``orchestrator.allocate_pipeline_id``
- Distinct Ray namespaces from ``get_pipeline_namespace(pipeline_id)``
- Distinct ``cluster_device_mappings`` registered with the orchestrator
  AND threaded through ``MilesPipelineConfig.cluster_device_mappings``
  so ``MilesPipeline._build_placement_provider`` uses the right physical
  GPUs (rather than the default ``range(actor_count)`` / ``range(...)``)
- Distinct ``MILES_ROLLOUT_BASE_PORT`` per pipeline (15000 vs 16000) so
  the ``find_available_port`` calls in two concurrent RolloutManager
  actors never race for the same port window
- Distinct ``exp_name`` so any tracking dirs / log dirs do not collide
- W&B / TensorBoard / Prometheus disabled (``--use-wandb`` etc must be
  unset); per-pipeline tracking re-enable is M11.3 follow-up

Per scope F13 the driver MUST NOT have a top-level ``try/except`` and
MUST NOT call ``ray.shutdown()``: failure semantics = let exceptions
propagate naturally → driver exits → user runs ``ray stop`` to clean up.
"""

from __future__ import annotations

import copy
import os
import sys

# F08 / F41 — fail fast if RLix entry is invoked without the env var.
# The check must happen BEFORE any heavy import (torch / sglang /
# megatron) so CVD has a chance to take effect via Ray runtime_env.
if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
    sys.stderr.write(
        "examples/rlix/run_miles_dual.py requires RLIX_CONTROL_PLANE=rlix.\n"
        "    RLIX_CONTROL_PLANE=rlix python -m examples.rlix.run_miles_dual ...\n"
    )
    sys.exit(2)


def _split_pools_for_dual(
    *, num_gpus_per_node: int, infer_pool_size: int
) -> tuple[list[int], list[int]]:
    """Split a contiguous physical GPU pool into two disjoint per-pipeline pools.

    For a 4-GPU machine with infer_pool_size=2 returns
    ``([0,1], [2,3])``. The base args carry the PER-PIPELINE shape
    (actor_num_gpus_per_node = train size per pipeline,
    rollout_num_gpus = infer pool size per pipeline). The dual driver
    just maps each pipeline onto its own slice of the physical pool.

    M11.2-DISJOINT fallback only. Real M11.2 (overlap) uses
    ``_overlap_pools_from_env`` below.
    """
    needed = 2 * infer_pool_size
    if num_gpus_per_node < needed:
        raise ValueError(
            f"need {needed} GPUs for 2 pipelines (each infer_pool={infer_pool_size}), "
            f"have num_gpus_per_node={num_gpus_per_node}"
        )
    # F9 (m11-review.review-report.md §2): silent GPU leak when
    # num_gpus_per_node is not exactly 2*infer_pool_size. Prior behavior
    # took the first 2*infer_pool_size GPUs and silently ignored the rest
    # (e.g. on a 5-GPU box with infer_pool_size=2, GPU 4 would be
    # invisible to both pipelines, never freed back to the rlix scheduler).
    # Reject odd / extra-GPU layouts explicitly so the operator knows to
    # either re-shape the smoke (use the overlap env path) or shrink to a
    # supported count.
    if num_gpus_per_node != needed:
        raise ValueError(
            f"_split_pools_for_dual requires num_gpus_per_node ({num_gpus_per_node}) "
            f"== 2 * infer_pool_size ({needed}); extra GPUs would be silently "
            f"ignored. Use MILES_DUAL_P*_{{TRAIN,INFER}} env vars to specify "
            f"explicit pool mappings instead (see _overlap_pools_from_env)."
        )
    physical = list(range(num_gpus_per_node))
    return list(physical[:infer_pool_size]), list(physical[infer_pool_size : 2 * infer_pool_size])


def _parse_gpu_list(env_name: str) -> list[int] | None:
    """Parse ``MILES_DUAL_*`` env (comma-separated GPU IDs) into a list.

    Returns ``None`` if the env is unset / empty so callers can fall
    back to the disjoint default.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise ValueError(
            f"{env_name}={raw!r} is not a comma-separated list of GPU IDs"
        ) from exc


def _overlap_pools_from_env(num_gpus_per_node: int) -> (
    tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]] | None
):
    """Read per-pipeline mappings from ``MILES_DUAL_*`` env vars.

    Returns ``((p1_train, p1_infer), (p2_train, p2_infer))`` if **all four**
    env vars are set; otherwise ``None`` (caller falls back to
    ``_split_pools_for_dual``). Validates:
      - each GPU id is in ``[0, num_gpus_per_node)``
      - per-pipeline ``train ⊆ infer`` (partial-overlap inside pipeline)
      - no duplicate IDs within a single mapping
    Cross-pipeline overlap is INTENDED for real M11.2 and is NOT rejected
    here; the harness ``grep_overlap_log.sh`` asserts the overlap-non-empty
    condition end-to-end.
    """
    p1_train = _parse_gpu_list("MILES_DUAL_P1_TRAIN")
    p1_infer = _parse_gpu_list("MILES_DUAL_P1_INFER")
    p2_train = _parse_gpu_list("MILES_DUAL_P2_TRAIN")
    p2_infer = _parse_gpu_list("MILES_DUAL_P2_INFER")
    if None in (p1_train, p1_infer, p2_train, p2_infer):
        return None
    for label, mapping in (
        ("p1_train", p1_train), ("p1_infer", p1_infer),
        ("p2_train", p2_train), ("p2_infer", p2_infer),
    ):
        for g in mapping:
            if g < 0 or g >= num_gpus_per_node:
                raise ValueError(
                    f"{label}={mapping} contains GPU {g} outside "
                    f"[0, {num_gpus_per_node})"
                )
        if len(set(mapping)) != len(mapping):
            raise ValueError(f"{label}={mapping} has duplicate GPU ids")
    if not set(p1_train).issubset(set(p1_infer)):
        raise ValueError(
            f"p1_train={p1_train} not ⊆ p1_infer={p1_infer} "
            f"(per-pipeline partial-overlap invariant)"
        )
    if not set(p2_train).issubset(set(p2_infer)):
        raise ValueError(
            f"p2_train={p2_train} not ⊆ p2_infer={p2_infer} "
            f"(per-pipeline partial-overlap invariant)"
        )
    return (p1_train, p1_infer), (p2_train, p2_infer)


def _per_pipeline_args(
    base_args,
    *,
    pipeline_index: int,
    train_size: int | None = None,
    infer_size: int | None = None,
):
    """Deep-copy parsed args and tailor for one pipeline.

    When ``train_size`` / ``infer_size`` are provided, the per-pipeline
    ``actor_num_gpus_per_node`` (assuming ``actor_num_nodes=1``) and
    ``rollout_num_gpus`` are overridden so MilesPipeline's placement
    provider sees the right sizes. When omitted, base args' per-pipeline
    shape is preserved (M11.2-disjoint fallback behavior).
    """
    args = copy.deepcopy(base_args)

    if hasattr(args, "exp_name") and args.exp_name:
        args.exp_name = f"{args.exp_name}-mp{pipeline_index}"
    else:
        args.exp_name = f"miles_dual_mp{pipeline_index}"

    # Force fresh router allocation per pipeline; rollout.py's
    # _start_router calls find_available_port when sglang_router_port is
    # None.
    if hasattr(args, "sglang_router_port"):
        args.sglang_router_port = None

    # M11.2-OVERLAP: per-pipeline shape derived from explicit mappings.
    # Assumes actor_num_nodes=1 (single-machine; multi-node deferred to
    # M11.3+). The whole-machine num_gpus_per_node stays as base.
    if train_size is not None:
        if int(getattr(args, "actor_num_nodes", 1)) != 1:
            raise NotImplementedError(
                "run_miles_dual.py overlap mode requires actor_num_nodes=1 "
                "(multi-node deferred to M11.3+)"
            )
        args.actor_num_gpus_per_node = int(train_size)
    if infer_size is not None:
        args.rollout_num_gpus = int(infer_size)

    return args


def _build_pipeline(
    *,
    base_args,
    pipeline_index: int,
    train_mapping: list[int],
    infer_mapping: list[int],
    orchestrator,
    ray,
    MilesCoordinator,
    MilesPipelineConfig,
    get_coordinator_actor_name,
    get_pipeline_namespace,
    logger,
):
    """Allocate one pipeline_id, register, admit, create coordinator+pipeline.

    ``train_mapping`` / ``infer_mapping`` are the EXPLICIT physical GPU
    IDs for this pipeline. Overlap with the peer pipeline is allowed
    (and required for real M11.2). The per-pipeline ``train ⊆ infer``
    partial-overlap invariant is asserted; cross-pipeline overlap is
    asserted by ``grep_overlap_log.sh`` end-to-end.

    Returns ``(pipeline_id, namespace, coordinator_handle, pipeline_handle, args)``.
    """
    pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("miles"))
    pipeline_namespace = get_pipeline_namespace(pipeline_id)

    train_size = len(train_mapping)
    infer_size = len(infer_mapping)
    if not set(train_mapping).issubset(set(infer_mapping)):
        raise ValueError(
            f"mp{pipeline_index}: train_mapping={train_mapping} not ⊆ "
            f"infer_mapping={infer_mapping} (per-pipeline partial-overlap)"
        )
    args = _per_pipeline_args(
        base_args,
        pipeline_index=pipeline_index,
        train_size=train_size,
        infer_size=infer_size,
    )
    logger.info(
        "[run_miles_dual] mp%d allocated pipeline_id=%s namespace=%s "
        "train=%s infer=%s",
        pipeline_index, pipeline_id, pipeline_namespace,
        train_mapping, infer_mapping,
    )

    cluster_device_mappings = {
        "actor_train": train_mapping,
        "actor_infer": infer_mapping,
    }
    cluster_tp_configs = {
        "actor_train": int(args.actor_num_gpus_per_node),
        "actor_infer": int(args.rollout_num_gpus_per_engine),
    }
    ray.get(
        orchestrator.register_pipeline.remote(
            pipeline_id=pipeline_id,
            ray_namespace=pipeline_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )
    )
    ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))
    logger.info(
        "[run_miles_dual] mp%d registered+admitted pipeline_id=%s mappings=%s",
        pipeline_index, pipeline_id, cluster_device_mappings,
    )

    cfg = MilesPipelineConfig(
        miles_args=args,
        sglang_config=getattr(args, "sglang_config", None),
        verify_model_after_sync=bool(getattr(args, "verify_model_after_sync", False)),
        num_gpus_per_node=int(
            getattr(args, "num_gpus_per_node", None) or args.actor_num_gpus_per_node
        ),
        system_envs={},
        cluster_device_mappings=cluster_device_mappings,
    )

    pipeline_runtime_env_vars = {
        "PIPELINE_ID": str(pipeline_id),
        "ROLL_RAY_NAMESPACE": pipeline_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
        # Per-pipeline base port so the two RolloutManager actors do
        # not race for the same port window.
        "MILES_ROLLOUT_BASE_PORT": str(15000 + pipeline_index * 1000),
    }
    if pythonpath := os.environ.get("PYTHONPATH"):
        pipeline_runtime_env_vars["PYTHONPATH"] = pythonpath
    for _k in (
        "MILES_TMS_HOOK_MODE",
        "MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL",
        "MILES_MAX_RESIDUAL_GPU_MEM_GB",
        "MILES_SKIP_TMS_PAUSE",
        "MILES_SKIP_NODE_PG_PIN",
        "TMS_INIT_ENABLE_CPU_BACKUP",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_NVLS_ENABLE",
    ):
        if (_v := os.environ.get(_k)) is not None:
            pipeline_runtime_env_vars[_k] = _v

    coordinator = (
        ray.remote(MilesCoordinator)
        .options(
            name=get_coordinator_actor_name(pipeline_id),
            namespace=pipeline_namespace,
            lifetime="detached",
            num_cpus=0.01,
            # Mirror rlix's verified 2-pipeline reference
            # (examples/start_multi_pipeline_test.py: CoordinatorActor
            # uses COORDINATOR_MAX_CONCURRENCY). The MILES coordinator
            # services concurrent RPCs from multiple sources:
            #   - scheduler.resize_infer (engine wake / shrink)
            #   - report_progress_from_scheduler (hooks → aggregate → scheduler)
            #   - sync_base_weights_to_active (driven by _after_training)
            # With the default max_concurrency=1, report_progress events
            # queue behind whatever else is in flight. That stalls the
            # scheduler's view of fresh rollout demand between rollouts,
            # which the gap-ratio planner needs to fire promptly to wake
            # engines for rollout N+1.
            max_concurrency=4,
            runtime_env={"env_vars": pipeline_runtime_env_vars},
        )
        .remote(pipeline_id=pipeline_id, pipeline_config=cfg)
    )
    logger.info("[run_miles_dual] mp%d MilesCoordinator created", pipeline_index)

    pipeline = ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=cfg))
    ray.get(pipeline.initialize_pipeline.remote(coordinator_handle=coordinator))
    logger.info(
        "[run_miles_dual] mp%d MilesPipeline.initialize_pipeline complete pipeline_id=%s",
        pipeline_index, pipeline_id,
    )

    return pipeline_id, pipeline_namespace, coordinator, pipeline, args


def main():
    """Dual-pipeline entry. Imports heavy modules lazily so the env-var
    guard above fires before transitive ``import torch`` / ``import sglang``.
    """
    import asyncio
    import logging
    from dataclasses import dataclass, field
    from typing import Any, Optional

    import ray

    from miles.utils.arguments import parse_args
    from miles.utils.logging_utils import configure_logger
    from miles.utils.rlix_train_loop import run_async_train_loop
    from miles.utils.rlix_validation import assert_rlix_topology
    from rlix.pipeline.miles_coordinator import MilesCoordinator
    from rlix.protocol.types import (
        get_coordinator_actor_name,
        get_pipeline_namespace,
    )

    import rlix

    configure_logger()
    logger = logging.getLogger("run_miles_dual")
    base_args = parse_args()

    # F10 startup fail-fast on the BASE args. Per-pipeline arg overrides
    # below preserve the topology shape (just shrink the GPU pool).
    assert_rlix_topology(
        base_args, sglang_config=getattr(base_args, "sglang_config", None)
    )

    # --- Pipeline config dataclass with cluster_device_mappings field ---
    @dataclass
    class MilesPipelineConfig:
        miles_args: Any
        sglang_config: Optional[Any] = None
        verify_model_after_sync: bool = False
        num_gpus_per_node: int = 8
        system_envs: dict = field(default_factory=dict)
        # M11.2 — cluster_device_mappings flow into MilesPipeline so
        # _build_placement_provider can use per-pipeline physical GPUs.
        cluster_device_mappings: dict = field(default_factory=dict)

    # --- Topology: explicit overlap (env-driven) OR fallback disjoint -----
    # Real M11.2 path: set MILES_DUAL_P1_TRAIN / P1_INFER / P2_TRAIN / P2_INFER
    # in the smoke env to drive overlap topology. Fallback (any unset):
    # _split_pools_for_dual produces disjoint pools sized from base_args.
    num_gpus_per_node = int(getattr(base_args, "num_gpus_per_node", 0) or 0)
    if num_gpus_per_node <= 0:
        raise RuntimeError(
            "run_miles_dual.py requires --num-gpus-per-node to set the whole-"
            "machine GPU count."
        )
    overlap_topology = _overlap_pools_from_env(num_gpus_per_node)
    if overlap_topology is not None:
        (p1_train, p1_infer), (p2_train, p2_infer) = overlap_topology
        topology_mode = "OVERLAP (real M11.2)"
    else:
        pool_p1, pool_p2 = _split_pools_for_dual(
            num_gpus_per_node=num_gpus_per_node,
            infer_pool_size=int(base_args.rollout_num_gpus),
        )
        # Disjoint default: train = first N of pool, infer = full pool
        train_size = int(base_args.actor_num_nodes) * int(
            base_args.actor_num_gpus_per_node
        )
        p1_train, p1_infer = list(pool_p1[:train_size]), list(pool_p1)
        p2_train, p2_infer = list(pool_p2[:train_size]), list(pool_p2)
        topology_mode = "DISJOINT (fallback, Option A)"
    overlap_shared = sorted(set(p1_infer) & set(p2_infer))
    logger.info(
        "[run_miles_dual] topology=%s num_gpus_per_node=%d "
        "mp1_train=%s mp1_infer=%s mp2_train=%s mp2_infer=%s overlap=%s",
        topology_mode, num_gpus_per_node,
        p1_train, p1_infer, p2_train, p2_infer, overlap_shared,
    )

    # ---- 1. Connect to RLix; get the orchestrator. -----------------------
    orchestrator = rlix.init(create_if_missing=True)

    # ---- 2. Build both pipelines sequentially. ---------------------------
    # Sequential init avoids racing the RolloutManager construction; the
    # rlix orchestrator's allocate_pipeline_id is itself serialized.
    p1 = _build_pipeline(
        base_args=base_args,
        pipeline_index=1,
        train_mapping=p1_train,
        infer_mapping=p1_infer,
        orchestrator=orchestrator,
        ray=ray,
        MilesCoordinator=MilesCoordinator,
        MilesPipelineConfig=MilesPipelineConfig,
        get_coordinator_actor_name=get_coordinator_actor_name,
        get_pipeline_namespace=get_pipeline_namespace,
        logger=logger,
    )
    p2 = _build_pipeline(
        base_args=base_args,
        pipeline_index=2,
        train_mapping=p2_train,
        infer_mapping=p2_infer,
        orchestrator=orchestrator,
        ray=ray,
        MilesCoordinator=MilesCoordinator,
        MilesPipelineConfig=MilesPipelineConfig,
        get_coordinator_actor_name=get_coordinator_actor_name,
        get_pipeline_namespace=get_pipeline_namespace,
        logger=logger,
    )

    pipelines = [p1, p2]

    # ---- 3. Pull handles for each pipeline. ------------------------------
    handles = []
    for pid, ns, coord, pipe, args in pipelines:
        train_group = ray.get(pipe.get_train_group.remote())
        rollout_manager = ray.get(pipe.get_rollout_manager.remote())
        engine_count = int(ray.get(pipe.get_declared_engine_count.remote()))
        logger.info(
            "[run_miles_dual] handles ready pipeline_id=%s engines=%d",
            pid, engine_count,
        )
        handles.append((pid, ns, coord, pipe, args, train_group, rollout_manager))

    # ---- 4. Drive 2 concurrent rlix_train_loops via asyncio.gather. -----
    async def _run_one_pipeline(idx, pid, pipe, args, train_group, rollout_manager):
        async def _before(step: int) -> None:
            await pipe.before_training.remote(step)

        async def _after(step: int) -> None:
            await pipe.after_training.remote(step)

        async def _release_only(step: int) -> None:
            # R04-F1 cleanup hook: releases actor_train allocation only.
            await pipe.release_train_only.remote(step)

        # Per-rollout step_target = rollout_batch_size. See
        # MilesPipeline.signal_rollout_demand docstring for why pre-signalling
        # demand to the scheduler is required for 4-GPU 2-pipeline full
        # cross-overlap (without it, rollout 2+ hangs when both pipelines
        # release all DP workers between rollouts).
        _step_target = int(getattr(args, "rollout_batch_size", 0) or 0)

        async def _signal_demand(rollout_id: int) -> None:
            if _step_target <= 0:
                return
            await pipe.signal_rollout_demand.remote(rollout_id, _step_target)

        await run_async_train_loop(
            args,
            train_group=train_group,
            rollout_manager=rollout_manager,
            before_step=_before,
            after_step=_after,
            release_only=_release_only,
            signal_demand=_signal_demand,
        )
        logger.info("[run_miles_dual] mp%d training loop complete pipeline_id=%s", idx, pid)

    async def _async_main():
        # F4 fix (m11-review.review-report.md §2): use create_task + wait(
        # FIRST_EXCEPTION) instead of asyncio.gather(...). gather's default
        # semantics propagate the first exception immediately but DO NOT
        # cancel peer coroutines — pipeline B's coroutine continues
        # running orphaned in its Ray actor while the driver tears down.
        # FIRST_EXCEPTION + explicit cancel forces the peer to settle,
        # so Phase 1's try/finally inside run_async_train_loop fires
        # release_only on the CancelledError path and the scheduler
        # ledger stays consistent.
        tasks = [
            asyncio.create_task(
                _run_one_pipeline(i + 1, pid, pipe, args, train_group, rollout_manager)
            )
            for i, (pid, ns, coord, pipe, args, train_group, rollout_manager)
            in enumerate(handles)
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            first_exc = None
            for t in done:
                if t.exception() is not None:
                    first_exc = t.exception()
                    break
            if first_exc is not None:
                for t in pending:
                    t.cancel()
                # Wait for cancelled tasks to settle so release_only fires
                # inside each pipeline's run_async_train_loop finally.
                await asyncio.gather(*pending, return_exceptions=True)
                raise first_exc
        finally:
            # F3 fix (m11-review.review-report.md §2): shutdown_hard MUST
            # fire regardless of how _async_main exits. The prior code
            # ran shutdown after asyncio.run returned cleanly, so a driver
            # crash (mid-loop exception, OOM, KeyboardInterrupt) would
            # skip cleanup and leak the scheduler ledger. F13 hard
            # constraint ("no top-level try/except") is preserved — this
            # try/finally lives INSIDE _async_main and propagates
            # exceptions; only cleanup is added.
            #
            # Construct .remote() refs INSIDE the inner try so a synchronous
            # actor-handle failure (e.g. already-killed actor) is caught
            # by the except clause below, not propagated to mask the
            # original training exception. Codex Phase 7 review MEDIUM.
            try:
                shutdown_refs = [
                    pipe.shutdown_hard.remote() for _, _, _, pipe, _, _, _ in handles
                ]
                ray.get(shutdown_refs, timeout=60.0)
                for pid, _, _, _, _, _, _ in handles:
                    logger.info(
                        "[run_miles_dual] shutdown_hard complete pipeline_id=%s",
                        pid,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[run_miles_dual] shutdown_hard during cleanup failed: %r",
                    exc,
                )

    asyncio.run(_async_main())
    logger.info("[run_miles_dual] both training loops complete; shutting down")


if __name__ == "__main__":
    main()
