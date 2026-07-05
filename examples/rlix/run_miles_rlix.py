"""RLix-mode entry driver — `examples/rlix/run_miles_rlix.py`.

Standalone entry stays at ``train_async.py``; this script is the RLix-mode
entry. ``RLIX_CONTROL_PLANE=rlix`` MUST be set before any heavy import
(``import torch`` / ``import sglang``) resolves so per-actor CVD can be
honored cleanly via Ray ``runtime_env`` (scope F08 — transitive imports
are the hazard, not first-line imports).

Per scope F13 the driver MUST NOT have a top-level ``try / except`` and
MUST NOT call ``ray.shutdown()``: failure semantics = let exceptions
propagate naturally → driver exits → user runs ``ray stop`` to clean up.
"""

from __future__ import annotations

import os
import sys

# F08 / F41 — fail fast if RLix entry is invoked without the env var.
# The check must happen BEFORE any heavy import (torch / sglang /
# megatron) so CVD has a chance to take effect via Ray runtime_env.
if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
    sys.stderr.write(
        "examples/rlix/run_miles_rlix.py requires RLIX_CONTROL_PLANE=rlix.\n"
        "Use train_async.py for standalone runs, or set the env var:\n"
        "    RLIX_CONTROL_PLANE=rlix python -m examples.rlix.run_miles_rlix ...\n"
    )
    sys.exit(2)


def _build_cluster_device_mappings(args) -> dict[str, list[int]]:
    """F8 driver — derive cluster_device_mappings from existing args.

    First-build contiguous-mapping invariant (F35 / C6): train pool is
    ``range(actor_num_nodes * actor_num_gpus_per_node)``; infer pool is
    ``range(rollout_num_gpus)``. Both are zero-based shared (RLix mode
    convention) so train can be a strict subset of infer (partial
    overlap topology). No new device_mapping CLI args are introduced
    (Layer 1 forbidden).
    """
    actor_count = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
    rollout_count = int(args.rollout_num_gpus)
    return {
        "actor_train": list(range(actor_count)),
        "actor_infer": list(range(rollout_count)),
    }


def main():
    """RLix entry. Imports heavy modules lazily so the env-var guard above
    fires before transitive ``import torch`` / ``import sglang``.
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
        RLIX_NAMESPACE,
        get_coordinator_actor_name,
        get_pipeline_namespace,
    )

    import rlix

    configure_logger()
    logger = logging.getLogger("run_miles_rlix")
    args = parse_args()

    # F10 startup fail-fast — verify partial overlap topology + transport
    # constraints BEFORE allocating any GPUs. R08-F1: pass
    # ``args.sglang_config`` so C9 (PD-disaggregation forbidden) fires
    # at the entry path. ``assert_rlix_topology`` also has an internal
    # fallback to ``args.sglang_config`` when the kwarg is None.
    assert_rlix_topology(args, sglang_config=getattr(args, "sglang_config", None))

    cluster_device_mappings = _build_cluster_device_mappings(args)
    logger.info(
        "[run_miles_rlix] F10 startup validation passed; "
        "cluster_device_mappings=%s",
        cluster_device_mappings,
    )

    # MILES-side tracking backends (W&B / TensorBoard / Prometheus). Lazily
    # imported so the smoke run does not pull in wandb (which depends on a
    # newer protobuf than rlix pins). Only initialize when the user opted
    # in via --use-wandb / --use-tensorboard / --use-prometheus.
    if (
        getattr(args, "use_wandb", False)
        or getattr(args, "use_tensorboard", False)
        or getattr(args, "use_prometheus", False)
    ):
        from miles.utils.tracking_utils import init_tracking

        init_tracking(args)

    # ---- 1. Connect to RLix; get the orchestrator. ---------------------------
    # ``rlix.init`` aliases ``rlix.client.client.connect``; with
    # ``create_if_missing=True`` it creates the singleton orchestrator
    # actor on the head node. The orchestrator constructor calls
    # ``_ensure_scheduler_singleton`` which creates and initializes the
    # central scheduler — no separate scheduler bootstrap needed.
    orchestrator = rlix.init(create_if_missing=True)

    # ---- 2. Allocate, register, and admit the pipeline. ----------------------
    pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("miles"))
    pipeline_namespace = get_pipeline_namespace(pipeline_id)
    logger.info(
        "[run_miles_rlix] allocated pipeline_id=%s namespace=%s",
        pipeline_id,
        pipeline_namespace,
    )

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
        "[run_miles_rlix] pipeline registered + admitted pipeline_id=%s",
        pipeline_id,
    )

    # ---- 3. Build the MilesPipelineConfig wrapper. ---------------------------
    @dataclass
    class MilesPipelineConfig:
        miles_args: Any
        sglang_config: Optional[Any] = None
        verify_model_after_sync: bool = False
        num_gpus_per_node: int = 8
        # ``system_envs`` is writeable so MilesCoordinator._inject_pipeline_env_vars
        # can mutate the deepcopy without hitting a frozen-dataclass error.
        system_envs: dict = field(default_factory=dict)

    # ``num_gpus_per_node`` defaults to actor_num_gpus_per_node when the
    # arg is absent; RLix uses this for placement-group bundle sizing.
    cfg = MilesPipelineConfig(
        miles_args=args,
        sglang_config=getattr(args, "sglang_config", None),
        verify_model_after_sync=bool(getattr(args, "verify_model_after_sync", False)),
        num_gpus_per_node=int(
            getattr(args, "num_gpus_per_node", None) or args.actor_num_gpus_per_node
        ),
        system_envs={},
    )

    # ---- 4. Create the named MilesCoordinator actor. -------------------------
    # ROLL constants.py asserts ROLL_RAY_NAMESPACE + PIPELINE_ID are set when
    # ``RLIX_CONTROL_PLANE=rlix`` BEFORE roll.* is imported. The coordinator's
    # __init__ lazily imports ``roll.distributed.scheduler.resource_manager``
    # so we propagate the identity vars via Ray runtime_env. Also set them on
    # the driver's own env so any later in-driver roll import (e.g. via the
    # placement provider) finds them.
    pipeline_runtime_env_vars = {
        "PIPELINE_ID": str(pipeline_id),
        "ROLL_RAY_NAMESPACE": pipeline_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
    }
    if pythonpath := os.environ.get("PYTHONPATH"):
        pipeline_runtime_env_vars["PYTHONPATH"] = pythonpath
    # Forward smoke-only escape hatches so MilesCoordinator + child actors
    # see them when reading os.environ (Ray runtime_env does not propagate
    # the parent driver's env by default).
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
    os.environ["PIPELINE_ID"] = str(pipeline_id)
    os.environ["ROLL_RAY_NAMESPACE"] = pipeline_namespace
    # Name + namespace must match how the rlix scheduler resolves this
    # coordinator, or its resize_infer/shrink_engines RPCs fail to resolve
    # the actor. Both sides build the name via get_coordinator_actor_name.
    coordinator = (
        ray.remote(MilesCoordinator)
        .options(
            name=get_coordinator_actor_name(pipeline_id),
            namespace=pipeline_namespace,
            lifetime="detached",
            num_cpus=0.01,
            runtime_env={"env_vars": pipeline_runtime_env_vars},
        )
        .remote(pipeline_id=pipeline_id, pipeline_config=cfg)
    )
    logger.info(
        "[run_miles_rlix] MilesCoordinator created pipeline_id=%s",
        pipeline_id,
    )

    # ---- 5. Create the per-pipeline MilesPipeline actor + initialize. --------
    pipeline = ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=cfg))
    ray.get(pipeline.initialize_pipeline.remote(coordinator_handle=coordinator))
    logger.info(
        "[run_miles_rlix] MilesPipeline.initialize_pipeline complete pipeline_id=%s",
        pipeline_id,
    )

    # ---- 6. Pull train_group + rollout_manager handles via the new accessors.
    train_group = ray.get(pipeline.get_train_group.remote())
    rollout_manager = ray.get(pipeline.get_rollout_manager.remote())
    declared_engine_count = int(ray.get(pipeline.get_declared_engine_count.remote()))
    logger.info(
        "[run_miles_rlix] pulled handles train_group=ok rollout_manager=ok engines=%d",
        declared_engine_count,
    )

    # ---- 7. Run the async training loop. ------------------------------------
    # set_rollout_manager and base v=-1 sync are now driven inside
    # MilesPipeline._init_phase_b_infer (Phase B steps 4b and 7), so the
    # driver only needs to start the per-step loop here.
    async def _async_main():
        async def _before(step: int) -> None:
            await pipeline.before_training.remote(step)

        async def _after(step: int) -> None:
            await pipeline.after_training.remote(step)

        async def _release_only(step: int) -> None:
            # R04-F1 cleanup hook: releases actor_train allocation only.
            await pipeline.release_train_only.remote(step)

        try:
            await run_async_train_loop(
                args,
                train_group=train_group,
                rollout_manager=rollout_manager,
                before_step=_before,
                after_step=_after,
                release_only=_release_only,
            )
            logger.info(
                "[run_miles_rlix] training loop complete pipeline_id=%s",
                pipeline_id,
            )
        finally:
            # F3 fix (m11-review.review-report.md §2): shutdown_hard MUST
            # fire regardless of how _async_main exits. The prior code
            # ran shutdown only on the success path; a mid-loop crash
            # (OOM, KeyboardInterrupt) would skip cleanup and leak the
            # scheduler ledger. F13 hard constraint ("no top-level
            # try/except") is preserved — this try/finally lives INSIDE
            # _async_main and propagates exceptions; only cleanup is
            # added.
            try:
                ray.get(pipeline.shutdown_hard.remote(), timeout=60.0)
                logger.info(
                    "[run_miles_rlix] shutdown_hard complete pipeline_id=%s — exiting",
                    pipeline_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[run_miles_rlix] shutdown_hard during cleanup failed pipeline_id=%s: %r",
                    pipeline_id, exc,
                )

    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
