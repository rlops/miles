"""F10 startup validation for RLix-mode partial-overlap topology.

This module defines the C1–C23 fail-fast guards specified in
``plans/miles-port-unified-plan.md`` §Feature 10 plus the F11 standalone-entry
helper ``train_devices_subset_of_infer`` and the ``is_rlix_mode`` /
``async_generation_enabled`` predicates.

Design notes
------------
- All derived quantities (``train_devices``, ``infer_devices``,
  ``infer_engine_count``) are computed from existing MILES args
  (``actor_num_nodes``, ``actor_num_gpus_per_node``, ``rollout_num_gpus``,
  ``rollout_num_gpus_per_engine``). NO new device-mapping CLI args are introduced
  (cf. plan §3.1 Layer 1 forbidden).
- ``assert_rlix_topology`` is the single entry callers (``MilesPipeline``,
  ``train_async`` standalone guard) use. It accepts an optional
  ``sglang_config`` for the PD-disaggregation check; if not supplied, the
  check is skipped (caller responsibility).
- All asserts raise :class:`RuntimeError` (not bare :func:`assert`) so they
  remain active under ``python -O``.
- Side-effect-free helpers (``train_devices_subset_of_infer``,
  ``single_updateable_model_and_server``, ``async_generation_enabled``,
  ``is_rlix_mode``) are exported for reuse by ``train_async.py`` (F11) and the
  RLix entry driver (F8/F11).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Predicates / helpers — used by F10 validators and by F11 standalone guard
# -----------------------------------------------------------------------------


def is_rlix_mode() -> bool:
    """Return True iff the current process is running under the RLix scheduler.

    Mirrors ROLL's ``DO_TIME_SHARING`` flag (``roll/utils/constants.py``).
    The env var is set by the RLix entry driver (``run_miles_rlix.py``) before
    any heavy import resolves; standalone (``train_async.py``) leaves it unset.
    """
    return os.environ.get("RLIX_CONTROL_PLANE") == "rlix"


def apply_rlix_offload_defaults(args: Any) -> None:
    """Force rollout offload on under RLix so SGLang can release VRAM on shrink.

    RLix time-sharing requires ``release_memory_occupation`` to actually return
    VRAM to the OS, which only happens when each engine launched with
    ``enable_memory_saver=True`` — and that flag is gated by
    ``args.offload_rollout`` (see ``backends/sglang_utils/sglang_engine.py``).
    Forcing it on under RLix means operators don't have to remember
    ``--offload-rollout``; without it, ``release_memory_occupation`` is a silent
    no-op and the first ``shrink_engines`` OOMs (M11.1 attempt-5 bug).

    Mutates ``args`` in place to match the surrounding ``miles_validate_args``
    normalization style. Idempotent: no-op when not in RLix mode or when offload
    is already enabled. ``offload_train`` is intentionally left untouched — that
    is a separate train-side knob with its own cost/benefit.
    """
    if is_rlix_mode() and not getattr(args, "offload_rollout", False):
        logger.info(
            "RLix mode (RLIX_CONTROL_PLANE=rlix): forcing offload_rollout=True so "
            "SGLang launches with enable_memory_saver and shrink_engines can "
            "release VRAM for actor_train."
        )
        args.offload_rollout = True


def _train_devices(args: Any) -> set[int]:
    n_nodes = int(getattr(args, "actor_num_nodes", 1))
    per_node = int(getattr(args, "actor_num_gpus_per_node", 0))
    return set(range(n_nodes * per_node))


def _infer_devices(args: Any) -> set[int]:
    rollout_num_gpus = getattr(args, "rollout_num_gpus", None)
    if rollout_num_gpus is None:
        return set()
    return set(range(int(rollout_num_gpus)))


def train_devices_subset_of_infer(args: Any) -> bool:
    """RLix-mode helper: detect partial-overlap topology from zero-based indices.

    Returns True iff zero-based ``train_devices`` is a strict (proper) subset of
    zero-based ``infer_devices``. **Only meaningful in RLix mode**, where the
    scheduler shares a single zero-based index space across train + infer pools.

    NOT a safe predicate for the standalone (``train_async.py``) path:
    ``create_placement_groups`` allocates rollout bundles at offset
    ``actor_num_nodes * actor_num_gpus_per_node``, so a standalone non-colocated
    config such as 4 actor GPUs + ``--rollout-num-gpus 8`` is fully disjoint
    even though the zero-based sets overlap. The standalone fail-fast in
    ``train_async.py`` therefore relies only on the ``RLIX_CONTROL_PLANE``
    env-var check; partial-overlap detection happens at the RLix entry driver
    via :func:`assert_rlix_topology` (C1).
    """
    train = _train_devices(args)
    infer = _infer_devices(args)
    if not train or not infer:
        return False
    return train.issubset(infer) and train != infer


def async_generation_enabled(args: Any) -> bool:
    """C3: partial overlap requires the fully-async rollout function path.

    Detected from ``args.rollout_function_path`` ending in
    ``fully_async_rollout.generate_rollout_fully_async``. Both the function
    path and a few other rollout-mode args may be absent on legacy configs.
    """
    fn_path = getattr(args, "rollout_function_path", "") or ""
    return fn_path.endswith("fully_async_rollout.generate_rollout_fully_async")


def single_updateable_model_and_server(args: Any) -> bool:
    """C19: reject configs that drive weight updates to multiple model/server slots.

    First-build supports exactly one updateable Megatron actor and one SGLang
    server group. critic / reward / RM / multi-server are out of scope; the
    sender side of ``MilesModelUpdateService`` is single-slot.
    """
    return (
        getattr(args, "critic_model_path", None) is None
        and getattr(args, "reward_model_path", None) is None
        and int(getattr(args, "sglang_secondary_server_count", 0) or 0) == 0
    )


def _infer_engine_count(args: Any) -> int:
    rollout_num_gpus = int(getattr(args, "rollout_num_gpus", 0) or 0)
    per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    if per_engine <= 0:
        return 0
    return rollout_num_gpus // per_engine


def _topology_has_non_colocate_engines(args: Any) -> bool:
    """S2: any infer engine GPU outside the train pool requires NCCL broadcast.

    Returns True iff at least one infer GPU is NOT a train GPU. NCCL broadcast
    (non-colocate path) requires per-bucket H2D staging on the cache_owner GPU,
    so the bucket-size + scratch budget must fit within post-wake free VRAM.
    """
    train = _train_devices(args)
    infer = _infer_devices(args)
    return bool(infer - train)


def _estimate_post_wake_free_vram(args: Any) -> int:
    """Conservative ceiling for post-wake free VRAM in bytes.

    SGLang weight + KV cache + cuda_graph occupy most of GPU memory after wake.
    Without a way to query the device at startup, we approximate from
    ``mem_fraction_static``: free fraction = 1 - mem_fraction_static, applied
    to a worst-case 80 GiB H100. Caller may pass a more precise estimate via
    ``args.miles_post_wake_free_vram_bytes`` if known.
    """
    override = getattr(args, "miles_post_wake_free_vram_bytes", None)
    if override is not None:
        return int(override)
    mem_fraction = float(getattr(args, "mem_fraction_static", 0.9) or 0.9)
    free_fraction = max(0.0, 1.0 - mem_fraction)
    return int(80 * (1024**3) * free_fraction)


# -----------------------------------------------------------------------------
# Aggregate validator — single entry called by MilesPipeline.initialize_pipeline
# -----------------------------------------------------------------------------


def assert_rlix_topology(args: Any, sglang_config: Any | None = None) -> None:
    """F10 startup fail-fast — verify partial-overlap topology preconditions.

    Raises :class:`RuntimeError` on any C1–C23 violation. Designed to run before
    any actor / engine spin-up so misconfiguration produces a single
    actionable error instead of a mid-init crash with a confusing stack.

    Parameters
    ----------
    args
        Parsed MILES args namespace.
    sglang_config
        Optional :class:`SglangConfig`. If unset, C9 falls back to
        ``args.sglang_config`` (set by ``arguments.py``) so the
        PD-disaggregation guard fires regardless of caller plumbing. The
        explicit argument exists for callers that have already built /
        mutated a separate config object and want to validate against
        it.
    """
    # Resolve the SglangConfig used for C9 once. Prefer an explicitly
    # supplied config; otherwise fall back to args.sglang_config so the
    # entry-driver path doesn't have to pre-build one (R08-F1 fix —
    # without this, callers passing sglang_config=None would silently
    # bypass C9, leaving PD-disaggregation configs unenforced).
    effective_sglang_config = sglang_config
    if effective_sglang_config is None:
        effective_sglang_config = getattr(args, "sglang_config", None)
    train = _train_devices(args)
    infer = _infer_devices(args)
    engine_count = _infer_engine_count(args)
    per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    transport = getattr(args, "model_update_transport", "cuda_ipc")

    # --- C1: train ⊂ infer (partial overlap)
    if not train.issubset(infer):
        raise RuntimeError(
            f"C1: partial overlap requires train_devices ⊂ infer_devices "
            f"(train={sorted(train)}, infer={sorted(infer)})"
        )

    # --- C2: infer_engine_count >= 2
    if engine_count < 2:
        raise RuntimeError(
            f"C2: partial overlap requires >= 2 inference engines (got {engine_count}; "
            f"derived from rollout_num_gpus / rollout_num_gpus_per_engine = "
            f"{int(getattr(args, 'rollout_num_gpus', 0))} / {per_engine})"
        )

    # --- C3: fully-async generation enabled
    if not async_generation_enabled(args):
        raise RuntimeError(
            "C3: partial overlap requires fully-async rollout (rollout_function_path "
            "must end with fully_async_rollout.generate_rollout_fully_async)"
        )

    # --- C4: at least 1 inference engine fully outside the train pool
    non_overlap_size = len(infer - train)
    if non_overlap_size < per_engine:
        raise RuntimeError(
            f"C4: at least one full inference engine ({per_engine} GPUs) must stay "
            f"active after a worst-case shrink (got {non_overlap_size} non-overlap GPUs)"
        )

    # --- C19: single updateable model + single SGLang server group
    if not single_updateable_model_and_server(args):
        raise RuntimeError(
            "C19: RLix MILES first build requires single updateable model + single "
            "SGLang server group (critic / reward / RM / multi-server out of scope)"
        )

    # --- C5: offload_train must be True
    if not bool(getattr(args, "offload_train", False)):
        raise RuntimeError(
            "C5: RLix-mode partial overlap requires offload_train=True. Without it, "
            "actor_train cannot release the overlap GPU after each step → OOM at "
            "infer wake_up."
        )

    # --- M7 / C12: async_save not supported in first build
    if bool(getattr(args, "async_save", False)):
        raise RuntimeError(
            "C12: first build does not support args.async_save — background ckpt "
            "flush races with actor.sleep() torch_memory_saver.pause() and "
            "segfaults. Implement maybe_finalize_async_save(blocking=True) + "
            "cuda.synchronize() in MegatronTrainRayActor.sleep() prologue as a "
            "follow-up."
        )

    # --- C6: rollout_num_gpus divisibility
    rollout_num_gpus = int(getattr(args, "rollout_num_gpus", 0) or 0)
    if rollout_num_gpus % per_engine != 0:
        raise RuntimeError(
            f"C6: rollout_num_gpus ({rollout_num_gpus}) must divide evenly by "
            f"rollout_num_gpus_per_engine ({per_engine}); otherwise scheduler "
            f"placement_group world_size and MILES engine count diverge."
        )

    # --- C7: sglang dp == 1
    if int(getattr(args, "sglang_data_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            f"C7: RLix mode requires sglang_data_parallel_size == 1 "
            f"(got {getattr(args, 'sglang_data_parallel_size', None)!r})"
        )

    # --- C9: PD disaggregation forbidden. Reads ``effective_sglang_config``
    # which falls back to ``args.sglang_config`` when no explicit config
    # is supplied — guarantees the guard fires from the entry-driver
    # path even when sglang_config kwarg is None (R08-F1 fix).
    if effective_sglang_config is not None and bool(
        getattr(effective_sglang_config, "has_pd_disaggregation", False)
    ):
        raise RuntimeError("C9: PD disaggregation is out of scope for this milestone")

    # --- C8: MoE / EP forbidden
    if int(getattr(args, "expert_model_parallel_size", 1) or 1) != 1:
        raise RuntimeError(
            "C8: MoE / EP is out of scope; F4 CPU bucket cache covers dense Megatron only"
        )
    if int(getattr(args, "moe_router_topk", 0) or 0) != 0:
        raise RuntimeError("C8: MoE configs not allowed in RLix mode")

    # --- C10: streaming generate forbidden (router metadata injection requires JSON body)
    if bool(getattr(args, "rollout_force_stream", False)):
        raise RuntimeError(
            "C10: RLix mode requires non-streaming generate; metadata injection "
            "requires a JSON response body"
        )

    # --- C11: M11.1 RLix mode forces cpu_serialize transport
    if is_rlix_mode():
        if transport != "cpu_serialize":
            raise RuntimeError(
                f"C11: M11.1 RLix mode forces model_update_transport='cpu_serialize' "
                f"(got {transport!r}). cuda_ipc colocate adapter is M11.6 follow-up."
            )

    # --- C7-engine: rollout_num_gpus_per_engine <= num_gpus_per_node (cross-node engine forbidden)
    num_gpus_per_node = int(getattr(args, "num_gpus_per_node", 8) or 8)
    if per_engine > num_gpus_per_node:
        raise RuntimeError(
            f"C7-engine: rollout_num_gpus_per_engine ({per_engine}) must be <= "
            f"num_gpus_per_node ({num_gpus_per_node}); cross-node single rollout "
            f"engine is out of scope (M11.3 was skipped)."
        )

    # --- C6-mapping: contiguous ordered infer device mapping (first build)
    infer_device_mapping = sorted(infer)
    if infer_device_mapping != list(range(len(infer_device_mapping))):
        raise RuntimeError(
            f"C6-mapping: RLix MILES first build requires sorted contiguous "
            f"infer_device_mapping starting at 0 (got {infer_device_mapping}); "
            f"non-contiguous / custom ordering is a follow-up adapter (F12 / A18)."
        )
    for engine_index, start in enumerate(range(0, len(infer_device_mapping), per_engine)):
        group = infer_device_mapping[start : start + per_engine]
        expected = list(range(group[0], group[0] + per_engine))
        if group != expected:
            raise RuntimeError(
                f"C6-mapping: infer engine {engine_index} must occupy contiguous "
                f"GPUs in first build; got {group}, expected {expected}"
            )

    # --- C17: RLix mode disables RadixTreeMiddleware (partial_rollout + radix_tree forbidden)
    middleware_paths = getattr(args, "miles_router_middleware_paths", None) or []
    if any("RadixTreeMiddleware" in str(p) for p in middleware_paths):
        raise RuntimeError(
            "C17: RLix mode disables RadixTreeMiddleware; partial_rollout + "
            "radix_tree compatibility is follow-up after the main path stabilizes"
        )

    # --- C13/C14: Megatron internal parallelism divisibility
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    ep = int(getattr(args, "expert_model_parallel_size", 1) or 1)
    parallelism_product = tp * pp * cp * ep
    if parallelism_product > 0 and len(train) % parallelism_product != 0:
        raise RuntimeError(
            f"C13: train device count ({len(train)}) must divide evenly by "
            f"tp*pp*cp*ep ({parallelism_product})"
        )

    # --- S2: bucket size <= post-wake free VRAM (NCCL broadcast / cuda_ipc paths)
    bucket_size_bytes = (
        int(getattr(args, "miles_model_update_bucket_size_mb", 512) or 512) * 1024 * 1024
    )
    has_gpu_staging = _topology_has_non_colocate_engines(args) or transport == "cuda_ipc"
    if has_gpu_staging:
        post_wake_free = _estimate_post_wake_free_vram(args)
        scratch = 256 * 1024 * 1024
        if post_wake_free > 0 and bucket_size_bytes + scratch >= post_wake_free:
            raise RuntimeError(
                f"S2: bucket_size ({bucket_size_bytes} B) + transport scratch "
                f"({scratch} B) exceeds estimated post-wake free VRAM "
                f"({post_wake_free} B); reduce --miles-model-update-bucket-size-mb"
            )

    # --- S3a-2: cpu_serialize requires writable /dev/shm with enough capacity
    if transport == "cpu_serialize":
        if not (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)):
            raise RuntimeError(
                "S3a-2: cpu_serialize transport requires a writable /dev/shm; the "
                "container may need --shm-size (Docker) or a tmpfs mount."
            )
        try:
            shm_free = shutil.disk_usage("/dev/shm").free
        except OSError as exc:
            raise RuntimeError(f"S3a-2: cannot stat /dev/shm: {exc}") from exc
        required = bucket_size_bytes + 256 * 1024 * 1024
        if shm_free < required:
            raise RuntimeError(
                f"S3a-2: /dev/shm free space ({shm_free} B) < required "
                f"({required} B = bucket_size + 256 MiB margin); increase "
                f"--shm-size or reduce --miles-model-update-bucket-size-mb"
            )

    logger.info(
        "F10 startup validation passed (engines=%d, per_engine=%d, train=%d, "
        "infer=%d, overlap=%d)",
        engine_count,
        per_engine,
        len(train),
        len(infer),
        len(train & infer),
    )


def assert_partial_overlap_standalone_safe(args: Any) -> None:
    """Deprecated no-op kept for callsite stability.

    Originally intended to refuse partial-overlap topologies under the
    standalone entry. Removed because the standalone ``create_placement_groups``
    path offsets the rollout pool by ``actor_num_nodes * actor_num_gpus_per_node``,
    so zero-based ``train ⊂ infer`` is also true for valid disjoint
    standalone configs (e.g. 4 actor GPUs + ``--rollout-num-gpus 8``).
    Auto-classifying intent from args alone is unsafe; the env-var guard
    (``RLIX_CONTROL_PLANE=rlix``) is the only reliable signal at the standalone
    entry. RLix-mode partial-overlap correctness is enforced by
    :func:`assert_rlix_topology` (C1) inside the RLix entry driver.
    """
    # Intentional no-op — see docstring.
    del args
    return None


__all__ = [
    "is_rlix_mode",
    "train_devices_subset_of_infer",
    "async_generation_enabled",
    "single_updateable_model_and_server",
    "assert_rlix_topology",
    "assert_partial_overlap_standalone_safe",
]
