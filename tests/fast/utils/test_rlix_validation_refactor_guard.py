"""Unit tests for the C24 guard: RLix mode + experimental rollout refactor.

The refactored ``call_rollout_function`` path does not thread ``rlix_hooks``
through to the rollout function, so under RLix the F9 progress channel would
silently die. ``assert_rlix_topology`` must fail fast on the combination.

Imports only ``rlix_validation`` (stdlib-only), so these run anywhere with no
torch / GPU dependency.
"""

from types import SimpleNamespace

import pytest

from miles.utils.rlix_validation import assert_rlix_topology


def _valid_rlix_args() -> SimpleNamespace:
    """Minimal args that satisfy C1-C23 for a 2-train / 4-infer topology."""
    return SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        rollout_function_path="examples.fully_async.fully_async_rollout.generate_rollout_fully_async",
        critic_model_path=None,
        reward_model_path=None,
        sglang_secondary_server_count=0,
        offload_train=True,
        async_save=False,
        sglang_data_parallel_size=1,
        expert_model_parallel_size=1,
        moe_router_topk=0,
        rollout_force_stream=False,
        model_update_transport="cpu_serialize",
        num_gpus_per_node=8,
        miles_router_middleware_paths=[],
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        miles_model_update_bucket_size_mb=512,
        mem_fraction_static=0.9,
        sglang_config=None,
    )


def _patch_shm_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the S3a-2 /dev/shm capacity check pass on hosts without tmpfs."""
    import os as _os
    import shutil as _shutil

    real_isdir = _os.path.isdir
    monkeypatch.setattr(
        _os.path, "isdir", lambda p: True if p == "/dev/shm" else real_isdir(p)
    )
    real_access = _os.access
    monkeypatch.setattr(
        _os, "access", lambda p, mode: True if p == "/dev/shm" else real_access(p, mode)
    )
    real_disk_usage = _shutil.disk_usage
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda p: SimpleNamespace(total=2**34, used=0, free=2**34)
        if p == "/dev/shm"
        else real_disk_usage(p),
    )


def test_rlix_plus_experimental_refactor_raises_c24(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
    with pytest.raises(RuntimeError, match="C24"):
        assert_rlix_topology(_valid_rlix_args())


def test_rlix_without_refactor_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    monkeypatch.delenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", raising=False)
    _patch_shm_available(monkeypatch)
    assert_rlix_topology(_valid_rlix_args())  # must not raise


def test_standalone_with_refactor_not_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard is RLix-scoped: standalone runs keep the experimental
    # refactor available. (assert_rlix_topology is normally not called on
    # the standalone path; this pins the guard's env-scoping regardless.)
    monkeypatch.delenv("RLIX_CONTROL_PLANE", raising=False)
    monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
    args = _valid_rlix_args()
    args.model_update_transport = "cuda_ipc"  # C11 is rlix-scoped too
    assert_rlix_topology(args)  # must not raise C24