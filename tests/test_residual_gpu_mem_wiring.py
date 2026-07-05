from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlix_drivers_forward_residual_gpu_mem_env_var() -> None:
    for relpath in (
        "examples/rlix/run_miles_rlix.py",
        "examples/rlix/run_miles_dual.py",
    ):
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert (
            '"MILES_MAX_RESIDUAL_GPU_MEM_GB"' in source
        ), f"{relpath} must forward residual threshold env into runtime_env"


def test_shrink_logs_sglang_residual_diagnostics() -> None:
    source = (REPO_ROOT / "miles" / "ray" / "rollout.py").read_text(
        encoding="utf-8"
    )
    # Miles logs SGLang attribution diagnostics; the hard whole-GPU gate runs in RLix.
    assert "log_post_sleep_residual_diagnostics" in source
    assert "post-sleep SGLang residual diagnostics" in source
    assert "whole-GPU hard gate runs in RLix" in source
