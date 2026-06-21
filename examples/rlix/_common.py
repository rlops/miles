"""Shared helpers for the RLix-mode example drivers.

Used by both ``run_miles_rlix.py`` (single pipeline) and
``run_miles_dual.py`` (dual pipeline) so the env guard, pipeline-config
shape, and Ray ``runtime_env`` construction live in one place. This module
imports only the stdlib so it is safe to import before the per-actor
CUDA_VISIBLE_DEVICES guard fires.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# Smoke-only escape hatches forwarded to child actors. Ray runtime_env does
# not inherit the driver's env, so the coordinator + its children only see
# these if we copy them in explicitly.
_FORWARDED_ENV_KEYS = (
    "MILES_TMS_HOOK_MODE",
    "MILES_SKIP_TMS_PAUSE",
    "MILES_SKIP_NODE_PG_PIN",
    "TMS_INIT_ENABLE_CPU_BACKUP",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
)


def require_rlix_control_plane(script_path: str, module_path: str) -> None:
    """Exit unless ``RLIX_CONTROL_PLANE=rlix`` is set.

    Must run before any heavy import (torch / sglang / megatron) so per-actor
    CUDA_VISIBLE_DEVICES can take effect via Ray runtime_env.
    """
    if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
        sys.stderr.write(
            f"{script_path} requires RLIX_CONTROL_PLANE=rlix.\n"
            f"Use train_async.py for standalone runs, or set the env var:\n"
            f"    RLIX_CONTROL_PLANE=rlix python -m {module_path} ...\n"
        )
        sys.exit(2)


@dataclass
class MilesPipelineConfig:
    """Config wrapper passed to MilesCoordinator / MilesPipeline."""

    miles_args: Any
    sglang_config: Optional[Any] = None
    verify_model_after_sync: bool = False
    num_gpus_per_node: int = 8
    # Writeable so MilesCoordinator._inject_pipeline_env_vars can mutate the
    # deepcopy without hitting a frozen-dataclass error.
    system_envs: dict = field(default_factory=dict)
    # Per-pipeline physical GPU mappings; empty in single-pipeline mode.
    cluster_device_mappings: dict = field(default_factory=dict)


def build_pipeline_runtime_env(
    pipeline_id, pipeline_namespace, *, extra: Optional[dict] = None
) -> dict:
    """Build the Ray ``runtime_env`` env_vars for a pipeline's coordinator.

    Sets the identity vars that roll.* asserts on import, forwards PYTHONPATH
    and the smoke-only escape hatches, then merges any ``extra`` overrides.
    """
    env_vars = {
        "PIPELINE_ID": str(pipeline_id),
        "ROLL_RAY_NAMESPACE": pipeline_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
    }
    if pythonpath := os.environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = pythonpath
    for key in _FORWARDED_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env_vars[key] = value
    if extra:
        env_vars.update(extra)
    return env_vars
