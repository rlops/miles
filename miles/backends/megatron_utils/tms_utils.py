"""torch_memory_saver (tms) hook-mode safety helpers.

tms exposes two hook modes:

- ``preload`` (its default): an ``LD_PRELOAD`` libc-malloc interposer. Broadest
  catchment (every allocation, incl. NCCL / raw ``cudaMalloc``), but segfaults on
  Blackwell-class GPUs under pre-CUDA-13 wheels (observed on 12.9) — a raw
  SIGSEGV with no Python traceback, typically on the first allocation (e.g.
  ``build_cpu_bucket_cache``). Verified fixed on cu130 wheels.
- ``torch``: PyTorch's ``CUDAPluggableAllocator``. Narrower (only torch
  allocations), but stable across architectures.

This module keeps the mode-resolution + arch-compatibility guard in one small,
torch-only place so it can be unit-tested without importing the heavy Megatron
actor stack.
"""

from __future__ import annotations

import logging
import os

import torch
from packaging.version import parse

logger = logging.getLogger(__name__)

# Blackwell-class GPUs (datacenter B100/B200 == sm_100, consumer RTX 50xx ==
# sm_120) have compute-capability major >= 10. ``preload`` segfaults there under
# pre-CUDA-13 wheels (observed on 12.9; verified fixed on cu130); ``torch`` mode
# is required on those older stacks.
TMS_PRELOAD_UNSAFE_CC_MAJOR = 10

# preload verified safe on Blackwell from CUDA 13 wheels (RTX 5090 +
# torch 2.11.0+cu130, 2026-07-05 audit: mock pause/resume + dual E2E both
# pass); cu12.x wheels keep the historical tms 0.0.9 segfault.
TMS_PRELOAD_SAFE_CUDA_MAJOR = 13

# Escape hatch: set to "1" to proceed with preload on Blackwell anyway (e.g. once
# a fixed tms/CUDA build is confirmed). Mirrors the repo's MILES_SKIP_* knobs.
TMS_ALLOW_PRELOAD_ON_BLACKWELL_ENV = "MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL"


def resolve_tms_hook_mode(env_mode: str | None) -> str:
    """Return the hook mode torch_memory_saver will actually use.

    An explicit, valid ``MILES_TMS_HOOK_MODE`` wins; anything else (unset or
    unrecognised) leaves tms on its own default, which is ``"preload"``.
    """
    return env_mode if env_mode in ("torch", "preload") else "preload"


def _torch_cuda_major() -> int | None:
    """Major version of the CUDA runtime the torch wheel was built against.

    ``torch.version.cuda`` is ``None`` on CPU-only builds; treat that (or an
    unparseable value) as "unknown" and return ``None`` so callers stay
    conservative.
    """
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version is None:
        return None
    try:
        return parse(str(cuda_version)).major
    except Exception:  # unparseable (e.g. vendor-patched string) -> unknown
        return None


def assert_tms_hook_mode_matches_arch(env_mode: str | None) -> None:
    """Fail fast when the resolved tms hook mode will crash on this GPU.

    On Blackwell-class GPUs the ``"preload"`` hook segfaults inside libc on the
    first allocation under pre-CUDA-13 wheels (observed on 12.9; verified fixed
    on cu130). Because the crash is a tracebackless SIGSEGV, we refuse to
    proceed on cu12.x/unknown stacks and tell the operator exactly which knob
    to set. Torch wheels built against CUDA >= 13 are allowed through. Set
    ``MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL=1`` to bypass on an older stack once
    a fixed tms/CUDA combination is confirmed.

    Raises ``RuntimeError`` (not bare ``assert``) so it remains active under
    ``python -O``.
    """
    if not torch.cuda.is_available():
        return
    if resolve_tms_hook_mode(env_mode) != "preload":
        return
    major, minor = torch.cuda.get_device_capability()
    if major < TMS_PRELOAD_UNSAFE_CC_MAJOR:
        return
    cuda_major = _torch_cuda_major()
    if cuda_major is not None and cuda_major >= TMS_PRELOAD_SAFE_CUDA_MAJOR:
        logger.info(
            "torch_memory_saver hook_mode 'preload' allowed on Blackwell (sm_%d%d) "
            "because torch wheel CUDA %s >= %d (segfault verified fixed on cu130).",
            major,
            minor,
            torch.version.cuda,
            TMS_PRELOAD_SAFE_CUDA_MAJOR,
        )
        return
    if os.environ.get(TMS_ALLOW_PRELOAD_ON_BLACKWELL_ENV) == "1":
        logger.warning(
            "torch_memory_saver hook_mode 'preload' on Blackwell (sm_%d%d) is "
            "known to segfault under pre-CUDA-13 wheels (torch wheel CUDA: %s); "
            "proceeding anyway because %s=1.",
            major,
            minor,
            getattr(torch.version, "cuda", None),
            TMS_ALLOW_PRELOAD_ON_BLACKWELL_ENV,
        )
        return
    raise RuntimeError(
        f"torch_memory_saver hook_mode resolved to 'preload' on a Blackwell-class "
        f"GPU ({torch.cuda.get_device_name()}, compute capability sm_{major}{minor}) "
        f"with a pre-CUDA-13 torch wheel (torch.version.cuda="
        f"{getattr(torch.version, 'cuda', None)!r}). "
        f"'preload' uses an LD_PRELOAD libc-malloc hook that segfaults on Blackwell "
        f"with pre-CUDA-13 wheels — a raw SIGSEGV with no Python traceback, "
        f"typically during build_cpu_bucket_cache. "
        f"Fixes: (a) export MILES_TMS_HOOK_MODE=torch before launch "
        f"(MILES_TMS_HOOK_MODE was {'unset' if env_mode is None else repr(env_mode)}); "
        f"(b) upgrade to a cu13+ torch wheel (preload is verified safe there); or "
        f"(c) after confirming a fixed tms/CUDA build, set "
        f"{TMS_ALLOW_PRELOAD_ON_BLACKWELL_ENV}=1."
    )
