"""Unit tests for tms hook-mode / GPU-arch safety guard.

Imports only ``tms_utils`` (torch-only) so the heavy Megatron actor stack is not
pulled in. GPU capability is monkeypatched, so these run on any machine.
"""

import pytest
import torch

from miles.backends.megatron_utils.tms_utils import (
    assert_tms_hook_mode_matches_arch,
    resolve_tms_hook_mode,
)


@pytest.mark.parametrize(
    "env_mode, expected",
    [
        (None, "preload"),  # unset -> tms default
        ("", "preload"),  # empty -> tms default
        ("bogus", "preload"),  # unrecognised -> tms default
        ("preload", "preload"),
        ("torch", "torch"),
    ],
)
def test_resolve_hook_mode(env_mode, expected):
    assert resolve_tms_hook_mode(env_mode) == expected


def _patch_gpu(monkeypatch, *, available=True, cc=(8, 9), name="NVIDIA L4", cuda_version="12.9"):
    # cuda_version default "12.9" preserves the historical (pre-cu13-guard) semantics
    # and keeps the raise-tests deterministic on any host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cc)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: name)
    monkeypatch.setattr(torch.version, "cuda", cuda_version, raising=False)


# (major, minor) for Blackwell-class parts: B100/B200 == sm_100, RTX 50xx == sm_120
@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
# all of these resolve to preload, which is the unsafe mode on Blackwell + cu12.x
@pytest.mark.parametrize("env_mode", [None, "preload", "bogus"])
def test_raises_on_preload_blackwell(monkeypatch, cc, env_mode):
    monkeypatch.delenv("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL", raising=False)
    _patch_gpu(monkeypatch, cc=cc, name="NVIDIA B200", cuda_version="12.9")
    with pytest.raises(RuntimeError, match="MILES_TMS_HOOK_MODE=torch"):
        assert_tms_hook_mode_matches_arch(env_mode)


# preload verified safe on Blackwell from cu13 wheels (RTX 5090 + torch cu130)
@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
@pytest.mark.parametrize("cuda_version", ["13.0", "13.1"])
# None resolves to preload too — the guard must allow both spellings on cu13+
@pytest.mark.parametrize("env_mode", [None, "preload"])
def test_no_raise_preload_blackwell_cu13(monkeypatch, cc, cuda_version, env_mode):
    monkeypatch.delenv("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL", raising=False)
    _patch_gpu(monkeypatch, cc=cc, name="NVIDIA B200", cuda_version=cuda_version)
    assert_tms_hook_mode_matches_arch(env_mode)  # cu13+ wheel: must not raise


# unknown wheel CUDA (CPU build or unparseable string): stay conservative -> raise
@pytest.mark.parametrize("cuda_version", [None, "not-a-version"])
def test_raises_on_preload_blackwell_unknown_cuda(monkeypatch, cuda_version):
    monkeypatch.delenv("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL", raising=False)
    _patch_gpu(monkeypatch, cc=(12, 0), name="NVIDIA RTX 5090", cuda_version=cuda_version)
    with pytest.raises(RuntimeError, match="MILES_TMS_HOOK_MODE=torch"):
        assert_tms_hook_mode_matches_arch("preload")


# V100 (sm_70), A100 (sm_80), L4/Ada (sm_89), H100 (sm_90): preload is fine
@pytest.mark.parametrize("cc", [(7, 0), (8, 0), (8, 9), (9, 0)])
def test_no_raise_preload_pre_blackwell(monkeypatch, cc):
    _patch_gpu(monkeypatch, cc=cc, cuda_version="12.9")
    assert_tms_hook_mode_matches_arch("preload")  # must not raise
    assert_tms_hook_mode_matches_arch(None)  # unset -> preload, still pre-Blackwell


@pytest.mark.parametrize("cc", [(10, 0), (12, 0)])
def test_torch_mode_always_safe(monkeypatch, cc):
    _patch_gpu(monkeypatch, cc=cc, name="NVIDIA B200")
    assert_tms_hook_mode_matches_arch("torch")  # torch mode never raises, even on Blackwell


def test_escape_hatch_allows_preload_on_blackwell(monkeypatch):
    # escape hatch still matters on cu12.x/unknown stacks
    monkeypatch.setenv("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL", "1")
    _patch_gpu(monkeypatch, cc=(12, 0), name="NVIDIA RTX 5090", cuda_version="12.9")
    assert_tms_hook_mode_matches_arch("preload")  # bypassed -> no raise


def test_no_cuda_no_raise(monkeypatch):
    _patch_gpu(monkeypatch, available=False)
    assert_tms_hook_mode_matches_arch("preload")  # CPU-only: nothing to guard
