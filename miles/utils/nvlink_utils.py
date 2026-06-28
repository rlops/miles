"""Node NVLink detection.

Used to decide ``NCCL_NVLS_ENABLE``: NVLink SHARP (NVLS) should only be
enabled when the node actually has active NVLink connections.

Detection prefers NVML (``pynvml``) — the same data source ``nvidia-smi``
reads — and falls back to the legacy ``nvidia-smi topo`` shell probe when
``pynvml`` is not installed (it is an optional dependency here, see
``miles/ray/train_actor.py``). When *neither* source can determine topology
(no driver, no ``nvidia-smi`` binary), detection returns ``False`` and logs a
warning, so callers keep the safe default of "NVLS disabled" instead of
crashing or *silently* mis-detecting — the latter being the gap in the old
``nvidia-smi topo -m | grep NV | wc -l`` one-liner, which could not tell a
genuine no-NVLink machine apart from one simply missing ``nvidia-smi``.
"""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _has_nvlink_via_nvml() -> bool | None:
    """Probe NVLink via NVML.

    Returns ``True``/``False`` when NVML answers, or ``None`` when NVML is
    unavailable (pynvml not installed, or no driver/library) so the caller
    can fall back to the nvidia-smi probe.
    """
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 — driver/library not present
        logger.debug("NVML init failed (%r); falling back to nvidia-smi probe.", exc)
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                    return True
            except pynvml.NVMLError:
                # Link index unsupported on this GPU — treat as inactive.
                continue
        return False
    except pynvml.NVMLError as exc:
        logger.debug("NVML NVLink query failed (%r); falling back to nvidia-smi probe.", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _has_nvlink_via_nvidia_smi() -> bool | None:
    """Legacy probe: count NVLink entries in ``nvidia-smi topo -m``.

    Returns ``None`` when the ``nvidia-smi`` binary is absent, so the caller
    can distinguish "no NVLink" from "cannot probe".
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["bash", "-c", "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi NVLink probe failed (%r).", exc)
        return None
    return int(completed.stdout.strip() or "0") > 0


def has_nvlink() -> bool:
    """Return ``True`` iff the node has at least one active NVLink.

    Tries NVML first, then the ``nvidia-smi`` shell probe. When neither can
    determine topology, logs a warning and returns ``False`` so NVLS stays
    disabled (the safe default).
    """
    result = _has_nvlink_via_nvml()
    if result is None:
        result = _has_nvlink_via_nvidia_smi()
    if result is None:
        logger.warning(
            "Could not detect NVLink (no pynvml and no nvidia-smi); assuming "
            "none and leaving NCCL NVLS disabled."
        )
        return False
    return result
