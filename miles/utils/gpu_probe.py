"""GPU residual probing helpers — pure, dependency-free, unit-testable.

Used by :class:`SGLangEngine` to measure a SGLang server's REAL per-process
resident GPU memory after offload, via ``nvidia-smi`` compute-apps over the
engine's process tree. Kept free of sglang/torch imports so the parsing and
fail-open logic can be unit-tested without a GPU.

Semantics: ``MILES_MAX_RESIDUAL_GPU_MEM_GB`` is the **max per-GPU** resident
residual for an engine — for each GPU, sum the engine's process-tree usage on
that GPU, then take the max across GPUs. This supports TP>1 without summing
across cards (which would over-count and falsely trip the gate).
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def build_process_tree(root_pid, proc_root: str = "/proc") -> set:
    """Return ``root_pid`` plus all descendant PIDs by reading
    ``<proc_root>/<pid>/stat`` ppid links. Pure /proc walk, no psutil.

    ``self.process.pid`` is the multiprocessing spawn parent; the real
    GPU-resident process is the ``sglang::scheduler`` child, so the whole
    tree must be walked.
    """
    import os

    try:
        entries = [int(p) for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return {root_pid}
    children: dict = {}
    for pid in entries:
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), "rb") as f:
                data = f.read()
        except OSError:
            continue
        # comm (2nd field) is paren-wrapped and may contain spaces/parens;
        # ppid is the 2nd whitespace token after the final ')'.
        try:
            rparen = data.rindex(b")")
            ppid = int(data[rparen + 2:].split()[1])
        except (ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)
    tree = {root_pid}
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, ()):
            if ch not in tree:
                tree.add(ch)
                stack.append(ch)
    return tree


def parse_compute_apps_per_gpu_max_gb(nvidia_csv: str, tree_pids: set):
    """Parse ``nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory``.

    For PIDs in ``tree_pids``: sum ``used_memory`` (MiB) per GPU
    (keyed by ``gpu_bus_id``), then take the MAX across GPUs and return GiB.
    This is the ``MILES_MAX_RESIDUAL_GPU_MEM_GB`` semantics: the engine's
    worst single-GPU resident residual (TP-safe — no cross-card summing).

    Returns ``None`` (fail-open) if no tree PID appears in the listing.
    """
    per_gpu: dict = {}
    matched = False
    for line in nvidia_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        bus_id = parts[0]
        try:
            pid = int(parts[1])
            used = float(parts[2])
        except ValueError:
            continue
        if pid in tree_pids:
            per_gpu[bus_id] = per_gpu.get(bus_id, 0.0) + used
            matched = True
    if not matched:
        return None
    return max(per_gpu.values()) / 1024.0


def parse_compute_apps_used_gb(nvidia_csv: str, tree_pids: set):
    """Fallback parser for the legacy 2-col ``pid,used_memory`` query (no
    ``gpu_bus_id``). Sums all matched rows -> GiB. Used only when the
    GPU-aware query is unavailable; it cannot distinguish per-GPU, so it
    over-estimates for a multi-GPU engine.

    Returns ``None`` (fail-open) if no tree PID appears.
    """
    total_mib = 0.0
    matched = False
    for line in nvidia_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used = float(parts[1])
        except ValueError:
            continue
        if pid in tree_pids:
            total_mib += used
            matched = True
    if not matched:
        return None
    return total_mib / 1024.0


def _run_nvidia_smi(args, timeout_s: float):
    try:
        return subprocess.check_output(
            ["nvidia-smi"] + args, stderr=subprocess.STDOUT, timeout=timeout_s
        ).decode("utf-8", errors="replace")
    except (subprocess.SubprocessError, OSError):
        return None


def query_process_tree_gpu_used_gb(root_pid, timeout_s: float = 5.0,
                                   proc_root: str = "/proc"):
    """Max per-GPU resident GPU memory (GiB) of ``root_pid``'s process tree.

    Prefers the GPU-aware query (``gpu_bus_id,pid,used_memory``): per-GPU
    sum, max across GPUs. Falls back to the legacy 2-col query
    (``pid,used_memory``, summed) with a warning if the GPU-aware query is
    unsupported by this nvidia-smi.

    Returns ``None`` (fail-open) when nvidia-smi is missing, the call fails,
    or no tree PID appears (PID-namespace mismatch inside a container).
    Callers MUST treat ``None`` as "cannot measure", never as 0.
    """
    if root_pid is None or shutil.which("nvidia-smi") is None:
        return None
    tree = build_process_tree(root_pid, proc_root=proc_root)
    if not tree:
        return None
    out = _run_nvidia_smi(
        ["--query-compute-apps=gpu_bus_id,pid,used_memory",
         "--format=csv,noheader,nounits"],
        timeout_s,
    )
    if out is not None:
        return parse_compute_apps_per_gpu_max_gb(out, tree)
    # GPU-aware query unsupported -> legacy 2-col fallback (summed).
    logger.warning(
        "nvidia-smi gpu_bus_id query unavailable; falling back to "
        "pid,used_memory (summed; cannot distinguish per-GPU)"
    )
    out = _run_nvidia_smi(
        ["--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        timeout_s,
    )
    if out is None:
        return None
    return parse_compute_apps_used_gb(out, tree)
