import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    # "gpu" is the LOCAL index inside this actor's CUDA_VISIBLE_DEVICES
    # slice; report the physical id too so per-GPU logs are unambiguous
    # when several actors each see their own GPU as cuda:0.
    import os as _os

    cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [x for x in cvd.split(",") if x.strip()]
    physical = visible[device] if device < len(visible) else str(device)
    return {
        "gpu": str(device),
        "physical_gpu": physical,
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def log_nontorch(label: str):
    """Escape-audit probe: log the gap between whole-GPU physical usage and
    this process's torch allocator — growth in that gap across a phase means
    the phase allocated GPU memory OUTSIDE torch (context/module loading,
    NCCL, driver pools) or outside the tms interesting region. Deltas between
    consecutive probes attribute the ~2.5 GB unpausable tail phase by phase;
    same-GPU co-tenants (the sleeping engine) are constant and cancel out.
    """
    device = torch.cuda.current_device()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(device)
    used = _byte_to_gb(total - free)
    reserved = _byte_to_gb(torch.cuda.memory_reserved(device))
    info = available_memory()
    logger.info(
        f"[NONTORCH-AUDIT] {label}: physical_gpu={info['physical_gpu']} "
        f"whole_used={used} reserved={reserved} non_torch={round(used - reserved, 2)}"
    )


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
