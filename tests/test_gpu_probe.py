"""Unit tests for miles.utils.gpu_probe — the per-process GPU residual probe.

Key properties under test:
- FAIL-OPEN: when none of the engine's process-tree PIDs appear in nvidia-smi
  compute-apps (e.g. PID-namespace mismatch in a container), parsers return
  None ("cannot measure"), never 0 — otherwise a hard gate would falsely pass.
- MAX-PER-GPU: MILES_MAX_RESIDUAL_GPU_MEM_GB = sum within a GPU, max across
  GPUs (TP-safe; never sum across cards).

Dependency-free: runnable via pytest OR directly (`python3 tests/test_gpu_probe.py`).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miles.utils.gpu_probe import (
    build_process_tree,
    parse_compute_apps_per_gpu_max_gb,
    parse_compute_apps_used_gb,
)


# --- P2: GPU-aware parser (sum within a GPU, max across GPUs) ---

def test_per_gpu_max_takes_max_across_gpus():
    csv = "0000:98:00.0, 100, 500\n0000:98:00.0, 200, 1024\n0000:A8:00.0, 100, 2000"
    # bus 98 = 500+1024 = 1524 MiB; bus A8 = 2000 MiB; max = 2000
    got = parse_compute_apps_per_gpu_max_gb(csv, {100, 200})
    assert abs(got - 2000 / 1024.0) < 1e-6


def test_per_gpu_max_sums_within_same_gpu():
    # two engine procs on the SAME gpu must be summed, not max'd
    csv = "0000:98:00.0,100,500\n0000:98:00.0,200,256"
    got = parse_compute_apps_per_gpu_max_gb(csv, {100, 200})
    assert abs(got - 756 / 1024.0) < 1e-6


def test_per_gpu_max_excludes_non_tree_pids():
    csv = "0000:98:00.0,100,500\n0000:A8:00.0,999,9000"
    got = parse_compute_apps_per_gpu_max_gb(csv, {100})
    assert abs(got - 500 / 1024.0) < 1e-6


def test_per_gpu_max_no_match_returns_none_not_zero():
    assert parse_compute_apps_per_gpu_max_gb("0000:A8:00.0,999,9000", {100, 200}) is None


def test_per_gpu_max_empty_returns_none():
    assert parse_compute_apps_per_gpu_max_gb("", {100}) is None


# --- legacy 2-col fallback parser ---

def test_fallback_sums_only_tree_pids():
    got = parse_compute_apps_used_gb("100, 500\n200, 1024\n999, 8000", {100, 200})
    assert abs(got - (1524 / 1024.0)) < 1e-6


def test_fallback_no_match_returns_none_not_zero():
    assert parse_compute_apps_used_gb("999, 8000\n888, 4000", {100, 200}) is None


def test_fallback_skips_unparsable_rows():
    csv = "100, [N/A]\n100, 512\nbad line\n, \n200,256"
    got = parse_compute_apps_used_gb(csv, {100, 200})
    assert abs(got - (768 / 1024.0)) < 1e-6


# --- process-tree walk ---

def _stat(proc, pid, comm, ppid):
    d = os.path.join(proc, str(pid))
    os.makedirs(d)
    with open(os.path.join(d, "stat"), "w") as f:
        f.write(f"{pid} ({comm}) S {ppid} 0 0 0 0\n")


def test_build_process_tree_walks_descendants():
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "proc")
        os.makedirs(proc)
        _stat(proc, 100, "spawn_main", 1)
        _stat(proc, 200, "sglang::sched", 100)
        _stat(proc, 300, "sglang::detok", 200)
        _stat(proc, 999, "unrelated", 1)
        assert build_process_tree(100, proc_root=proc) == {100, 200, 300}


def test_build_process_tree_comm_with_spaces_and_parens():
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "proc")
        os.makedirs(proc)
        _stat(proc, 100, "spawn", 1)
        d = os.path.join(proc, "200")
        os.makedirs(d)
        with open(os.path.join(d, "stat"), "w") as f:
            f.write("200 (weird (c o m m)) S 100 0\n")
        assert build_process_tree(100, proc_root=proc) == {100, 200}


def test_build_process_tree_missing_proc_returns_root():
    assert build_process_tree(100, proc_root="/nonexistent_proc_xyz") == {100}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} passed")
