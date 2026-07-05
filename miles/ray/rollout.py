import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.utils import dumper_utils, tracking_utils
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    init_http_client,
    is_port_available,
    wait_for_server_ready,
)
from miles.utils.iter_utils import group_by
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from miles.utils.misc import load_function
from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F2 EngineInfo — RLix-mode 5-state machine (single source of truth)
# ---------------------------------------------------------------------------

# State semantics (plan §F2 / scope F24):
#   - "shell"      : construction / pre-init slot reserved. No Ray actor, no SGLang
#                    server, no GPU. ``bundle_index`` / ``gpu_ids`` / ``placement``
#                    are populated; the only thing missing is the actor handle.
#                    INIT transitions ``shell → loading`` via expand_engines (full
#                    INIT only, iter 5).
#   - "active"     : engine is serving traffic (router has admitted it,
#                    weight version is set). Default state under standalone
#                    full-init.
#   - "disabling"  : router admission has been closed; abort-drain-sleep is in
#                    progress. Transitions to ``offloaded`` after
#                    release_memory_occupation completes.
#   - "offloaded"  : engine actor is alive but its SGLang server has released
#                    weights / KV / cuda_graph. Runtime expand transitions
#                    ``offloaded → loading`` via wake_up + selective sync.
#   - "loading"    : either INIT (post-create, pre-finish_init_offload) or
#                    runtime (post-wake_up, pre-activate_routing). Transitions
#                    to ``offloaded`` (INIT) or ``active`` (runtime) once
#                    finish_init_offload / activate_routing completes.
EngineState = Literal["shell", "active", "disabling", "offloaded", "loading"]


@dataclasses.dataclass
class EngineInfo:
    """Per-engine metadata for the RolloutManager state machine.

    The ``handle`` may be ``None`` only in the ``shell`` state. ``bundle_index``,
    ``gpu_ids``, and ``node_rank`` are populated for every state including
    ``shell`` (so RLix-mode INIT-time placement is fully described before any
    actor is created).
    """

    engine_index: int
    state: EngineState
    handle: Any | None = None
    bundle_index: int | None = None
    gpu_ids: tuple[int, ...] = ()
    node_rank: int = 0

    def is_shell(self) -> bool:
        return self.state == "shell"

    def is_alive(self) -> bool:
        """True iff a Ray actor handle exists (state in {active, disabling,
        offloaded, loading})."""
        return self.state != "shell" and self.handle is not None


# ---------------------------------------------------------------------------
# ServerGroup / RolloutServer abstractions
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False
    model_path: str | None = None
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index -> next free port.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }
            env_vars.update(dumper_utils.get_sglang_env(self.args))

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # Per-pipeline base port via MILES_ROLLOUT_BASE_PORT env var.
            # Two MilesPipeline actors on the same Ray cluster both call
            # find_available_port concurrently; a shared default of 15000
            # races on the first probe (both check, both see free, both
            # try to bind, one crashes). The dual-pipeline driver passes
            # disjoint base ports per pipeline (e.g. 15000, 16000) so the
            # find-free-port windows never overlap.
            _env_base = os.environ.get("MILES_ROLLOUT_BASE_PORT")
            _default_base = int(_env_base) if _env_base else 15000
            base_port = max(port_cursors.values()) if port_cursors else _default_base
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups."""
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        values = {g.nodes_per_engine for g in self.server_groups}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


# ---------------------------------------------------------------------------
# RolloutManager
# ---------------------------------------------------------------------------


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        args,
        pg,
        *,
        all_engine_placements: list | None = None,
        active_engine_indices: frozenset[int] | None = None,
    ):
        """Initialize a RolloutManager.

        Standalone path passes only ``args`` and ``pg``; behavior is unchanged.

        RLix-mode path additionally passes:
          - ``all_engine_placements``: a list of ``WorkerPlacement`` (or
            duck-typed equivalent) of length ``rollout_num_gpus //
            rollout_num_gpus_per_engine``. Provided for the M11.2 init flow
            where a Pipeline first allocates all engine slots and then expands
            a subset; iter 14 introduces the concrete ``WorkerPlacement``
            class. iter 4 stores it for later iters and uses it only to
            populate :attr:`_engines` slot identities.
          - ``active_engine_indices``: ``frozenset[int]`` of engine indices
            that should be opened for routing at construction time. M11.2
            init passes ``frozenset()`` and grants engines via expand_engines
            (iter 5). Standalone passes ``None`` (legacy) and every server
            engine starts in the ``active`` state.
        """
        configure_logger()

        self.pg = pg
        self.args = args
        self._all_engine_placements: list = list(all_engine_placements or [])
        self._active_engine_indices: frozenset[int] = (
            frozenset(active_engine_indices) if active_engine_indices is not None else frozenset()
        )
        # RLix-mode progress hook. Standalone miles leaves this as ``None``
        # and the rollout function falls back to :class:`NoOpRLixHooks`.
        # In RLix mode the pipeline driver injects a :class:`MilesRLixHooks`
        # via :meth:`set_rlix_hooks` so the scheduler receives
        # ``begin_progress_batch`` / ``bump_completed`` events and can wake
        # engines for the next rollout after each ``_after_training``.
        self._rlix_hooks = None
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_experimental_refactor = enable_experimental_rollout_refactor()
        if self.use_experimental_refactor:
            input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
            self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
            self.eval_generate_rollout = load_rollout_function(input, self.args.eval_function_path)
        else:
            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
            _start_session_server(args)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        # F2 EngineInfo state machine. Populated after start_rollout_servers
        # so updatable-server engines map onto engine indices 0..N-1 (matches
        # F12 contiguous-mapping invariant). Standalone path leaves every
        # alive engine in the ``active`` state; RLix-mode path may pass
        # ``active_engine_indices=frozenset()`` to start every engine in
        # ``offloaded`` (M11.2 init) and grant subsets via expand_engines
        # (iter 5).
        self._engines: dict[int, EngineInfo] = {}
        self._init_engine_info_table()

        self._metric_checker = MetricChecker.maybe_create(args)
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        for monitor in self._health_monitors:
            monitor.stop()

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> RolloutServer | None:
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        if self.use_experimental_refactor:
            result = call_rollout_function(self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id))
        else:
            result = call_rollout_fn(
                self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True
            )
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    # ------------------------------------------------------------------
    # F2 EngineInfo accessors + subset offload/onload (RLix-mode entry points)
    # ------------------------------------------------------------------

    def _init_engine_info_table(self) -> None:
        """Populate :attr:`_engines` from the updatable server's engines.

        Iter 4 scope: only the standalone-style construction is wired here —
        every engine handle that ``start_rollout_servers`` returned is alive
        with weights already loaded, so its state is ``active``. Missing /
        dead handles map to ``shell``.

        Note: the RLix M11.2 init pattern (``all_engine_placements`` plus
        ``active_engine_indices=frozenset()`` to build only metadata slots
        with NO SGLang server creation, then ``expand_engines`` grants
        subsets) requires bypassing ``start_rollout_servers`` entirely. That
        construction path lands with iter 5 (compound ops) and iter 26
        (MilesPipeline init bootstrap). Iter 4 stores the RLix kwargs for
        those iters to consume but does NOT mark engines ``offloaded`` here:
        marking them offloaded without an actual ``release_memory_occupation``
        call would lie about resident memory.
        """
        srv = self._get_updatable_server()
        engines = list(srv.engines) if srv else []
        # M11.2 Option β: when MILES_INIT_DEFER_ADD_WORKER=1, engines were
        # constructed without /add_worker; land them in "loading" state so
        # MilesPipeline.initialize_pipeline can drive finish_init_offload
        # (rollout.py:1009-1034 — requires state=="loading") to bring them
        # to "offloaded" with VRAM released. F40 Runtime branch then handles
        # wake → sync → activate_routing on first _expand_workers.
        init_state = (
            "loading"
            if os.environ.get("MILES_INIT_DEFER_ADD_WORKER") == "1"
            else "active"
        )
        for idx, handle in enumerate(engines):
            if handle is None:
                self._engines[idx] = EngineInfo(engine_index=idx, state="shell", handle=None)
                continue
            self._engines[idx] = EngineInfo(
                engine_index=idx,
                state=init_state,
                handle=handle,
            )

    def _resolve_engine_indices(self, engine_indices: Iterable[int] | None) -> list[int]:
        """Return a sorted list of indices whose engines have a live handle.

        ``None`` means "all alive engines" (legacy behavior). Shell engines
        are always excluded; subset operations targeting them must go through
        ``expand_engines`` (iter 5) first.
        """
        if engine_indices is None:
            return sorted(idx for idx, info in self._engines.items() if info.is_alive())
        requested = sorted(set(int(i) for i in engine_indices))
        for idx in requested:
            if idx not in self._engines:
                raise KeyError(f"unknown engine_index {idx}; known: {sorted(self._engines)}")
            if not self._engines[idx].is_alive():
                raise RuntimeError(
                    f"engine_index {idx} is in state {self._engines[idx].state!r}; "
                    f"call expand_engines first"
                )
        return requested

    def _engine_handles(self, engine_indices: Iterable[int] | None) -> list[Any]:
        return [self._engines[idx].handle for idx in self._resolve_engine_indices(engine_indices)]

    def offload(
        self,
        tags: list[str] | None = None,
        engine_indices: Iterable[int] | None = None,
    ):
        """Release memory occupation on the engines.

        ``engine_indices=None`` matches legacy behavior (all alive engines).
        Pass a subset to release just those engines (used by F2
        shrink_engines, iter 5).
        """
        self.health_monitoring_pause()
        if engine_indices is not None:
            indices = self._resolve_engine_indices(engine_indices)
            handles = [self._engines[idx].handle for idx in indices]
            results = (
                ray.get([h.release_memory_occupation.remote(tags=tags) for h in handles])
                if handles
                else []
            )
            # Subset offload moves engines toward ``offloaded``; the full F2
            # disable lifecycle (active → disabling → offloaded) lands in iter
            # 5 via shrink_engines. Iter 4's subset path is the data-plane
            # primitive only.
            for idx in indices:
                self._engines[idx].state = "offloaded"
            return results
        if tags is not None:
            handles = [
                engine.release_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
            return ray.get(handles) if handles else []
        for srv in self.servers.values():
            srv.offload()

    def onload(
        self,
        tags: list[str] | None = None,
        engine_indices: Iterable[int] | None = None,
    ):
        if engine_indices is not None:
            indices = self._resolve_engine_indices(engine_indices)
            handles = [self._engines[idx].handle for idx in indices]
            results = (
                ray.get([h.resume_memory_occupation.remote(tags=tags) for h in handles])
                if handles
                else []
            )
            # Subset onload is the data-plane primitive; the full
            # ``offloaded → loading → active`` transition is driven by
            # expand_engines + activate_routing (iter 5).
            for idx in indices:
                if self._engines[idx].state == "offloaded":
                    self._engines[idx].state = "loading"
            return results
        for srv in self.servers.values():
            srv.onload(tags)

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def onload_weights(self, engine_indices: Iterable[int] | None = None):
        if engine_indices is not None:
            indices = self._resolve_engine_indices(engine_indices)
            handles = [self._engines[idx].handle for idx in indices]
            return (
                ray.get(
                    [h.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for h in handles]
                )
                if handles
                else []
            )
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self, engine_indices: Iterable[int] | None = None):
        if engine_indices is not None:
            indices = self._resolve_engine_indices(engine_indices)
            handles = [self._engines[idx].handle for idx in indices]
            return (
                ray.get(
                    [
                        h.resume_memory_occupation.remote(
                            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
                        )
                        for h in handles
                    ]
                )
                if handles
                else []
            )
        for srv in self.servers.values():
            srv.onload_kv()

    # ------------------------------------------------------------------
    # F2 compound ops + admin (RLix-mode lifecycle)
    # ------------------------------------------------------------------

    def get_engine_count(self) -> int:
        """Return the declared engine count (length of :attr:`_engines` table).

        Includes shell entries. Iter 26 (MilesPipeline init) uses this for
        the consistency assert "scheduler engine_count == declared engine_count".
        """
        return len(self._engines)

    def set_rlix_hooks(self, hooks) -> None:
        """Inject an RLix progress hook for downstream rollout calls.

        Called by ``MilesPipeline._init_phase_b_infer`` after the manager is
        created but before the first rollout dispatch. ``hooks`` must
        implement the :class:`miles.utils.rlix_hooks.RLixHooks` protocol —
        in production this is :class:`rlix.pipeline.miles_hooks.MilesRLixHooks`
        wrapping the per-pipeline coordinator handle, so the scheduler
        receives ``begin_progress_batch`` / ``bump_completed`` events for
        every rollout and can wake engines for rollout N+1 after the prior
        ``_after_training`` released ``actor_train``.

        Standalone miles never calls this; ``self._rlix_hooks`` stays
        ``None`` and :func:`call_rollout_fn` lets the rollout fn fall back
        to :class:`NoOpRLixHooks`.
        """
        self._rlix_hooks = hooks
        logger.info("[RolloutManager] set_rlix_hooks installed (kind=%s)", type(hooks).__name__)

    def get_engine_handles(self, engine_indices: Iterable[int]) -> dict[int, Any]:
        """Read-only snapshot of per-engine handles for the given indices.

        Used by ``MilesModelUpdateService.sync_selected_workers`` (iter 19/20)
        at sync entry to fetch handles once and drive per-engine cpu_serialize
        and NCCL broadcast RPCs without re-querying the manager.

        Raises ``RuntimeError`` if any requested index is in ``shell`` state
        (no handle to give).
        """
        snapshot: dict[int, Any] = {}
        for idx in sorted(set(int(i) for i in engine_indices)):
            if idx not in self._engines:
                raise KeyError(f"unknown engine_index {idx}")
            info = self._engines[idx]
            if info.is_shell() or info.handle is None:
                raise RuntimeError(
                    f"engine_index {idx} is in state {info.state!r} with no live handle; "
                    f"call expand_engines first"
                )
            snapshot[idx] = info.handle
        return snapshot

    def get_engine_states(self, engine_indices: Iterable[int]) -> dict[int, str]:
        """Read-only entry-state snapshot — sibling to :meth:`get_engine_handles`.

        ``MilesCoordinator._expand_workers`` (iter 23) reads this to dispatch
        between INIT branch (entry state == ``shell``) and Runtime GENERATION
        branch (entry state == ``offloaded``). Heterogeneous entry states
        across the requested set must be detected upstream.
        """
        snapshot: dict[int, str] = {}
        for idx in sorted(set(int(i) for i in engine_indices)):
            if idx not in self._engines:
                raise KeyError(f"unknown engine_index {idx}")
            snapshot[idx] = self._engines[idx].state
        return snapshot

    def set_weight_version(
        self,
        version: int,
        engine_indices: Iterable[int] | None = None,
    ) -> int:
        """Fan-out per-engine ``update_weight_version`` call.

        Called only by ``MilesModelUpdateService.sync_selected_workers`` at the
        end of the atomic sync unit (plan §F4 / scope F21: one publish per
        sync). Pipeline / coordinator MUST NOT call this directly.

        ``engine_indices=None`` falls back to all currently-alive engines
        (shell engines are always skipped — they have no SGLang server to
        receive the update). "Alive" includes ``loading`` because the
        runtime-expand path publishes the version BEFORE
        ``activate_routing`` flips state to ``active``; if we filtered on
        ``state == "active"`` here, runtime-expanded engines would come up
        active with stale weight_version. Both branches now share the
        ``_resolve_engine_indices`` predicate so explicit-list and None
        produce the same fan-out for the same logical set (per scope F21
        engine-table contract).

        Returns the version actually published (echoed back for caller logging).
        """
        indices = self._resolve_engine_indices(engine_indices)
        handles = [self._engines[idx].handle for idx in indices]
        if handles:
            # SGLangEngine.update_weight_version(self, weight_version: str)
            # — sglang_engine.py:671. Use the exact kwarg name; passing
            # `version=` would raise TypeError inside the Ray actor.
            ray.get(
                [h.update_weight_version.remote(weight_version=str(version)) for h in handles]
            )
        return int(version)

    def _abort_engines(self, engine_indices: Iterable[int]) -> None:
        """Idempotency-cached abort fan-out.

        ``_preempted_engines`` is the abort-idempotency cache for this
        method (per A19 / scope F03): abort-once-per-admission-cycle. The
        cache MUST NOT be promoted to routing / dispatch / attribution /
        resize-safety state — that responsibility lives entirely in
        ``router.enabled_workers``. The only sanctioned mutation surface
        is this method plus the ``_release_abort_idempotency_for`` /
        ``_reset_abort_idempotency_for`` helpers below — together they
        own the cache lifecycle.
        """
        if not hasattr(self, "_preempted_engines"):
            self._preempted_engines: set[int] = set()
        indices = self._resolve_engine_indices(engine_indices)
        new_targets = [idx for idx in indices if idx not in self._preempted_engines]
        if not new_targets:
            return
        handles = [self._engines[idx].handle for idx in new_targets]
        ray.get([h.abort_all_requests.remote() for h in handles])
        self._preempted_engines.update(new_targets)

    def _release_abort_idempotency_for(self, engine_indices: Iterable[int]) -> None:
        """Drop ``_preempted_engines`` cache entries on successful offload.

        Companion to ``_abort_engines`` — clears the abort-idempotency
        cache for indices that have transitioned to ``offloaded`` so the
        next admission cycle starts fresh. Lives next to ``_abort_engines``
        so the cache lifecycle (init / update / discard) is owned in one
        place per A19 / scope F03.
        """
        if not hasattr(self, "_preempted_engines"):
            return
        for idx in engine_indices:
            self._preempted_engines.discard(int(idx))

    def _reset_abort_idempotency_for(self, engine_indices: Iterable[int]) -> None:
        """Drop cache entries when a shrink cycle fails so retry re-aborts.

        On ``shrink_engines`` failure between abort and offload-flip, the
        cached indices are still ``disabling`` and any new in-flights
        that arrived during the failure must be aborted on retry. This
        helper makes that drop explicit.
        """
        # Same body as _release_abort_idempotency_for; named distinctly
        # so callers document intent.
        self._release_abort_idempotency_for(engine_indices)

    def get_router_enabled_workers(self) -> list[str]:
        """M11.2 Option β 3e: snapshot the router's ``enabled_workers`` set.

        ``MilesPipeline._init_phase_b_infer`` asserts this returns an
        empty list at the end of Phase B INIT when running with
        ``MILES_INIT_DEFER_ADD_WORKER=1`` (Codex KT review Q5-a).
        """
        srv = self._get_updatable_server()
        router = getattr(srv, "router", None) if srv is not None else None
        if router is None:
            return []
        # `router.enabled_workers` is a set of URLs maintained as the
        # source of truth for the live worker set (router.py:497).
        try:
            return sorted(router.enabled_workers)
        except AttributeError:
            return []

    def shrink_engines(
        self,
        engine_indices: Iterable[int],
        *,
        post_sleep_vram_threshold_gb: float | None = None,
    ) -> list[int]:
        """F2 abort-drain-sleep ordering for a subset of active engines.

        Sequence (per scope F23):
          1. Mark each target ``active → disabling``.
          2. Abort all in-flight requests (``_abort_engines``).
          3. Drain via ``is_idle`` poll until every target reports zero
             outstanding requests.
          4. ``release_memory_occupation()`` to release weights/KV/graph.
          5. Optional: assert post-sleep VRAM below threshold (Anti-regression
             invariant #8).
          6. Mark each target ``disabling → offloaded``.

        Returns the sorted list of engine indices actually shrunk.
        """
        indices = self._resolve_engine_indices(engine_indices)
        if not indices:
            return []
        # Step 1: announce intent (admission close happens at the router via
        # F3, iter 7+ — manager state alone doesn't gate dispatch in iter 5).
        for idx in indices:
            if self._engines[idx].state == "active":
                self._engines[idx].state = "disabling"
        # M11.2 Option β 3f (Codex KT review Q5-b + Phase 3 review HIGH):
        # close router admission BEFORE memory release. Under
        # MILES_INIT_DEFER_ADD_WORKER=1 the router is the single source
        # of truth for live workers; calling /disable_worker first
        # guarantees no new request is dispatched to an engine whose VRAM
        # is about to drop. unregister_from_router raises on non-2xx,
        # propagating into the outer try/except (L967-971) which resets
        # the abort idempotency cache and re-raises — so the manager
        # never frees GPU memory on a router that hasn't closed admission.
        handles = [self._engines[idx].handle for idx in indices]
        ray.get([h.unregister_from_router.remote() for h in handles])
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[RolloutManager] shrink_engines: disabled router workers "
            "prior to release engine_indices=%s",
            indices,
        )
        # Steps 2-5 are wrapped so a mid-sequence failure resets the abort
        # idempotency cache — otherwise retry's _abort_engines would skip
        # the already-cached indices and the drain would re-stall.
        try:
            # Steps 2 + 3: abort + drain.
            self._abort_engines(indices)
            handles = [self._engines[idx].handle for idx in indices]
            deadline = time.time() + 30.0  # bounded test-side drain; production hardening = M11.5.
            while time.time() < deadline:
                verdicts = ray.get([h.is_idle.remote() for h in handles])
                if all(verdicts):
                    break
                time.sleep(0.1)
            else:
                still_busy = [
                    idx for idx, idle in zip(indices, ray.get([h.is_idle.remote() for h in handles]))
                    if not idle
                ]
                raise RuntimeError(
                    f"shrink_engines drain timeout after 30s; still busy: {still_busy}"
                )
            # Step 3.5 (rlix-mode safety): pause the SGLang scheduler with
            # mode="retract" before release_memory_occupation, so any
            # in-flight Triton kernel (e.g. write_req_to_token_pool_triton
            # launched for the last decode iteration) finishes against
            # GPU-resident persistent buffers before release moves them
            # to CPU. Without this, SGLang crashes with ``Pointer
            # argument cannot be accessed from Triton (cpu tensor?)``
            # because release_memory_occupation moves persistent
            # token-pool buffers to CPU mid-iteration. The pause API
            # blocks until the scheduler reaches a safe checkpoint.
            if os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
                # F7 (m11-review.review-report.md §2): documented SGLang
                # /pause_generation API contract:
                #   - mode="retract" — waits until the SGLang scheduler reaches
                #     a safe checkpoint (no in-flight Triton kernels referencing
                #     persistent token-pool buffers); then halts new batch
                #     dispatch. The scheduler thread keeps running but does not
                #     spawn new work.
                #   - Idempotent (documented intent per the prior comment
                #     "if SGLang rejects the request e.g. already paused";
                #     NOT empirically verified across all sglang versions).
                #   - 4xx ONLY on misuse (unknown mode, missing engine). 5xx
                #     means SGLang internal error — caller should escalate
                #     (current path swallows; future M11.5 hardening should
                #     differentiate 4xx vs 5xx).
                # Caught Exception is intentionally broad here because the
                # release sequence MUST continue: without release, the
                # scheduler ledger leaks. We log at WARNING; the post-sleep
                # VRAM assert (step 5) will catch any actual memory leak.
                try:
                    ray.get([h.pause_generation.remote(mode="retract") for h in handles])
                except Exception as exc:  # noqa: BLE001
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "shrink_engines: pause_generation pre-release failed "
                        "(engine_indices=%s, swallowing to keep release path "
                        "unblocked; post-sleep VRAM assert will catch leak): %r",
                        indices, exc,
                    )
            # Step 4: release memory.
            ray.get([h.release_memory_occupation.remote(tags=None) for h in handles])
            # Step 5: attribution diagnostics. The hard residual gate is
            # whole-GPU memory.used in RLix; this logs each SGLang engine's
            # process-resident memory and /server_info accounting so a high
            # whole-GPU residual can be attributed to SGLang vs non-SGLang
            # co-tenants (Megatron/Miles/vLLM/orphan processes).
            if post_sleep_vram_threshold_gb is not None:
                observed_resident_gbs = ray.get(
                    [
                        h.log_post_sleep_residual_diagnostics.remote(
                            threshold_gb=post_sleep_vram_threshold_gb
                        )
                        for h in handles
                    ]
                )
                measured = [v for v in observed_resident_gbs if v is not None]
                logger.info(
                    "shrink_engines: post-sleep SGLang residual diagnostics "
                    "process_resident_max=%s GiB per_engine=%s "
                    "whole_gpu_threshold=%.3f GiB engine_indices=%s "
                    "(whole-GPU hard gate runs in RLix)",
                    ("%.3f" % max(measured)) if measured else "n/a",
                    [None if v is None else round(float(v), 3) for v in observed_resident_gbs],
                    float(post_sleep_vram_threshold_gb),
                    indices,
                )
        except Exception:
            # Reset the abort cache on failure so retry re-aborts new
            # in-flights that arrived during the failed cycle.
            self._reset_abort_idempotency_for(indices)
            raise
        # Step 6: state transition + abort-cache cleanup. Cache cleanup
        # owned by _abort_engines's companion helper so the Layer-1
        # invariant ("cache lifecycle owned by _abort_engines and its
        # private helpers") reads literally.
        for idx in indices:
            self._engines[idx].state = "offloaded"
        self._release_abort_idempotency_for(indices)
        return indices

    def expand_engines(
        self,
        engine_indices: Iterable[int],
    ) -> list[int]:
        """Wake offloaded engines, leaving them in the ``loading`` state.

        Iter 5 only handles the runtime path (``offloaded → loading``) via
        ``resume_memory_occupation``. The ``shell → loading`` (full INIT)
        branch needs a placement provider + actor-creation flow and lands
        with iters 14/15/26.

        After ``expand_engines`` returns, the engines are warm with weights
        loaded but routing is NOT open. The full transition to ``active``
        requires:
          - ``MilesModelUpdateService.sync_selected_workers`` (iter 19/20) to
            push a current weight version onto the engines, and
          - ``activate_routing`` to add the engines to the router's enabled
            set.
        """
        indices = sorted(set(int(i) for i in engine_indices))
        for idx in indices:
            if idx not in self._engines:
                raise KeyError(f"unknown engine_index {idx}")
            info = self._engines[idx]
            if info.state == "shell":
                raise RuntimeError(
                    f"engine_index {idx} is shell; iter 5 does not implement the "
                    f"shell → loading INIT branch (lands with iter 14/15/26 once "
                    f"placement provider + actor-creation flow is in place)"
                )
            if info.state != "offloaded":
                raise RuntimeError(
                    f"engine_index {idx} state={info.state!r}, expected 'offloaded'"
                )
        handles = [self._engines[idx].handle for idx in indices]
        if handles:
            ray.get([h.resume_memory_occupation.remote(tags=None) for h in handles])
        for idx in indices:
            self._engines[idx].state = "loading"
        return indices

    def finish_init_offload(self, engine_indices: Iterable[int]) -> list[int]:
        """``loading → offloaded`` transition for the INIT path.

        Used by iter 26 ``MilesPipeline.initialize_pipeline`` Step 7: full INIT
        creates engines with weights loaded (state ``loading``); then drops
        weights/KV/graph WITHOUT a service.sync, version publish, or router
        activation. After this call the engine is parked, ready to be granted
        a runtime ``expand_engines + sync_selected_workers + activate_routing``
        cycle when the scheduler signals.
        """
        indices = sorted(set(int(i) for i in engine_indices))
        for idx in indices:
            info = self._engines.get(idx)
            if info is None:
                raise KeyError(f"unknown engine_index {idx}")
            if info.state != "loading":
                raise RuntimeError(
                    f"finish_init_offload requires state=='loading'; engine_index "
                    f"{idx} is {info.state!r}"
                )
        handles = [self._engines[idx].handle for idx in indices]
        if handles:
            ray.get([h.release_memory_occupation.remote(tags=None) for h in handles])
        for idx in indices:
            self._engines[idx].state = "offloaded"
        return indices

    def activate_routing(self, engine_indices: Iterable[int]) -> list[int]:
        """``loading → active`` transition + router /add_worker (M11.2 Option β).

        Iter 5 only updated manager-side state and assumed engines had
        already registered with the router at ``_init_normal``. Under
        ``MILES_INIT_DEFER_ADD_WORKER=1`` (real M11.2 overlap mode) that
        register was skipped so the engine could initialize with empty
        router state; ``activate_routing`` now drives the just-in-time
        register via ``SGLangEngine.register_with_router``. Standalone
        miles also calls register_with_router — idempotent at the router
        (`_add_worker_internal` discards from ``dead_workers`` on re-add)
        and keeps router state aligned with manager state on every
        ``F40 Runtime`` expand cycle.
        """
        indices = sorted(set(int(i) for i in engine_indices))
        for idx in indices:
            info = self._engines.get(idx)
            if info is None:
                raise KeyError(f"unknown engine_index {idx}")
            if info.state != "loading":
                raise RuntimeError(
                    f"activate_routing requires state=='loading'; engine_index "
                    f"{idx} is {info.state!r}"
                )
        # Codex Phase 3 review MEDIUM: register_with_router raises on
        # non-2xx so we never mark engines "active" against a router
        # that doesn't know about them. The exception propagates up to
        # the coordinator's F40 Runtime branch which can retry or fail
        # the expand cycle.
        handles = [self._engines[idx].handle for idx in indices]
        ray.get([h.register_with_router.remote() for h in handles])
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[RolloutManager] activate_routing: registered router workers "
            "engine_indices=%s",
            indices,
        )
        for idx in indices:
            self._engines[idx].state = "active"
        return indices

    def shutdown_hard(self) -> None:
        """M4 minimal hard cleanup — terminate every alive engine actor.

        Used by ``MilesPipeline`` (iter 27) on init-failure / dispose paths to
        guarantee scheduler / Ray ledger consistency. CUDA context lives in
        the SGLang server child processes, so killing the Ray actor handle is
        only the first step; the SGLang server tree is killed via the actor's
        existing ``shutdown`` method (which calls ``kill_process_tree`` on
        ``self.process.pid``). We invoke that BEFORE ``ray.kill`` so the OS
        children get SIGTERM rather than being orphaned.

        Forbidden in M11.1 (Layer 3 deferred): graceful drain RPC, abort RPC,
        30s + force-kill timeout, cleanup daemon, VRAM-threshold gate. This
        is the intentionally minimal version.
        """
        # Stop background monitors first so they don't race with engine death.
        for monitor in self._health_monitors:
            try:
                monitor.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"shutdown_hard: monitor.stop failed: {exc!r}")
        # Best-effort SGLang server-tree shutdown, then ray.kill.
        for idx, info in self._engines.items():
            handle = info.handle
            if handle is None:
                continue  # shell — nothing to kill.
            try:
                # `shutdown` performs router /remove_worker + kill_process_tree.
                ray.get(handle.shutdown.remote(), timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"shutdown_hard: engine {idx} shutdown.remote() failed: {exc!r}"
                )
            try:
                ray.kill(handle, no_restart=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"shutdown_hard: engine {idx} ray.kill failed: {exc!r}")
            info.handle = None
            info.state = "shell"

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            if self.use_experimental_refactor:
                data = call_rollout_function(self.generate_rollout, RolloutFnTrainInput(rollout_id=rollout_id))
            else:
                data = call_rollout_fn(
                    self.generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=False,
                    rlix_hooks=self._rlix_hooks,
                )
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        # Round down to a multiple of dp_size to ensure only one training step
        dynamic_gbs = (num_samples // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        # Calculate how many samples will be discarded
        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if any(sample.weight_versions for sample in samples):
            train_data["weight_versions"] = [sample.weight_versions for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        # Pass dynamic global_batch_size to training side
        assert self.args.use_dynamic_global_batch_size == hasattr(self, "_dynamic_global_batch_size")
        if hasattr(self, "_dynamic_global_batch_size"):
            train_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
                "weight_versions",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
                "dynamic_global_batch_size",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


# ---------------------------------------------------------------------------
# Port allocation helpers
# ---------------------------------------------------------------------------


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()
            # Always allocate a unique engine_info_bootstrap_port per engine
            addr_and_ports[current_rank]["engine_info_bootstrap_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


# ---------------------------------------------------------------------------
# Router + server bootstrap
# ---------------------------------------------------------------------------


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sgl router or miles router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set and ``force_new`` is False,
    skip launching and return the existing values.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_miles_router:
        import copy

        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
        from miles.router.router import run_router

        router_args = copy.copy(args)
        router_args.sglang_router_ip = router_ip
        router_args.sglang_router_port = router_port

    else:
        from sglang_router.launch_router import RouterArgs

        from miles.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = router_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if args.sglang_router_policy:
            router_args.policy = args.sglang_router_policy

        if has_pd_disaggregation:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    port = router_port
    if not is_port_available(port):
        raise RuntimeError(
            f"Port {port} is already in use — a stale router process may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    wait_for_server_ready(router_ip, router_port, process, timeout=30)
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        offset += args.critic_num_nodes * args.critic_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        num += args.critic_num_nodes * args.critic_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        all_init_handles: list = []
        port_cursors: dict[int, int] = {}

        for group_cfg in model_cfg.server_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            # rlix-mode override: miles' static rollout_pg_offset model
            # places engines after the train pool (group_abs_start ≥
            # megatron_num_gpus → needs_offload=False), but rlix's
            # cluster_device_mappings can map engines onto the same
            # physical GPUs as train (partial overlap). When the rlix
            # scheduler resizes infer to free overlap GPUs for
            # actor_train, SGLang's release_memory_occupation must
            # actually return memory to the OS — which requires
            # enable_memory_saver=True. Force it on under rlix when
            # offload_rollout is set.
            if args.offload_rollout and os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
                needs_offload = True
            overrides = dict(group_cfg.overrides)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            server_groups.append(group)

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        if all_init_handles:
            ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


# ---------------------------------------------------------------------------
# Logging / metrics helpers (unchanged)
# ---------------------------------------------------------------------------


def _start_session_server(args):
    """Start a standalone session server when ``--use-session-server`` is set.

    The session server runs as a separate process with its own port and proxies
    inference requests directly to SGLang worker engines.  It is always started
    as a standalone process regardless of whether ``--use-miles-router`` is active.
    """
    if not getattr(args, "use_session_server", False):
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    if getattr(args, "session_server_ip", None) is None:
        args.session_server_ip = args.sglang_router_ip
    if getattr(args, "session_server_port", None) is None:
        args.session_server_port = find_available_port(random.randint(5000, 6000))
    if getattr(args, "session_server_instance_id", None) is None:
        args.session_server_instance_id = uuid.uuid4().hex

    ip, port = args.session_server_ip, args.session_server_port
    if not is_port_available(port):
        raise RuntimeError(
            f"Port {port} is already in use — a stale session server may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

    from miles.rollout.session.session_server import run_session_server

    process = multiprocessing.Process(target=run_session_server, args=(args, router_url))
    process.daemon = True
    process.start()
    wait_for_server_ready(ip, port, process, timeout=30)
    logger.info(f"Session server launched at {ip}:{port}")


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len(set(s.weight_versions)) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        log_dict["tito_session_mismatch_rate"] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"tito_session_mismatch_rate/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            for strict_type in ("special_token_count", "special_token_type", "non_assistant_text"):
                rate = log_dict.get(f"tito_session_mismatch_rate/{strict_type}", 0)
                assert rate == 0, (
                    f"tito_session_mismatch_rate/{strict_type}={rate:.4f} must be 0 — "
                    "this indicates a bug in the TITO algorithm or chat template. "
                    "Please check your tito model and chat template."
                )
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(all_sample_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = counts.get("0.0", 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = counts.get("1.0", 0) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
