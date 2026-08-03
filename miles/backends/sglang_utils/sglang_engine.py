import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, convert_target_modules_to_hf, is_lora_enabled
from miles.ray.ray_actor import RayActor
from miles.utils.gpu_probe import query_process_tree_gpu_used_gb
from miles.utils.env_report import collect_and_print_node_env_report
from miles.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def _register_miles_admin_routes(sglang_app) -> None:
    """Inject MILES F4 admin routes into the SGLang FastAPI app.

    The host site is ``miles/backends/sglang_utils/sglang_engine.py``;
    no SGLang fork is required (the route is added at runtime via
    FastAPI's ``add_api_route``). Idempotent: re-registering an existing
    route is a no-op (we check ``app.router.routes`` for the path).

    The handler currently returns HTTP 501 — iter 13 only establishes
    the route contract. The real loader callout (drive SGLang
    ``tokenizer_manager.update_weights_from_tensor`` under
    ``model_update_lock.writer_lock``, F27) is wired by
    :class:`MilesModelUpdateService` (iters 19/20) once the wire-format
    + per-bucket metadata flow are in place. Failing closed prevents
    iter 12's run_sync_session from misreporting a successful sync
    while the serving model is still on old weights.
    """
    try:
        existing_paths = {route.path for route in sglang_app.router.routes}
    except AttributeError:
        existing_paths = set()
    if "/update_weights_from_cpu_bucket" in existing_paths:
        return

    # Define the body schema at registration time so FastAPI parses the
    # JSON body correctly. Without a Pydantic model FastAPI treats the
    # `request` parameter as a query field and rejects the POST as 400.
    from pydantic import BaseModel as _BaseModel

    class _UpdateBucketBody(_BaseModel):
        payload_path: str
        bucket_index: int = -1
        sync_id: str = ""

    async def _route_update_weights_from_cpu_bucket(body: _UpdateBucketBody):
        """Receiver-side F4d loader: read the bucket payload from the
        sender-written tmpfs file, deserialize the named tensors, and
        forward to SGLang's tokenizer_manager.update_weights_from_tensor.

        The sender (`SGLangEngine.update_weights_from_cpu_bucket`) writes
        a torch-pickled ``dict[name, tensor]`` (constructed by
        `MegatronTrainRayActor._dispatch_cpu_serialize_bucket`) to
        ``body.payload_path`` and POSTs the body to this route.
        """
        # Lazy imports inside the handler so module import order stays
        # robust under multiprocessing spawn.
        import os as _os

        import torch
        from fastapi.responses import JSONResponse

        from sglang.srt.entrypoints.http_server import _global_state
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from sglang.srt.utils import MultiprocessingSerializer

        payload_path = body.payload_path
        if not payload_path or not _os.path.exists(payload_path):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"payload_path missing or not on disk: {payload_path!r}",
                },
            )

        try:
            named_tensors_dict = torch.load(payload_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fall back if the torch version requires weights_only=False.
            named_tensors_dict = torch.load(payload_path, map_location="cpu")

        named_tensors = [(str(k), v) for k, v in named_tensors_dict.items()]
        tp_size = int(_global_state.tokenizer_manager.server_args.tp_size)
        serialized_named_tensors = [
            MultiprocessingSerializer.serialize(named_tensors) for _ in range(tp_size)
        ]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=True,
        )
        try:
            success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
                obj, None
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"update_weights_from_tensor: {exc!r}"},
            )

        return JSONResponse(
            status_code=200 if success else 500,
            content={
                "success": bool(success),
                "message": str(message),
                "bucket_index": int(body.bucket_index),
                "sync_id": str(body.sync_id),
            },
        )

    sglang_app.add_api_route(
        "/update_weights_from_cpu_bucket",
        _route_update_weights_from_cpu_bucket,
        methods=["POST"],
    )


def _miles_launch_server_target(server_args: ServerArgs) -> None:
    """Child-process entry point used by :func:`launch_server_process`.

    Registers the MILES F4 admin routes inside the SGLang server process
    (the parent's registration on its own ``app`` instance does NOT
    propagate under ``multiprocessing.set_start_method('spawn')``) and
    then delegates to SGLang's ``launch_server``.
    """
    from sglang.srt.entrypoints.http_server import app as _child_app, launch_server

    _register_miles_admin_routes(_child_app)
    launch_server(server_args)


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    # The child target re-imports SGLang's http_server module and then
    # registers MILES routes against THAT app instance before launching.
    p = multiprocessing.Process(target=_miles_launch_server_target, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break

            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        engine_info_bootstrap_port=None,
    ):
        if env_report := self.args.env_report:
            collect_and_print_node_env_report(
                role="rollout",
                rank=self.rank,
                partial_env_report=env_report,
            )

        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            engine_info_bootstrap_port=engine_info_bootstrap_port,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        import os as _os

        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        if self.node_rank == 0 and self.router_ip and self.router_port:
            # M11.2 Option β: when MILES_INIT_DEFER_ADD_WORKER=1, skip the
            # router /add_worker POST so engines initialize with empty
            # router enabled_workers. The coordinator's F40 Runtime expand
            # (miles_coordinator.py:498-512) calls activate_routing later
            # which adds workers as engines wake. Standalone miles (no
            # rlix) leaves the env unset — existing behavior preserved.
            if _os.environ.get("MILES_INIT_DEFER_ADD_WORKER") == "1":
                logger.info(
                    "[sglang_engine] MILES_INIT_DEFER_ADD_WORKER=1 — skipping "
                    "router /add_worker at init (Option β / Gate 4(c)) for "
                    "url=http://%s:%d",
                    self.server_host, self.server_port,
                )
                return
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_miles_router:
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or miles router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                payload = {
                    "url": f"http://{self.server_host}:{self.server_port}",
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def register_with_router(self) -> None:
        """M11.2 Option β: post /add_worker to the local router on demand.

        Called by ``RolloutManager.activate_routing`` for engines that
        skipped router registration during ``_init_normal`` (because
        ``MILES_INIT_DEFER_ADD_WORKER=1`` was set). **Raises** on
        non-2xx so the caller can abort the wake cycle rather than mark
        the engine ``active`` against a router that doesn't know about
        it (Codex Phase 3 review HIGH). Idempotent at the router layer
        (`_add_worker_internal` discards from ``dead_workers`` on re-add).

        No-op for non-rank-0 / no-router engines (returns silently).
        """
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return
        url = f"http://{self.server_host}:{self.server_port}"
        r = requests.post(
            f"http://{self.router_ip}:{self.router_port}/add_worker?url={url}"
        )
        r.raise_for_status()
        logger.info(
            "[sglang_engine] register_with_router OK url=%s status=%d",
            url, r.status_code,
        )

    def unregister_from_router(self) -> None:
        """M11.2 Option β / 3f: post /disable_worker before memory release.

        Called by ``RolloutManager.shrink_engines`` ahead of
        ``release_memory_occupation`` so the router admission is closed
        before VRAM is dropped. **Raises** on non-2xx so the caller
        aborts the release sequence rather than free GPU memory while
        the router can still dispatch to this URL (Codex Phase 3 review
        HIGH).

        No-op for non-rank-0 / no-router engines (returns silently).
        """
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return
        url = f"http://{self.server_host}:{self.server_port}"
        r = requests.post(
            f"http://{self.router_ip}:{self.router_port}/disable_worker?url={url}"
        )
        r.raise_for_status()
        logger.info(
            "[sglang_engine] unregister_from_router OK url=%s status=%d",
            url, r.status_code,
        )

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def get_remote_instance_transfer_engine_info(self, rank: int):
        # TODO: will be changed to `remote_instance_transfer_engine_info` when the sglang side is ready.
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/get_remote_instance_transfer_engine_info",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()["remote_instance_transfer_engine_info"]

    def get_parallelism_info(self, rank: int):
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/parallelism_config",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    def get_server_info(self):
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/server_info",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        serialized_tensors: str,
        config_dict: dict,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
    ):
        """Load a LoRA adapter from serialized tensor data."""
        payload = {
            "lora_name": lora_name,
            "serialized_tensors": serialized_tensors,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config

        return self._make_request(
            "load_lora_adapter_from_tensors",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_miles_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                try:
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        base = f"http://{self.server_host}:{self.server_port}"
        # new sglang change api from /get_weight_version to /model_info
        for endpoint in ("/model_info", "/get_weight_version"):
            response = requests.get(f"{base}{endpoint}")
            if response.status_code == 200:
                return response.json()["weight_version"]
        response.raise_for_status()

    def unload_lora_adapter(self, lora_name: str):
        """Unload LoRA adapter."""
        return self._make_request(
            "unload_lora_adapter",
            {"lora_name": lora_name},
        )

    def _log_whole_gpu(self, label: str) -> None:
        """SGL offload audit: whole-GPU used (nvidia-smi) for this engine's
        visible GPUs — before/after release/resume shows how much physical
        memory the engine actually returned, independent of PID-namespace
        issues that break per-process attribution on some containers."""
        import subprocess

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            )
            rows = dict(line.split(", ") for line in out.strip().splitlines())
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            visible = [x.strip() for x in cvd.split(",") if x.strip()] or sorted(rows)
            usage = {g: f"{rows.get(g)} MiB" for g in visible}
            # print (not logger): the engine actor process has no logging
            # handler configured, so logger.info is silently dropped; Ray
            # forwards actor stdout unconditionally.
            print(
                f"[SGL-OFFLOAD-AUDIT] {label} engine={self.server_host}:{self.server_port} gpus={usage}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SGL-OFFLOAD-AUDIT] {label}: probe failed {exc!r}", flush=True)

    def release_memory_occupation(self, tags: list[str] = None):
        """Release memory occupation. Available tags: weights, kv_cache."""
        self._log_whole_gpu("before release_memory_occupation")
        self.flush_cache()
        result = self._make_request(
            "release_memory_occupation",
            {"tags": tags},
        )
        self._log_whole_gpu("after release_memory_occupation")
        return result

    # ------------------------------------------------------------------
    # F1 RLix-mode sleep/wake helpers (used by F2 RolloutManager.shrink_engines)
    # ------------------------------------------------------------------

    def is_idle(self, timeout_s: float = 5.0) -> bool:
        """Return True iff the SGLang server has no in-flight or queued requests.

        Reads ``GET /v1/loads`` and inspects each per-DP entry's
        ``num_total_reqs`` (running + waiting). Current SGLang
        (``sglang/srt/entrypoints/v1_loads.py:get_loads``) returns:

            {
                "timestamp": ..., "version": ..., "dp_rank_count": N,
                "loads": [{ "num_running_reqs": ..., "num_waiting_reqs": ...,
                            "num_total_reqs": ..., ... }, ...],
                "aggregate": {"total_running_reqs": ..., "total_waiting_reqs": ...,
                              "total_reqs": ..., ...}
            }

        The plan explicitly forbids reading ``/server_info``'s
        ``num_running_reqs`` for this purpose (it is missing the waiting-queue
        depth and is version-dependent).

        Returns ``True`` when ``aggregate.total_reqs == 0``; falls back to
        scanning ``loads[*].num_total_reqs`` if ``aggregate`` is missing.
        Connection errors and HTTP failures bubble up to the caller.
        """
        if self.node_rank != 0:
            # Non-rank-0 nodes don't talk HTTP; defer to rank 0's verdict.
            return True
        url = f"http://{self.server_host}:{self.server_port}/v1/loads"
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Unexpected /v1/loads payload shape: {body!r}")
        aggregate = body.get("aggregate")
        if isinstance(aggregate, dict) and "total_reqs" in aggregate:
            return int(aggregate["total_reqs"]) == 0
        loads = body.get("loads")
        if not isinstance(loads, list):
            raise RuntimeError(
                f"/v1/loads response missing both 'aggregate.total_reqs' and "
                f"'loads' list: {body!r}"
            )
        for entry in loads:
            running = int(entry.get("num_running_reqs", 0))
            waiting = int(entry.get("num_waiting_reqs", 0))
            total = int(entry.get("num_total_reqs", running + waiting))
            if total != 0:
                return False
        return True

    def abort_all_requests(self, timeout_s: float = 10.0) -> dict:
        """POST ``/abort_request {"abort_all": true}`` to drain the engine.

        Used by F2 abort-drain-sleep ordering: admission close →
        ``_abort_engines`` → drain via ``is_idle`` →
        ``release_memory_occupation`` → post-sleep VRAM assert. Returns the
        SGLang JSON response so the caller can log the count of aborted reqs.
        """
        if self.node_rank != 0:
            return {}
        url = f"http://{self.server_host}:{self.server_port}/abort_request"
        response = requests.post(url, json={"abort_all": True}, timeout=timeout_s)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def assert_post_sleep_vram_below_threshold(
        self, threshold_gb: float, timeout_s: float = 5.0
    ) -> float:
        """Anti-regression invariant #8 — verify post-sleep VRAM is below threshold.

        Reads SGLang ``/server_info`` after ``release_memory_occupation`` /
        sleep, sums the per-DP ``internal_states[i].memory_usage`` categories
        (``weight + kvcache + graph``, all in GiB), takes the max across DPs,
        and raises if it exceeds ``threshold_gb``.

        Current SGLang exposes memory in:

            internal_states[i].memory_usage = {
                "weight": <GiB>,
                "kvcache": <GiB>,
                "token_capacity": <int>,  # not memory; skip
                "graph": <GiB>,
            }

        This is the *server-side* assert (distinct from the train-side
        ``torch.cuda.memory_allocated`` assert which is Layer 3 / M11.5
        follow-up). Returns the observed maximum memory_usage in GiB so
        callers can log it.

        Notes:
        - ``threshold_gb`` is typically
          ``args.miles_post_sleep_vram_threshold_gb`` (default 1.0 GiB).
        - On non-rank-0 nodes this is a no-op returning ``0.0``.
        """
        if self.node_rank != 0:
            return 0.0
        url = f"http://{self.server_host}:{self.server_port}/server_info"
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
        internal_states = body.get("internal_states") if isinstance(body, dict) else None
        if not isinstance(internal_states, list) or not internal_states:
            raise RuntimeError(
                "/server_info response missing 'internal_states' list; cannot enforce "
                "post-sleep VRAM threshold (Anti-regression invariant #8)"
            )
        memory_categories = ("weight", "kvcache", "graph")
        observed_max_gb = 0.0
        for state in internal_states:
            mem = state.get("memory_usage") if isinstance(state, dict) else None
            if not isinstance(mem, dict):
                raise RuntimeError(
                    "/server_info internal_states entry missing 'memory_usage' dict "
                    "(Anti-regression invariant #8)"
                )
            total_gb = sum(float(mem.get(k, 0.0) or 0.0) for k in memory_categories)
            observed_max_gb = max(observed_max_gb, total_gb)
        if observed_max_gb > float(threshold_gb):
            raise RuntimeError(
                f"Post-sleep VRAM {observed_max_gb:.3f} GiB (max across DPs, "
                f"weight+kvcache+graph) exceeds threshold "
                f"{float(threshold_gb):.3f} GiB on engine "
                f"{self.server_host}:{self.server_port} — torch_memory_saver may "
                f"have leaked. Check release_memory_occupation tags."
            )
        return observed_max_gb

    def _server_info_residual_gb(self, timeout_s: float = 5.0):
        """SGLang /server_info weight+kvcache+graph, max across DPs (GiB).

        This is *accounting* (KV static-pool size). It does NOT drop after a
        torch_memory_saver pause, so it is logged for diagnostics only and is
        never used as a hard gate. Returns None if unavailable.
        """
        try:
            body = self.get_server_info()
        except Exception:
            return None
        internal_states = body.get("internal_states") if isinstance(body, dict) else None
        if not isinstance(internal_states, list) or not internal_states:
            return None
        observed_max_gb = 0.0
        for state in internal_states:
            mem = state.get("memory_usage") if isinstance(state, dict) else None
            if not isinstance(mem, dict):
                continue
            total_gb = sum(float(mem.get(k, 0.0) or 0.0) for k in ("weight", "kvcache", "graph"))
            observed_max_gb = max(observed_max_gb, total_gb)
        return observed_max_gb

    def log_post_sleep_residual_diagnostics(
        self, threshold_gb: float | None = None, timeout_s: float = 5.0
    ):
        """Log attribution diagnostics after ``release_memory_occupation``.

        The hard residual gate is whole-GPU ``memory.used`` in RLix. This
        engine-side diagnostic still records this SGLang process tree's real
        resident GPU memory and ``/server_info`` accounting so high whole-GPU
        residual can be attributed to SGLang vs non-SGLang co-tenants.

        Returns the measured process-resident GiB, or ``None`` when
        unmeasurable (nvidia-smi missing / PID-namespace mismatch).
        """
        if self.node_rank != 0:
            return None
        _log = logging.getLogger(__name__)
        account_gb = self._server_info_residual_gb(timeout_s=timeout_s)
        root = getattr(self, "process", None)
        resident_gb = query_process_tree_gpu_used_gb(
            getattr(root, "pid", None), timeout_s=timeout_s
        )
        _log.info(
            "post-sleep residual diagnostic engine=%s:%s "
            "process_resident=%s GiB "
            "server_info_accounting(weight+kvcache+graph)=%s GiB "
            "whole_gpu_threshold=%s GiB",
            self.server_host,
            self.server_port,
            ("%.3f" % resident_gb) if resident_gb is not None else "n/a",
            ("%.3f" % account_gb) if account_gb is not None else "n/a",
            ("%.3f" % float(threshold_gb)) if threshold_gb is not None else "n/a",
        )
        if resident_gb is None:
            _log.warning(
                "post-sleep process-resident diagnostic unavailable on engine "
                "%s:%s (nvidia-smi missing or PID-namespace mismatch).",
                self.server_host,
                self.server_port,
            )
        return resident_gb

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        self._log_whole_gpu("before resume_memory_occupation")
        result = self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )
        self._log_whole_gpu("after resume_memory_occupation")
        return result

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron:
        after offload, weights are restored from disk instead of CPU cache.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self, mode: str = "retract"):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/pause_generation",
            json={"mode": mode},
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
        post_load_weights: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
                "post_load_weights": post_load_weights,
            },
        )

    def update_weight_version(self, weight_version: str):
        return self._make_request(
            "update_weight_version",
            {"new_version": weight_version},
        )

    # ------------------------------------------------------------------
    # F4d RLix-mode receiver methods (driven by run_sync_session iter 12).
    #
    # Iter 13 lands the actor-method surface area; full integration with
    # the SGLang server's tokenizer_manager weight loader is wired by
    # MilesModelUpdateService (iter 19/20), which knows how to materialize
    # tmpfs payload files (cpu_serialize) and orchestrate NCCL groups.
    # ------------------------------------------------------------------

    def update_weights_from_cpu_bucket(
        self,
        payload_bytes: bytes,
        bucket_index: int,
        sync_id: str,
    ) -> dict:
        """Receiver-side F4d entry point — write the bucket payload to tmpfs
        and POST it to ``/update_weights_from_cpu_bucket``.

        ``payload_bytes`` arrives as raw bytes. Per scope: Ray auto-derefs
        an ObjectRef wrapping a top-level argument, but we explicitly pass
        bytes (NOT an ObjectRef) so the wire shape is unambiguous.
        Tmpfs file lifecycle (F28) is wrapper-owned: this method writes
        the file, POSTs the SGLang admin route, then unconditionally
        unlinks in ``finally``. Per-bucket invocation is serial so peak
        ``/dev/shm`` usage is 1× bucket size.
        """
        if self.node_rank != 0:
            return {}
        import os as _os

        # F28 / F66: tmpfs dir + leak-detection-friendly file name.
        tmpfs_dir = "/dev/shm" if _os.path.isdir("/dev/shm") else "/tmp"
        from miles.backends.megatron_utils.update_weight.cpu_bucket_cache import (
            CPUBucketCache,
        )

        filename = CPUBucketCache.make_tmpfs_filename(bucket_index)
        path = _os.path.join(tmpfs_dir, filename)
        try:
            with open(path, "wb") as f:
                f.write(payload_bytes)
            url = f"http://{self.server_host}:{self.server_port}/update_weights_from_cpu_bucket"
            response = requests.post(
                url,
                json={
                    "payload_path": path,
                    "bucket_index": int(bucket_index),
                    "sync_id": str(sync_id),
                },
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()
        finally:
            try:
                _os.unlink(path)
            except FileNotFoundError:
                pass

    def setup_collective_group(
        self,
        group_name: str,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
    ) -> dict:
        """Receiver-side F25 NCCL collective group setup.

        Calls SGLang's ``/init_weights_update_group`` admin route to
        create the dynamic NCCL broadcast group used for the non-
        colocate transport. ``world_size`` MUST match the value the
        sender passes when it sets up its end of the NCCL group
        (cache_owner + every receiver participating in the broadcast,
        i.e. ``sum(receiver_engine_gpu_counts) + 1`` for the existing
        standalone updater pattern); 0 forbidden because NCCL group
        creation requires a positive world size.
        """
        if self.node_rank != 0:
            return {}
        if int(master_port) == 0:
            raise ValueError(
                "F26 / C16: master_port=0 forbidden in setup_collective_group"
            )
        if int(world_size) <= 0:
            raise ValueError(
                f"setup_collective_group requires world_size > 0; got {world_size}"
            )
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_addr,
                "master_port": int(master_port),
                "rank_offset": int(rank),
                "world_size": int(world_size),
                "group_name": group_name,
                "backend": "nccl",
            },
        )

    def destroy_collective_group(self, group_name: str) -> dict:
        """Receiver-side F25 / Anti-regression invariant #3 — destroy
        the named collective group with an ``is_group_exist`` no-op
        guard.

        SGLang exposes ``/destroy_weights_update_group`` which returns
        HTTP 200 on success and HTTP 400 BAD_REQUEST when the group does
        not exist (sglang/srt/entrypoints/http_server.py). 404 never
        appears for this route. We must tolerate the missing-group case
        — the cache_owner may have already torn down the group via a
        prior session — by treating 400 with a "group does not exist"
        message as a no-op success.
        """
        if self.node_rank != 0:
            return {}
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {"group_name": group_name},
            )
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            if response is None:
                raise
            status = response.status_code
            # SGLang returns 400 BAD_REQUEST on missing group; 404 kept
            # as a defensive catch in case future SGLang versions move to
            # 404 for the same condition.
            if status in (400, 404):
                body_text = ""
                try:
                    body_text = response.text or ""
                except Exception:  # noqa: BLE001
                    body_text = ""
                if status == 404 or "group does not exist" in body_text.lower() or "does not exist" in body_text.lower():
                    return {"status": "noop"}
            raise

    def broadcast_parameter(
        self,
        sync_id: str,
        bucket_index: int,
        group_name: str,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
    ) -> dict:
        """Receiver-side per-bucket broadcast trigger.

        SGLang's ``/update_weights_from_distributed`` admin route reads
        a list of named tensors from the dynamic NCCL group. Per
        existing distributed-updater pattern (broadcast.py /
        UpdateWeightFromDistributed) the request body carries
        ``names`` / ``dtypes`` / ``shapes`` lists in the same order.
        ``sync_id`` and ``bucket_index`` are echoed only for
        observability (logged into SGLang's response).

        :class:`MilesModelUpdateService` (iters 19/20) is responsible
        for matching the sender's ``dist.broadcast`` order to these
        metadata lists.
        """
        if self.node_rank != 0:
            return {}
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": list(names),
                "dtypes": list(dtypes),
                "shapes": list(shapes),
                "group_name": group_name,
                "_sync_id": str(sync_id),
                "_bucket_index": int(bucket_index),
            },
        )

    def finalize_weight_update(self) -> dict:
        """Receiver-side hook called once per sync after all bucket
        broadcasts complete. Invokes SGLang's ``/flush_cache`` to ensure
        inflight requests pick up the new weights on next prefill (the
        actual ``update_weight_version`` publish happens later via
        :meth:`update_weight_version` driven by
        ``RolloutManager.set_weight_version``).
        """
        if self.node_rank != 0:
            return {}
        self.flush_cache()
        return {"status": "finalized"}

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    engine_info_bootstrap_port: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
    }

    if sglang_overrides:
        kwargs.update(sglang_overrides)

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    if is_lora_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = 1
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)

        if args.lora_adapter_path is not None:
            kwargs["lora_paths"] = {LORA_ADAPTER_NAME: args.lora_adapter_path}
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
