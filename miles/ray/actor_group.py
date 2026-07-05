import asyncio
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class RayTrainGroup:
    """
    A group of ray actors

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]] | None = None,
        *,
        worker_placements: list | None = None,
        num_gpus_per_actor: float = 1,
        role: str,
        with_ref: bool,
    ) -> None:
        """Allocate train actors against a placement group.

        Standalone path: ``pg=(PlacementGroup, indices, gpu_ids)``,
        ``worker_placements=None``. Behavior unchanged from the legacy
        implementation.

        RLix-mode path: ``pg=None``, ``worker_placements=[WorkerPlacement, ...]``
        (length == ``num_nodes * num_gpus_per_node``). Each worker is
        scheduled on its own bundle inside the WorkerPlacement's
        placement_group; CVD is set per worker via runtime_env so the
        post-CVD process sees ``cuda:0`` (matches F33 / F34 invariants).

        Exactly one of ``pg`` / ``worker_placements`` must be supplied.

        M4 self-cleanup (scope F36): if the per-actor allocation loop
        raises (Ray placement timeout, OOM, etc.) any handles created
        before the failure are explicitly killed before the error
        propagates so the scheduler / Ray ledger stays consistent.
        """
        if (pg is None) == (worker_placements is None):
            raise ValueError(
                "RayTrainGroup requires exactly one of pg= or worker_placements="
                " (mutually exclusive). pg is the legacy standalone shape; "
                "worker_placements is the RLix per-worker view from "
                "MilesPlacementProvider.get_train_workers()."
            )
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role
        self.with_ref = with_ref
        self._actor_handles: list = []

        try:
            # Both helpers append to self._actor_handles as each actor
            # is created so the except block below sees PARTIAL
            # allocations on failure and can still kill them.
            if worker_placements is not None:
                self._allocate_gpus_via_placements(
                    worker_placements, num_gpus_per_actor
                )
            else:
                self._actor_handles = self._allocate_gpus_for_actor(pg, num_gpus_per_actor)
        except Exception:
            # M4 self-cleanup: kill any actors created before the failure
            # so the scheduler / Ray ledger doesn't think GPUs are free
            # while actors still hold them. Best-effort: each ray.kill is
            # wrapped to swallow secondary errors so the ORIGINAL
            # exception still propagates.
            for h in self._actor_handles:
                if h is None:
                    continue
                try:
                    ray.kill(h, no_restart=True)
                except Exception:  # noqa: BLE001
                    pass
            self._actor_handles = []
            raise

    def _allocate_gpus_via_placements(self, worker_placements, num_gpus_per_actor) -> None:
        """RLix per-worker placement path. F33 / F34 / F36.

        Appends each actor handle to ``self._actor_handles`` *as soon as
        it is created* so the M4 cleanup path in :meth:`__init__` can
        kill partial allocations on failure (codex review of iter 15).
        """
        import os as _os

        env_vars_base = {
            "NCCL_CUMEM_ENABLE": _os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }
        if source_patcher_config := self.args.dumper_source_patcher_config_train:
            env_vars_base["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert _os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."
            env_vars_base["LD_PRELOAD"] = dynlib_path
            env_vars_base["TMS_INIT_ENABLE"] = "1"
            env_vars_base["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
            # Per-actor switch into torch_memory_saver "torch" hook mode
            # (CUDAPluggableAllocator) which avoids the LD_PRELOAD libc
            # malloc hook that segfaults during build_cpu_bucket_cache on
            # Blackwell with pre-CUDA-13 wheels (the guard is CUDA-version
            # aware: preload is allowed on cu13+ Blackwell). The actor
            # reads this env at init.
            if (mode := _os.environ.get("MILES_TMS_HOOK_MODE")):
                env_vars_base["MILES_TMS_HOOK_MODE"] = mode
            # The guard's escape hatch is read inside the actor process;
            # forward it so it works under Ray runtime_env isolation.
            if (allow := _os.environ.get("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL")):
                env_vars_base["MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL"] = allow

        backend = self.args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor
        else:
            from miles.backends.experimental.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        world_size = len(worker_placements)
        master_addr: str | None = None
        master_port: int | None = None
        for rank, wp in enumerate(worker_placements):
            # Per-worker CVD so the post-import process sees cuda:0 only.
            cvd = ",".join(str(g) for g in wp.gpu_ids)
            env_vars = dict(env_vars_base)
            env_vars["CUDA_VISIBLE_DEVICES"] = cvd
            TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=wp.placement_group,
                    placement_group_bundle_index=wp.bundle_index,
                ),
            ).remote(world_size, rank, master_addr, master_port, local_rank=0)
            # Append IMMEDIATELY so __init__'s except block sees the
            # partial allocation and can kill it. master_addr fetch
            # below may raise; the cleanup path picks the actor up.
            self._actor_handles.append(actor)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())

    def collect_cache_owner_roles(self) -> list[tuple[int, bool, object]]:
        """F18 cache_owner uniqueness — fan-out report_cache_owner_role.

        Returns ``[(rank, is_owner, actor_handle), ...]`` so the F8
        driver / coordinator (iter 23 / 25) can pin the cache_owner
        actor handle into ``MilesModelUpdateService`` via
        :meth:`MilesCoordinator.register_model_update_resources`.
        """
        results = ray.get(
            [actor.report_cache_owner_role.remote() for actor in self._actor_handles]
        )
        return [
            (int(rank), bool(is_owner), self._actor_handles[i])
            for i, (rank, is_owner) in enumerate(results)
        ]

    async def build_cpu_bucket_cache(self, step: int) -> int:
        """F4b fan-out: drive build_cpu_bucket_cache(step) on every actor.

        Cache_owner builds + stores HF buckets; non-owner ranks
        participate in the collective gather and call put_empty_step
        in lockstep.
        """
        return await self._broadcast("build_cpu_bucket_cache", step)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if source_patcher_config := self.args.dumper_source_patcher_config_train:
            env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
            # Forward MILES_TMS_HOOK_MODE for consistency with the
            # placement path above (_allocate_gpus_via_placements).
            if (mode := os.environ.get("MILES_TMS_HOOK_MODE")):
                env_vars["MILES_TMS_HOOK_MODE"] = mode
            if (allow := os.environ.get("MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL")):
                env_vars["MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL"] = allow

        backend = self.args.train_backend
        if backend == "megatron":
            from miles.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor

        else:
            from miles.backends.experimental.fsdp_utils import FSDPTrainRayActor

            actor_impl = FSDPTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        # Create worker actors
        actor_handles = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            actor_handles.append(actor)

        return actor_handles

    async def init(self):
        """
        Allocate GPU resourced and initialize model, optimizer, local ckpt, etc.
        """
        return await self._broadcast("init", self.args, self.role, with_ref=self.with_ref)

    async def train(self, rollout_id, rollout_data_ref):
        """Do one rollout training"""
        await self._broadcast("train", rollout_id, rollout_data_ref)

    async def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        await self._broadcast("save_model", rollout_id, force_sync=force_sync)

    async def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        await self._broadcast("update_weights")

    async def onload(self):
        await self._broadcast("wake_up")

    async def offload(self):
        await self._broadcast("sleep")

    async def clear_memory(self):
        await self._broadcast("clear_memory")

    async def connect(self, critic_group):
        refs = [
            actor.connect_actor_critic.remote(critic)
            for actor, critic in zip(self._actor_handles, critic_group._actor_handles, strict=False)
        ]
        await asyncio.gather(*refs)

    async def set_rollout_manager(self, rollout_manager):
        await self._broadcast("set_rollout_manager", rollout_manager)

    async def _broadcast(self, method_name: str, *args, **kwargs) -> list:
        refs = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actor_handles]
        return await asyncio.gather(*refs)
