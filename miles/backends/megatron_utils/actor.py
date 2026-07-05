import logging
import random
import socket
from argparse import Namespace
from contextlib import nullcontext
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig

from miles.ray.train_actor import TrainRayActor
from miles.utils import train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint
from .initialize import init, is_megatron_main_rank
from .lora_utils import is_lora_enabled
from .model import forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state
from .replay_utils import get_register_replay_list_func
from .tms_utils import assert_tms_hook_mode_matches_arch
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref)

        init(args)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_main_rank = is_megatron_main_rank()

        if self._is_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            # MILES_TMS_HOOK_MODE=torch switches torch_memory_saver into
            # PyTorch's CUDAPluggableAllocator path, avoiding the
            # LD_PRELOAD libc malloc hook that segfaults during
            # build_cpu_bucket_cache on CUDA 12.9 / Blackwell. Must be
            # set BEFORE any tms call that triggers _ensure_initialized.
            import os as _os

            mode = _os.environ.get("MILES_TMS_HOOK_MODE")
            # Fail fast on the preload-on-Blackwell segfault: a clear, actionable
            # error beats a tracebackless SIGSEGV during build_cpu_bucket_cache.
            assert_tms_hook_mode_matches_arch(mode)
            if mode in ("torch", "preload"):
                logger.info(f"Set torch_memory_saver.hook_mode to {mode!r}")
                torch_memory_saver.hook_mode = mode  # type: ignore[assignment]
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay")
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "broadcast":
                update_weight_cls = UpdateWeightFromDistributed
            else:
                update_weight_cls = UpdateWeightP2P
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            is_lora=is_lora_enabled(args),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        # MILES_SKIP_TMS_PAUSE=1 turns sleep/wake_up into a near no-op:
        # don't destroy the process groups (subsequent Phase A steps
        # call dist.get_rank() and the cache_owner sync uses NCCL), and
        # don't call torch_memory_saver.pause (crashes on CUDA 12.9 /
        # Blackwell + tms 0.0.9). 0.5B fits 32GB without aggressive
        # offload, so skipping is fine for the smoke.
        import os as _os

        skip_tms = _os.environ.get("MILES_SKIP_TMS_PAUSE") == "1"

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if not skip_tms:
            destroy_process_groups()

        tag = "default" if is_lora_enabled(self.args) else None
        if not skip_tms:
            torch_memory_saver.pause(tag=tag)
        else:
            torch.cuda.empty_cache()

        print_memory("after offload model")

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        tag = "default" if is_lora_enabled(self.args) else None
        import os as _os

        skip_tms = _os.environ.get("MILES_SKIP_TMS_PAUSE") == "1"

        if not skip_tms:
            torch_memory_saver.resume(tag=tag)

        clear_memory()
        if not skip_tms:
            reload_process_groups()
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    def _fill_replay_data(
        self,
        data_iterator,
        num_microbatches,
        rollout_data,
        data_key: str,
        replay_list: list,
        register_replay_list_func,
        if_sp_region=True,
    ):
        if data_key not in rollout_data:
            raise ValueError(f"{data_key} is required in rollout_data for replay.")

        for iterator in data_iterator:
            iterator.reset()

        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size
        qkv_format = self.args.qkv_format

        def pad_func(data, pad):
            _, num_layers, topk = data.shape
            pad_tensor = torch.full(
                (pad, num_layers, topk),
                fill_value=-1,
                device=data.device,
                dtype=data.dtype,
            )
            return torch.cat([data, pad_tensor], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
            replay_data = batch[data_key]
            tokens = batch["tokens"]
            assert len(replay_data) == len(tokens)
            for a, b in zip(replay_data, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # TODO: maybe extract a common process function for here and get_batch?

            if qkv_format == "bshd":
                max_seqlen = batch["max_seq_lens"][0]
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
                replay_data = torch.stack(replay_data, dim=0)
                batch_size, seqlen, num_layers, topk = replay_data.shape
                replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                replay_data = torch.cat(replay_data, dim=0)
                pad_size = parallel_state.tp.size * self.args.data_pad_size_multiplier
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

            if self.args.sequence_parallel and if_sp_region:
                seqlen = replay_data.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                replay_data = replay_data[start:end]

            register_replay_list_func(replay_list, replay_data, self.model)

        del rollout_data[data_key]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                self._fill_replay_data(
                    data_iterator,
                    num_microbatches,
                    rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=get_register_replay_list_func(m),
                    if_sp_region=m.if_sp_region,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.offload_train:
                destroy_process_groups()
            return

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and not is_lora_enabled(self.args):
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    # ------------------------------------------------------------------
    # F4 sender-side API (RLix mode) — top-level Ray methods consumed by
    # MilesModelUpdateService.run_sync_session via a single composite RPC
    # in iter 12. This iter (11) lands the cache build + cache_owner role
    # report only; the run_sync_session composite Ray method itself is
    # iter 12's concern.
    #
    # F18 cache_owner uniqueness: exactly one rank reports
    # is_cache_owner=True at init Step 6.5 (pp0 + dp0 + tp0 + cp0). Other
    # ranks participate in the collective gather (so the cache_owner can
    # produce HF-format weights) but discard the resulting tensors and
    # advance their own _cache_ready_step pointer in lockstep.
    # ------------------------------------------------------------------

    def _ensure_cpu_bucket_cache(self):
        """Lazily initialize the per-rank :class:`CPUBucketCache`.

        Constructed once and reused across every training step's bucket
        rebuild. Bucket size cap comes from
        ``args.miles_model_update_bucket_size_mb`` (cf. F10 S2/S3a-2
        startup checks).
        """
        if not hasattr(self, "_cpu_bucket_cache") or self._cpu_bucket_cache is None:
            from .update_weight.cpu_bucket_cache import CPUBucketCache

            max_bytes = int(self.args.miles_model_update_bucket_size_mb) * 1024 * 1024
            self._cpu_bucket_cache = CPUBucketCache(max_bucket_size_bytes=max_bytes)
            # F20: bucket build / sync session each acquire this lock for
            # the whole critical section (single-method-single-critical-
            # section). Cross-RPC locking is forbidden (Layer 1 / F04).
            import threading as _threading

            self._cache_lock = _threading.Lock()
        return self._cpu_bucket_cache

    @staticmethod
    def _is_cache_owner_rank() -> bool:
        """F18: exactly one rank returns True (pp0 + dp0 + tp0 + cp0).

        Mirrors the existing
        ``UpdateWeightFromTensor._is_distributed_src_rank`` predicate so
        the cache_owner agrees with the existing weight-update sender
        rank. cp_rank == 0 is implicit when cp_size == 1; the
        intra_dp_cp.rank predicate already covers it for cp_size > 1.
        """
        # ``get_parallel_state`` lives in ``..training_utils.parallel`` (already
        # imported at module top); ``mpu`` is megatron.core.parallel_state.
        from megatron.core import parallel_state as mpu

        return (
            get_parallel_state().intra_dp_cp.rank == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        )

    def report_cache_owner_role(self) -> tuple[int, bool]:
        """Top-level Ray method — return (rank, is_cache_owner) for this rank.

        Called once at init Step 6.5 by ``RayTrainGroup.collect_cache_owner_roles``
        (iter 15) so the orchestrator can construct the cache_owner-actor
        handle pair the :class:`MilesModelUpdateService` needs.
        """
        return int(dist.get_rank()), bool(self._is_cache_owner_rank())

    def build_cpu_bucket_cache(self, step: int) -> int:
        """F4 build: gather HF-format weights into CPU buckets for `step`.

        Reuses the existing ``HfWeightIteratorBase`` pipeline so the
        produced tensors are full-shape HF-named (PP broadcast + EP
        broadcast + TP all-gather + ``convert_to_hf`` already applied).
        Plain ``named_params_and_buffers`` would only iterate per-rank
        Megatron shards — wrong for the F4 receiver which expects whole
        HF-format weights by name.

        Every rank participates in the gather (so the cache_owner sees a
        complete set). Only the cache_owner stores the resulting tensors;
        other ranks call ``put_empty_step`` so their
        ``_cache_ready_step`` pointer matches.

        ``step == -1`` is the init bootstrap path (base from-checkpoint
        weights); ``step >= 0`` is a post-training-step refresh.

        Returns the published step (echoed for caller logging).
        """
        from .update_weight.cpu_bucket_cache import BucketEntry
        from .update_weight.hf_weight_iterator_base import HfWeightIteratorBase

        cache = self._ensure_cpu_bucket_cache()
        is_owner = self._is_cache_owner_rank()

        # Construct the HF iterator the same way self.weight_updater does
        # internally, so the cache_owner sees identical name/shape/dtype
        # conventions to the existing standalone broadcast path.
        iterator = HfWeightIteratorBase.create(
            self.args,
            self.model,
            model_name=(
                type(self.hf_config).__name__.lower()
                if self.args.model_name is None
                else self.args.model_name
            ),
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )
        # The HF iterator pulls megatron-local weights from the same
        # weights_backuper used by the standalone path; reuse the
        # 'actor' tag so the rebuilt cache reflects the current actor
        # weights.
        megatron_local_weights = self.weights_backuper.get("actor")

        with self._cache_lock:
            if not is_owner:
                # Non-owner ranks must still drive the collective gather
                # (each chunk is implicitly cross-rank inside
                # _get_megatron_full_params + all_gather_params_async),
                # but they discard the resulting tensors and advance
                # their pointer in lockstep.
                for _ in iterator.get_hf_weight_chunks(megatron_local_weights):
                    pass
                cache.put_empty_step(int(step))
                return int(step)

            # F4 bucket layout (cache_owner only): pack (name, tensor)
            # pairs into buckets up to max_bucket_size_bytes each. The
            # iterator yields chunks of (name, hf_tensor); tensors come
            # back on the GPU device (cuda.current_device()) for the
            # standalone broadcast path, so we materialize to CPU here
            # before storing — BucketEntry rejects CUDA tensors per the
            # cpu_serialize transport contract.
            max_bytes = cache.max_bucket_size_bytes
            buckets: list[BucketEntry] = []
            current: dict[str, torch.Tensor] = {}
            current_bytes = 0
            current_elements = 0
            current_idx = 0
            for chunk in iterator.get_hf_weight_chunks(megatron_local_weights):
                for name, tensor in chunk:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.is_cuda:
                        tensor = tensor.detach().to("cpu")
                    tensor_bytes = tensor.element_size() * tensor.numel()
                    if current_bytes + tensor_bytes > max_bytes and current:
                        buckets.append(
                            BucketEntry(
                                bucket_index=current_idx,
                                params=current,
                                size_bytes=current_bytes,
                                element_count=current_elements,
                            )
                        )
                        current_idx += 1
                        current = {}
                        current_bytes = 0
                        current_elements = 0
                    current[name] = tensor
                    current_bytes += tensor_bytes
                    current_elements += tensor.numel()
            if current:
                buckets.append(
                    BucketEntry(
                        bucket_index=current_idx,
                        params=current,
                        size_bytes=current_bytes,
                        element_count=current_elements,
                    )
                )

            cache.put_step(int(step), buckets)
        return int(step)

    def run_sync_session(self, plan) -> int:
        """F4 sender-side composite RPC — single top-level Ray method.

        Per scope F04 (Layer 1 forbidden) the cache_owner exposes ONE
        top-level Ray method for transporting a sync session; helpers
        (per-engine cpu_serialize, NCCL group setup/broadcast/destroy)
        are in-method private helpers, NOT separate Ray RPCs. The
        ``_cache_lock`` is held for the whole transport phase so the
        bucket list snapshot + payload generation cannot be torn by a
        concurrent build_cpu_bucket_cache.

        ``plan`` is a plain mapping (per cozy-plan §Shared protocol
        contract — RLix may use a frozen dataclass internally but
        crosses the Ray boundary as ``dict[str, Any]`` so MILES has
        zero RLix import dependency). Required keys:

          - ``sync_id``: opaque str, traced through SGLang receiver logs.
          - ``version``: int weight version; -1 means base from-init.
          - ``group_name``: str, NCCL collective group name.
          - ``master_addr``: str, NCCL master rendezvous host.
          - ``master_port``: int, NCCL master rendezvous port (must
            be != 0 per F26 / scope C16).
          - ``timeout_s``: float (currently advisory; bounded
            asyncio.wait_for is enforced by service.sync_selected_workers).
          - ``target_handles``: dict[engine_index, ray_actor_handle]
            populated by service from manager.get_engine_handles.
          - ``cpu_serialize_local_ranks``: set[int] — engine indices that
            receive via the cpu_serialize tmpfs path.
          - ``broadcast_local_ranks``: set[int] — engine indices that
            receive via the NCCL broadcast non-colocate path.
          - ``comm_ranks``: dict[engine_index, int] — per-engine NCCL
            rank within the dynamic broadcast group (cache_owner ==
            rank 0).

        Non-cache_owner ranks raise: only the cache_owner has the bucket
        data and may drive transport. The MilesModelUpdateService picks
        the right actor handle via report_cache_owner_role.
        """
        if not self._is_cache_owner_rank():
            raise RuntimeError(
                "run_sync_session called on non-cache_owner rank "
                f"({dist.get_rank()}); service.sync_selected_workers must "
                "drive only the cache_owner actor (selected via "
                "report_cache_owner_role at init Step 6.5)."
            )

        if not isinstance(plan, dict):
            raise TypeError(
                f"run_sync_session expects plan as a dict (cross-Ray-boundary "
                f"plain mapping); got {type(plan).__name__}"
            )

        required = (
            "sync_id",
            "version",
            "group_name",
            "master_addr",
            "master_port",
            "timeout_s",
            "target_handles",
            "cpu_serialize_local_ranks",
            "broadcast_local_ranks",
            "comm_ranks",
            "world_size",
        )
        missing = [k for k in required if k not in plan]
        if missing:
            raise KeyError(f"run_sync_session plan missing keys: {missing}")

        sync_id = str(plan["sync_id"])
        version = int(plan["version"])
        group_name = str(plan["group_name"])
        master_addr = str(plan["master_addr"])
        master_port = int(plan["master_port"])
        if master_port == 0:
            raise ValueError(
                "F26 / C16: master_port=0 forbidden (other ranks cannot "
                "discover the ephemeral port); MilesModelUpdateService.run "
                "must claim a deterministic port via SharedStorage."
            )
        target_handles: dict[int, Any] = dict(plan["target_handles"])
        cpu_serialize_local_ranks: set[int] = set(plan["cpu_serialize_local_ranks"])
        broadcast_local_ranks: set[int] = set(plan["broadcast_local_ranks"])
        comm_ranks: dict[int, int] = dict(plan["comm_ranks"])
        world_size: int = int(plan["world_size"])

        cache = self._ensure_cpu_bucket_cache()
        with self._cache_lock:
            buckets = cache.get_step(version)
            if not buckets:
                logger.info(
                    "run_sync_session sync_id=%s version=%s found 0 buckets — empty publish",
                    sync_id,
                    version,
                )
                return version

            # Path A: cpu_serialize per-engine RPC (tmpfs payload). The
            # wrapper owns the tmpfs file lifecycle (try/finally
            # os.unlink) per scope F28; payload is materialized once
            # per (bucket, engine) pair so peak /dev/shm = 1× bucket
            # size (serial per-bucket receiver invocation).
            for bucket in buckets:
                if not cpu_serialize_local_ranks:
                    break
                self._dispatch_cpu_serialize_bucket(
                    sync_id=sync_id,
                    bucket=bucket,
                    target_handles={
                        idx: target_handles[idx]
                        for idx in cpu_serialize_local_ranks
                        if idx in target_handles
                    },
                )

            # Path B: NCCL broadcast non-colocate path. Set up a dynamic
            # group with TCP rendezvous, broadcast each bucket from
            # cache_owner (rank 0), and tear the group down after the
            # last bucket. F25: warmup allreduce on every CREATE; F26
            # already enforced master_port != 0 above; F03/Anti-regression
            # invariant #3: is_group_exist no-op guard on destroy is
            # provided by SGLangEngine.destroy_collective_group.
            if broadcast_local_ranks:
                self._dispatch_nccl_broadcast(
                    sync_id=sync_id,
                    buckets=buckets,
                    target_handles={
                        idx: target_handles[idx]
                        for idx in broadcast_local_ranks
                        if idx in target_handles
                    },
                    group_name=group_name,
                    master_addr=master_addr,
                    master_port=master_port,
                    comm_ranks=comm_ranks,
                    world_size=world_size,
                )

        return version

    def _dispatch_cpu_serialize_bucket(
        self,
        *,
        sync_id: str,
        bucket,  # BucketEntry
        target_handles: dict[int, Any],
    ) -> None:
        """In-method helper: cpu_serialize one bucket to its target engines.

        Iter 12 establishes the dispatch shape; the actual SGLang
        receiver methods (update_weights_from_cpu_bucket, route
        registration) land in iter 13. For now, drive the per-engine
        RPC with payload_bytes (Ray auto-derefs ObjectRef at the top
        level; we pass bytes directly so there is no double-deref).
        """
        if not target_handles:
            return
        # Serialize once per bucket: torch.save into a bytes buffer.
        # Iter 13 wraps this onto tmpfs; for iter 12 the bytes form is
        # sufficient for the dispatch shape.
        import io as _io

        buffer = _io.BytesIO()
        torch.save(dict(bucket.params), buffer)
        payload_bytes = buffer.getvalue()
        # Per-engine RPC, serial per-bucket so peak /dev/shm = 1x bucket
        # (cf. scope F28 tmpfs lifecycle).
        for engine_index, handle in target_handles.items():
            ray.get(
                handle.update_weights_from_cpu_bucket.remote(
                    payload_bytes=payload_bytes,
                    bucket_index=int(bucket.bucket_index),
                    sync_id=sync_id,
                )
            )

    def _dispatch_nccl_broadcast(
        self,
        *,
        sync_id: str,
        buckets,
        target_handles: dict[int, Any],
        group_name: str,
        master_addr: str,
        master_port: int,
        comm_ranks: dict[int, int],
        world_size: int,
    ) -> None:
        """In-method helper: dynamic NCCL broadcast of bucket payloads.

        **MILES-side self-guard (cross-cutting review P1-8)**: the sender-
        side ``init_process_group`` + per-bucket ``dist.broadcast`` +
        ``dist.destroy_process_group`` is NOT yet wired here. The
        receiver-side fan-out below (``setup_collective_group`` +
        ``broadcast_parameter`` + ``destroy_collective_group``) would
        block forever on ``init_weights_update_group`` waiting for an
        absent rank-0 sender. Until the sender path lands, refuse to
        even attempt the receiver-side setup so MILES does not depend
        on the RLix-side service guard for safety. The MILES contract
        is "zero RLix import dependency"; MILES must self-guard.
        """
        if not target_handles:
            return
        raise NotImplementedError(
            "broadcast transport requires sender-side NCCL "
            "(init_process_group + dist.broadcast on the cache_owner). "
            "Until the sender-side path lands, plans must route every "
            "target through cpu_serialize. MilesModelUpdateService "
            "already raises in iter 19/20; this MILES-side guard makes "
            "the same invariant explicit at the receiver fan-out."
        )
        if world_size <= 0:
            raise ValueError(
                f"_dispatch_nccl_broadcast requires world_size > 0; got {world_size}"
            )
        # Receiver-side group create with the same world_size value the
        # sender uses (cache_owner == rank 0, plus one entry per
        # receiver engine that participates in the broadcast).
        ray.get(
            [
                handle.setup_collective_group.remote(
                    group_name=group_name,
                    master_addr=master_addr,
                    master_port=master_port,
                    rank=comm_ranks[engine_index],
                    world_size=int(world_size),
                )
                for engine_index, handle in target_handles.items()
            ]
        )
        try:
            for bucket in buckets:
                # Per-bucket metadata for SGLang's
                # update_weights_from_distributed admin route.
                names: list[str] = []
                dtypes: list[str] = []
                shapes: list[list[int]] = []
                for name, tensor in bucket.params.items():
                    names.append(name)
                    dtypes.append(str(tensor.dtype).replace("torch.", ""))
                    shapes.append(list(tensor.shape))
                ray.get(
                    [
                        handle.broadcast_parameter.remote(
                            sync_id=sync_id,
                            bucket_index=int(bucket.bucket_index),
                            group_name=group_name,
                            names=names,
                            dtypes=dtypes,
                            shapes=shapes,
                        )
                        for handle in target_handles.values()
                    ]
                )
        finally:
            ray.get(
                [
                    handle.destroy_collective_group.remote(group_name=group_name)
                    for handle in target_handles.values()
                ]
            )

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
