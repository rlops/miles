"""F12 placement provider — adapter between RLix-declared placements and the
per-worker view MILES needs.

The standalone path uses :func:`miles.ray.placement_group.create_placement_groups`
which builds Ray placement groups directly from MILES args. The RLix path
delegates allocation to the RLix scheduler (via a
``RollResourceManagerProxy`` injected by the coordinator), then translates
the resulting allocation into per-worker placements MILES code expects.

Design notes (per scope F33 / F102 / F35):

- ``WorkerPlacement`` is per-worker and node-local: ``placement_group`` is
  a Ray PlacementGroup; ``bundle_index`` indexes within the PG;
  ``gpu_ids`` is a tuple of node-local GPU ids the worker may claim.
  Multi-node-compatible structurally (no global GPU id assumption).
- ``MilesPlacementProvider`` is constructed with declared train and
  infer device mappings (NOT computed inside the provider — they come
  from the F8 driver's ``cluster_device_mappings`` so multi-pipeline
  scenarios don't double-allocate).
- ``get_all_rollout_engine_placements()`` returns the FULL engine table
  (length == ``rollout_num_gpus // rollout_num_gpus_per_engine``). Init
  bootstrap (iter 26) creates every engine; runtime expand activates
  subsets.
- F33: SGLang ``base_gpu_id`` in RLix mode is the engine's first
  *physical* GPU id (from ``reordered_gpu_ids``), NOT 0. Unlike the
  train path, rollout actors are launched with visible devices left
  unset (``NOSET_VISIBLE_DEVICES_ENV_VARS_LIST``), so no per-worker CVD
  remapping happens and ``_to_local_gpu_id`` returns the physical id
  unchanged. The placement provider records physical ids in
  ``WorkerPlacement.gpu_ids`` for diagnostics. (If rollout actors are
  later given per-worker CVD like the train path, this note and the
  SGLang launch must switch to ``base_gpu_id=0`` together.)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class WorkerPlacement:
    """Per-worker view of a Ray placement group bundle.

    ``placement_group`` is the Ray PG (opaque object — only Ray APIs
    interpret it). ``bundle_index`` is the logical bundle index within
    the PG, used by ``PlacementGroupSchedulingStrategy(
    placement_group=pg, placement_group_bundle_index=...)``.
    ``gpu_ids`` is a tuple of node-local GPU ids. ``node_rank`` is the
    Ray node rank (0 for single-node).

    Frozen / hashable so the provider can return immutable views and
    callers can use placements as dict keys for engine_index lookups.
    """

    placement_group: object
    bundle_index: int
    gpu_ids: tuple[int, ...]
    node_rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_ids, tuple):
            raise TypeError("gpu_ids must be a tuple (frozen)")
        if any(g < 0 for g in self.gpu_ids):
            raise ValueError(f"gpu_ids must be non-negative; got {self.gpu_ids}")
        if list(self.gpu_ids) != sorted(self.gpu_ids):
            raise ValueError(
                f"gpu_ids must be sorted (first-build contiguous-mapping invariant, "
                f"per scope F35 startup structural assert); got {self.gpu_ids}"
            )


class MilesPlacementProvider:
    """Adapter from RLix-declared mappings to MILES per-worker placements.

    Constructed by the F8 driver / coordinator with:
      - ``resource_manager_proxy``: opaque proxy delegating allocation to
        the RLix scheduler. Iter 14 keeps it as ``Any`` (the concrete
        type lives in the rlix package). Provider does NOT instantiate
        a new proxy — the caller injects one (per F12 forbidden:
        provider must NOT self-construct the manager).
      - ``train_device_mapping``: list of physical GPU ids the train
        actors will claim (length ==
        ``actor_num_nodes * actor_num_gpus_per_node``).
      - ``infer_device_mapping``: list of physical GPU ids the inference
        engines will claim (length == ``rollout_num_gpus``).
      - ``rollout_num_gpus_per_engine``: tp_size for each engine.
      - ``num_gpus_per_node``: declared.

    The mappings are passed in (not derived) so multi-pipeline
    scenarios can have different declared mappings without the
    provider re-deriving per-pipeline conflicts.
    """

    def __init__(
        self,
        *,
        resource_manager_proxy,
        train_device_mapping: list[int],
        infer_device_mapping: list[int],
        rollout_num_gpus_per_engine: int,
        num_gpus_per_node: int,
    ):
        if rollout_num_gpus_per_engine <= 0:
            raise ValueError(
                f"rollout_num_gpus_per_engine must be > 0; got {rollout_num_gpus_per_engine}"
            )
        if num_gpus_per_node <= 0:
            raise ValueError(f"num_gpus_per_node must be > 0; got {num_gpus_per_node}")
        if len(infer_device_mapping) % rollout_num_gpus_per_engine != 0:
            raise ValueError(
                f"infer_device_mapping ({infer_device_mapping}) must divide evenly by "
                f"rollout_num_gpus_per_engine ({rollout_num_gpus_per_engine}); "
                f"this mirrors the F10 C6 startup assert."
            )
        # First-build contiguous-mapping structural assert (scope F35).
        # Non-contiguous / custom-ordered mappings need an explicit
        # scheduler_dp_rank -> engine_index adapter (A18 / F95) that is
        # use-case-triggered, not part of M11.1.
        if list(infer_device_mapping) != sorted(infer_device_mapping):
            raise ValueError(
                f"first build requires sorted infer_device_mapping; got "
                f"{infer_device_mapping}. Non-contiguous mapping requires the "
                f"A18 / F95 scheduler_dp_rank adapter (follow-up)."
            )
        # First-build also requires GAP-free (gpu_ids in each engine slice
        # must be a contiguous integer run). E.g. infer_device_mapping=[0, 2]
        # with tp=2 is sorted but the slice (0, 2) is not contiguous.
        for engine_idx in range(len(infer_device_mapping) // int(rollout_num_gpus_per_engine)):
            start = engine_idx * int(rollout_num_gpus_per_engine)
            slice_ids = infer_device_mapping[start : start + int(rollout_num_gpus_per_engine)]
            expected = list(range(slice_ids[0], slice_ids[0] + int(rollout_num_gpus_per_engine)))
            if list(slice_ids) != expected:
                raise ValueError(
                    f"first build requires gap-free GPU ids per engine; engine "
                    f"{engine_idx} got {slice_ids}, expected {expected}. Non-"
                    f"contiguous mapping requires the A18 / F95 adapter."
                )

        self._proxy = resource_manager_proxy
        self._train_device_mapping = list(train_device_mapping)
        self._infer_device_mapping = list(infer_device_mapping)
        self._per_engine = int(rollout_num_gpus_per_engine)
        self._num_gpus_per_node = int(num_gpus_per_node)

    @property
    def engine_count(self) -> int:
        return len(self._infer_device_mapping) // self._per_engine

    def get_all_rollout_engine_placements(self) -> list[WorkerPlacement]:
        """Full engine table — length == :attr:`engine_count`.

        Independent of any runtime-allocated subset: full INIT (iter 26)
        creates every engine, then runtime grants wake selected indices.
        Each engine's ``gpu_ids`` is the contiguous slice of
        ``infer_device_mapping`` covering its tp_size GPUs.

        The Ray PlacementGroup itself is requested from
        ``resource_manager_proxy.allocate_placement_group`` so the
        scheduler can satisfy multi-pipeline allocation policy. F35
        startup structural asserts run in :meth:`assert_structural`.
        """
        engine_count = self.engine_count
        # Ask the proxy for a PG covering the inference pool. ROLL's
        # ``ResourceManager.allocate_placement_group`` returns
        # ``List[List[Dict]]`` (length=world_size, each inner list one dict
        # per GPU with ``placement_group`` keying the Ray PG). In single-
        # node deployments every dict references the same node-PG (a
        # single-bundle PG containing all node GPUs); ``bundle_index`` is
        # therefore always 0. We retain the per-engine indexing so the
        # multi-node path stays correct when each engine's PG differs.
        allocated = self._proxy.allocate_placement_group(
            world_size=len(self._infer_device_mapping),
            device_mapping=tuple(self._infer_device_mapping),
        )
        placements: list[WorkerPlacement] = []
        for engine_idx in range(engine_count):
            start = engine_idx * self._per_engine
            global_slice = tuple(
                self._infer_device_mapping[start : start + self._per_engine]
            )
            pg = allocated[start][0]["placement_group"]
            # F102: derive node_rank from the first GLOBAL physical GPU
            # id of this slice. num_gpus_per_node tells us node
            # boundaries on a homogeneous cluster. Proxy is expected to
            # pin each engine slice to a single node — verify that
            # invariant here so multi-node configs fail fast at startup
            # instead of producing invalid CVDs at actor spawn.
            node_rank = global_slice[0] // self._num_gpus_per_node
            slice_node_ranks = {g // self._num_gpus_per_node for g in global_slice}
            if slice_node_ranks != {int(node_rank)}:
                raise ValueError(
                    f"engine_index={engine_idx} spans multiple nodes "
                    f"(global_slice={global_slice}, node_ranks={sorted(slice_node_ranks)}); "
                    f"first build requires each engine slice pinned to a single node."
                )
            # Store NODE-LOCAL gpu ids in WorkerPlacement.gpu_ids per
            # the docstring/plan invariant — single-node deployments
            # see global == node-local; multi-node deployments need the
            # mod so CVD construction in actor_group.py:141 produces
            # ids the per-node CUDA driver actually enumerates (R09-F1
            # fix).
            node_local_gpu_ids = tuple(g % self._num_gpus_per_node for g in global_slice)
            placements.append(
                WorkerPlacement(
                    placement_group=pg,
                    bundle_index=0,
                    gpu_ids=node_local_gpu_ids,
                    node_rank=int(node_rank),
                )
            )
        return placements

    def get_active_engine_indices(
        self,
        allocated_gpus: Iterable[int],
        tp_size: int,
    ) -> frozenset[int]:
        """Map a runtime-allocated GPU set back to engine indices.

        Used by the coordinator's runtime-expand path (iter 23) to
        translate ``scheduler.allocate_inference_resources`` results
        into engine indices the manager can wake. Each engine's
        tp_size GPUs must be either fully present or fully absent in
        ``allocated_gpus`` — partial allocation is invalid.
        """
        if int(tp_size) != self._per_engine:
            raise ValueError(
                f"tp_size mismatch: provider was constructed with "
                f"rollout_num_gpus_per_engine={self._per_engine}, got tp_size={tp_size}"
            )
        allocated = set(int(g) for g in allocated_gpus)
        active: set[int] = set()
        for engine_idx in range(self.engine_count):
            start = engine_idx * self._per_engine
            slice_gpu_ids = set(self._infer_device_mapping[start : start + self._per_engine])
            in_allocated = slice_gpu_ids & allocated
            if not in_allocated:
                continue
            if in_allocated != slice_gpu_ids:
                raise ValueError(
                    f"partial-engine allocation for engine_index={engine_idx}: "
                    f"declared GPUs={sorted(slice_gpu_ids)}, allocated subset="
                    f"{sorted(in_allocated)}. Schedulers must allocate full "
                    f"engine slices."
                )
            active.add(engine_idx)
        return frozenset(active)

    def get_train_workers(self) -> list[WorkerPlacement]:
        """Per-worker placements for the train pool.

        ROLL's ``ResourceManager.allocate_placement_group`` returns
        ``List[List[Dict]]`` (one outer entry per worker, inner entries one
        per worker GPU; each dict carries ``placement_group``). For the
        single-node, one-GPU-per-train-worker case every dict points at
        the same node-PG (a single-bundle PG containing all node GPUs),
        so ``bundle_index`` is always 0.
        """
        allocated = self._proxy.allocate_placement_group(
            world_size=len(self._train_device_mapping),
            device_mapping=tuple(self._train_device_mapping),
        )
        placements: list[WorkerPlacement] = []
        for idx, gpu_id in enumerate(self._train_device_mapping):
            pg = allocated[idx][0]["placement_group"]
            # Each train worker holds exactly one GPU. Convert the
            # global id to node-local for WorkerPlacement.gpu_ids per
            # the multi-node invariant (R09-F1).
            global_gpu = int(gpu_id)
            node_rank = global_gpu // self._num_gpus_per_node
            node_local_gpu = global_gpu % self._num_gpus_per_node
            placements.append(
                WorkerPlacement(
                    placement_group=pg,
                    bundle_index=0,
                    gpu_ids=(node_local_gpu,),
                    node_rank=int(node_rank),
                )
            )
        return placements

    def assert_structural(self, placements: list[WorkerPlacement]) -> None:
        """F35 startup structural asserts on a returned engine table.

        Verifies length == engine_count, each placement covers exactly
        ``rollout_num_gpus_per_engine`` GPUs, and the slice is
        contiguous starting at the expected stride.
        """
        if len(placements) != self.engine_count:
            raise RuntimeError(
                f"expected {self.engine_count} placements; got {len(placements)}"
            )
        for engine_idx, wp in enumerate(placements):
            if len(wp.gpu_ids) != self._per_engine:
                raise RuntimeError(
                    f"engine_index={engine_idx}: expected {self._per_engine} GPUs, "
                    f"got {wp.gpu_ids}"
                )
            expected_start = engine_idx * self._per_engine
            global_expected = tuple(
                self._infer_device_mapping[expected_start : expected_start + self._per_engine]
            )
            # WorkerPlacement.gpu_ids is node-local (R09-F1) so compare
            # against the node-local projection of the declared global
            # slice. node_rank derived from the global slice's first id;
            # all ids must share the rank (multi-node-spanning slice
            # rejected at construction in get_all_rollout_engine_placements).
            expected_node_rank = global_expected[0] // self._num_gpus_per_node
            expected_node_local = tuple(
                g % self._num_gpus_per_node for g in global_expected
            )
            if tuple(wp.gpu_ids) != expected_node_local:
                raise RuntimeError(
                    f"engine_index={engine_idx}: expected node-local gpu_ids="
                    f"{expected_node_local} (from global slice {global_expected}), "
                    f"got {wp.gpu_ids}; first-build contiguous-mapping invariant"
                )
            if int(wp.node_rank) != int(expected_node_rank):
                raise RuntimeError(
                    f"engine_index={engine_idx}: expected node_rank="
                    f"{expected_node_rank}, got {wp.node_rank}"
                )


__all__ = ["WorkerPlacement", "MilesPlacementProvider"]
