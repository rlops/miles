"""Unit tests for the sender-side NCCL broadcast dispatch (rlops/rlix#42).

Covers plan miles-nccl-broadcast v7 evidence items:

- E1: dispatch ordering — receiver setup refs BEFORE sender join; metadata
  refs BEFORE tensor broadcasts; sender-owned teardown in ``finally`` on
  success AND on mid-bucket exception.
- E7: receiver-fails-during-setup — sender aborts bounded (setup ray.get
  failure or rendezvous timeout), still runs teardown, surfaces the error.
- E8: memory-preflight degradation — low free VRAM switches staging to
  tensor-by-tensor with the broadcast sequence/metadata order unchanged.

All Ray / torch.distributed / CUDA interactions are patched at the
``actor`` module namespace; ``_run_sender_broadcast`` is exercised
directly (module-level helper — no MegatronTrainRayActor construction).
"""

import types
import unittest
from unittest import mock

import torch

import miles.backends.megatron_utils.actor as actor_mod
from miles.backends.megatron_utils.update_weight.cpu_bucket_cache import BucketEntry


def _make_bucket(idx: int = 0, n_tensors: int = 2, numel: int = 4) -> BucketEntry:
    params = {f"w{idx}_{i}": torch.zeros(numel) for i in range(n_tensors)}
    size = sum(t.element_size() * t.numel() for t in params.values())
    return BucketEntry(
        bucket_index=idx,
        params=params,
        size_bytes=size,
        element_count=n_tensors * numel,
    )


class _FakeRef:
    def __init__(self, kind: str, engine_index: int, kwargs: dict):
        self.kind = kind
        self.engine_index = engine_index
        self.kwargs = kwargs


class _FakeRemoteMethod:
    def __init__(self, log: list, kind: str, engine_index: int):
        self._log = log
        self._kind = kind
        self._engine_index = engine_index

    def remote(self, **kwargs):
        self._log.append((f"{self._kind}.remote", self._engine_index))
        return _FakeRef(self._kind, self._engine_index, kwargs)


class _FakeHandle:
    def __init__(self, log: list, engine_index: int):
        self.setup_collective_group = _FakeRemoteMethod(log, "setup", engine_index)
        self.broadcast_parameter = _FakeRemoteMethod(log, "metadata", engine_index)
        self.destroy_collective_group = _FakeRemoteMethod(log, "destroy", engine_index)


class _Harness:
    """Patches ray/dist/init_process_group/staging seams in the actor
    module namespace and records a global event log."""

    def __init__(self, fail_get_kinds: set | None = None, join_error: Exception | None = None):
        self.log: list = []
        self.join_calls: list = []
        self.get_refs: list = []
        self._fail_get_kinds = fail_get_kinds or set()
        self._join_error = join_error
        self.sender_group = object()

    # --- fakes -------------------------------------------------------
    def ray_get(self, refs, timeout=None):
        kinds = tuple(sorted({r.kind for r in refs}))
        self.log.append(("ray.get", kinds, timeout))
        self.get_refs.extend(refs)
        for kind in kinds:
            if kind in self._fail_get_kinds:
                raise RuntimeError(f"injected {kind} failure")
        return [None for _ in refs]

    def ray_cancel(self, ref):
        self.log.append(("ray.cancel", ref.kind, ref.engine_index))

    def dist_broadcast(self, tensor, src, group=None):
        assert src == 0
        assert group is self.sender_group
        self.log.append(("broadcast", int(tensor.numel())))

    def dist_destroy(self, group):
        assert group is self.sender_group
        self.log.append(("sender_destroy",))

    def join(self, **kwargs):
        self.log.append(("sender_join",))
        self.join_calls.append(kwargs)
        if self._join_error is not None:
            raise self._join_error
        return self.sender_group

    # --- helpers -----------------------------------------------------
    def patches(self, free_bytes: int = 10**12):
        fake_ray = types.SimpleNamespace(get=self.ray_get, cancel=self.ray_cancel)
        fake_dist = types.SimpleNamespace(
            broadcast=self.dist_broadcast, destroy_process_group=self.dist_destroy
        )
        return [
            mock.patch.object(actor_mod, "ray", fake_ray),
            mock.patch.object(actor_mod, "dist", fake_dist),
            mock.patch.object(actor_mod, "init_process_group", self.join),
            mock.patch.object(
                actor_mod, "_resolve_staging_device", lambda: torch.device("cpu")
            ),
            mock.patch.object(actor_mod, "_query_free_bytes", lambda device: free_bytes),
        ]

    def run(self, *, buckets, handles, free_bytes: int = 10**12, timeout_s: float = 150.0):
        patches = self.patches(free_bytes=free_bytes)
        for p in patches:
            p.start()
        try:
            actor_mod._run_sender_broadcast(
                sync_id="sync-test",
                buckets=buckets,
                target_handles=handles,
                group_name="grp",
                master_addr="10.0.0.1",
                master_port=29500,
                comm_ranks={i: i + 1 for i in handles},
                world_size=1 + len(handles),
                timeout_s=timeout_s,
            )
        finally:
            for p in patches:
                p.stop()

    # --- log queries -------------------------------------------------
    def indices(self, predicate):
        return [i for i, ev in enumerate(self.log) if predicate(ev)]

    def first(self, predicate):
        idx = self.indices(predicate)
        assert idx, f"no matching event in log: {self.log}"
        return idx[0]

    def last(self, predicate):
        idx = self.indices(predicate)
        assert idx, f"no matching event in log: {self.log}"
        return idx[-1]


class TestE1DispatchOrdering(unittest.TestCase):
    def test_happy_path_ordering(self):
        h = _Harness()
        handles = {0: _FakeHandle(h.log, 0), 1: _FakeHandle(h.log, 1)}
        h.run(buckets=[_make_bucket(0), _make_bucket(1)], handles=handles)

        setup_last = h.last(lambda ev: ev[0] == "setup.remote")
        join_idx = h.first(lambda ev: ev[0] == "sender_join")
        setup_get = h.first(lambda ev: ev[0] == "ray.get" and ev[1] == ("setup",))
        # (1) every receiver setup dispatched before the sender joins;
        # (2) sender joins before the setup refs are awaited.
        self.assertLess(setup_last, join_idx)
        self.assertLess(join_idx, setup_get)

        # (3) per bucket: metadata refs dispatched before the first tensor
        # broadcast of that bucket; bucket refs awaited after broadcasts.
        first_meta = h.first(lambda ev: ev[0] == "metadata.remote")
        first_bcast = h.first(lambda ev: ev[0] == "broadcast")
        self.assertLess(first_meta, first_bcast)
        # 2 buckets x 2 tensors = 4 tensor broadcasts, once each (to the
        # group — not per receiver).
        self.assertEqual(len(h.indices(lambda ev: ev[0] == "broadcast")), 4)

        # (4) teardown ran at the end: receiver destroy fan-out + sender
        # destroy after the last broadcast.
        last_bcast = h.last(lambda ev: ev[0] == "broadcast")
        destroy_first = h.first(lambda ev: ev[0] == "destroy.remote")
        sender_destroy = h.first(lambda ev: ev[0] == "sender_destroy")
        self.assertLess(last_bcast, destroy_first)
        self.assertLess(destroy_first, sender_destroy)

    def test_teardown_on_mid_bucket_receiver_exception(self):
        h = _Harness(fail_get_kinds={"metadata"})
        handles = {0: _FakeHandle(h.log, 0)}
        with self.assertRaisesRegex(RuntimeError, "injected metadata failure"):
            h.run(buckets=[_make_bucket(0)], handles=handles)
        # Sender-owned finally still cancelled refs and destroyed both sides.
        self.assertTrue(h.indices(lambda ev: ev[0] == "ray.cancel"))
        self.assertTrue(h.indices(lambda ev: ev[0] == "destroy.remote"))
        self.assertTrue(h.indices(lambda ev: ev[0] == "sender_destroy"))

    def test_empty_target_handles_is_noop(self):
        h = _Harness()
        h.run(buckets=[_make_bucket(0)], handles={})
        self.assertEqual(h.log, [])

    def test_invalid_world_size_raises(self):
        h = _Harness()
        with self.assertRaises(ValueError):
            with mock.patch.object(actor_mod, "ray"), mock.patch.object(actor_mod, "dist"):
                actor_mod._run_sender_broadcast(
                    sync_id="s",
                    buckets=[],
                    target_handles={0: _FakeHandle(h.log, 0)},
                    group_name="g",
                    master_addr="a",
                    master_port=1,
                    comm_ranks={0: 1},
                    world_size=0,
                    timeout_s=1.0,
                )

    def test_missing_comm_rank_raises(self):
        h = _Harness()
        with self.assertRaises(KeyError):
            actor_mod._run_sender_broadcast(
                sync_id="s",
                buckets=[],
                target_handles={0: _FakeHandle(h.log, 0), 7: _FakeHandle(h.log, 7)},
                group_name="g",
                master_addr="a",
                master_port=1,
                comm_ranks={0: 1},
                world_size=3,
                timeout_s=1.0,
            )


class TestE7ReceiverFailureAbort(unittest.TestCase):
    def test_setup_failure_aborts_and_tears_down(self):
        h = _Harness(fail_get_kinds={"setup"})
        handles = {0: _FakeHandle(h.log, 0), 1: _FakeHandle(h.log, 1)}
        with self.assertRaisesRegex(RuntimeError, "injected setup failure"):
            h.run(buckets=[_make_bucket(0)], handles=handles)
        # No tensor was broadcast; teardown still ran on both sides.
        self.assertEqual(h.indices(lambda ev: ev[0] == "broadcast"), [])
        self.assertTrue(h.indices(lambda ev: ev[0] == "destroy.remote"))
        self.assertTrue(h.indices(lambda ev: ev[0] == "sender_destroy"))

    def test_sender_join_timeout_aborts_bounded(self):
        h = _Harness(join_error=RuntimeError("rendezvous timed out"))
        handles = {0: _FakeHandle(h.log, 0)}
        with self.assertRaisesRegex(RuntimeError, "rendezvous timed out"):
            h.run(buckets=[_make_bucket(0)], handles=handles)
        # Receiver destroy fan-out still dispatched; the sender group was
        # never created so sender-side destroy is correctly absent.
        self.assertTrue(h.indices(lambda ev: ev[0] == "destroy.remote"))
        self.assertEqual(h.indices(lambda ev: ev[0] == "sender_destroy"), [])

    def test_join_timeout_uses_rendezvous_budget(self):
        h = _Harness()
        handles = {0: _FakeHandle(h.log, 0)}
        h.run(buckets=[], handles=handles, timeout_s=150.0)
        self.assertEqual(len(h.join_calls), 1)
        timeout = h.join_calls[0]["timeout"]
        self.assertAlmostEqual(timeout.total_seconds(), 0.15 * 150.0)

    def test_budget_hierarchy_nests_inside_session_deadline(self):
        rendezvous, transport, teardown = actor_mod._broadcast_budgets(150.0)
        self.assertAlmostEqual(rendezvous, 0.15 * 150.0)
        self.assertAlmostEqual(transport, 0.5 * 150.0)
        self.assertAlmostEqual(teardown, 0.1 * 150.0)
        # C6 invariant: sender worst-case unwind — rendezvous + transport
        # + one in-flight collective (bounded by the pg timeout ==
        # rendezvous budget, NCCL watchdog) + teardown — < session
        # deadline.
        self.assertLess(rendezvous + transport + rendezvous + teardown, 150.0)
        # Fallback budgets when the service timeout is disabled.
        self.assertEqual(
            actor_mod._broadcast_budgets(0.0), actor_mod._BROADCAST_FALLBACK_BUDGETS_S
        )

    def test_disabled_async_error_handling_refuses_to_run(self):
        # codex impl-r2 high: an explicitly disabled NCCL watchdog would
        # reintroduce unbounded native collectives — the sender must
        # refuse to start rather than silently overwrite the env.
        for var in ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_ASYNC_ERROR_HANDLING"):
            h = _Harness()
            handles = {0: _FakeHandle(h.log, 0)}
            with mock.patch.dict("os.environ", {var: "0"}):
                with self.assertRaisesRegex(RuntimeError, "NCCL watchdog"):
                    h.run(buckets=[_make_bucket(0)], handles=handles)
            # Refused before any rendezvous: no join, no receiver setup get.
            self.assertEqual(h.indices(lambda ev: ev[0] == "sender_join"), [])

    def test_unset_async_error_handling_is_pinned_on(self):
        h = _Harness()
        handles = {0: _FakeHandle(h.log, 0)}
        import os

        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("TORCH_NCCL_ASYNC_ERROR_HANDLING", None)
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            h.run(buckets=[], handles=handles)
            self.assertEqual(os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"), "1")

    def test_nonzero_async_error_handling_values_accepted(self):
        h = _Harness()
        handles = {0: _FakeHandle(h.log, 0)}
        with mock.patch.dict("os.environ", {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "2"}):
            h.run(buckets=[], handles=handles)
        self.assertTrue(h.indices(lambda ev: ev[0] == "sender_join"))

    def test_stalled_transport_hits_cumulative_deadline_and_tears_down(self):
        # Receiver setup succeeds but the transport clock burns past the
        # cumulative deadline (e.g. a stalled peer eating watchdog-bounded
        # collectives): the sender must abort between collectives and
        # still run its finally teardown (codex impl-r1 high).
        h = _Harness()
        handles = {0: _FakeHandle(h.log, 0)}
        clock = {"now": 0.0}

        def _fake_monotonic():
            # Every observation advances the clock far beyond the budget.
            clock["now"] += 1000.0
            return clock["now"]

        with mock.patch.object(actor_mod, "_monotonic", _fake_monotonic):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                h.run(
                    buckets=[_make_bucket(0, n_tensors=2)],
                    handles=handles,
                    timeout_s=150.0,
                )
        self.assertTrue(h.indices(lambda ev: ev[0] == "destroy.remote"))
        self.assertTrue(h.indices(lambda ev: ev[0] == "sender_destroy"))


class TestE8MemoryPreflight(unittest.TestCase):
    def test_staging_plan_pure_decision(self):
        self.assertEqual(actor_mod._plan_bucket_staging(100, 200, 50), "bucket")
        self.assertEqual(actor_mod._plan_bucket_staging(100, 149, 50), "tensor")
        self.assertEqual(actor_mod._plan_bucket_staging(100, 150, 50), "bucket")

    def test_low_memory_degrades_without_changing_order(self):
        rich = _Harness()
        handles_rich = {0: _FakeHandle(rich.log, 0)}
        rich.run(buckets=[_make_bucket(0, n_tensors=3)], handles=handles_rich)

        poor = _Harness()
        handles_poor = {0: _FakeHandle(poor.log, 0)}
        poor.run(
            buckets=[_make_bucket(0, n_tensors=3)], handles=handles_poor, free_bytes=0
        )

        # Same broadcast sequence either way (3 tensors, same sizes) —
        # degradation changes batching only, not wire behavior.
        rich_bcasts = [ev for ev in rich.log if ev[0] == "broadcast"]
        poor_bcasts = [ev for ev in poor.log if ev[0] == "broadcast"]
        self.assertEqual(rich_bcasts, poor_bcasts)
        # Metadata names preserve bucket insertion order in both runs.
        rich_meta = [r for r in rich.get_refs if r.kind == "metadata"]
        poor_meta = [r for r in poor.get_refs if r.kind == "metadata"]
        self.assertEqual(rich_meta[0].kwargs["names"], poor_meta[0].kwargs["names"])
        self.assertEqual(rich_meta[0].kwargs["names"], ["w0_0", "w0_1", "w0_2"])

    def test_staging_margin_env_override(self):
        with mock.patch.dict("os.environ", {actor_mod._BROADCAST_STAGING_MARGIN_ENV: "2.5"}):
            self.assertEqual(actor_mod._staging_margin_bytes(), int(2.5 * 1024**3))
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop(actor_mod._BROADCAST_STAGING_MARGIN_ENV, None)
            self.assertEqual(actor_mod._staging_margin_bytes(), 1024**3)
        with mock.patch.dict(
            "os.environ", {actor_mod._BROADCAST_STAGING_MARGIN_ENV: "not-a-float"}
        ):
            with self.assertRaises(ValueError):
                actor_mod._staging_margin_bytes()
        with mock.patch.dict("os.environ", {actor_mod._BROADCAST_STAGING_MARGIN_ENV: "-1"}):
            with self.assertRaises(ValueError):
                actor_mod._staging_margin_bytes()


if __name__ == "__main__":
    unittest.main()
