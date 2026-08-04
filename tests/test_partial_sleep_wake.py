"""F78 — DEV-ONLY-MVP scaffolding for F1–F3 (sleep/wake + admission + redispatch).

SCAFFOLDING ONLY — these tests do NOT execute as Gate acceptance evidence.
They land as a starting point for the post-coding GPU smoke run; logic
may need iteration once the tests actually run against real engines.

F1 — SGLang sleep/wake helpers (is_idle, abort_all_requests, post-sleep
     VRAM assert).
F2 — RolloutManager EngineInfo state machine + subset offload/onload +
     compound ops (shrink_engines / expand_engines / activate_routing /
     finish_init_offload).
F3 — Router admission lifecycle + multi_turn redispatch + scheduler-
     preempt classification.

All tests use mocks where GPU state is required so the file imports /
parses cleanly on a CPU box.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# Ensure the test environment can locate the package under test even
# when MILES is not installed editably.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


class TestEngineInfoStateMachine(unittest.TestCase):
    """F2 — EngineInfo 5-state shell/active/disabling/offloaded/loading."""

    def test_shell_state_has_no_handle(self):
        from miles.ray.rollout import EngineInfo

        info = EngineInfo(engine_index=0, state="shell", handle=None)
        self.assertTrue(info.is_shell())
        self.assertFalse(info.is_alive())

    def test_active_state_requires_handle(self):
        from miles.ray.rollout import EngineInfo

        # is_alive() short-circuits on state=="shell" first.
        info = EngineInfo(engine_index=0, state="active", handle=mock.Mock())
        self.assertFalse(info.is_shell())
        self.assertTrue(info.is_alive())


class TestRouterAdmissionLifecycle(unittest.TestCase):
    """F3a — admission lifecycle 4-state machine.

    Router instance is constructed with a stub args namespace; tests
    drive the sync helpers directly (the async _use_url path requires
    an event loop, covered separately).
    """

    def _build_router(self):
        from miles.router.router import MilesRouter

        args = mock.Mock()
        args.miles_router_max_connections = 8
        args.miles_router_timeout = 5.0
        args.sglang_server_concurrency = 1
        args.rollout_num_gpus = 4
        args.rollout_num_gpus_per_engine = 2
        args.rollout_health_check_interval = 1.0
        args.miles_router_health_check_failure_threshold = 3
        args.miles_router_middleware_paths = None
        return MilesRouter(args, verbose=False)

    def test_add_worker_admits_by_default(self):
        router = self._build_router()
        router._add_worker_internal("http://w1:8000")
        self.assertIn("http://w1:8000", router.enabled_workers)
        self.assertEqual(router.worker_request_counts["http://w1:8000"], 0)
        self.assertTrue(router._admission_declared)

    def test_disable_worker_keeps_request_counts(self):
        router = self._build_router()
        router._add_worker_internal("http://w1:8000")
        router._disable_worker_internal("http://w1:8000")
        self.assertNotIn("http://w1:8000", router.enabled_workers)
        # Preserved so in-flight balance accounting stays consistent.
        self.assertIn("http://w1:8000", router.worker_request_counts)
        self.assertEqual(router.worker_failure_counts["http://w1:8000"], 0)

    def test_remove_worker_drops_all_state(self):
        router = self._build_router()
        router._add_worker_internal("http://w1:8000")
        router._remove_worker_internal("http://w1:8000")
        self.assertNotIn("http://w1:8000", router.worker_request_counts)
        self.assertNotIn("http://w1:8000", router.enabled_workers)


class TestSchedulerPreemptClassification(unittest.TestCase):
    """F3 / F31 — _is_scheduler_preempt strict missing-metadata check."""

    def test_standalone_always_returns_false(self):
        from miles.rollout.generate_hub.multi_turn import _is_scheduler_preempt

        # Standalone mode: no meta_info or absent admission flag must
        # NOT classify as preempt.
        self.assertFalse(_is_scheduler_preempt({}, rlix_mode=False))
        self.assertFalse(
            _is_scheduler_preempt({"meta_info": {}}, rlix_mode=False)
        )

    def test_rlix_mode_missing_metadata_raises(self):
        from miles.rollout.base_types import RLixRouterMetadataError
        from miles.rollout.generate_hub.multi_turn import _is_scheduler_preempt

        with self.assertRaises(RLixRouterMetadataError):
            _is_scheduler_preempt({}, rlix_mode=True)
        with self.assertRaises(RLixRouterMetadataError):
            _is_scheduler_preempt({"meta_info": {}}, rlix_mode=True)

    def test_rlix_mode_classifies_admission_disabled(self):
        from miles.rollout.generate_hub.multi_turn import _is_scheduler_preempt

        out = {"meta_info": {"miles_admission_disabled": True}}
        self.assertTrue(_is_scheduler_preempt(out, rlix_mode=True))
        out = {"meta_info": {"miles_admission_disabled": False}}
        self.assertFalse(_is_scheduler_preempt(out, rlix_mode=True))


if __name__ == "__main__":
    unittest.main()
