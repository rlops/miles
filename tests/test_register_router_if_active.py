"""Contract test for the REAL RolloutManager.register_router_if_active
(rlix#42 sync-under-load bracket; codex impl-r15 asked for a test against
the production class, not coordinator-side fakes).

The manager instance is built via ``object.__new__`` with a hand-rolled
``_engines`` table; ``ray.get`` in the rollout module is patched to
unwrap the fakes' eager returns.
"""

import types
import unittest
from unittest import mock

import miles.ray.rollout as rollout_mod


def _manager_cls():
    cls = rollout_mod.RolloutManager
    meta = getattr(cls, "__ray_metadata__", None)
    return meta.modified_class if meta is not None else cls


class _FakeHandle:
    def __init__(self, log, idx, fail=False):
        def _register():
            if fail:
                raise RuntimeError(f"router add_worker failed for {idx}")
            log.append(idx)
            return None

        self.register_with_router = types.SimpleNamespace(remote=lambda: _register())


def _info(state, handle):
    return types.SimpleNamespace(state=state, handle=handle)


class TestRegisterRouterIfActive(unittest.TestCase):
    def _manager(self, engines):
        mgr = object.__new__(_manager_cls())
        mgr._engines = engines
        return mgr

    def test_registers_only_active_engines(self):
        log: list = []
        mgr = self._manager(
            {
                0: _info("offloaded", _FakeHandle(log, 0)),
                1: _info("active", _FakeHandle(log, 1)),
                2: _info("disabling", _FakeHandle(log, 2)),
                3: _info("active", _FakeHandle(log, 3)),
            }
        )
        with mock.patch.object(rollout_mod, "ray", types.SimpleNamespace(get=lambda r: r)):
            out = mgr.register_router_if_active([0, 1, 2, 3])
        self.assertEqual(out, [1, 3])
        self.assertEqual(sorted(log), [1, 3])

    def test_unknown_index_is_skipped_not_raised(self):
        log: list = []
        mgr = self._manager({1: _info("active", _FakeHandle(log, 1))})
        with mock.patch.object(rollout_mod, "ray", types.SimpleNamespace(get=lambda r: r)):
            out = mgr.register_router_if_active([1, 99])
        self.assertEqual(out, [1])

    def test_empty_probe_returns_empty(self):
        # The rlix coordinator uses an empty-list call as a version-skew
        # capability probe at registration time — must be a cheap no-op.
        mgr = self._manager({})
        out = mgr.register_router_if_active([])
        self.assertEqual(out, [])

    def test_register_failure_propagates(self):
        log: list = []
        mgr = self._manager({1: _info("active", _FakeHandle(log, 1, fail=True))})
        with mock.patch.object(rollout_mod, "ray", types.SimpleNamespace(get=lambda r: r)):
            with self.assertRaisesRegex(RuntimeError, "add_worker failed"):
                mgr.register_router_if_active([1])


if __name__ == "__main__":
    unittest.main()
