"""Tests for the rlix#42 single-mode explicit-mapping override
(`MILES_SINGLE_TRAIN_GPUS` / `MILES_SINGLE_INFER_GPUS` in
examples/rlix/run_miles_rlix.py::_build_cluster_device_mappings).

Covers codex impl-r9: matching, undersized, oversized, duplicate,
partial-intersection, subset, disjoint, half-set, and unset-legacy cases.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("RLIX_CONTROL_PLANE", "rlix")  # module import guard

from examples.rlix.run_miles_rlix import _build_cluster_device_mappings  # noqa: E402


def _args(actor_gpus=1, rollout_gpus=2):
    return types.SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=actor_gpus,
        rollout_num_gpus=rollout_gpus,
    )


def _env(train=None, infer=None):
    env = {}
    if train is not None:
        env["MILES_SINGLE_TRAIN_GPUS"] = train
    if infer is not None:
        env["MILES_SINGLE_INFER_GPUS"] = infer
    return mock.patch.dict("os.environ", env, clear=False)


class TestSingleMappingOverride(unittest.TestCase):
    def setUp(self):
        for var in ("MILES_SINGLE_TRAIN_GPUS", "MILES_SINGLE_INFER_GPUS"):
            os.environ.pop(var, None)

    def test_unset_envs_keep_legacy_range_derivation(self):
        out = _build_cluster_device_mappings(_args(1, 2))
        self.assertEqual(out, {"actor_train": [0], "actor_infer": [0, 1]})

    def test_disjoint_override_accepted(self):
        with _env("0", "1,2"):
            out = _build_cluster_device_mappings(_args(1, 2))
        self.assertEqual(out, {"actor_train": [0], "actor_infer": [1, 2]})

    def test_subset_override_accepted(self):
        with _env("0", "0,1"):
            out = _build_cluster_device_mappings(_args(1, 2))
        self.assertEqual(out, {"actor_train": [0], "actor_infer": [0, 1]})

    def test_partial_intersection_rejected(self):
        with _env("0,1", "1,2"):
            with self.assertRaisesRegex(ValueError, "partially intersects"):
                _build_cluster_device_mappings(_args(2, 2))

    def test_undersized_infer_rejected(self):
        with _env("0", "1"):
            with self.assertRaisesRegex(ValueError, "lengths must match"):
                _build_cluster_device_mappings(_args(1, 2))

    def test_oversized_infer_rejected(self):
        with _env("0", "1,2,3"):
            with self.assertRaisesRegex(ValueError, "lengths must match"):
                _build_cluster_device_mappings(_args(1, 2))

    def test_train_length_mismatch_rejected(self):
        with _env("0,3", "1,2"):
            with self.assertRaisesRegex(ValueError, "lengths must match"):
                _build_cluster_device_mappings(_args(1, 2))

    def test_duplicate_ids_rejected(self):
        with _env("0", "1,1"):
            with self.assertRaisesRegex(ValueError, "duplicates"):
                _build_cluster_device_mappings(_args(1, 2))

    def test_half_set_envs_rejected(self):
        with _env(train="0"):
            with self.assertRaisesRegex(ValueError, "must be .*set together"):
                _build_cluster_device_mappings(_args(1, 2))


if __name__ == "__main__":
    unittest.main()
