"""Regression checks for cache-comparison statistics."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "compare_grasp_caches",
    Path(__file__).parents[1] / "scripts/compare_grasp_caches.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_singleton_parent_has_finite_zero_spread():
    cache = {
        "q": torch.zeros(2, 22),
        "q_cmd": torch.zeros(2, 22),
        "object_pose_palm": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        ),
        "source_seed_index": torch.tensor([0, 1]),
        "contact_bits": torch.ones(2, 5),
    }
    report = module.inspect(cache, torch.tensor([[-1.0, 1.0]]).repeat(22, 1))
    assert report["within_parent_position_spread_mm"]["max"] == 0
    json.dumps(report, allow_nan=False)


def test_orientation_distance_ignores_quaternion_sign_and_scale():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    torch.testing.assert_close(
        module.nearest_orientation(q, -0.999 * q), torch.zeros(2)
    )


def test_histogram_difference_for_disjoint_and_identical_positions():
    a = np.zeros((5, 3), dtype=int)
    b = np.ones((5, 3), dtype=int)
    assert module.histogram_difference(a, a)["jensen_shannon_bits"] == 0
    report = module.histogram_difference(a, b)
    assert report["jensen_shannon_bits"] == 1
    assert report["reference_bin_coverage"] == 0


def test_nearest_preserves_small_distances_with_large_common_offset():
    query = torch.tensor([[0.15, 0.10, 0.20]])
    bank = query.repeat(32, 1)
    bank[:, 0] += torch.arange(1, 33) * 1e-5
    expected = float((query[0] - bank[0]).norm())
    assert abs(float(module.nearest(query, bank)[0]) - expected) < 1e-9
