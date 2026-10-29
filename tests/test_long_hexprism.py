"""Long-object geometry coverage and the requested 15 mm / 15 degree task."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from poise.sim.pose_success import full_pose_success_terms
from poise.sim.assets.sharpa_manual_hex_prisms import get_manual_hex_prism_spec

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "source/poise/sim/recipes/hexprism_d40_l160_omniwrist_tactile_fullpose15_v1.json"


def test_long_object_cover_reaches_both_ends():
    cfg = json.loads(RECIPE.read_text())
    spec = get_manual_hex_prism_spec(cfg["primitive_object_ids"][0])
    points = np.asarray(cfg["pose6d_feasible_target_forbidden_sdf_proxy_points_local_m"])
    radii = np.asarray(cfg["pose6d_feasible_target_forbidden_sdf_proxy_radii_m"])
    # Include cell boundaries and the ends: a reused 40-mm cover fails here.
    samples = np.asarray(list(itertools.product(*[
        np.linspace(-extent / 2, extent / 2, divisions + 1)
        for extent, divisions in zip(spec.extents_xyz_m, (3, 3, 12))
    ])))
    distances = np.linalg.norm(samples[:, None] - points[None], axis=-1)
    assert ((distances - radii[None]).min(axis=1) < 1e-10).all()
    assert max(radii) < .012
    assert cfg["dexndm_omni_wrist_ornt"]
    assert cfg["observation_profile"] == "sharpa_pose6d_v1_tactile60_actor304_critic552"
    assert (cfg["observation_space"], cfg["state_space"]) == (304, 552)


@pytest.mark.parametrize("mm,angle,expected", [(14, 14, True), (16, 14, False), (14, 16, False)])
def test_full_pose_15mm_15deg(mm, angle, expected):
    cfg = json.loads(RECIPE.read_text())
    half = math.radians(angle) / 2
    current = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=torch.float64)
    target = torch.tensor([[mm / 1000, 0., 0., math.cos(half), 0., 0., math.sin(half)]], dtype=torch.float64)
    p, r, _, _ = full_pose_success_terms(current, target,
        cfg["pose6d_success_position_threshold_m"], cfg["pose6d_success_orientation_threshold_rad"])
    assert bool((p & r).item()) == expected
    assert cfg["pose6d_success_criterion"] == "full_pose"
    assert cfg["success_steps"] == 20
