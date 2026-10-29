"""Regressions for the paper's object-origin and full SO(3) tolerances."""

import json
import math
from importlib.resources import files

import pytest
import torch

from poise.sim.pose_success import full_pose_success_terms


def pose(position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    return torch.tensor([(*position, *quaternion)], dtype=torch.float64)


def check(current, target):
    return full_pose_success_terms(current, target, 0.01, math.radians(10))


def test_combined_translation_rejects_componentwise_false_success():
    position_ok, rotation_ok, position_error, _ = check(pose(), pose((0.009, 0.009, 0)))
    assert not position_ok.item()
    assert rotation_ok.item()
    assert position_error.item() == pytest.approx(math.sqrt(2) * 0.009)


def test_combined_twist_swing_rejects_componentwise_false_success():
    c, s = math.cos(math.radians(9) / 2), math.sin(math.radians(9) / 2)
    # A 9-degree X swing composed with a 9-degree Z twist.
    position_ok, rotation_ok, _, angle = check(pose(), pose(quaternion=(c*c, s*c, -s*s, c*s)))
    assert position_ok.item()
    assert not rotation_ok.item()
    assert math.degrees(angle.item()) == pytest.approx(12.7213723, abs=1e-6)


@pytest.mark.parametrize("degrees,accepted", [(0, True), (9.9999, True), (10.0001, False), (180, False), (350.0001, True)])
def test_shortest_rotation_angle_and_quaternion_sign(degrees, accepted):
    half = math.radians(degrees) / 2
    target = pose(quaternion=(math.cos(half), 0, math.sin(half), 0))
    baseline = check(pose(), target)
    assert baseline[1].item() == accepted
    assert math.degrees(baseline[3].item()) == pytest.approx(min(degrees, 360-degrees))
    target[:, 3:] *= -2.0
    current = pose(quaternion=(-3, 0, 0, 0))
    signed = check(current, target)
    torch.testing.assert_close(signed[3], baseline[3])


@pytest.mark.parametrize("distance,accepted", [(0.009999, True), (0.01, False), (0.010001, False)])
def test_position_tolerance_is_strict(distance, accepted):
    assert check(pose(), pose((distance, 0, 0)))[0].item() == accepted


def test_public_recipe_preserves_actor_corruption_with_clean_critic():
    recipe = json.loads(files("poise.sim.recipes").joinpath("hexprism40.json").read_text())
    assert recipe["pose6d_success_criterion"] == "full_pose"
    assert recipe["pose6d_success_position_threshold_m"] == 0.01
    assert recipe["pose6d_success_orientation_threshold_rad"] == pytest.approx(math.radians(10))
    assert recipe["success_steps"] == 20
    assert recipe["object_com_randomization_range_m"] == 0.005
    assert recipe["dexndm_grasp_wrench_maintenance_reward_scale"] == 0.05
    assert recipe["dexndm_joint_noise_scale"] == 0
    assert recipe["sharpa_deployable_joint_sensing_noise_scale"] == 0.02
    assert recipe["sharpa_deployable_object_pose_obs_delay_noise_enabled"]
    assert recipe["sharpa_deployable_object_pose_obs_pos_noise_std_m"] == 0.01
    assert recipe["sharpa_deployable_object_pose_obs_rot_noise_deg"] == 5


def test_fullpose_omniwrist_is_only_success_criterion_ablation():
    recipes = files("poise.sim.recipes")
    baseline = json.loads(recipes.joinpath("hexprism40_omniwrist_aug16_v1.json").read_text())
    variant = json.loads(recipes.joinpath("hexprism40_omniwrist_fullpose_v1.json").read_text())
    assert {k for k in baseline.keys() | variant.keys() if baseline.get(k) != variant.get(k)} == {
        "pose6d_success_criterion"
    }
    assert variant["pose6d_success_criterion"] == "full_pose"
    assert variant["pose6d_success_position_threshold_m"] == 0.01
    assert variant["pose6d_success_orientation_threshold_rad"] == pytest.approx(math.radians(10))
    assert variant["observation_profile"] == "sharpa_pose6d_deployable_v1_notactile"
    assert (variant["observation_space"], variant["state_space"]) == (244, 492)
