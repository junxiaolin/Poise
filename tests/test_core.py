"""Regression checks for the shared observation and policy interface."""

from dataclasses import asdict
import hashlib
import json
import math

import pytest
import torch

from poise.core.observation import SharpaPose6DObservationBuilder, make_pose6d_target
from poise.core.policy_bundle import PolicyBundle, export_fused_policy
from poise.core.schema import (
    ACTION_DIM,
    OBSERVATION_CONTRACT,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    joint_interface_manifest,
    observation_contract_from_manifest,
)


def builder(descriptor=None, **kwargs):
    return SharpaPose6DObservationBuilder(
        torch.full((22,), -2.0),
        torch.full((22,), 2.0),
        object_descriptor=torch.linspace(-1, 1, 64) if descriptor is None else descriptor,
        **kwargs,
    )


def test_observation_history_bps_and_velocity():
    obs = builder()
    pose = torch.tensor([0.1, 0.02, 0.08, 1, 0, 0, 0])
    desired = pose.clone()
    desired[0] += 0.03
    target = make_pose6d_target(pose, desired)
    obs.reset(torch.zeros(22), torch.ones(22), pose)
    first = obs.observe(target, pose_tracker_confidence=0.8, gravity_direction_palm=[1, 0, 0])
    assert first.shape == (244,)
    torch.testing.assert_close(first[:22], torch.zeros(22))
    torch.testing.assert_close(first[22:44], torch.full((22,), 0.5))
    torch.testing.assert_close(first[:44], first[88:132])
    torch.testing.assert_close(first[132:139], pose)
    torch.testing.assert_close(first[139:146], desired)
    torch.testing.assert_close(first[176:240], torch.linspace(-1, 1, 64))
    torch.testing.assert_close(first[240:], torch.tensor([0.8, 1, 0, 0]))
    moved = pose.clone()
    moved[0] += 0.005
    second = obs.step(torch.full((22,), 0.2), torch.ones(22), moved, target,
                      gravity_direction_palm=[1, 0, 0])
    torch.testing.assert_close(second[88:110], torch.full((22,), 0.1))
    torch.testing.assert_close(second[170:176], torch.tensor([0.1, 0, 0, 0, 0, 0]))


def test_target_rotation_and_anchor():
    half = math.sqrt(0.5)
    target = make_pose6d_target(
        torch.tensor([0., 0, 0, 1, 0, 0, 0]),
        torch.tensor([0.03, 0, 0, half, 0, 0, half]),
        torch.tensor([0.01, 0, 0]),
    )
    torch.testing.assert_close(target.twist_axis_palm, torch.tensor([0., 0, 1]))
    torch.testing.assert_close(target.twist_angle_rad, torch.tensor(math.pi / 2))
    torch.testing.assert_close(target.target_anchor_pos_palm, torch.tensor([0.03, 0.01, 0]))


def test_batched_descriptor_and_quaternion_sign():
    descriptors = torch.stack([torch.zeros(64), torch.ones(64)])
    obs = builder(descriptors)
    pose = torch.tensor([[0.1, 0, 0.1, 1, 0, 0, 0], [0.1, 0, 0.1, -1, 0, 0, 0]])
    obs.reset(torch.zeros(2, 22), torch.zeros(2, 22), pose)
    result = obs.observe(make_pose6d_target(pose, pose), gravity_direction_palm=[1, 0, 0])
    torch.testing.assert_close(result[:, 176:240], descriptors)
    torch.testing.assert_close(result[0, :176], result[1, :176])


@pytest.mark.parametrize("options", [{"object_descriptor_dim": 0}, {"include_gravity_palm": False}])
def test_reject_legacy_layout(options):
    with pytest.raises(ValueError, match="BPS64"):
        builder(**options)


def test_reject_missing_and_nonfinite_geometry():
    with pytest.raises(ValueError, match="requires object_descriptor"):
        SharpaPose6DObservationBuilder(torch.full((22,), -1.), torch.ones(22))
    descriptor = torch.zeros(64)
    descriptor[3] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        builder(descriptor)


@pytest.mark.parametrize("record", [None, {}, {"dimension": 180}, {
    "contract": OBSERVATION_CONTRACT.contract_id, "dimension": 244, "include_gravity_palm": False,
}])
def test_manifest_requires_explicit_bps64_contract(record):
    with pytest.raises(ValueError):
        observation_contract_from_manifest(record)


def test_export_load_and_action_limits(tmp_path):
    rng = torch.Generator().manual_seed(17)
    state = {
        "actor.0.weight": torch.randn(8, OBSERVATION_DIM, generator=rng) * 0.01,
        "actor.0.bias": torch.randn(8, generator=rng) * 0.01,
        "actor.2.weight": torch.randn(ACTION_DIM, 8, generator=rng) * 0.01,
        "actor.2.bias": torch.zeros(ACTION_DIM),
        "actor_obs_normalizer._mean": torch.linspace(-0.1, 0.1, OBSERVATION_DIM)[None],
        "actor_obs_normalizer._std": torch.linspace(0.5, 1.5, OBSERVATION_DIM)[None],
    }
    checkpoint = tmp_path / "user_checkpoint.pt"
    torch.save({"iter": 7, "model_state_dict": state}, checkpoint)
    policy = tmp_path / "policy.ts"
    metadata = export_fused_policy(checkpoint, policy, activation="elu")
    assert metadata["checkpoint_iteration"] == 7
    observation = asdict(OBSERVATION_CONTRACT)
    observation["contract"] = observation.pop("contract_id")
    observation["object_descriptor"] = {
        "kind": "mesh_bps64", "dimension": 64, "object_id": "example", "values": [0.] * 64,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "joint_interface": joint_interface_manifest(),
        "observation": observation,
        "physical_object": {"stable_cache_object_id": "example"},
        "policy": {**metadata, "filename": "policy.ts", "sha256": hashlib.sha256(policy.read_bytes()).hexdigest()},
        "sim_contract": {"joint_lower": [-1.] * 22, "joint_upper": [1.] * 22,
                         "control_dt_s": 0.05, "target_translation_scale_m": 0.03},
        "control": {"action_scale_rad": 1 / 24, "action_smoothing": 1.0},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    bundle = PolicyBundle.load(tmp_path)
    inputs = torch.randn(3, 244, generator=rng)
    normalized = (inputs - state["actor_obs_normalizer._mean"]) / (state["actor_obs_normalizer._std"] + 0.01)
    hidden = torch.nn.functional.elu(torch.nn.functional.linear(normalized, state["actor.0.weight"], state["actor.0.bias"]))
    expected = torch.nn.functional.linear(hidden, state["actor.2.weight"], state["actor.2.bias"])
    torch.testing.assert_close(bundle.infer(inputs, clip_action=False), expected, atol=1e-6, rtol=1e-6)
    assert bundle.infer(inputs[0]).shape == (22,)
    torch.testing.assert_close(bundle.infer(inputs[0], clip_action=False), expected[0], atol=1e-6, rtol=1e-6)
    # Use the real exported normalizer/actor: its 1x244 buffers used to introduce
    # an extra batch axis and make PolicyController reject every first command.
    from poise.deploy.control import PolicyController
    pose = torch.tensor([0.07, 0, 0.09, 1., 0, 0, 0])
    controller = PolicyController(bundle, torch.zeros(22), torch.zeros(22), pose, pose,
                                  [-1, 0, 0], timestamp_ms=1000)
    command = controller.propose(torch.zeros(22), pose, timestamp_ms=1050, pose_age_s=0.01, joint_age_s=0.01)
    assert command.shape == (22,)
    controller.acknowledge(command)
    assert controller.pending is None
    torch.testing.assert_close(bundle.next_joint_target(torch.zeros(22), torch.full((22,), 2.)), torch.full((22,), 1 / 24))
    torch.testing.assert_close(bundle.next_joint_target(torch.full((22,), 0.99), torch.ones(22)), torch.ones(22))
    torch.testing.assert_close(bundle.action_to_sdk(torch.arange(22)), torch.arange(22))
    inputs[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        bundle.infer(inputs)
    manifest["policy"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PolicyBundle.load(tmp_path)
