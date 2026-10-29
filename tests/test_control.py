"""Hardware-free checks of command acknowledgment and sensor fault handling."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poise.core.policy_bundle import PolicyBundle
from poise.deploy.control import PolicyController


class FakeBundle:
    object_descriptor = torch.zeros(64)
    manifest = {
        "sim_contract": {"joint_lower": [-1.0] * 22, "joint_upper": [1.0] * 22,
                         "control_dt_s": 0.05, "target_translation_scale_m": 0.03},
        "control": {"action_scale_rad": 1 / 24, "action_smoothing": 1.0},
        "policy": {"clip_actions": 1.0},
    }
    next_joint_target = PolicyBundle.next_joint_target
    device = "cpu"

    def infer(self, observation):
        assert observation.shape == (244,)
        return torch.full((22,), 2.0)


def controller():
    pose = torch.tensor([0.07, 0, 0.09, 1, 0, 0, 0], dtype=torch.float32)
    return PolicyController(FakeBundle(), torch.zeros(22), torch.zeros(22), pose, pose,
                            [-1, 0, 0], timestamp_ms=1000), pose


def test_only_acknowledged_targets_enter_command_history():
    control, pose = controller()
    command = control.propose(torch.zeros(22), pose, timestamp_ms=1050, pose_age_s=0.01, joint_age_s=0.01)
    np.testing.assert_allclose(command, 1 / 24)
    assert torch.equal(control.q_cmd, torch.zeros(22))
    with pytest.raises(RuntimeError, match="unacknowledged"):
        control.propose(torch.zeros(22), pose, timestamp_ms=1100, pose_age_s=0.01, joint_age_s=0.01)
    control.acknowledge(command)
    next_command = control.propose(torch.zeros(22), pose, timestamp_ms=1100, pose_age_s=0.01, joint_age_s=0.01)
    np.testing.assert_allclose(next_command, 2 / 24)


@pytest.mark.parametrize("timestamp,pose_age,joint_age", [(1000, 0.01, 0.01), (1050, 0.11, 0.01), (1050, 0.01, 0.21), (float("nan"), 0, 0)])
def test_timing_faults_latch_controller_stop(timestamp, pose_age, joint_age):
    control, pose = controller()
    with pytest.raises((ValueError, RuntimeError)):
        control.propose(torch.zeros(22), pose, timestamp_ms=timestamp, pose_age_s=pose_age, joint_age_s=joint_age)
    assert control.failed
    with pytest.raises(RuntimeError, match="stopped"):
        control.propose(torch.zeros(22), pose, timestamp_ms=1100, pose_age_s=0.01, joint_age_s=0.01)


def test_pose_jump_and_bad_ack_are_rejected():
    control, pose = controller()
    jumped = pose.clone()
    jumped[0] += 0.02
    with pytest.raises(RuntimeError, match="jumped"):
        control.propose(torch.zeros(22), jumped, timestamp_ms=1050, pose_age_s=0.01, joint_age_s=0.01)
    control, pose = controller()
    command = control.propose(torch.zeros(22), pose, timestamp_ms=1050, pose_age_s=0.01, joint_age_s=0.01)
    with pytest.raises(RuntimeError, match="differs"):
        control.acknowledge(command + 0.001)


@pytest.mark.parametrize("q", [torch.zeros(21), torch.full((22,), float("nan"))])
def test_invalid_joint_feedback_latches_stop(q):
    control, pose = controller()
    with pytest.raises(ValueError, match="measured joints"):
        control.propose(q, pose, timestamp_ms=1050, pose_age_s=0, joint_age_s=0)
    assert control.failed


@pytest.mark.parametrize("fault", ["timestamp", "quaternion", "q_cmd"])
def test_initial_state_is_checked_before_observation_history(fault):
    pose = torch.tensor([0.07, 0, 0.09, 1., 0, 0, 0])
    command = torch.zeros(22)
    timestamp = 1000
    if fault == "timestamp":
        timestamp = float("nan")
    elif fault == "quaternion":
        pose[3:] = 0
    else:
        command[:] = 2
    with pytest.raises(ValueError):
        PolicyController(FakeBundle(), torch.zeros(22), command, pose, pose,
                         [-1, 0, 0], timestamp_ms=timestamp)
