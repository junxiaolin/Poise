"""Capture inputs for independent simulator/deployment numerical verification."""
from typing import Any
import torch
from poise.sim.env import sim_to_action_order

def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def capture_frame(base_env: Any, policy_observation: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture independent builder inputs after the simulator made its observation."""

    q = sim_to_action_order(base_env.hand.data.joint_pos, base_env.mapping.action_joint_indices)
    object_pose = base_env._get_object_pose_obs()
    confidence = torch.full(
        (base_env.num_envs, 1),
        float(getattr(base_env.cfg, "sharpa_deployable_pose_tracker_confidence", 1.0)),
        device=base_env.device,
        dtype=torch.float32,
    )
    frame = {
        "policy_observation": _cpu_clone(policy_observation),
        "joint_position": _cpu_clone(q),
        "joint_command": _cpu_clone(base_env.action_targets),
        "object_pose_palm": _cpu_clone(object_pose),
        "target_pose_palm": _cpu_clone(base_env.target_pose_palm),
        "source_anchor_local_pos_m": _cpu_clone(base_env.source_anchor_local_pos_m),
        "target_anchor_pos_palm": _cpu_clone(base_env.target_anchor_pos_palm),
        "target_twist_axis_palm": _cpu_clone(base_env.target_twist_axis_palm),
        "target_twist_angle_rad": _cpu_clone(base_env.target_twist_angle),
        "target_translation_axis_palm": _cpu_clone(base_env.target_translation_axis_palm),
        "target_translation_delta_m": _cpu_clone(base_env.target_translation_delta),
        "pose_tracker_confidence": _cpu_clone(confidence),
        "gravity_direction_palm": _cpu_clone(base_env._get_gravity_palm_obs()),
        "sim_history": _cpu_clone(base_env.sharpa_deployable_obs_history),
        "sim_object_velocity_est_palm": _cpu_clone(base_env.sharpa_deployable_object_velocity_est_palm),
    }
    object_descriptor = base_env._get_sharpa_deployable_object_descriptor_obs()
    if object_descriptor.shape[-1] > 0:
        frame["object_descriptor"] = _cpu_clone(object_descriptor)
    return frame


def stack_frames(frames: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = frames[0].keys()
    return {key: torch.stack([frame[key] for frame in frames], dim=0) for key in keys}
