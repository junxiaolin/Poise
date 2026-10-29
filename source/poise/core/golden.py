"""Offline verification for IsaacLab-generated deployment golden references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .observation import Pose6DTarget, SharpaPose6DObservationBuilder
from .policy_bundle import PolicyBundle
from .schema import ACTION_DIM, ObservationField, observation_contract_from_manifest


GOLDEN_SCHEMA_VERSION = "sharpa_pose6d_sim_golden_v1"


def _target_at(frames: dict[str, torch.Tensor], index: int) -> Pose6DTarget:
    return Pose6DTarget(
        target_pose_palm=frames["target_pose_palm"][index],
        source_anchor_local_pos_m=frames["source_anchor_local_pos_m"][index],
        target_anchor_pos_palm=frames["target_anchor_pos_palm"][index],
        twist_axis_palm=frames["target_twist_axis_palm"][index],
        twist_angle_rad=frames["target_twist_angle_rad"][index],
        translation_axis_palm=frames["target_translation_axis_palm"][index],
        translation_delta_m=frames["target_translation_delta_m"][index],
    )


def _observation_field_name(
    flat_index: int,
    fields: tuple[ObservationField, ...],
) -> str:
    for field in fields:
        if field.start <= flat_index < field.stop:
            return field.name
    return "unknown"


def _error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    observation_fields: tuple[ObservationField, ...] = (),
) -> dict[str, Any]:
    absolute = torch.abs(actual - expected)
    flat_index = int(torch.argmax(absolute).item())
    unraveled = torch.unravel_index(torch.tensor(flat_index), absolute.shape)
    coordinates = [int(value.item()) for value in unraveled]
    result: dict[str, Any] = {
        "max_abs_error": float(absolute.reshape(-1)[flat_index].item()),
        "mean_abs_error": float(absolute.mean().item()),
        "max_error_coordinates": coordinates,
    }
    if observation_fields and actual.shape[-1] == observation_fields[-1].stop:
        result["max_error_field"] = _observation_field_name(
            coordinates[-1],
            observation_fields,
        )
    return result


def verify_golden_reference(
    golden_path: str | Path,
    bundle_dir: str | Path,
    *,
    observation_atol: float = 2.0e-5,
    action_atol: float = 2.0e-5,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Rebuild every observation and action without importing Isaac Lab."""

    golden_path = Path(golden_path).expanduser().resolve()
    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    if not isinstance(golden, dict) or golden.get("schema_version") != GOLDEN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported golden reference schema in {golden_path}.")
    frames = golden.get("frames")
    contract = golden.get("sim_contract")
    if not isinstance(frames, dict) or not isinstance(contract, dict):
        raise ValueError("Golden reference is missing frames or sim_contract.")

    bundle = PolicyBundle.load(bundle_dir, device="cpu", verify_hash=True)
    observation_contract = observation_contract_from_manifest(
        bundle.manifest.get("observation")
    )
    contract_id = contract.get("observation_contract")
    if contract_id is not None and contract_id != observation_contract.contract_id:
        raise ValueError(
            "Golden sim_contract observation contract does not match the bundle: "
            f"{contract_id!r} != {observation_contract.contract_id!r}."
        )
    include_gravity = contract.get("include_gravity_palm")
    if (
        include_gravity is not None
        and bool(include_gravity) != observation_contract.include_gravity_palm
    ):
        raise ValueError("Golden sim_contract gravity flag does not match the bundle.")
    observation_dim = observation_contract.dimension

    observation = frames["policy_observation"].to(dtype=torch.float32)
    if observation.ndim != 3 or observation.shape[-1] != observation_dim:
        raise ValueError(
            f"policy_observation must have shape (frames, envs, {observation_dim}), "
            f"got {tuple(observation.shape)}."
        )
    frame_count, env_count = observation.shape[:2]
    if frame_count < 2:
        raise ValueError("Golden reference must contain at least two observation frames.")
    joint_position = frames["joint_position"].to(dtype=torch.float32)
    joint_command = frames["joint_command"].to(dtype=torch.float32)
    object_pose = frames["object_pose_palm"].to(dtype=torch.float32)
    if joint_position.shape != (frame_count, env_count, ACTION_DIM):
        raise ValueError("Golden joint_position shape is invalid.")
    if joint_command.shape != joint_position.shape:
        raise ValueError("Golden joint_command shape is invalid.")
    if object_pose.shape != (frame_count, env_count, 7):
        raise ValueError("Golden object_pose_palm shape is invalid.")

    descriptor = bundle.object_descriptor
    recorded_descriptor = frames.get("object_descriptor")
    if observation_contract.object_descriptor_dim:
        expected_descriptor = descriptor.reshape(1, 1, -1).expand(
            frame_count,
            env_count,
            -1,
        )
        if not isinstance(recorded_descriptor, torch.Tensor):
            raise ValueError("Golden reference is missing the BPS64 object descriptor.")
        recorded_descriptor = recorded_descriptor.to(dtype=torch.float32)
        if recorded_descriptor.shape != expected_descriptor.shape:
            raise ValueError(
                "Golden object_descriptor has shape "
                f"{tuple(recorded_descriptor.shape)}, expected {tuple(expected_descriptor.shape)}."
            )
        descriptor_error = _error_summary(recorded_descriptor, expected_descriptor)
    else:
        if recorded_descriptor is not None:
            raise ValueError("Golden object_descriptor is forbidden by this observation contract.")
        descriptor_error = None

    builder = SharpaPose6DObservationBuilder(
        torch.as_tensor(contract["joint_lower"], dtype=torch.float32),
        torch.as_tensor(contract["joint_upper"], dtype=torch.float32),
        control_dt_s=float(contract["control_dt_s"]),
        target_translation_scale_m=float(contract["target_translation_scale_m"]),
        linear_velocity_clip_mps=float(contract["linear_velocity_clip_mps"]),
        angular_velocity_clip_radps=float(contract["angular_velocity_clip_radps"]),
        include_gravity_palm=observation_contract.include_gravity_palm,
        object_descriptor=descriptor,
        object_descriptor_dim=observation_contract.object_descriptor_dim,
    )
    builder.reset(joint_position[0], joint_command[0], object_pose[0])
    rebuilt = [
        builder.observe(
            _target_at(frames, 0),
            pose_tracker_confidence=frames["pose_tracker_confidence"][0],
            gravity_direction_palm=frames["gravity_direction_palm"][0],
        )
    ]
    for frame_index in range(1, frame_count):
        rebuilt.append(
            builder.step(
                joint_position[frame_index],
                joint_command[frame_index],
                object_pose[frame_index],
                _target_at(frames, frame_index),
                pose_tracker_confidence=frames["pose_tracker_confidence"][frame_index],
                gravity_direction_palm=frames["gravity_direction_palm"][frame_index],
            )
        )
    rebuilt_observation = torch.stack(rebuilt)
    observation_error = _error_summary(
        rebuilt_observation,
        observation,
        observation_fields=observation_contract.fields,
    )

    golden_metadata = golden.get("metadata", {})
    golden_contract = golden_metadata.get("observation_contract")
    if golden_contract is not None and golden_contract != observation_contract.contract_id:
        raise ValueError(
            "Golden observation contract does not match the loaded bundle: "
            f"{golden_contract!r} != {observation_contract.contract_id!r}."
        )
    expected_policy_sha = bundle.manifest.get("policy", {}).get("sha256")
    if golden_metadata.get("policy_sha256") != expected_policy_sha:
        raise ValueError(
            "Golden reference policy SHA256 does not match the loaded bundle: "
            f"{golden_metadata.get('policy_sha256')} != {expected_policy_sha}."
        )
    expected_checkpoint_sha = (
        bundle.manifest.get("audit", {})
        .get("source_artifacts", {})
        .get("checkpoint", {})
        .get("sha256")
    )
    if expected_checkpoint_sha and golden_metadata.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError(
            "Golden reference checkpoint SHA256 does not match the loaded bundle: "
            f"{golden_metadata.get('checkpoint_sha256')} != {expected_checkpoint_sha}."
        )
    policy_input = observation[:-1].reshape(-1, observation_dim)
    with torch.inference_mode():
        rebuilt_action = bundle.module(policy_input).reshape(frame_count - 1, env_count, ACTION_DIM)
    reference_action = golden["reference_action"].to(dtype=torch.float32)
    if reference_action.shape != rebuilt_action.shape:
        raise ValueError(
            f"reference_action has shape {tuple(reference_action.shape)}, expected {tuple(rebuilt_action.shape)}."
        )
    action_error = _error_summary(rebuilt_action, reference_action)

    recorded_bundle_action = golden.get("bundle_action")
    recorded_bundle_error = None
    if isinstance(recorded_bundle_action, torch.Tensor):
        recorded_bundle_error = _error_summary(
            rebuilt_action,
            recorded_bundle_action.to(dtype=torch.float32),
        )

    observation_pass = observation_error["max_abs_error"] <= float(observation_atol)
    action_pass = action_error["max_abs_error"] <= float(action_atol)
    bundle_recording_pass = (
        recorded_bundle_error is None or recorded_bundle_error["max_abs_error"] <= float(action_atol)
    )
    descriptor_pass = (
        descriptor_error is None
        or descriptor_error["max_abs_error"] <= float(observation_atol)
    )
    passed = observation_pass and action_pass and bundle_recording_pass and descriptor_pass
    report = {
        "passed": passed,
        "golden_path": str(golden_path),
        "bundle_dir": str(Path(bundle_dir).expanduser().resolve()),
        "frame_count": int(frame_count),
        "environment_count": int(env_count),
        "observation_tolerance": float(observation_atol),
        "action_tolerance": float(action_atol),
        "observation": {
            **observation_error,
            "passed": observation_pass,
        },
        "action": {
            **action_error,
            "passed": action_pass,
        },
        "recorded_bundle_action": (
            {**recorded_bundle_error, "passed": bundle_recording_pass}
            if recorded_bundle_error is not None
            else None
        ),
        "object_descriptor": (
            {**descriptor_error, "passed": descriptor_pass}
            if descriptor_error is not None
            else None
        ),
    }
    if raise_on_failure and not passed:
        raise AssertionError(
            "Golden verification failed: "
            f"observation max={observation_error['max_abs_error']:.3e}, "
            f"action max={action_error['max_abs_error']:.3e}."
        )
    return report
