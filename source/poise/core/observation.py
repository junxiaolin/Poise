"""Independent 244D BPS64 observation builder for Sharpa pose-6D policies.

This module intentionally imports only PyTorch. Pose tensors use
``[x, y, z, qw, qx, qy, qz]`` in the Sharpa hand-root (palm) frame and joint
tensors use the finger-grouped policy/action order declared in :mod:`schema`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .schema import (
    ACTION_DIM,
    OBJECT_DESCRIPTOR_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
)


def normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion in scalar-first convention."""

    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-8)


def positive_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion and choose the non-negative scalar hemisphere."""

    quat = normalize_quat_wxyz(quat)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return the quaternion conjugate."""

    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product of scalar-first quaternions."""

    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a normalized scalar-first quaternion."""

    quat = normalize_quat_wxyz(quat)
    quat_vector = quat[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quat_vector, vector, dim=-1)
    return vector + quat[..., :1] * twice_cross + torch.linalg.cross(quat_vector, twice_cross, dim=-1)


def quat_to_rotvec_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to the shortest axis-angle rotation vector."""

    quat = positive_quat_wxyz(quat)
    vector = quat[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half.squeeze(-1), torch.clamp(quat[..., 0], min=1.0e-8))
    scale = torch.where(
        sin_half > 1.0e-6,
        angle.unsqueeze(-1) / torch.clamp(sin_half, min=1.0e-6),
        torch.full_like(sin_half, 2.0),
    )
    return vector * scale


def scale_joint_position(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Scale joint positions to the policy interval, retaining the sim's guard band."""

    denominator = torch.clamp(upper - lower, min=1.0e-6)
    return torch.clamp(2.0 * (value - lower) / denominator - 1.0, -1.5, 1.5)


def _as_batch(
    value: Any,
    width: int,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    single = tensor.ndim == 1
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (batch, {width}), got {tuple(tensor.shape)}.")
    if batch_size is not None:
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {tensor.shape[0]} does not match expected {batch_size}.")
    return tensor, single


def _as_column(
    value: Any,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(batch_size).unsqueeze(-1)
    if tensor.ndim == 1:
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size)
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} has batch size {tensor.shape[0]}, expected {batch_size}.")
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2 and tensor.shape == (batch_size, 1):
        return tensor
    raise ValueError(f"{name} must be scalar, (batch,), or (batch, 1), got {tuple(tensor.shape)}.")


@dataclass(frozen=True)
class Pose6DTarget:
    """Target fields consumed by the Sharpa Wave BPS64 actor."""

    target_pose_palm: torch.Tensor
    source_anchor_local_pos_m: torch.Tensor
    target_anchor_pos_palm: torch.Tensor
    twist_axis_palm: torch.Tensor
    twist_angle_rad: torch.Tensor
    translation_axis_palm: torch.Tensor
    translation_delta_m: torch.Tensor


def make_pose6d_target(
    source_pose_palm: torch.Tensor,
    target_pose_palm: torch.Tensor,
    source_anchor_local_pos_m: torch.Tensor | None = None,
) -> Pose6DTarget:
    """Derive a target contract from source and desired object poses.

    The helper uses the same local point for source and target anchors. More
    specialized target samplers may construct :class:`Pose6DTarget` directly.
    """

    source = (
        source_pose_palm
        if isinstance(source_pose_palm, torch.Tensor)
        else torch.as_tensor(source_pose_palm)
    )
    target = torch.as_tensor(target_pose_palm, device=source.device, dtype=source.dtype)
    single = source.ndim == 1
    if single:
        source = source.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if source.ndim != 2 or source.shape[-1] != 7 or target.shape != source.shape:
        raise ValueError(
            "source_pose_palm and target_pose_palm must share shape (7,) or (batch, 7); "
            f"got {tuple(source.shape)} and {tuple(target.shape)}."
        )

    source = source.clone()
    target = target.clone()
    source[:, 3:7] = positive_quat_wxyz(source[:, 3:7])
    target[:, 3:7] = positive_quat_wxyz(target[:, 3:7])
    batch_size = source.shape[0]
    if source_anchor_local_pos_m is None:
        anchor_local = torch.zeros((batch_size, 3), device=source.device, dtype=source.dtype)
    else:
        anchor_local, _ = _as_batch(
            source_anchor_local_pos_m,
            3,
            name="source_anchor_local_pos_m",
            device=source.device,
            dtype=source.dtype,
            batch_size=batch_size,
        )

    delta_quat = positive_quat_wxyz(
        quat_mul_wxyz(target[:, 3:7], quat_conjugate_wxyz(source[:, 3:7]))
    )
    rotation_vector = quat_to_rotvec_wxyz(delta_quat)
    twist_angle = torch.linalg.norm(rotation_vector, dim=-1)
    default_axis = torch.zeros_like(rotation_vector)
    default_axis[:, 0] = 1.0
    twist_axis = torch.where(
        (twist_angle > 1.0e-8).unsqueeze(-1),
        rotation_vector / torch.clamp(twist_angle.unsqueeze(-1), min=1.0e-8),
        default_axis,
    )

    translation = target[:, :3] - source[:, :3]
    translation_delta = torch.linalg.norm(translation, dim=-1)
    translation_axis = torch.where(
        (translation_delta > 1.0e-8).unsqueeze(-1),
        translation / torch.clamp(translation_delta.unsqueeze(-1), min=1.0e-8),
        default_axis,
    )
    target_anchor = target[:, :3] + quat_apply_wxyz(target[:, 3:7], anchor_local)

    fields = (
        target,
        anchor_local,
        target_anchor,
        twist_axis,
        twist_angle,
        translation_axis,
        translation_delta,
    )
    if single:
        fields = tuple(field.squeeze(0) for field in fields)
    return Pose6DTarget(*fields)


def _target_as_batch(
    target: Pose6DTarget,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Pose6DTarget:
    target_pose, _ = _as_batch(
        target.target_pose_palm,
        7,
        name="target.target_pose_palm",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    source_anchor, _ = _as_batch(
        target.source_anchor_local_pos_m,
        3,
        name="target.source_anchor_local_pos_m",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    target_anchor, _ = _as_batch(
        target.target_anchor_pos_palm,
        3,
        name="target.target_anchor_pos_palm",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    twist_axis, _ = _as_batch(
        target.twist_axis_palm,
        3,
        name="target.twist_axis_palm",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    twist_angle = _as_column(
        target.twist_angle_rad,
        name="target.twist_angle_rad",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    ).squeeze(-1)
    translation_axis, _ = _as_batch(
        target.translation_axis_palm,
        3,
        name="target.translation_axis_palm",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    translation_delta = _as_column(
        target.translation_delta_m,
        name="target.translation_delta_m",
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    ).squeeze(-1)
    target_pose = target_pose.clone()
    target_pose[:, 3:7] = positive_quat_wxyz(target_pose[:, 3:7])
    return Pose6DTarget(
        target_pose,
        source_anchor,
        target_anchor,
        twist_axis,
        twist_angle,
        translation_axis,
        translation_delta,
    )


def compose_observation(
    history: torch.Tensor,
    object_pose_palm: torch.Tensor,
    target: Pose6DTarget,
    object_velocity_palm: torch.Tensor,
    pose_tracker_confidence: torch.Tensor,
    gravity_direction_palm: torch.Tensor | None = None,
    object_descriptor: torch.Tensor | None = None,
    *,
    include_gravity_palm: bool = True,
    object_descriptor_dim: int = OBJECT_DESCRIPTOR_DIM,
    target_translation_scale_m: float = 0.03,
    linear_velocity_clip_mps: float = 10.0,
    angular_velocity_clip_radps: float = 50.0,
) -> torch.Tensor:
    """Compose the flat policy observation from already sampled components."""

    if object_descriptor_dim != OBJECT_DESCRIPTOR_DIM or not include_gravity_palm:
        raise ValueError("POISE requires BPS64 and palm-frame gravity (244D observation).")
    if history.ndim != 3 or history.shape[1:] != (HISTORY_LENGTH, HISTORY_FRAME_DIM):
        raise ValueError(
            f"history must have shape (batch, {HISTORY_LENGTH}, {HISTORY_FRAME_DIM}), "
            f"got {tuple(history.shape)}."
        )
    batch_size = history.shape[0]
    if object_pose_palm.shape != (batch_size, 7):
        raise ValueError(f"object_pose_palm must have shape ({batch_size}, 7).")
    if object_velocity_palm.shape != (batch_size, 6):
        raise ValueError(f"object_velocity_palm must have shape ({batch_size}, 6).")
    if pose_tracker_confidence.shape != (batch_size, 1):
        raise ValueError(f"pose_tracker_confidence must have shape ({batch_size}, 1).")
    if object_descriptor_dim:
        if object_descriptor is None or object_descriptor.shape != (
            batch_size,
            object_descriptor_dim,
        ):
            actual = None if object_descriptor is None else tuple(object_descriptor.shape)
            raise ValueError(
                "object_descriptor must have shape "
                f"({batch_size}, {object_descriptor_dim}), got {actual}."
            )
    elif object_descriptor is not None:
        raise ValueError("object_descriptor is forbidden by this observation contract.")
    if include_gravity_palm:
        if gravity_direction_palm is None or gravity_direction_palm.shape != (batch_size, 3):
            actual = None if gravity_direction_palm is None else tuple(gravity_direction_palm.shape)
            raise ValueError(
                f"gravity_direction_palm must have shape ({batch_size}, 3) for the "
                f"244D contract, got {actual}."
            )

    object_pose = object_pose_palm.clone()
    object_pose[:, 3:7] = positive_quat_wxyz(object_pose[:, 3:7])
    target_pose = target.target_pose_palm
    pose_delta_position = target_pose[:, :3] - object_pose[:, :3]
    pose_delta_quat = positive_quat_wxyz(
        quat_mul_wxyz(target_pose[:, 3:7], quat_conjugate_wxyz(object_pose[:, 3:7]))
    )
    pose_error = torch.cat((pose_delta_position, pose_delta_quat), dim=-1)

    current_anchor = object_pose[:, :3] + quat_apply_wxyz(
        object_pose[:, 3:7],
        target.source_anchor_local_pos_m,
    )
    anchor_delta = target.target_anchor_pos_palm - current_anchor
    anchor_observation = torch.cat((current_anchor, target.target_anchor_pos_palm, anchor_delta), dim=-1)

    translation_scale = max(float(target_translation_scale_m), 1.0e-6)
    target_command = torch.cat(
        (
            target.twist_axis_palm,
            (target.twist_angle_rad / torch.pi).unsqueeze(-1),
            target.translation_axis_palm,
            (target.translation_delta_m / translation_scale).unsqueeze(-1),
        ),
        dim=-1,
    )
    linear_velocity = torch.clamp(
        object_velocity_palm[:, :3],
        -float(linear_velocity_clip_mps),
        float(linear_velocity_clip_mps),
    )
    angular_velocity = torch.clamp(
        object_velocity_palm[:, 3:6],
        -float(angular_velocity_clip_radps),
        float(angular_velocity_clip_radps),
    )

    fields = [
        history.reshape(batch_size, -1),
        object_pose,
        target_pose,
        pose_error,
        anchor_observation,
        target_command,
        linear_velocity,
        angular_velocity,
    ]
    if object_descriptor_dim:
        assert object_descriptor is not None
        fields.append(object_descriptor)
    fields.append(pose_tracker_confidence)
    if include_gravity_palm:
        assert gravity_direction_palm is not None
        fields.append(gravity_direction_palm)
    observation = torch.cat(fields, dim=-1).to(dtype=torch.float32)
    expected_dim = OBSERVATION_DIM
    if observation.shape != (batch_size, expected_dim):
        raise RuntimeError(
            f"Observation builder produced shape {tuple(observation.shape)}, "
            f"expected ({batch_size}, {expected_dim})."
        )
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Observation contains NaN or Inf.")
    return observation


class SharpaPose6DObservationBuilder:
    """Stateful history and finite-difference builder matching the simulator."""

    def __init__(
        self,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        *,
        control_dt_s: float = 0.05,
        target_translation_scale_m: float = 0.03,
        linear_velocity_clip_mps: float = 10.0,
        angular_velocity_clip_radps: float = 50.0,
        include_gravity_palm: bool = True,
        object_descriptor: torch.Tensor | None = None,
        object_descriptor_dim: int = OBJECT_DESCRIPTOR_DIM,
    ) -> None:
        lower = (
            joint_lower.detach().to(dtype=torch.float32)
            if isinstance(joint_lower, torch.Tensor)
            else torch.as_tensor(joint_lower, dtype=torch.float32)
        )
        upper = torch.as_tensor(joint_upper, device=lower.device, dtype=torch.float32)
        if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
            raise ValueError(f"joint limits must each have shape ({ACTION_DIM},).")
        if not bool(torch.all(upper > lower)):
            raise ValueError("Every joint upper limit must be greater than its lower limit.")
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive.")
        self.joint_lower = lower
        self.joint_upper = upper
        self.control_dt_s = float(control_dt_s)
        self.target_translation_scale_m = float(target_translation_scale_m)
        self.linear_velocity_clip_mps = float(linear_velocity_clip_mps)
        self.angular_velocity_clip_radps = float(angular_velocity_clip_radps)
        self.include_gravity_palm = bool(include_gravity_palm)
        self.object_descriptor_dim = int(object_descriptor_dim)
        if self.object_descriptor_dim != OBJECT_DESCRIPTOR_DIM or not self.include_gravity_palm:
            raise ValueError("POISE requires BPS64 and palm-frame gravity (244D observation).")
        if self.object_descriptor_dim:
            if object_descriptor is None:
                raise ValueError("The 244D BPS64 contract requires object_descriptor.")
            descriptor = torch.as_tensor(
                object_descriptor,
                device=lower.device,
                dtype=torch.float32,
            )
            if descriptor.ndim == 1:
                descriptor = descriptor.unsqueeze(0)
            if descriptor.ndim != 2 or descriptor.shape[1] != self.object_descriptor_dim:
                raise ValueError(
                    "object_descriptor must have shape "
                    f"({self.object_descriptor_dim},) or (batch, "
                    f"{self.object_descriptor_dim}), got {tuple(descriptor.shape)}."
                )
            if not bool(torch.isfinite(descriptor).all()):
                raise ValueError("object_descriptor contains NaN or Inf.")
            self.object_descriptor = descriptor
            self.observation_dim = OBSERVATION_DIM

        self._history: torch.Tensor | None = None
        self._object_pose: torch.Tensor | None = None
        self._object_velocity: torch.Tensor | None = None
        self._single_input = False

    @property
    def initialized(self) -> bool:
        return self._history is not None

    @property
    def batch_size(self) -> int:
        if self._history is None:
            raise RuntimeError("Observation builder has not been reset.")
        return int(self._history.shape[0])

    def reset(
        self,
        joint_position: torch.Tensor,
        joint_command: torch.Tensor,
        object_pose_palm: torch.Tensor,
    ) -> None:
        """Initialize all history frames and zero the object velocity estimate."""

        q, single = _as_batch(
            joint_position,
            ACTION_DIM,
            name="joint_position",
            device=self.joint_lower.device,
            dtype=torch.float32,
        )
        q_cmd, _ = _as_batch(
            joint_command,
            ACTION_DIM,
            name="joint_command",
            device=self.joint_lower.device,
            dtype=torch.float32,
            batch_size=q.shape[0],
        )
        pose, _ = _as_batch(
            object_pose_palm,
            7,
            name="object_pose_palm",
            device=self.joint_lower.device,
            dtype=torch.float32,
            batch_size=q.shape[0],
        )
        pose = pose.clone()
        pose[:, 3:7] = positive_quat_wxyz(pose[:, 3:7])
        q_scaled = scale_joint_position(q, self.joint_lower, self.joint_upper)
        q_cmd_scaled = scale_joint_position(q_cmd, self.joint_lower, self.joint_upper)
        frame = torch.cat((q_scaled, q_cmd_scaled), dim=-1)
        self._history = frame.unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
        self._object_pose = pose
        self._object_velocity = torch.zeros((q.shape[0], 6), device=q.device, dtype=torch.float32)
        self._single_input = single

    def _require_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._history is None or self._object_pose is None or self._object_velocity is None:
            raise RuntimeError("Call reset() before observe() or step().")
        return self._history, self._object_pose, self._object_velocity

    def _object_descriptor_batch(self, history: torch.Tensor) -> torch.Tensor | None:
        if self.object_descriptor is None:
            return None
        descriptor = self.object_descriptor
        if descriptor.shape[0] == 1 and self.batch_size > 1:
            descriptor = descriptor.expand(self.batch_size, -1)
        elif descriptor.shape[0] != self.batch_size:
            raise ValueError(
                f"object_descriptor batch size {descriptor.shape[0]} does not "
                f"match observation batch size {self.batch_size}."
            )
        return descriptor.to(device=history.device, dtype=history.dtype)

    def observe(
        self,
        target: Pose6DTarget,
        *,
        pose_tracker_confidence: torch.Tensor | float = 1.0,
        gravity_direction_palm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build an observation without advancing history or finite differences."""

        history, pose, velocity = self._require_state()
        target_batch = _target_as_batch(
            target,
            device=history.device,
            dtype=history.dtype,
            batch_size=self.batch_size,
        )
        confidence = _as_column(
            pose_tracker_confidence,
            name="pose_tracker_confidence",
            device=history.device,
            dtype=history.dtype,
            batch_size=self.batch_size,
        )
        gravity = None
        if self.include_gravity_palm:
            if gravity_direction_palm is None:
                raise ValueError("The 244D observation contract requires gravity_direction_palm.")
            gravity, _ = _as_batch(
                gravity_direction_palm,
                3,
                name="gravity_direction_palm",
                device=history.device,
                dtype=history.dtype,
                batch_size=self.batch_size,
            )
        observation = compose_observation(
            history,
            pose,
            target_batch,
            velocity,
            confidence,
            gravity,
            self._object_descriptor_batch(history),
            include_gravity_palm=self.include_gravity_palm,
            object_descriptor_dim=self.object_descriptor_dim,
            target_translation_scale_m=self.target_translation_scale_m,
            linear_velocity_clip_mps=self.linear_velocity_clip_mps,
            angular_velocity_clip_radps=self.angular_velocity_clip_radps,
        )
        return observation.squeeze(0) if self._single_input else observation

    def step(
        self,
        joint_position: torch.Tensor,
        joint_command: torch.Tensor,
        object_pose_palm: torch.Tensor,
        target: Pose6DTarget,
        *,
        pose_tracker_confidence: torch.Tensor | float = 1.0,
        gravity_direction_palm: torch.Tensor | None = None,
        dt_s: float | None = None,
    ) -> torch.Tensor:
        """Advance one control frame and return its policy observation."""

        history, previous_pose, _ = self._require_state()
        q, _ = _as_batch(
            joint_position,
            ACTION_DIM,
            name="joint_position",
            device=history.device,
            dtype=history.dtype,
            batch_size=self.batch_size,
        )
        q_cmd, _ = _as_batch(
            joint_command,
            ACTION_DIM,
            name="joint_command",
            device=history.device,
            dtype=history.dtype,
            batch_size=self.batch_size,
        )
        pose, _ = _as_batch(
            object_pose_palm,
            7,
            name="object_pose_palm",
            device=history.device,
            dtype=history.dtype,
            batch_size=self.batch_size,
        )
        pose = pose.clone()
        pose[:, 3:7] = positive_quat_wxyz(pose[:, 3:7])
        frame = torch.cat(
            (
                scale_joint_position(q, self.joint_lower, self.joint_upper),
                scale_joint_position(q_cmd, self.joint_lower, self.joint_upper),
            ),
            dim=-1,
        )
        self._history = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)

        delta_t = self.control_dt_s if dt_s is None else float(dt_s)
        if delta_t <= 0.0:
            raise ValueError("dt_s must be positive.")
        linear_velocity = (pose[:, :3] - previous_pose[:, :3]) / delta_t
        delta_quat = quat_mul_wxyz(
            pose[:, 3:7],
            quat_conjugate_wxyz(previous_pose[:, 3:7]),
        )
        angular_velocity = quat_to_rotvec_wxyz(delta_quat) / delta_t
        self._object_pose = pose
        self._object_velocity = torch.cat((linear_velocity, angular_velocity), dim=-1)
        return self.observe(
            target,
            pose_tracker_confidence=pose_tracker_confidence,
            gravity_direction_palm=gravity_direction_palm,
        )
