"""Stateful policy execution with acknowledged joint commands and fresh poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from poise.core.observation import SharpaPose6DObservationBuilder, make_pose6d_target
from poise.core.policy_bundle import PolicyBundle
from poise.deploy.calibration import as_transform, rotation_matrix_to_quat_wxyz


def transform_to_pose(transform):
    transform = as_transform(np.asarray(transform, dtype=np.float64))
    return torch.tensor([*transform[:3, 3], *rotation_matrix_to_quat_wxyz(transform[:3, :3])], dtype=torch.float32)


@dataclass(frozen=True)
class ControlLimits:
    pose_age_s: float = 0.10
    joint_age_s: float = 0.20
    translation_jump_m: float = 0.01
    rotation_jump_deg: float = 15.0


class PolicyController:
    """Produce one command at a time; commands enter history after acknowledgment."""

    def __init__(self, bundle: PolicyBundle, q, q_cmd, pose, target_pose, gravity, *, timestamp_ms, limits=ControlLimits()):
        self.bundle, self.limits = bundle, limits
        contract = bundle.manifest["sim_contract"]
        q = self._vector(q, 22, "measured joints")
        q_cmd = self._vector(q_cmd, 22, "commanded joints")
        pose = self._pose(pose)
        target_pose = self._pose(target_pose)
        if not np.isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("Initial camera timestamp must be finite and nonnegative")
        lower, upper = torch.as_tensor(contract["joint_lower"]), torch.as_tensor(contract["joint_upper"])
        if (q_cmd < lower).any() or (q_cmd > upper).any():
            raise ValueError("Initial command is outside policy joint limits")
        self.builder = SharpaPose6DObservationBuilder(
            contract["joint_lower"], contract["joint_upper"],
            control_dt_s=contract["control_dt_s"],
            target_translation_scale_m=contract["target_translation_scale_m"],
            object_descriptor=bundle.object_descriptor,
        )
        self.q_cmd = torch.as_tensor(q_cmd, dtype=torch.float32).clone()
        self.pose = torch.as_tensor(pose, dtype=torch.float32).clone()
        self.target = make_pose6d_target(self.pose, torch.as_tensor(target_pose, dtype=torch.float32), source_anchor_local_pos_m=torch.zeros(3))
        self.gravity = torch.as_tensor(gravity, dtype=torch.float32)
        if self.gravity.shape != (3,) or not torch.isfinite(self.gravity).all() or not torch.isclose(self.gravity.norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError("gravity must be a finite unit vector in the palm frame")
        self.builder.reset(torch.as_tensor(q), self.q_cmd, self.pose)
        self.timestamp_ms = float(timestamp_ms)
        self.pending = None
        self.pending_pose = None
        self.pending_timestamp_ms = None
        self.failed = False

    @staticmethod
    def _vector(value, width, name):
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.shape != (width,) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite {width}D vector")
        return value

    @classmethod
    def _pose(cls, value):
        value = cls._vector(value, 7, "object pose")
        if not torch.isclose(value[3:].norm(), torch.tensor(1.0), atol=1e-4):
            raise ValueError("Object pose must have a unit quaternion")
        return value

    def propose(self, q, pose, *, timestamp_ms, pose_age_s, joint_age_s):
        if self.failed or self.pending is not None:
            raise RuntimeError("Controller stopped or previous command unacknowledged")
        try:
            if not np.isfinite([timestamp_ms, pose_age_s, joint_age_s]).all():
                raise ValueError("Non-finite sensor timing")
            if not 0 <= pose_age_s <= self.limits.pose_age_s or not 0 <= joint_age_s <= self.limits.joint_age_s:
                raise RuntimeError("Stale sensor sample")
            if timestamp_ms <= self.timestamp_ms:
                raise RuntimeError("Camera timestamps must increase")
            q = self._vector(q, 22, "measured joints")
            pose = self._pose(pose)
            distance = (pose[:3] - self.pose[:3]).norm().item()
            cosine = torch.clamp(torch.abs(torch.dot(pose[3:], self.pose[3:])), 0, 1)
            angle = torch.rad2deg(2 * torch.acos(cosine)).item()
            if distance > self.limits.translation_jump_m or angle > self.limits.rotation_jump_deg:
                raise RuntimeError("Object tracking jumped")
            obs = self.builder.step(torch.as_tensor(q), self.q_cmd, pose, self.target,
                                    gravity_direction_palm=self.gravity, dt_s=(timestamp_ms - self.timestamp_ms) / 1000)
            action = self.bundle.infer(obs)
            command = self.bundle.next_joint_target(self.q_cmd, action)
            if command.shape != (22,) or not torch.isfinite(command).all():
                raise RuntimeError("Invalid policy command")
            self.pending = command.detach().cpu().clone()
            self.pending_pose, self.pending_timestamp_ms = pose.clone(), float(timestamp_ms)
            return self.pending.numpy().copy()
        except BaseException:
            self.failed = True
            raise

    def acknowledge(self, accepted):
        accepted = torch.as_tensor(accepted, dtype=torch.float32)
        if self.pending is None or accepted.shape != (22,) or not torch.allclose(accepted, self.pending, atol=1e-7, rtol=0):
            self.failed = True
            raise RuntimeError("Acknowledged command differs from policy command")
        self.q_cmd, self.pose, self.timestamp_ms = self.pending, self.pending_pose, self.pending_timestamp_ms
        self.pending = self.pending_pose = self.pending_timestamp_ms = None
