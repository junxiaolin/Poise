"""Sharpa Wave pose reaching with stable-grasp resets and adaptive SE(3) targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from poise.core.tactile import TACTILE_PROFILE, TactileHistory, contact_frame
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

from poise.sim.assets.cuboid import get_cuboid_cfg
from poise.sim.assets.capsule import get_capsule_cfg
from poise.sim.assets.cube_3cm import CUBE_3CM_SIDE_LENGTH_M, get_cube_3cm_cfg
from poise.sim.assets.cube_4cm import CUBE_4CM_SIDE_LENGTH_M, get_cube_4cm_cfg
from poise.sim.assets.cube_5cm import CUBE_5CM_SIDE_LENGTH_M, get_cube_5cm_cfg
from poise.sim.assets.cylinder import get_cylinder_cfg
from poise.sim.assets.irregular import (
    IRREGULAR_SHAPE_NAMES,
    get_irregular_object_cfg,
    irregular_shape_name_from_object_id,
    is_irregular_object_id,
)
from poise.sim.assets.sphere import get_sphere_cfg
from poise.sim.assets.sharpa_manual_convex_objects import (
    SHARPA_MANUAL_CONVEX_OBJECT_IDS,
    get_manual_convex_object_spec,
    get_sharpa_manual_convex_object_cfg,
)
from poise.sim.assets.sharpa_manual_hex_prisms import (
    SHARPA_MANUAL_HEX_PRISM_FAMILY,
    SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS,
    get_sharpa_manual_hex_prism_cfg,
)
from poise.sim.assets.sharpa_hand import SHARPA_DEPLOY_DOF_LIMITS_DEG, resolve_sharpa_root_rot
from poise.sim.curriculum_utils import adaptive_soft_promotion_is_eligible
from poise.sim.pose_success import full_pose_success_terms
from poise.sim.config import PoiseEnvCfg
from poise.sim.hb1_sim2real import (
    compound_cuboid_mass_properties,
    composite_deploy_usd_limits,
    conservative_saturated_grasp_wrench_coverage,
    contact_gated_excess_load_terms,
    cuboid_inertia_diagonal,
    geometry_cache_anchor_ids,
    lookup_mesh_surface_field,
    normalized_grasp_wrench_maintenance_reward,
    mass_bucket_ids,
    mass_size_scaled_inertia_diagonal,
    sample_density_correlated_cuboid_masses,
    sample_density_correlated_cube_masses,
    sample_independent_two_sided_scales,
    sample_inertia_multipliers,
    sample_nearcube_geometry_cpu,
    sample_nominal_log_uniform_masses,
    sample_real_to_legacy_cuboid_inertia_diagonal,
    scale_cache_bucket_ids,
    scaled_signed_usd_limits,
    timebase_matched_trigger_probability,
    timebase_matched_wrench_scale,
    world_torque_for_target_angular_acceleration,
)
from poise.sim.joint_mapping import (
    SHARPA_CSGWCD_COLLISION_BODIES,
    SHARPA_CSGWCD_GROUP_IDS,
    SHARPA_CSGWCD_PRIMARY_BODIES,
    SHARPA_CSGWCD_SECONDARY_BODIES,
    resolve_hand_mapping,
    resolve_hand_spec,
)
from poise.sim.metrics import (
    cuboid_keypoints_from_pose,
    cuboid_pose_distance,
    cube_keypoints_from_pose,
    cube_pose_distance,
    normalize_quat_wxyz,
    quat_apply_wxyz,
)
from poise.grasp.cache import load_stable_grasp_cache


from poise.paths import resolve_path as resolve_workspace_path


def _tuple_cfg(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _filter_tuple_by_indices(value: Any, indices: tuple[int, ...], attr_name: str, expected_len: int) -> tuple[Any, ...]:
    items = _tuple_cfg(value)
    if len(items) == 0:
        return ()
    if len(items) != expected_len:
        raise ValueError(
            f"{attr_name} must be empty or have length {expected_len} for primitive rank object partition; "
            f"got length {len(items)}."
        )
    return tuple(items[index] for index in indices)


def _partition_indices_for_rank(
    total: int,
    rank: int,
    world_size: int,
    mode: str,
    objects_per_rank: int = 0,
    cycle_offset: int = 0,
) -> tuple[int, ...]:
    if mode == "strided":
        indices = tuple(index for index in range(total) if index % world_size == rank)
    elif mode == "contiguous":
        base = total // world_size
        remainder = total % world_size
        start = rank * base + min(rank, remainder)
        stop = start + base + (1 if rank < remainder else 0)
        indices = tuple(range(start, stop))
    else:
        raise ValueError(
            "primitive_rank_object_partition_mode must be one of 'strided' or 'contiguous', "
            f"got {mode!r}."
        )
    if len(indices) == 0:
        raise ValueError(
            "Primitive rank object partition produced an empty object subset. "
            f"total={total}, rank={rank}, world_size={world_size}, mode={mode!r}."
        )
    if objects_per_rank > 0 and objects_per_rank < len(indices):
        start = cycle_offset % len(indices)
        indices = (indices[start:] + indices[:start])[:objects_per_rank]
    return indices


def _apply_primitive_rank_object_partition(cfg: PoiseEnvCfg) -> None:
    """Optionally split primitive objects across DDP ranks to reduce per-rank physics load."""

    if not bool(getattr(cfg, "primitive_rank_object_partition_enabled", False)):
        return
    if not bool(getattr(cfg, "primitive_multi_shape_enabled", False)):
        raise ValueError("primitive_rank_object_partition_enabled requires primitive_multi_shape_enabled=True.")

    object_ids = tuple(str(item) for item in _tuple_cfg(getattr(cfg, "primitive_object_ids", ())))
    total = len(object_ids)
    if total <= 1:
        return

    rank_cfg = int(getattr(cfg, "primitive_rank_object_partition_rank", -1))
    world_size_cfg = int(getattr(cfg, "primitive_rank_object_partition_world_size", 0))
    rank = rank_cfg if rank_cfg >= 0 else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = world_size_cfg if world_size_cfg > 0 else int(os.environ.get("WORLD_SIZE", "1"))
    mode = str(getattr(cfg, "primitive_rank_object_partition_mode", "strided")).lower()
    objects_per_rank = max(int(getattr(cfg, "primitive_rank_object_partition_objects_per_rank", 0)), 0)
    cycle_offset = max(int(getattr(cfg, "primitive_rank_object_partition_cycle_offset", 0)), 0)
    if world_size <= 1 and objects_per_rank <= 0:
        return
    if world_size <= 1:
        rank = 0
        world_size = 1
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid primitive rank object partition rank/world_size: rank={rank}, world_size={world_size}."
        )

    indices = _partition_indices_for_rank(total, rank, world_size, mode, objects_per_rank, cycle_offset)
    selected_object_ids = tuple(object_ids[index] for index in indices)

    cfg.primitive_object_ids = selected_object_ids
    cfg.primitive_dims_m = _filter_tuple_by_indices(getattr(cfg, "primitive_dims_m", ()), indices, "primitive_dims_m", total)
    cfg.primitive_object_shape_names = _filter_tuple_by_indices(
        getattr(cfg, "primitive_object_shape_names", ()), indices, "primitive_object_shape_names", total
    )
    selected_cache_paths = _filter_tuple_by_indices(
        getattr(cfg, "primitive_stable_cache_paths", ()),
        indices,
        "primitive_stable_cache_paths",
        total,
    )
    cfg.primitive_stable_cache_paths = selected_cache_paths
    if len(selected_cache_paths) > 0:
        cfg.stable_cache_path = selected_cache_paths[0]
    cfg.primitive_post_success_reset_cache_paths = _filter_tuple_by_indices(
        getattr(cfg, "primitive_post_success_reset_cache_paths", ()),
        indices,
        "primitive_post_success_reset_cache_paths",
        total,
    )
    cfg.primitive_post_success_reset_sample_probs = _filter_tuple_by_indices(
        getattr(cfg, "primitive_post_success_reset_sample_probs", ()),
        indices,
        "primitive_post_success_reset_sample_probs",
        total,
    )

    anchor_ids = tuple(str(item) for item in _tuple_cfg(getattr(cfg, "pose6d_anchor_per_object_ids", ())))
    if len(anchor_ids) > 0:
        selected_set = set(selected_object_ids)
        anchor_indices = tuple(index for index, object_id in enumerate(anchor_ids) if object_id in selected_set)
        cfg.pose6d_anchor_per_object_ids = tuple(anchor_ids[index] for index in anchor_indices)
        cfg.pose6d_anchor_per_object_success_thresholds_m = _filter_tuple_by_indices(
            getattr(cfg, "pose6d_anchor_per_object_success_thresholds_m", ()),
            anchor_indices,
            "pose6d_anchor_per_object_success_thresholds_m",
            len(anchor_ids),
        )
        cfg.pose6d_anchor_per_object_swing_thresholds_rad = _filter_tuple_by_indices(
            getattr(cfg, "pose6d_anchor_per_object_swing_thresholds_rad", ()),
            anchor_indices,
            "pose6d_anchor_per_object_swing_thresholds_rad",
            len(anchor_ids),
        )

    print(
        "[INFO] primitive rank object partition: "
        f"rank={rank}/{world_size}, mode={mode}, objects_per_rank={objects_per_rank}, "
        f"cycle_offset={cycle_offset}, indices={list(indices)}, objects={list(selected_object_ids)}",
        flush=True,
    )


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., (1, 2, 3, 0)]


def quat_to_rotvec_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    vec = quat[..., 1:]
    sin_half = torch.linalg.norm(vec, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half.squeeze(-1), torch.clamp(quat[..., 0], min=1.0e-8))
    scale = torch.where(
        sin_half > 1.0e-6,
        angle.unsqueeze(-1) / torch.clamp(sin_half, min=1.0e-6),
        torch.full_like(sin_half, 2.0),
    )
    return vec * scale


def quat_from_angle_axis_wxyz(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
    half_angle = 0.5 * angle
    quat = torch.cat((torch.cos(half_angle).unsqueeze(-1), axis * torch.sin(half_angle).unsqueeze(-1)), dim=-1)
    return normalize_quat_wxyz(quat)


def randomize_rotation_rpy_wxyz(
    rand0: torch.Tensor,
    rand1: torch.Tensor,
    rand2: torch.Tensor,
    x_unit: torch.Tensor,
    y_unit: torch.Tensor,
    z_unit: torch.Tensor,
) -> torch.Tensor:
    qx = quat_from_angle_axis_wxyz(rand0 * torch.pi, x_unit)
    qy = quat_from_angle_axis_wxyz(rand1 * torch.pi, y_unit)
    qz = quat_from_angle_axis_wxyz(rand2 * torch.pi, z_unit)
    return normalize_quat_wxyz(quat_mul_wxyz(quat_mul_wxyz(qx, qy), qz))


def positive_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def quat_angle_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = positive_quat_wxyz(quat)
    return 2.0 * torch.atan2(
        torch.linalg.norm(quat[..., 1:], dim=-1),
        torch.clamp(quat[..., 0], min=1.0e-8),
    )


def quat_angle_error_wxyz(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    delta = quat_mul_wxyz(target, quat_conjugate_wxyz(current))
    return quat_angle_wxyz(delta)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def signed_twist_angle_wxyz(quat: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-8)
    projected_vec = torch.sum(quat[..., 1:] * axis, dim=-1, keepdim=True) * axis
    twist_raw = torch.cat((quat[..., :1], projected_vec), dim=-1)
    twist_norm = torch.linalg.norm(twist_raw, dim=-1, keepdim=True)
    identity = torch.zeros_like(twist_raw)
    identity[..., 0] = 1.0
    twist = torch.where(twist_norm > 1.0e-8, twist_raw / torch.clamp(twist_norm, min=1.0e-8), identity)
    signed_angle = 2.0 * torch.atan2(torch.sum(twist[..., 1:] * axis, dim=-1), twist[..., 0])
    return wrap_to_pi(signed_angle)


def swing_angle_wxyz(quat: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat_wxyz(quat)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-8)
    projected_vec = torch.sum(quat[..., 1:] * axis, dim=-1, keepdim=True) * axis
    twist_raw = torch.cat((quat[..., :1], projected_vec), dim=-1)
    twist_norm = torch.linalg.norm(twist_raw, dim=-1, keepdim=True)
    identity = torch.zeros_like(twist_raw)
    identity[..., 0] = 1.0
    twist = torch.where(twist_norm > 1.0e-8, twist_raw / torch.clamp(twist_norm, min=1.0e-8), identity)
    swing = quat_mul_wxyz(quat, quat_conjugate_wxyz(twist))
    return quat_angle_wxyz(swing)


def action_to_sim_order(action_values: torch.Tensor, action_joint_indices: tuple[int, ...], num_joints: int) -> torch.Tensor:
    result = torch.zeros(action_values.shape[0], num_joints, device=action_values.device, dtype=action_values.dtype)
    result[:, list(action_joint_indices)] = action_values
    return result


def sim_to_action_order(sim_values: torch.Tensor, action_joint_indices: tuple[int, ...]) -> torch.Tensor:
    return sim_values[:, list(action_joint_indices)]


def scale_to_unit(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(upper - lower, min=1.0e-6)
    scaled = 2.0 * (value - lower) / denom - 1.0
    return torch.clamp(scaled, -1.5, 1.5)


def rank01(values: torch.Tensor) -> torch.Tensor:
    """Return percentile-like ranks in [0, 1], with larger input values ranked higher."""

    values = values.detach().float()
    num = int(values.numel())
    if num <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.linspace(0.0, 1.0, num, device=values.device, dtype=values.dtype)
    return ranks


MULTISIZE_CUBE_SPECS = {
    "cube_3cm": (CUBE_3CM_SIDE_LENGTH_M, get_cube_3cm_cfg, "Cube3cm"),
    "cube_4cm": (CUBE_4CM_SIDE_LENGTH_M, get_cube_4cm_cfg, "Cube4cm"),
    "cube_5cm": (CUBE_5CM_SIDE_LENGTH_M, get_cube_5cm_cfg, "Cube5cm"),
}


MULTICUBOID_VISUAL_COLORS = (
    (0.95, 0.68, 0.32),
    (0.30, 0.78, 0.67),
    (0.42, 0.74, 0.94),
    (0.80, 0.62, 0.92),
)

DEFAULT_PRIMITIVE_SHAPE_NAMES = ("cuboid", "cylinder", "sphere", "capsule") + IRREGULAR_SHAPE_NAMES
LOCAL_Y_CYLINDER_OBJECT_IDS = {"cylinder_2x20cm", "cylinder_2cm_20cm_y"}



def primitive_shape_name_from_object_id(object_id: str) -> str:
    """Return the primitive shape family encoded in an object id."""

    if object_id in MULTISIZE_CUBE_SPECS or object_id.startswith("cube_"):
        return "cuboid"
    if object_id in SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS:
        return SHARPA_MANUAL_HEX_PRISM_FAMILY
    if object_id in SHARPA_MANUAL_CONVEX_OBJECT_IDS:
        return get_manual_convex_object_spec(object_id).shape_family
    if is_irregular_object_id(object_id):
        return irregular_shape_name_from_object_id(object_id)
    for shape_name in DEFAULT_PRIMITIVE_SHAPE_NAMES:
        if object_id.startswith(f"{shape_name}_"):
            return shape_name
    raise ValueError(
        f"Unsupported primitive object_id={object_id!r}; expected one of "
        f"{', '.join(f'{name}_*' for name in DEFAULT_PRIMITIVE_SHAPE_NAMES)}."
    )


def is_local_y_cylinder_object_id(object_id: str) -> bool:
    """Return whether a cylinder id uses the local-Y long-axis USD convention."""

    return object_id in LOCAL_Y_CYLINDER_OBJECT_IDS or (
        object_id.startswith("cylinder_") and object_id.endswith("_y")
    )


def validate_primitive_dims(object_id: str, dims_m: tuple[float, float, float]) -> str:
    """Validate primitive dimensions and return the shape family name."""

    shape_name = primitive_shape_name_from_object_id(object_id)
    if len(dims_m) != 3 or any(v <= 0.0 for v in dims_m):
        raise ValueError(f"Invalid dimensions for {object_id!r}: {dims_m!r}.")
    if (object_id in MULTISIZE_CUBE_SPECS or object_id.startswith("cube_")) and (
        abs(dims_m[0] - dims_m[1]) > 1.0e-6 or abs(dims_m[0] - dims_m[2]) > 1.0e-6
    ):
        raise ValueError(f"Cube dimensions must encode side, side, side; got {dims_m!r}.")
    if (
        shape_name in {"cylinder", "capsule"}
        and not is_local_y_cylinder_object_id(object_id)
        and abs(dims_m[0] - dims_m[1]) > 1.0e-6
    ):
        raise ValueError(f"{shape_name} dimensions must encode diameter, diameter, height; got {dims_m!r}.")
    if is_local_y_cylinder_object_id(object_id) and (
        abs(dims_m[0] - dims_m[2]) > 1.0e-6 or dims_m[1] <= dims_m[0]
    ):
        raise ValueError(f"Local-Y cylinder dimensions must encode diameter, length, diameter; got {dims_m!r}.")
    if shape_name == "capsule" and dims_m[2] <= dims_m[0]:
        raise ValueError(f"Capsule bbox height must be larger than diameter, got {dims_m!r}.")
    if shape_name == "sphere" and (abs(dims_m[0] - dims_m[1]) > 1.0e-6 or abs(dims_m[0] - dims_m[2]) > 1.0e-6):
        raise ValueError(f"Sphere dimensions must encode diameter, diameter, diameter; got {dims_m!r}.")
    if shape_name in {"cone", "frustum", "asym_capsule"} and abs(dims_m[0] - dims_m[1]) > 1.0e-6:
        raise ValueError(f"{shape_name} dimensions must encode diameter, diameter, height; got {dims_m!r}.")
    return shape_name


def capsule_shaft_height_from_bbox(dims_m: tuple[float, float, float]) -> float:
    """Return USD capsule shaft height from bbox dimensions."""

    return float(dims_m[2]) - float(dims_m[0])


def multiobject_prim_name(object_id: str) -> str:
    """Return a compact, stable prim name for a spawned object id."""

    return "".join(part[:1].upper() + part[1:] for part in object_id.split("_") if part)


class PoiseEnv(DirectRLEnv):
    """Direct RL environment for Wuji in-hand cube reorientation."""

    cfg: PoiseEnvCfg

    def __init__(self, cfg: PoiseEnvCfg, render_mode: str | None = None, **kwargs):
        self.pose6d_success_criterion = str(getattr(cfg, "pose6d_success_criterion", "legacy_components"))
        if self.pose6d_success_criterion not in {"legacy_components", "full_pose"}:
            raise ValueError("pose6d_success_criterion must be 'full_pose' or 'legacy_components'")
        if self.pose6d_success_criterion == "full_pose" and (
            cfg.pose6d_success_position_threshold_m <= 0.0 or cfg.pose6d_success_orientation_threshold_rad <= 0.0
        ):
            raise ValueError("Full-pose success thresholds must be positive")
        hand_spec = resolve_hand_spec(str(cfg.hand_type))
        self.hand_spec = hand_spec
        self.action_dim = hand_spec.action_dim
        hand_root_rot = resolve_sharpa_root_rot(str(cfg.hand_root_orientation))
        cfg.action_space = hand_spec.action_dim
        cfg.robot_cfg.init_state.rot = hand_root_rot
        if bool(getattr(cfg, "dexndm_low_gain_drive_enabled", False)):
            if bool(getattr(cfg, "dexndm_torque_control", False)):
                raise ValueError("dexndm_low_gain_drive_enabled and dexndm_torque_control cannot both be true.")
            for actuator_cfg in cfg.robot_cfg.actuators.values():
                actuator_cfg.stiffness = float(cfg.dexndm_low_gain_drive_stiffness)
                actuator_cfg.damping = float(cfg.dexndm_low_gain_drive_damping)
                actuator_cfg.effort_limit_sim = float(cfg.dexndm_low_gain_drive_effort_limit_sim)
        torque_gain_source = str(getattr(cfg, "dexndm_torque_control_gain_source", "scalar")).lower()
        if torque_gain_source not in {"scalar", "default_joint_pd"}:
            raise ValueError(
                "dexndm_torque_control_gain_source must be 'scalar' or 'default_joint_pd', "
                f"got {torque_gain_source!r}."
            )
        torque_velocity_source = str(getattr(cfg, "dexndm_torque_control_velocity_source", "finite_difference")).lower()
        if torque_velocity_source not in {"finite_difference", "joint_vel"}:
            raise ValueError(
                "dexndm_torque_control_velocity_source must be 'finite_difference' or 'joint_vel', "
                f"got {torque_velocity_source!r}."
            )
        if bool(getattr(cfg, "dexndm_torque_control", False)) and torque_gain_source != "default_joint_pd":
            for actuator_cfg in cfg.robot_cfg.actuators.values():
                actuator_cfg.stiffness = 0.0
                actuator_cfg.damping = 0.0
        _apply_primitive_rank_object_partition(cfg)
        observation_profile = str(cfg.observation_profile).lower()
        self.tactile_enabled = observation_profile == TACTILE_PROFILE
        if observation_profile not in ("sharpa_pose6d_deployable_v1_notactile", TACTILE_PROFILE):
            raise ValueError("POISE requires the BPS64 + gravity observation profile.")
        object_pose_obs_delay_mode = str(
            getattr(cfg, "sharpa_deployable_object_pose_obs_delay_mode", "episode_fixed")
        ).lower()
        if object_pose_obs_delay_mode not in {"episode_fixed", "per_step_random"}:
            raise ValueError(
                "sharpa_deployable_object_pose_obs_delay_mode must be 'episode_fixed' or "
                f"'per_step_random', got {object_pose_obs_delay_mode!r}."
            )
        object_pose_obs_dropout_mode = str(
            getattr(cfg, "sharpa_deployable_object_pose_obs_dropout_mode", "hold_last")
        ).lower()
        if object_pose_obs_dropout_mode != "hold_last":
            raise ValueError(
                "sharpa_deployable_object_pose_obs_dropout_mode currently supports only "
                f"'hold_last', got {object_pose_obs_dropout_mode!r}."
            )
        qtarget_history_enabled = False
        fingertip_full_state_obs_enabled = False
        force_obs_enabled = False
        all_body_contact_force_obs_enabled = False
        object_full_state_obs_enabled = False
        dexndm_guided_rotation_obs_enabled = False
        contact_sensor_required = True
        cfg.observation_space = 304 if self.tactile_enabled else 244
        cfg.state_space = 552 if self.tactile_enabled else 492
        cfg.cube_cfg.spawn.activate_contact_sensors = True
        cfg.scene.clone_in_fabric = False
        super().__init__(cfg, render_mode, **kwargs)
        if self.tactile_enabled:
            self.tactile = TactileHistory(self.num_envs, self.device, cfg.seed,
                                         getattr(cfg, "tactile_hold_probability", 0.005))
            self.tactile_body_ids = [self.hand.body_names.index(n) for n in hand_spec.force_sensor_bodies]
        self.training_metric_env_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._apply_cube_inertia_override()
        self.hand_spec = hand_spec
        self.action_dim = hand_spec.action_dim
        self.hand_root_pos = tuple(float(v) for v in cfg.robot_cfg.init_state.pos)
        self.hand_root_rot = tuple(float(v) for v in hand_root_rot)
        self.dexndm_omni_wrist_ornt = bool(getattr(self.cfg, "dexndm_omni_wrist_ornt", False))
        self.observation_profile = observation_profile
        self.sharpa_deployable_profile = True
        self.sharpa_deployable_v2_profile = False
        self.sharpa_deployable_tactile_obs_enabled = False
        self.sharpa_deployable_asymmetric_critic_enabled = True
        self.sharpa_deployable_gravity_palm_obs_enabled = True
        self.sharpa_deployable_object_pose_obs_delay_noise_enabled = bool(
            getattr(self.cfg, "sharpa_deployable_object_pose_obs_delay_noise_enabled", False)
        )
        self.sharpa_deployable_object_pose_obs_delay_max_steps = max(
            0,
            int(getattr(self.cfg, "sharpa_deployable_object_pose_obs_delay_max_steps", 0)),
        )
        self.sharpa_deployable_object_pose_obs_delay_mode = object_pose_obs_delay_mode
        self.sharpa_deployable_object_pose_obs_pos_noise_std_m = max(
            0.0,
            float(getattr(self.cfg, "sharpa_deployable_object_pose_obs_pos_noise_std_m", 0.0)),
        )
        self.sharpa_deployable_object_pose_obs_rot_noise_rad = max(
            0.0,
            float(getattr(self.cfg, "sharpa_deployable_object_pose_obs_rot_noise_deg", 0.0)) * torch.pi / 180.0,
        )
        self.sharpa_deployable_object_pose_obs_rot_noise_distribution = str(
            getattr(
                self.cfg,
                "sharpa_deployable_object_pose_obs_rot_noise_distribution",
                "gaussian_clipped",
            )
        ).strip().lower()
        if self.sharpa_deployable_object_pose_obs_rot_noise_distribution not in {
            "gaussian_clipped",
            "uniform_bounded",
        }:
            raise ValueError(
                "sharpa_deployable_object_pose_obs_rot_noise_distribution must be "
                "'gaussian_clipped' or 'uniform_bounded'."
            )
        self.sharpa_deployable_object_pose_obs_dropout_prob = min(
            max(float(getattr(self.cfg, "sharpa_deployable_object_pose_obs_dropout_prob", 0.0)), 0.0),
            1.0,
        )
        self.sharpa_deployable_object_pose_obs_dropout_confidence = float(
            getattr(self.cfg, "sharpa_deployable_object_pose_obs_dropout_confidence", 0.0)
        )
        self.dexndm_reward_mode = bool(getattr(self.cfg, "dexndm_reward_mode", False))
        self.object_keypoint_side_length_m = float(getattr(self.cfg, "object_keypoint_side_length_m", 0.04))
        self.use_hora_observation_profile = False
        self.action_target_obs_enabled = observation_profile != 'hora_strict' and bool(self.cfg.enable_action_target_obs)
        self.target_error_obs_enabled = observation_profile != 'hora_strict' and bool(self.cfg.enable_target_error_obs)
        self.object_pose_obs_enabled = observation_profile != 'hora_strict' and bool(self.cfg.enable_object_pose_obs)
        self.object_full_state_obs_enabled = object_full_state_obs_enabled
        self.legacy_contact_sensor_obs_enabled = False
        self.legacy_fingertip_contact_force_obs_enabled = False
        self.legacy_contact_bit_obs_enabled = False

        self.mapping = resolve_hand_mapping(self.hand_spec, self.hand.joint_names, self.hand.body_names)
        self.hand_body_name_to_idx = {name: idx for idx, name in enumerate(self.hand.body_names)}
        if len(self.mapping.action_joint_indices) != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} {self.hand_spec.hand_id} action joints, "
                f"got {len(self.mapping.action_joint_indices)}."
            )

        scale_cache_manifest_cfg = str(
            getattr(self.cfg, "scale_conditioned_stable_cache_manifest_path", "") or ""
        )
        geometry_cache_manifest_cfg = str(
            getattr(self.cfg, "geometry_conditioned_stable_cache_manifest_path", "") or ""
        )
        if scale_cache_manifest_cfg and geometry_cache_manifest_cfg:
            raise ValueError(
                "scale_conditioned_stable_cache_manifest_path and "
                "geometry_conditioned_stable_cache_manifest_path are mutually exclusive."
            )
        self.scale_conditioned_stable_cache_enabled = bool(scale_cache_manifest_cfg)
        self.geometry_conditioned_stable_cache_enabled = bool(geometry_cache_manifest_cfg)
        self.scale_conditioned_stable_cache_manifest_path: Path | None = None
        self.geometry_conditioned_stable_cache_manifest_path: Path | None = None
        self.stable_cache_path = resolve_workspace_path(self.cfg.stable_cache_path)
        if self.geometry_conditioned_stable_cache_enabled:
            self.geometry_conditioned_stable_cache_manifest_path = resolve_workspace_path(
                geometry_cache_manifest_cfg
            )
            self.stable_cache = self._load_geometry_conditioned_stable_cache_bank(
                self.geometry_conditioned_stable_cache_manifest_path
            )
        elif self.scale_conditioned_stable_cache_enabled:
            self.scale_conditioned_stable_cache_manifest_path = resolve_workspace_path(scale_cache_manifest_cfg)
            self.stable_cache = self._load_scale_conditioned_stable_cache_bank(
                self.scale_conditioned_stable_cache_manifest_path
            )
        else:
            self.stable_cache = self._load_stable_cache_to_device(self.stable_cache_path)
        self.num_stable_grasps = int(self.stable_cache["q"].shape[0])
        self.multisize_cube_enabled = bool(getattr(self.cfg, "multisize_cube_enabled", False))
        self.multicuboid_enabled = bool(getattr(self.cfg, "multicuboid_enabled", False))
        self.primitive_multi_shape_enabled = bool(getattr(self.cfg, "primitive_multi_shape_enabled", False))
        self.primitive_single_asset_per_env_enabled = bool(
            getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)
        )
        if self.primitive_single_asset_per_env_enabled and not self.primitive_multi_shape_enabled:
            raise ValueError("primitive_single_asset_per_env_enabled requires primitive_multi_shape_enabled=True.")
        enabled_multiobject_modes = sum(
            int(v) for v in (self.multisize_cube_enabled, self.multicuboid_enabled, self.primitive_multi_shape_enabled)
        )
        if enabled_multiobject_modes > 1:
            raise ValueError(
                "multisize_cube_enabled, multicuboid_enabled, and primitive_multi_shape_enabled "
                "are mutually exclusive."
            )
        self.multiobject_enabled = (
            self.multisize_cube_enabled or self.multicuboid_enabled or self.primitive_multi_shape_enabled
        )
        self.post_success_reset_sample_prob = min(
            max(float(getattr(self.cfg, "post_success_reset_sample_prob", 0.0)), 0.0),
            1.0,
        )
        self.post_success_stable_cache: dict[str, Any] | None = None
        self.num_post_success_stable_grasps = 0
        post_success_path_cfg = str(getattr(self.cfg, "post_success_reset_cache_path", "") or "")
        if self.post_success_reset_sample_prob > 0.0 and not self.multiobject_enabled:
            if self.scale_conditioned_stable_cache_enabled or self.geometry_conditioned_stable_cache_enabled:
                cache_kind = "Scale" if self.scale_conditioned_stable_cache_enabled else "Geometry"
                raise ValueError(
                    f"{cache_kind}-conditioned stable-cache reset does not support a single shared post-success cache. "
                    f"Provide {cache_kind.lower()}-conditioned post-success buckets before enabling "
                    "post_success_reset_sample_prob."
                )
            if not post_success_path_cfg:
                raise ValueError(
                    "post_success_reset_sample_prob > 0 requires post_success_reset_cache_path to be set."
                )
            self.post_success_stable_cache_path = resolve_workspace_path(post_success_path_cfg)
            self.post_success_stable_cache = self._load_stable_cache_to_device(self.post_success_stable_cache_path)
            self.num_post_success_stable_grasps = int(self.post_success_stable_cache["q"].shape[0])
            if self.num_post_success_stable_grasps <= 0:
                raise ValueError(f"Post-success reset cache is empty: {self.post_success_stable_cache_path}")
        self.multisize_cube_size_obs_enabled = bool(getattr(self.cfg, "multisize_cube_size_obs_enabled", False))
        self.multicuboid_dim_obs_enabled = bool(getattr(self.cfg, "multicuboid_dim_obs_enabled", False))
        self.primitive_dim_obs_enabled = bool(getattr(self.cfg, "primitive_dim_obs_enabled", False))
        self.primitive_shape_obs_enabled = bool(getattr(self.cfg, "primitive_shape_obs_enabled", False))
        self.primitive_geometry_latent_obs_enabled = bool(
            getattr(self.cfg, "primitive_geometry_latent_obs_enabled", False)
        )
        self.primitive_geometry_latent_source = str(
            getattr(self.cfg, "primitive_geometry_latent_source", "lookup")
        ).lower()
        if self.primitive_geometry_latent_source not in {"lookup", "hand_surface_delta"}:
            raise ValueError(
                "primitive_geometry_latent_source must be one of 'lookup' or 'hand_surface_delta', "
                f"got {getattr(self.cfg, 'primitive_geometry_latent_source', None)!r}."
            )
        self.primitive_geometry_latent_dim = int(getattr(self.cfg, "primitive_geometry_latent_dim", 0))
        self.sharpa_deployable_mesh_bps_obs_enabled = bool(
            getattr(self.cfg, "sharpa_deployable_mesh_bps_obs_enabled", False)
        )
        if self.primitive_geometry_latent_obs_enabled:
            if not self.primitive_multi_shape_enabled:
                raise ValueError("primitive_geometry_latent_obs_enabled requires primitive_multi_shape_enabled=True.")
            if self.primitive_geometry_latent_dim <= 0:
                raise ValueError("primitive_geometry_latent_dim must be positive when geometry latent obs is enabled.")
            if self.primitive_geometry_latent_source == "hand_surface_delta" and self.primitive_geometry_latent_dim != 63:
                raise ValueError(
                    "primitive_geometry_latent_source='hand_surface_delta' expects "
                    "primitive_geometry_latent_dim=63 for palm+link234+fingertips nearest deltas."
                )
        self.multisize_object_ids = tuple(
            str(item)
            for item in getattr(
                self.cfg,
                (
                    "primitive_object_ids"
                    if self.primitive_multi_shape_enabled
                    else "multicuboid_object_ids"
                    if self.multicuboid_enabled
                    else "multisize_cube_object_ids"
                ),
                (),
            )
        )
        if self.primitive_single_asset_per_env_enabled:
            self.fixed_object_size_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self.fixed_object_size_ids.remainder_(len(self.multisize_object_ids))
            print(
                "[INFO] primitive single-asset-per-env: "
                f"envs={self.num_envs}, objects={len(self.multisize_object_ids)}, "
                f"counts={torch.bincount(self.fixed_object_size_ids, minlength=len(self.multisize_object_ids)).tolist()}",
                flush=True,
            )
        self.multisize_side_lengths = torch.empty(0, device=self.device, dtype=torch.float32)
        self.multiobject_dims_m = torch.empty((0, 3), device=self.device, dtype=torch.float32)
        self.multiobject_is_cylinder = torch.empty(0, device=self.device, dtype=torch.bool)
        self.multiobject_shape_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self.multiobject_geometry_latents = torch.empty(
            (0, max(self.primitive_geometry_latent_dim, 0)), device=self.device, dtype=torch.float32
        )
        self.multiobject_surface_points_obj = torch.empty((0, 0, 3), device=self.device, dtype=torch.float32)
        self.multiobject_mesh_com_m = torch.empty((0, 3), device=self.device, dtype=torch.float32)
        self.multiobject_mesh_inertia_unit_mass_kg_m2 = torch.empty((0, 3, 3), device=self.device, dtype=torch.float32)
        self.multiobject_csgwcd_surface_delta_grid_m = torch.empty((0, 0, 0, 0, 3), device=self.device)
        self.multiobject_csgwcd_outward_normal_grid = torch.empty((0, 0, 0, 0, 3), device=self.device)
        self.csgwcd_mesh_grid_half_extent_m = 0.0
        self.primitive_shape_names = tuple(str(v) for v in getattr(self.cfg, "primitive_shape_names", DEFAULT_PRIMITIVE_SHAPE_NAMES))
        if len(self.primitive_shape_names) == 0:
            self.primitive_shape_names = DEFAULT_PRIMITIVE_SHAPE_NAMES
        self.primitive_shape_name_to_id = {name: idx for idx, name in enumerate(self.primitive_shape_names)}
        self.primitive_shape_size_ids: dict[str, torch.Tensor] = {}
        self.multiobject_prim_names: dict[str, str] = {}
        self.multisize_stable_caches: dict[str, dict[str, Any]] = {}
        self.multisize_num_stable_grasps: dict[str, int] = {}
        self.multisize_post_success_stable_caches: dict[str, dict[str, Any]] = {}
        self.multisize_num_post_success_stable_grasps: dict[str, int] = {}
        self.multisize_post_success_reset_sample_probs = torch.empty(0, device=self.device, dtype=torch.float32)
        if self.multisize_cube_enabled:
            if len(self.multisize_object_ids) == 0:
                raise ValueError("multisize_cube_enabled requires non-empty multisize_cube_object_ids.")
            side_lengths = tuple(float(v) for v in getattr(self.cfg, "multisize_cube_side_lengths_m", ()))
            cache_paths = tuple(str(v) for v in getattr(self.cfg, "multisize_cube_stable_cache_paths", ()))
            if len(side_lengths) != len(self.multisize_object_ids) or len(cache_paths) != len(self.multisize_object_ids):
                raise ValueError(
                    "multisize_cube_object_ids, multisize_cube_side_lengths_m, and "
                    "multisize_cube_stable_cache_paths must have the same length."
                )
            unknown = sorted(set(self.multisize_object_ids) - set(MULTISIZE_CUBE_SPECS))
            if unknown:
                raise ValueError(f"Unsupported multisize cube object ids: {unknown}")
            self.multisize_side_lengths = torch.tensor(side_lengths, device=self.device, dtype=torch.float32)
            self.multiobject_dims_m = torch.stack((self.multisize_side_lengths,) * 3, dim=-1)
            self.multiobject_is_cylinder = torch.zeros(
                len(self.multisize_object_ids), device=self.device, dtype=torch.bool
            )
            self.multiobject_shape_ids = torch.zeros(len(self.multisize_object_ids), device=self.device, dtype=torch.long)
            self.multiobject_prim_names = {
                object_id: MULTISIZE_CUBE_SPECS[object_id][2] for object_id in self.multisize_object_ids
            }
            for object_id, cache_path in zip(self.multisize_object_ids, cache_paths, strict=True):
                cache = self._load_stable_cache_to_device(resolve_workspace_path(cache_path))
                cache_object_id = str(cache.get("object_id", ""))
                if cache_object_id != object_id:
                    raise ValueError(
                        f"Configured cache for {object_id!r} has object_id={cache_object_id!r}: {cache_path}"
                    )
                self.multisize_stable_caches[object_id] = cache
                self.multisize_num_stable_grasps[object_id] = int(cache["q"].shape[0])
        elif self.multicuboid_enabled:
            if len(self.multisize_object_ids) == 0:
                raise ValueError("multicuboid_enabled requires non-empty multicuboid_object_ids.")
            dims = tuple(
                tuple(float(v) for v in item)
                for item in getattr(self.cfg, "multicuboid_dims_m", ())
            )
            cache_paths = tuple(str(v) for v in getattr(self.cfg, "multicuboid_stable_cache_paths", ()))
            if len(dims) != len(self.multisize_object_ids) or len(cache_paths) != len(self.multisize_object_ids):
                raise ValueError(
                    "multicuboid_object_ids, multicuboid_dims_m, and "
                    "multicuboid_stable_cache_paths must have the same length."
                )
            for object_id, dim in zip(self.multisize_object_ids, dims, strict=True):
                if not object_id.startswith("cuboid_"):
                    raise ValueError(f"multicuboid object_id must start with 'cuboid_', got {object_id!r}.")
                if len(dim) != 3 or any(v <= 0.0 for v in dim):
                    raise ValueError(f"Invalid dimensions for {object_id!r}: {dim!r}.")
            self.multiobject_dims_m = torch.tensor(dims, device=self.device, dtype=torch.float32)
            self.multisize_side_lengths = self.multiobject_dims_m[:, 0].contiguous()
            self.multiobject_is_cylinder = torch.zeros(
                len(self.multisize_object_ids), device=self.device, dtype=torch.bool
            )
            self.multiobject_shape_ids = torch.zeros(len(self.multisize_object_ids), device=self.device, dtype=torch.long)
            self.multiobject_prim_names = {
                object_id: multiobject_prim_name(object_id) for object_id in self.multisize_object_ids
            }
            for object_id, expected_dims, cache_path in zip(
                self.multisize_object_ids, dims, cache_paths, strict=True
            ):
                cache = self._load_stable_cache_to_device(resolve_workspace_path(cache_path))
                cache_object_id = str(cache.get("object_id", ""))
                if cache_object_id != object_id:
                    raise ValueError(
                        f"Configured cache for {object_id!r} has object_id={cache_object_id!r}: {cache_path}"
                    )
                cache_dims_raw = cache.get("object_dims_m", None)
                if cache_dims_raw is None and isinstance(cache.get("metadata"), dict):
                    cache_dims_raw = cache["metadata"].get("object_dims_m", None)
                if cache_dims_raw is not None:
                    cache_dims = torch.as_tensor(cache_dims_raw, dtype=torch.float32)
                    expected = torch.tensor(expected_dims, device=cache_dims.device, dtype=torch.float32)
                    if tuple(cache_dims.shape) != (3,) or not bool(torch.allclose(cache_dims, expected, atol=1.0e-6)):
                        raise ValueError(
                            f"Configured cache for {object_id!r} has object_dims_m={cache_dims_raw!r}, "
                            f"expected {expected_dims!r}: {cache_path}"
                        )
                self.multisize_stable_caches[object_id] = cache
                self.multisize_num_stable_grasps[object_id] = int(cache["q"].shape[0])
        elif self.primitive_multi_shape_enabled:
            if len(self.multisize_object_ids) == 0:
                raise ValueError("primitive_multi_shape_enabled requires non-empty primitive_object_ids.")
            dims = tuple(tuple(float(v) for v in item) for item in getattr(self.cfg, "primitive_dims_m", ()))
            cache_paths = tuple(str(v) for v in getattr(self.cfg, "primitive_stable_cache_paths", ()))
            if len(dims) != len(self.multisize_object_ids) or len(cache_paths) != len(self.multisize_object_ids):
                raise ValueError(
                    "primitive_object_ids, primitive_dims_m, and primitive_stable_cache_paths must have the same length."
                )
            explicit_shape_names = tuple(
                str(v) for v in getattr(self.cfg, "primitive_object_shape_names", ())
            )
            if explicit_shape_names and len(explicit_shape_names) != len(self.multisize_object_ids):
                raise ValueError("primitive_object_shape_names must match primitive_object_ids.")
            is_cylinder_items: list[bool] = []
            shape_id_items: list[int] = []
            for object_index, (object_id, dim) in enumerate(zip(self.multisize_object_ids, dims, strict=True)):
                inferred_shape_name = validate_primitive_dims(object_id, dim)
                shape_name = explicit_shape_names[object_index] if explicit_shape_names else inferred_shape_name
                if shape_name not in self.primitive_shape_name_to_id:
                    raise ValueError(
                        f"primitive_shape_names={self.primitive_shape_names!r} does not include "
                        f"shape {shape_name!r} for object_id={object_id!r}."
                    )
                is_cylinder = shape_name == "cylinder"
                is_cylinder_items.append(is_cylinder)
                shape_id_items.append(self.primitive_shape_name_to_id[shape_name])
            self.multiobject_dims_m = torch.tensor(dims, device=self.device, dtype=torch.float32)
            self.multisize_side_lengths = self.multiobject_dims_m[:, 0].contiguous()
            self.multiobject_is_cylinder = torch.tensor(is_cylinder_items, device=self.device, dtype=torch.bool)
            self.multiobject_shape_ids = torch.tensor(shape_id_items, device=self.device, dtype=torch.long)
            self.primitive_shape_size_ids = {
                shape_name: torch.nonzero(
                    self.multiobject_shape_ids == shape_id, as_tuple=False
                ).squeeze(-1)
                for shape_name, shape_id in self.primitive_shape_name_to_id.items()
            }
            self.multiobject_prim_names = {
                object_id: multiobject_prim_name(object_id) for object_id in self.multisize_object_ids
            }
            for object_id, expected_dims, cache_path in zip(
                self.multisize_object_ids, dims, cache_paths, strict=True
            ):
                cache = self._load_stable_cache_to_device(resolve_workspace_path(cache_path))
                cache_object_id = str(cache.get("object_id", ""))
                if cache_object_id != object_id:
                    raise ValueError(
                        f"Configured cache for {object_id!r} has object_id={cache_object_id!r}: {cache_path}"
                    )
                cache_dims_raw = cache.get("object_dims_m", None)
                if cache_dims_raw is None and isinstance(cache.get("metadata"), dict):
                    cache_dims_raw = cache["metadata"].get("object_dims_m", None)
                if cache_dims_raw is not None:
                    cache_dims = torch.as_tensor(cache_dims_raw, dtype=torch.float32)
                    expected = torch.tensor(expected_dims, device=cache_dims.device, dtype=torch.float32)
                    if tuple(cache_dims.shape) != (3,) or not bool(torch.allclose(cache_dims, expected, atol=1.0e-6)):
                        raise ValueError(
                            f"Configured cache for {object_id!r} has object_dims_m={cache_dims_raw!r}, "
                            f"expected {expected_dims!r}: {cache_path}"
                        )
                self.multisize_stable_caches[object_id] = cache
                self.multisize_num_stable_grasps[object_id] = int(cache["q"].shape[0])
        if self.multiobject_enabled:
            if self.primitive_multi_shape_enabled:
                post_cache_paths = tuple(
                    str(v) for v in getattr(self.cfg, "primitive_post_success_reset_cache_paths", ())
                )
                post_sample_probs = tuple(
                    float(v) for v in getattr(self.cfg, "primitive_post_success_reset_sample_probs", ())
                )
            elif self.multicuboid_enabled:
                post_cache_paths = tuple(
                    str(v) for v in getattr(self.cfg, "multicuboid_post_success_reset_cache_paths", ())
                )
                post_sample_probs = tuple(
                    float(v) for v in getattr(self.cfg, "multicuboid_post_success_reset_sample_probs", ())
                )
            else:
                post_cache_paths = tuple(
                    str(v) for v in getattr(self.cfg, "multisize_cube_post_success_reset_cache_paths", ())
                )
                post_sample_probs = tuple(
                    float(v) for v in getattr(self.cfg, "multisize_cube_post_success_reset_sample_probs", ())
                )
            if post_cache_paths and len(post_cache_paths) != len(self.multisize_object_ids):
                raise ValueError("post-success reset cache paths must match the multi-object id count.")
            if post_sample_probs and len(post_sample_probs) != len(self.multisize_object_ids):
                raise ValueError("post-success reset sample probs must match the multi-object id count.")
            if not post_cache_paths:
                post_cache_paths = ("",) * len(self.multisize_object_ids)
            if not post_sample_probs:
                default_prob = self.post_success_reset_sample_prob if post_success_path_cfg else 0.0
                post_sample_probs = (default_prob,) * len(self.multisize_object_ids)

            post_probs = []
            expected_dims_by_object = self.multiobject_dims_m.detach().cpu().tolist()
            for object_id, expected_dims, cache_path, sample_prob in zip(
                self.multisize_object_ids,
                expected_dims_by_object,
                post_cache_paths,
                post_sample_probs,
                strict=True,
            ):
                prob = min(max(float(sample_prob), 0.0), 1.0)
                cache_path = str(cache_path or "")
                if prob > 0.0:
                    if not cache_path:
                        raise ValueError(
                            f"post-success reset sample prob for {object_id!r} is {prob}, "
                            "but its cache path is empty."
                        )
                    cache = self._load_stable_cache_to_device(resolve_workspace_path(cache_path))
                    self._validate_multiobject_cache_metadata(cache, object_id, expected_dims, cache_path)
                    num_grasps = int(cache["q"].shape[0])
                    if num_grasps <= 0:
                        raise ValueError(f"Post-success reset cache for {object_id!r} is empty: {cache_path}")
                    self.multisize_post_success_stable_caches[object_id] = cache
                    self.multisize_num_post_success_stable_grasps[object_id] = num_grasps
                post_probs.append(prob)
            self.multisize_post_success_reset_sample_probs = torch.tensor(
                post_probs,
                device=self.device,
                dtype=torch.float32,
            )
        if self.primitive_geometry_latent_obs_enabled and self.primitive_geometry_latent_source == "lookup":
            self.multiobject_geometry_latents = self._load_multiobject_geometry_latents(
                self.multisize_object_ids,
                self.multiobject_dims_m,
                resolve_workspace_path(str(getattr(self.cfg, "primitive_geometry_latent_path", ""))),
                self.primitive_geometry_latent_dim,
            )
        if self.primitive_geometry_latent_obs_enabled and self.primitive_geometry_latent_source == "hand_surface_delta":
            self.multiobject_surface_points_obj = self._load_multiobject_surface_points(
                self.multisize_object_ids,
                self.multiobject_dims_m,
                resolve_workspace_path(str(getattr(self.cfg, "primitive_hand_surface_pointcloud_path", ""))),
            )
        if bool(getattr(self.cfg, "object_mesh_mass_properties_enabled", False)) or bool(
            getattr(self.cfg, "grasp_wrench_mesh_projection_enabled", False)
        ):
            self._load_multiobject_mesh_data(
                self.multisize_object_ids,
                self.multiobject_dims_m,
                resolve_workspace_path(str(getattr(self.cfg, "object_mesh_data_path", ""))),
            )
        self.pose6d_anchor_success_threshold_by_size = torch.empty(0, device=self.device, dtype=torch.float32)
        self.pose6d_anchor_swing_success_threshold_by_size = torch.empty(0, device=self.device, dtype=torch.float32)
        if self.multiobject_enabled:
            anchor_ids = tuple(str(v) for v in getattr(self.cfg, "pose6d_anchor_per_object_ids", ()))
            anchor_success_values = tuple(
                float(v) for v in getattr(self.cfg, "pose6d_anchor_per_object_success_thresholds_m", ())
            )
            swing_values = tuple(
                float(v) for v in getattr(self.cfg, "pose6d_anchor_per_object_swing_thresholds_rad", ())
            )
            if anchor_ids:
                if anchor_success_values and len(anchor_success_values) != len(anchor_ids):
                    raise ValueError(
                        "pose6d_anchor_per_object_success_thresholds_m must match "
                        "pose6d_anchor_per_object_ids length."
                    )
                if swing_values and len(swing_values) != len(anchor_ids):
                    raise ValueError(
                        "pose6d_anchor_per_object_swing_thresholds_rad must match "
                        "pose6d_anchor_per_object_ids length."
                    )
                unknown_ids = tuple(object_id for object_id in anchor_ids if object_id not in self.multisize_object_ids)
                if unknown_ids:
                    raise ValueError(
                        f"pose6d_anchor_per_object_ids contains unknown primitive ids: {unknown_ids!r}."
                    )
            id_to_success = dict(zip(anchor_ids, anchor_success_values, strict=False))
            id_to_swing = dict(zip(anchor_ids, swing_values, strict=False))
            default_anchor_threshold = float(getattr(self.cfg, "pose6d_anchor_success_threshold_m", 0.010))
            default_swing_threshold = float(getattr(self.cfg, "twist_swing_success_threshold_rad", 0.174533))
            self.pose6d_anchor_success_threshold_by_size = torch.tensor(
                [id_to_success.get(object_id, default_anchor_threshold) for object_id in self.multisize_object_ids],
                device=self.device,
                dtype=torch.float32,
            )
            self.pose6d_anchor_swing_success_threshold_by_size = torch.tensor(
                [id_to_swing.get(object_id, default_swing_threshold) for object_id in self.multisize_object_ids],
                device=self.device,
                dtype=torch.float32,
            )
        self.target_rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed or 0) + 101)
        self.reset_source = self._resolve_reset_source()
        if (self.scale_conditioned_stable_cache_enabled or self.geometry_conditioned_stable_cache_enabled) and self.reset_source != "stable_cache":
            cache_kind = "Scale" if self.scale_conditioned_stable_cache_enabled else "Geometry"
            raise ValueError(
                f"{cache_kind}-conditioned stable-cache manifests require reset_source='stable_cache'; "
                f"got {self.reset_source!r}."
            )
        pass
        pass
        pass

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.qtarget_history_len = max(1, int(self.cfg.qtarget_history_len))
        self.qtarget_history_enabled = qtarget_history_enabled
        self.fingertip_full_state_obs_enabled = fingertip_full_state_obs_enabled
        self.force_obs_enabled = force_obs_enabled
        self.all_body_contact_force_obs_enabled = all_body_contact_force_obs_enabled
        self.dexndm_guided_rotation_obs_enabled = dexndm_guided_rotation_obs_enabled
        self.object_velocity_obs_source = str(self.cfg.object_velocity_obs_source).lower()
        if self.object_velocity_obs_source not in {"physx", "pose_finite_difference", "none"}:
            raise ValueError(
                "object_velocity_obs_source must be one of "
                f"'physx', 'pose_finite_difference', or 'none'; got {self.cfg.object_velocity_obs_source!r}."
            )
        self.dexndm_torque_control = bool(getattr(self.cfg, "dexndm_torque_control", False))
        self.dexndm_torque_control_gain_source = str(
            getattr(self.cfg, "dexndm_torque_control_gain_source", "scalar")
        ).lower()
        self.dexndm_torque_control_velocity_source = str(
            getattr(self.cfg, "dexndm_torque_control_velocity_source", "finite_difference")
        ).lower()
        self.dexndm_controller_pgain = float(getattr(self.cfg, "dexndm_controller_pgain", 3.0))
        self.dexndm_controller_dgain = float(getattr(self.cfg, "dexndm_controller_dgain", 0.1))
        self.dexndm_torque_limit = float(getattr(self.cfg, "dexndm_torque_limit", 0.5))
        self.dexndm_contact_gated_excess_load_penalty_enabled = bool(
            getattr(self.cfg, "dexndm_contact_gated_excess_load_penalty_enabled", False)
        )
        self.dexndm_contact_gated_excess_load_penalty_scale = float(
            getattr(self.cfg, "dexndm_contact_gated_excess_load_penalty_scale", 0.0)
        )
        self.dexndm_contact_gated_excess_load_reference_lead_rad = float(
            getattr(self.cfg, "dexndm_contact_gated_excess_load_reference_lead_rad", math.radians(5.0))
        )
        self.dexndm_contact_gated_excess_load_contact_threshold_n = float(
            getattr(self.cfg, "dexndm_contact_gated_excess_load_contact_threshold_n", 0.05)
        )
        self.dexndm_contact_gated_excess_load_normalized_clip = float(
            getattr(self.cfg, "dexndm_contact_gated_excess_load_normalized_clip", 3.0)
        )
        # The old deficit-penalty keys remain accepted for external configs,
        # but the maintained/base contract exposes the positive-reward names.
        self.dexndm_grasp_wrench_maintenance_reward_enabled = bool(
            getattr(
                self.cfg,
                "dexndm_grasp_wrench_deficit_penalty_enabled",
                getattr(self.cfg, "dexndm_grasp_wrench_maintenance_reward_enabled", False),
            )
        )
        self.dexndm_grasp_wrench_maintenance_reward_scale = float(
            getattr(
                self.cfg,
                "dexndm_grasp_wrench_deficit_penalty_scale",
                getattr(self.cfg, "dexndm_grasp_wrench_maintenance_reward_scale", 0.0),
            )
        )
        self.dexndm_grasp_wrench_contact_on_n = float(
            getattr(self.cfg, "dexndm_grasp_wrench_contact_on_n", 0.05)
        )
        self.dexndm_grasp_wrench_contact_sat_n = float(
            getattr(self.cfg, "dexndm_grasp_wrench_contact_sat_n", 0.15)
        )
        self.dexndm_grasp_wrench_elastomer_friction = float(
            getattr(self.cfg, "dexndm_grasp_wrench_elastomer_friction", 0.25)
        )
        self.dexndm_grasp_wrench_dp_friction = float(
            getattr(self.cfg, "dexndm_grasp_wrench_dp_friction", 0.05)
        )
        self.dexndm_grasp_wrench_body_mode = str(
            getattr(self.cfg, "dexndm_grasp_wrench_body_mode", "primary_10")
        ).lower()
        self.dexndm_grasp_wrench_secondary_friction = float(
            getattr(self.cfg, "dexndm_grasp_wrench_secondary_friction", 0.05)
        )
        self.dexndm_grasp_wrench_secondary_confidence_cap = float(
            getattr(self.cfg, "dexndm_grasp_wrench_secondary_confidence_cap", 0.25)
        )
        self.dexndm_grasp_wrench_secondary_group_confidence_cap = float(
            getattr(self.cfg, "dexndm_grasp_wrench_secondary_group_confidence_cap", 0.25)
        )
        self.dexndm_grasp_wrench_spatial_dedup_distance_m = float(
            getattr(self.cfg, "dexndm_grasp_wrench_spatial_dedup_distance_m", 0.003)
        )
        self.dexndm_grasp_wrench_reset_warmup_steps = int(
            getattr(self.cfg, "dexndm_grasp_wrench_reset_warmup_steps", 4)
        )
        self.dexndm_grasp_wrench_quality_floor = float(
            getattr(self.cfg, "dexndm_grasp_wrench_quality_floor", 0.08)
        )
        self.dexndm_grasp_wrench_quality_cap = float(
            getattr(self.cfg, "dexndm_grasp_wrench_quality_cap", 0.35)
        )
        self.dexndm_grasp_wrench_force_margin_erosion = float(
            getattr(self.cfg, "dexndm_grasp_wrench_force_margin_erosion", 0.0)
        )
        self.dexndm_grasp_wrench_torque_margin_erosion = float(
            getattr(self.cfg, "dexndm_grasp_wrench_torque_margin_erosion", 0.0)
        )
        self.dexndm_grasp_wrench_softmin_power = float(
            getattr(self.cfg, "dexndm_grasp_wrench_softmin_power", 8.0)
        )
        if self.dexndm_contact_gated_excess_load_penalty_scale < 0.0:
            raise ValueError("dexndm_contact_gated_excess_load_penalty_scale must be non-negative.")
        if self.dexndm_contact_gated_excess_load_reference_lead_rad <= 0.0:
            raise ValueError("dexndm_contact_gated_excess_load_reference_lead_rad must be positive.")
        if self.dexndm_contact_gated_excess_load_contact_threshold_n < 0.0:
            raise ValueError("dexndm_contact_gated_excess_load_contact_threshold_n must be non-negative.")
        if self.dexndm_contact_gated_excess_load_normalized_clip <= 0.0:
            raise ValueError("dexndm_contact_gated_excess_load_normalized_clip must be positive.")
        if self.dexndm_grasp_wrench_maintenance_reward_scale < 0.0:
            raise ValueError("dexndm_grasp_wrench_maintenance_reward_scale must be non-negative.")
        if not (
            0.0 <= self.dexndm_grasp_wrench_contact_on_n
            < self.dexndm_grasp_wrench_contact_sat_n
        ):
            raise ValueError(
                "Grasp-wrench contact thresholds must satisfy 0 <= contact_on_n < contact_sat_n."
            )
        if (
            self.dexndm_grasp_wrench_elastomer_friction < 0.0
            or self.dexndm_grasp_wrench_dp_friction < 0.0
            or self.dexndm_grasp_wrench_secondary_friction < 0.0
        ):
            raise ValueError("Grasp-wrench friction coefficients must be non-negative.")
        if self.dexndm_grasp_wrench_body_mode not in {
            "primary_10",
            "all_collision_capped_secondary",
        }:
            raise ValueError(
                "dexndm_grasp_wrench_body_mode must be 'primary_10' or "
                "'all_collision_capped_secondary'."
            )
        if self.dexndm_grasp_wrench_body_mode != "primary_10" and self.hand_spec.hand_id != "sharpa_wave_right":
            raise ValueError("All-collision CSGWCD is defined only for the pinned Sharpa Wave HB1 asset.")
        if not 0.0 <= self.dexndm_grasp_wrench_secondary_confidence_cap <= 1.0:
            raise ValueError("dexndm_grasp_wrench_secondary_confidence_cap must be within [0, 1].")
        if not 0.0 <= self.dexndm_grasp_wrench_secondary_group_confidence_cap <= 1.0:
            raise ValueError(
                "dexndm_grasp_wrench_secondary_group_confidence_cap must be within [0, 1]."
            )
        if self.dexndm_grasp_wrench_spatial_dedup_distance_m < 0.0:
            raise ValueError("dexndm_grasp_wrench_spatial_dedup_distance_m must be non-negative.")
        if self.dexndm_grasp_wrench_reset_warmup_steps < 0:
            raise ValueError("dexndm_grasp_wrench_reset_warmup_steps must be non-negative.")
        if not (
            0.0 <= self.dexndm_grasp_wrench_quality_floor
            <= self.dexndm_grasp_wrench_quality_cap
            <= 1.0
        ):
            raise ValueError("Grasp-wrench quality bounds must satisfy 0 <= floor <= cap <= 1.")
        if (
            self.dexndm_grasp_wrench_force_margin_erosion < 0.0
            or self.dexndm_grasp_wrench_torque_margin_erosion < 0.0
        ):
            raise ValueError("Grasp-wrench margin erosion values must be non-negative.")
        if self.dexndm_grasp_wrench_softmin_power <= 0.0:
            raise ValueError("dexndm_grasp_wrench_softmin_power must be positive.")
        self.num_joints = len(self.hand.joint_names)
        self.action_joint_ids = list(self.mapping.action_joint_indices)
        self.fingertip_body_ids = list(self.mapping.fingertip_body_indices)
        if self.primitive_geometry_latent_source == "hand_surface_delta":
            self.hand_surface_delta_body_names = self.hand_spec.surface_delta_bodies
            self.hand_surface_delta_body_ids = self._resolve_body_indices(self.hand_surface_delta_body_names)
        else:
            self.hand_surface_delta_body_names = ()
            self.hand_surface_delta_body_ids = []
        self.contact_sensor_body_names = self._contact_sensor_body_names_from_cfg()
        self.contact_sensor_body_name_to_idx = {name: idx for idx, name in enumerate(self.contact_sensor_body_names)}
        self.contact_sensor_obs_enabled = False
        self.contact_sensor_required = bool(
            self.contact_sensor_obs_enabled or self.force_obs_enabled or self.all_body_contact_force_obs_enabled or True
        )
        self.all_body_contact_obs_count = len(self.hand_spec.all_contact_bodies) + 1
        if self.contact_sensor_required and self.cube_contact_sensor is None:
            raise RuntimeError("Cube contact sensor was not created.")

        sharpa_joint_limit_profile = str(getattr(self.cfg, "sharpa_joint_limit_profile", "usd")).strip().lower()
        if bool(getattr(self.cfg, "sharpa_deploy_dof_limits_enabled", False)):
            # Backward-compatible interpretation for the pre-profile boolean.
            if sharpa_joint_limit_profile == "usd":
                sharpa_joint_limit_profile = "deployment"
        valid_sharpa_limit_profiles = {
            "usd",
            "usd_scaled",
            "deployment",
            "deployment_intersect_usd_scaled",
        }
        if sharpa_joint_limit_profile not in valid_sharpa_limit_profiles:
            raise ValueError(
                f"sharpa_joint_limit_profile must be one of {sorted(valid_sharpa_limit_profiles)}, "
                f"got {sharpa_joint_limit_profile!r}."
            )
        if sharpa_joint_limit_profile != "usd":
            if self.hand_spec.hand_id != "sharpa_wave_right":
                raise ValueError("Sharpa joint-limit profiles require hand_type='sharpa_wave_right'.")
            usd_limits_sim = self.hand.data.soft_joint_pos_limits[0].to(device=self.device, dtype=torch.float32)
            usd_limits_action = usd_limits_sim[self.action_joint_ids].clone()
            usd_endpoint_scale = float(getattr(self.cfg, "sharpa_usd_joint_limit_endpoint_scale", 0.9))
            if sharpa_joint_limit_profile == "usd_scaled":
                # Pure signed USDA endpoint scaling.  Deliberately do not
                # intersect with the separate deployment-limit table.
                applied_limits_action = scaled_signed_usd_limits(
                    usd_limits_action,
                    endpoint_scale=usd_endpoint_scale,
                )
            else:
                missing_limits = tuple(
                    name for name in self.hand_spec.joint_order if name not in SHARPA_DEPLOY_DOF_LIMITS_DEG
                )
                if missing_limits:
                    raise RuntimeError(f"Missing Sharpa deployment limits for joints: {missing_limits}.")
                if sharpa_joint_limit_profile == "deployment":
                    applied_limits_action = torch.deg2rad(
                        torch.tensor(
                            [SHARPA_DEPLOY_DOF_LIMITS_DEG[name] for name in self.hand_spec.joint_order],
                            device=self.device,
                            dtype=torch.float32,
                        )
                    )
                else:
                    applied_limits_action = composite_deploy_usd_limits(
                        usd_limits_action,
                        self.hand_spec.joint_order,
                        SHARPA_DEPLOY_DOF_LIMITS_DEG,
                        usd_endpoint_scale=usd_endpoint_scale,
                    )
            sim_limits = applied_limits_action.unsqueeze(0).expand(self.num_envs, -1, -1)
            self.hand.write_joint_position_limit_to_sim(
                sim_limits,
                joint_ids=self.action_joint_ids,
                warn_limit_violation=False,
            )
            self.sharpa_raw_usd_action_limits = usd_limits_action
            self.sharpa_applied_action_limits = applied_limits_action
        else:
            self.sharpa_raw_usd_action_limits = self.hand.data.soft_joint_pos_limits[0][self.action_joint_ids].clone()
            self.sharpa_applied_action_limits = self.sharpa_raw_usd_action_limits.clone()
        self.sharpa_joint_limit_profile = sharpa_joint_limit_profile

        limits = self.hand.data.soft_joint_pos_limits[0].to(device=self.device)
        self.joint_lower = limits[:, 0]
        self.joint_upper = limits[:, 1]
        self.action_lower = self.joint_lower[self.action_joint_ids]
        self.action_upper = self.joint_upper[self.action_joint_ids]
        self.dexndm_controller_pgain_action = torch.full(
            (self.num_envs, self.action_dim),
            self.dexndm_controller_pgain,
            device=self.device,
            dtype=torch.float32,
        )
        self.dexndm_controller_dgain_action = torch.full(
            (self.num_envs, self.action_dim),
            self.dexndm_controller_dgain,
            device=self.device,
            dtype=torch.float32,
        )
        if self.dexndm_torque_control and self.dexndm_torque_control_gain_source == "default_joint_pd":
            self.dexndm_controller_pgain_action[:] = sim_to_action_order(
                self.hand.data.default_joint_stiffness.to(device=self.device, dtype=torch.float32),
                self.mapping.action_joint_indices,
            )
            self.dexndm_controller_dgain_action[:] = sim_to_action_order(
                self.hand.data.default_joint_damping.to(device=self.device, dtype=torch.float32),
                self.mapping.action_joint_indices,
            )
            self.hand.data.default_joint_stiffness[:] = 0.0
            self.hand.data.default_joint_damping[:] = 0.0
            for actuator in self.hand.actuators.values():
                if getattr(actuator, "stiffness", None) is not None:
                    actuator.stiffness = torch.zeros_like(actuator.stiffness)
                if getattr(actuator, "damping", None) is not None:
                    actuator.damping = torch.zeros_like(actuator.damping)
        self.dexndm_controller_pgain_nominal_action = self.dexndm_controller_pgain_action.clone()
        self.dexndm_controller_dgain_nominal_action = self.dexndm_controller_dgain_action.clone()
        self.dexndm_excess_load_joint_finger_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self.dexndm_excess_load_contact_body_indices = torch.empty(
            (0, 2), device=self.device, dtype=torch.long
        )
        if self.dexndm_contact_gated_excess_load_penalty_enabled:
            finger_names = ("thumb", "index", "middle", "ring", "pinky")
            joint_finger_ids = []
            for joint_name in self.hand_spec.joint_order:
                matches = [
                    finger_id
                    for finger_id, finger_name in enumerate(finger_names)
                    if joint_name.startswith(f"right_{finger_name}_")
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Cannot map Sharpa joint {joint_name!r} to exactly one finger."
                    )
                joint_finger_ids.append(matches[0])
            contact_body_indices = []
            for finger_name in finger_names:
                body_pair = (
                    f"right_{finger_name}_elastomer",
                    f"right_{finger_name}_DP",
                )
                missing = tuple(
                    body_name
                    for body_name in body_pair
                    if body_name not in self.contact_sensor_body_name_to_idx
                )
                if missing:
                    raise RuntimeError(
                        "Contact-gated excess-load reward is missing sensor bodies: "
                        f"{missing}."
                    )
                contact_body_indices.append(
                    [self.contact_sensor_body_name_to_idx[body_name] for body_name in body_pair]
                )
            self.dexndm_excess_load_joint_finger_ids = torch.tensor(
                joint_finger_ids, device=self.device, dtype=torch.long
            )
            self.dexndm_excess_load_contact_body_indices = torch.tensor(
                contact_body_indices, device=self.device, dtype=torch.long
            )
        self.dexndm_grasp_wrench_contact_bodies = tuple(self.hand_spec.all_contact_bodies)
        if self.dexndm_grasp_wrench_body_mode == "all_collision_capped_secondary":
            self.dexndm_grasp_wrench_contact_bodies = SHARPA_CSGWCD_COLLISION_BODIES
        self.dexndm_grasp_wrench_contact_body_ids: list[int] = []
        self.dexndm_grasp_wrench_contact_friction = torch.empty(0, device=self.device)
        self.dexndm_grasp_wrench_contact_confidence_cap: torch.Tensor | None = None
        self.dexndm_grasp_wrench_contact_primary_mask: torch.Tensor | None = None
        self.dexndm_grasp_wrench_contact_group_ids: torch.Tensor | None = None
        if self.dexndm_grasp_wrench_maintenance_reward_enabled:
            self.dexndm_grasp_wrench_contact_body_ids = self._resolve_body_indices(
                self.dexndm_grasp_wrench_contact_bodies
            )
            self.dexndm_grasp_wrench_contact_friction = torch.tensor(
                [
                    self.dexndm_grasp_wrench_elastomer_friction
                    if body_name.endswith("_elastomer")
                    else (
                        self.dexndm_grasp_wrench_dp_friction
                        if body_name in SHARPA_CSGWCD_PRIMARY_BODIES
                        else self.dexndm_grasp_wrench_secondary_friction
                    )
                    for body_name in self.dexndm_grasp_wrench_contact_bodies
                ],
                device=self.device,
                dtype=torch.float32,
            )
            if self.dexndm_grasp_wrench_body_mode == "all_collision_capped_secondary":
                primary_count = len(SHARPA_CSGWCD_PRIMARY_BODIES)
                secondary_count = len(SHARPA_CSGWCD_SECONDARY_BODIES)
                self.dexndm_grasp_wrench_contact_confidence_cap = torch.tensor(
                    [1.0] * primary_count
                    + [self.dexndm_grasp_wrench_secondary_confidence_cap] * secondary_count,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.dexndm_grasp_wrench_contact_primary_mask = torch.tensor(
                    [True] * primary_count + [False] * secondary_count,
                    device=self.device,
                    dtype=torch.bool,
                )
                self.dexndm_grasp_wrench_contact_group_ids = torch.tensor(
                    SHARPA_CSGWCD_GROUP_IDS,
                    device=self.device,
                    dtype=torch.long,
                )
        self.dexndm_controller_pgain_scale = torch.ones_like(self.dexndm_controller_pgain_action)
        self.dexndm_controller_dgain_scale = torch.ones_like(self.dexndm_controller_dgain_action)
        self.dexndm_torque_pd_gain_randomization_enabled = bool(
            getattr(self.cfg, "dexndm_torque_pd_gain_randomization_enabled", False)
        )
        if self.dexndm_torque_pd_gain_randomization_enabled and not (
            self.dexndm_torque_control and self.dexndm_torque_control_gain_source == "default_joint_pd"
        ):
            raise ValueError(
                "dexndm_torque_pd_gain_randomization_enabled requires explicit torque control with "
                "dexndm_torque_control_gain_source='default_joint_pd'."
            )

        self.actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.action_targets = torch.zeros_like(self.actions)
        self.prev_action_targets = torch.zeros_like(self.actions)
        self.qtarget_history = torch.zeros((self.num_envs, self.qtarget_history_len, 2 * self.action_dim), device=self.device)
        self.qtarget_history_prev_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.dexndm_obs_history_len = 3
        self.dexndm_obs_history = torch.zeros(
            (self.num_envs, self.dexndm_obs_history_len, 2 * self.action_dim), device=self.device
        )
        self.dexndm_obs_history_prev_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.sharpa_deployable_history_len = 3
        self.sharpa_deployable_tactile_dim = (
            0
        )
        self.sharpa_deployable_frame_dim = 2 * self.action_dim + self.sharpa_deployable_tactile_dim
        self.sharpa_deployable_object_descriptor_dim = (
            self.primitive_geometry_latent_dim if self.sharpa_deployable_mesh_bps_obs_enabled else 0
        )
        self.sharpa_deployable_gravity_palm_obs_dim = 3
        self.sharpa_deployable_task_dim = (
            45 + self.sharpa_deployable_object_descriptor_dim + self.sharpa_deployable_gravity_palm_obs_dim
        )
        self.sharpa_deployable_observation_dim = (
            self.sharpa_deployable_history_len * self.sharpa_deployable_frame_dim
            + self.sharpa_deployable_task_dim
        )
        expected_sharpa_deployable_obs_dim = 192
        expected_sharpa_deployable_obs_dim -= self.sharpa_deployable_history_len * len(
            self.hand_spec.force_sensor_bodies
        )
        expected_sharpa_deployable_obs_dim += self.sharpa_deployable_gravity_palm_obs_dim
        if self.sharpa_deployable_mesh_bps_obs_enabled:
            expected_sharpa_deployable_obs_dim += self.primitive_geometry_latent_dim
        if self.sharpa_deployable_observation_dim != expected_sharpa_deployable_obs_dim:
            raise RuntimeError(
                f"Sharpa deployable observation layout must be {expected_sharpa_deployable_obs_dim}D: "
                f"history_len={self.sharpa_deployable_history_len}, "
                f"frame_dim={self.sharpa_deployable_frame_dim}, task_dim={self.sharpa_deployable_task_dim}, "
                f"obs_dim={self.sharpa_deployable_observation_dim}."
            )
        self.sharpa_deployable_obs_history = torch.zeros(
            (
                self.num_envs,
                self.sharpa_deployable_history_len,
                self.sharpa_deployable_frame_dim,
            ),
            device=self.device,
        )
        self.sharpa_deployable_obs_history_prev_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.hora_rotation_init_q = torch.zeros_like(self.actions)
        self.sim_joint_targets = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.dexndm_prev_control_q_action = torch.zeros_like(self.actions)
        self.dexndm_sim_joint_efforts = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.dexndm_commanded_torque_action = torch.zeros_like(self.actions)
        self.dexndm_raw_commanded_torque_action = torch.zeros_like(self.actions)

        self.reset_node_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_node_ids = torch.zeros_like(self.reset_node_ids)
        self.reset_grasp_indices = torch.zeros_like(self.reset_node_ids)
        self.graph_source_grasp_indices = torch.full_like(self.reset_node_ids, -1)
        self.graph_target_grasp_indices = torch.full_like(self.reset_node_ids, -1)
        self.edge_bin = torch.zeros_like(self.reset_node_ids)
        self.edge_dist = torch.zeros(self.num_envs, device=self.device)
        self.edge_translation = torch.zeros_like(self.edge_dist)
        self.edge_score = torch.zeros_like(self.edge_dist)
        self.edge_q_dist = torch.zeros_like(self.edge_dist)
        self.edge_contact_mismatch = torch.zeros_like(self.edge_dist)
        self.edge_q_limit_risk = torch.zeros_like(self.edge_dist)
        self.target_pose_palm = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_pose_palm[:, 3] = 1.0
        self.target_source_pose_palm = torch.zeros_like(self.target_pose_palm)
        self.target_source_pose_palm[:, 3] = 1.0
        self.target_keypoints_palm = torch.zeros((self.num_envs, 8, 3), device=self.device)
        self.target_twist_axis_palm = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_twist_axis_palm[:, 0] = 1.0
        self.target_translation_axis_palm = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_translation_axis_palm[:, 0] = 1.0
        self.prev_pose_error = torch.zeros(self.num_envs, device=self.device)
        self.best_pose_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.best_directional_rotation_coord = torch.zeros(self.num_envs, device=self.device)
        self.directional_rotation_coord = torch.zeros_like(self.best_directional_rotation_coord)
        self.directional_rotation_progress = torch.zeros_like(self.best_directional_rotation_coord)
        self.directional_rotation_target_angle = torch.zeros_like(self.best_directional_rotation_coord)
        self.directional_rotation_remaining = torch.zeros_like(self.best_directional_rotation_coord)
        self.directional_rotation_orthogonal_error = torch.zeros_like(self.best_directional_rotation_coord)
        self.target_twist_angle = torch.zeros_like(self.best_directional_rotation_coord)
        self.current_twist_angle = torch.zeros_like(self.best_directional_rotation_coord)
        self.twist_progress_raw = torch.zeros_like(self.best_directional_rotation_coord)
        self.twist_progress = torch.zeros_like(self.best_directional_rotation_coord)
        self.best_twist_progress = torch.zeros_like(self.best_directional_rotation_coord)
        self.twist_remaining = torch.zeros_like(self.best_directional_rotation_coord)
        self.twist_swing_drift = torch.zeros_like(self.best_directional_rotation_coord)
        self.twist_position_drift = torch.zeros_like(self.best_directional_rotation_coord)
        self.target_translation_delta = torch.zeros_like(self.best_directional_rotation_coord)
        self.translation_progress_raw = torch.zeros_like(self.best_directional_rotation_coord)
        self.translation_progress = torch.zeros_like(self.best_directional_rotation_coord)
        self.best_translation_progress = torch.zeros_like(self.best_directional_rotation_coord)
        self.translation_remaining = torch.zeros_like(self.best_directional_rotation_coord)
        self.translation_orthogonal_drift = torch.zeros_like(self.best_directional_rotation_coord)
        self.source_anchor_local_pos_m = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_anchor_local_pos_m = torch.zeros_like(self.source_anchor_local_pos_m)
        self.source_anchor_pos_palm = torch.zeros_like(self.source_anchor_local_pos_m)
        self.current_anchor_pos_palm = torch.zeros_like(self.source_anchor_local_pos_m)
        self.target_anchor_pos_palm = torch.zeros_like(self.source_anchor_local_pos_m)
        self.pose6d_anchor_error_m = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_resample_fail = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_workspace_reject = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_segment_overlap_count = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_min_centerline_z_m = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_forbidden_reject = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_dynamic_contact_count = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_anchor_dynamic_y_local_m = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_feasible_target_resample_fail = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_feasible_target_workspace_reject = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_feasible_target_forbidden_reject = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_feasible_target_proxy_hit_count = torch.zeros_like(self.best_directional_rotation_coord)
        self.pose6d_feasible_target_hit_proxy_y_m = torch.zeros_like(self.best_directional_rotation_coord)
        self.target_delta = torch.zeros_like(self.best_directional_rotation_coord)
        self.target_delta_sample_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        def curriculum_initial_max(start_name: str, end_name: str, initial_name: str, default: float) -> float:
            start = float(getattr(self.cfg, start_name, default))
            end = float(getattr(self.cfg, end_name, start))
            initial = float(getattr(self.cfg, initial_name, -1.0))
            if initial < 0.0:
                initial = start
            return min(max(initial, start), end)

        twist_initial_max = curriculum_initial_max(
            "twist_target_curriculum_start_rad",
            "twist_target_curriculum_end_rad",
            "twist_target_curriculum_initial_max_rad",
            0.0872665,
        )
        chain_twist_initial_max = curriculum_initial_max(
            "chain_post_success_target_curriculum_start_rad",
            "chain_post_success_target_curriculum_end_rad",
            "chain_post_success_target_curriculum_initial_max_rad",
            0.0872665,
        )
        if bool(getattr(self.cfg, "primary_chain_shared_rotation_curriculum_enabled", False)):
            chain_twist_initial_max = twist_initial_max
        translation_initial_max = curriculum_initial_max(
            "pose6d_translation_curriculum_start_m",
            "pose6d_translation_curriculum_end_m",
            "pose6d_translation_curriculum_initial_max_m",
            0.002,
        )
        chain_translation_initial_max = curriculum_initial_max(
            "chain_pose6d_translation_curriculum_start_m",
            "chain_pose6d_translation_curriculum_end_m",
            "chain_pose6d_translation_curriculum_initial_max_m",
            0.002,
        )
        self.target_delta_curriculum_max_at_reset = torch.full(
            (self.num_envs,), twist_initial_max, device=self.device
        )
        self.target_chain_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_from_post_success_cache = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        adaptive_tail_batches = max(int(getattr(self.cfg, "twist_target_adaptive_tail_batches", 50)), 1)
        self.adaptive_target_delta_max = torch.tensor(twist_initial_max, device=self.device)
        self.adaptive_frontier_success_history = torch.zeros(adaptive_tail_batches, device=self.device)
        self.adaptive_frontier_count_history = torch.zeros(adaptive_tail_batches, device=self.device)
        self.adaptive_frontier_success_history_count = 0
        self.adaptive_frontier_success_history_index = 0
        self.adaptive_frontier_success_rate = torch.zeros((), device=self.device)
        self.adaptive_frontier_done_count = torch.zeros((), device=self.device)
        self.adaptive_frontier_done_total = torch.zeros((), device=self.device)
        self.adaptive_curriculum_advances = torch.zeros((), device=self.device)
        self.adaptive_twist_chain_sync_limited_count = torch.zeros((), device=self.device)
        self.adaptive_chain_target_delta_max = torch.tensor(chain_twist_initial_max, device=self.device)
        self.adaptive_chain_frontier_success_history = torch.zeros(adaptive_tail_batches, device=self.device)
        self.adaptive_chain_frontier_count_history = torch.zeros(adaptive_tail_batches, device=self.device)
        self.adaptive_chain_frontier_success_history_count = 0
        self.adaptive_chain_frontier_success_history_index = 0
        self.adaptive_chain_frontier_success_rate = torch.zeros((), device=self.device)
        self.adaptive_chain_frontier_done_count = torch.zeros((), device=self.device)
        self.adaptive_chain_frontier_done_total = torch.zeros((), device=self.device)
        self.adaptive_chain_curriculum_advances = torch.zeros((), device=self.device)
        self.translation_delta_sample_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.translation_delta_curriculum_max_at_reset = torch.full(
            (self.num_envs,), translation_initial_max, device=self.device
        )
        translation_tail_batches = max(int(getattr(self.cfg, "pose6d_translation_adaptive_tail_batches", 50)), 1)
        self.adaptive_translation_delta_max = torch.tensor(translation_initial_max, device=self.device)
        self.adaptive_translation_frontier_success_history = torch.zeros(translation_tail_batches, device=self.device)
        self.adaptive_translation_frontier_count_history = torch.zeros(translation_tail_batches, device=self.device)
        self.adaptive_translation_frontier_success_history_count = 0
        self.adaptive_translation_frontier_success_history_index = 0
        self.adaptive_translation_frontier_success_rate = torch.zeros((), device=self.device)
        self.adaptive_translation_frontier_done_count = torch.zeros((), device=self.device)
        self.adaptive_translation_frontier_done_total = torch.zeros((), device=self.device)
        self.adaptive_translation_curriculum_advances = torch.zeros((), device=self.device)
        self.adaptive_translation_chain_sync_limited_count = torch.zeros((), device=self.device)
        self.adaptive_chain_translation_delta_max = torch.tensor(chain_translation_initial_max, device=self.device)
        self.adaptive_chain_translation_frontier_success_history = torch.zeros(translation_tail_batches, device=self.device)
        self.adaptive_chain_translation_frontier_count_history = torch.zeros(translation_tail_batches, device=self.device)
        self.adaptive_chain_translation_frontier_success_history_count = 0
        self.adaptive_chain_translation_frontier_success_history_index = 0
        self.adaptive_chain_translation_frontier_success_rate = torch.zeros((), device=self.device)
        self.adaptive_chain_translation_frontier_done_count = torch.zeros((), device=self.device)
        self.adaptive_chain_translation_frontier_done_total = torch.zeros((), device=self.device)
        self.adaptive_chain_translation_curriculum_advances = torch.zeros((), device=self.device)
        self.adaptive_soft_promotion_counts = torch.zeros(3, device=self.device)
        self.adaptive_soft_promotion_last_global_step = 0
        self.adaptive_soft_promotion_last_level_change_step = {
            "twist": 0,
            "chain_twist": 0,
            "translation": 0,
        }
        requested_per_object_curriculum = bool(
            getattr(self.cfg, "multiobject_per_object_curriculum_enabled", False)
        )
        if requested_per_object_curriculum and not self.multiobject_enabled:
            raise ValueError(
                "multiobject_per_object_curriculum_enabled requires a multi-object task."
            )
        curriculum_object_count = len(self.multisize_object_ids) if self.multiobject_enabled else 0
        self.multiobject_per_object_curriculum_enabled = (
            requested_per_object_curriculum and curriculum_object_count > 0
        )
        self.adaptive_soft_promotion_counts_by_object = torch.zeros(
            (curriculum_object_count, 3), device=self.device
        )
        self.adaptive_soft_promotion_last_global_step_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_soft_promotion_last_level_change_step_by_object = {
            "twist": torch.zeros(curriculum_object_count, device=self.device, dtype=torch.long),
            "chain_twist": torch.zeros(curriculum_object_count, device=self.device, dtype=torch.long),
            "translation": torch.zeros(curriculum_object_count, device=self.device, dtype=torch.long),
        }
        self.adaptive_target_delta_max_by_object = torch.full(
            (curriculum_object_count,), twist_initial_max, device=self.device
        )
        self.adaptive_frontier_success_history_by_object = torch.zeros(
            (curriculum_object_count, adaptive_tail_batches), device=self.device
        )
        self.adaptive_frontier_count_history_by_object = torch.zeros(
            (curriculum_object_count, adaptive_tail_batches), device=self.device
        )
        self.adaptive_frontier_success_history_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_frontier_success_history_index_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_frontier_success_rate_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_frontier_done_count_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_frontier_done_total_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_curriculum_advances_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_twist_chain_sync_limited_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_chain_target_delta_max_by_object = torch.full(
            (curriculum_object_count,), chain_twist_initial_max, device=self.device
        )
        self.adaptive_chain_frontier_success_history_by_object = torch.zeros(
            (curriculum_object_count, adaptive_tail_batches), device=self.device
        )
        self.adaptive_chain_frontier_count_history_by_object = torch.zeros(
            (curriculum_object_count, adaptive_tail_batches), device=self.device
        )
        self.adaptive_chain_frontier_success_history_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_chain_frontier_success_history_index_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_chain_frontier_success_rate_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_chain_frontier_done_count_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_chain_frontier_done_total_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_chain_curriculum_advances_by_object = torch.zeros(curriculum_object_count, device=self.device)
        self.adaptive_translation_delta_max_by_object = torch.full(
            (curriculum_object_count,), translation_initial_max, device=self.device
        )
        self.adaptive_translation_frontier_success_history_by_object = torch.zeros(
            (curriculum_object_count, translation_tail_batches), device=self.device
        )
        self.adaptive_translation_frontier_count_history_by_object = torch.zeros(
            (curriculum_object_count, translation_tail_batches), device=self.device
        )
        self.adaptive_translation_frontier_success_history_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_translation_frontier_success_history_index_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_translation_frontier_success_rate_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_translation_frontier_done_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_translation_frontier_done_total_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_translation_curriculum_advances_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_translation_chain_sync_limited_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_chain_translation_delta_max_by_object = torch.full(
            (curriculum_object_count,), chain_translation_initial_max, device=self.device
        )
        self.adaptive_chain_translation_frontier_success_history_by_object = torch.zeros(
            (curriculum_object_count, translation_tail_batches), device=self.device
        )
        self.adaptive_chain_translation_frontier_count_history_by_object = torch.zeros(
            (curriculum_object_count, translation_tail_batches), device=self.device
        )
        self.adaptive_chain_translation_frontier_success_history_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_chain_translation_frontier_success_history_index_by_object = torch.zeros(
            curriculum_object_count, device=self.device, dtype=torch.long
        )
        self.adaptive_chain_translation_frontier_success_rate_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_chain_translation_frontier_done_count_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_chain_translation_frontier_done_total_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )
        self.adaptive_chain_translation_curriculum_advances_by_object = torch.zeros(
            curriculum_object_count, device=self.device
        )

        def apply_per_object_curriculum_initial_max(
            config_name: str,
            target: torch.Tensor,
            minimum: float,
            maximum: float,
        ) -> None:
            values = tuple(float(value) for value in getattr(self.cfg, config_name, ()))
            if not values:
                return
            if len(values) != curriculum_object_count:
                raise ValueError(f"{config_name} must contain {curriculum_object_count} values.")
            target.copy_(torch.tensor(values, device=self.device).clamp(min=minimum, max=maximum))

        if self.multiobject_per_object_curriculum_enabled:
            apply_per_object_curriculum_initial_max(
                "multiobject_twist_curriculum_initial_max_rad_by_object",
                self.adaptive_target_delta_max_by_object,
                float(getattr(self.cfg, "twist_target_curriculum_start_rad", twist_initial_max)),
                float(getattr(self.cfg, "twist_target_curriculum_end_rad", twist_initial_max)),
            )
            apply_per_object_curriculum_initial_max(
                "multiobject_chain_twist_curriculum_initial_max_rad_by_object",
                self.adaptive_chain_target_delta_max_by_object,
                float(getattr(self.cfg, "chain_post_success_target_curriculum_start_rad", chain_twist_initial_max)),
                float(getattr(self.cfg, "chain_post_success_target_curriculum_end_rad", chain_twist_initial_max)),
            )
            apply_per_object_curriculum_initial_max(
                "multiobject_translation_curriculum_initial_max_m_by_object",
                self.adaptive_translation_delta_max_by_object,
                float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", translation_initial_max)),
                float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", translation_initial_max)),
            )
        self._sync_shared_rotation_curriculum_frontier_aliases()
        if self.multiobject_per_object_curriculum_enabled:
            self._sync_multiobject_curriculum_summaries()

        self.episode_success_rate = torch.zeros((), device=self.device)
        self.hora_prev_cube_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.hora_prev_cube_quat_w[:, 0] = 1.0
        self.object_velocity_obs = torch.zeros((self.num_envs, 6), device=self.device)
        self.object_velocity_obs_prev_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_velocity_obs_prev_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_velocity_obs_prev_quat_w[:, 0] = 1.0
        self.object_velocity_obs_prev_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.sharpa_deployable_object_velocity_est_palm = torch.zeros((self.num_envs, 6), device=self.device)
        self.sharpa_deployable_prev_object_pos_palm = torch.zeros((self.num_envs, 3), device=self.device)
        self.sharpa_deployable_prev_object_quat_palm = torch.zeros((self.num_envs, 4), device=self.device)
        self.sharpa_deployable_prev_object_quat_palm[:, 0] = 1.0
        self.sharpa_deployable_velocity_prev_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.sharpa_deployable_object_pose_obs_queue_len = (
            self.sharpa_deployable_object_pose_obs_delay_max_steps + 1
        )
        self.sharpa_deployable_object_pose_obs_queue = torch.zeros(
            (self.num_envs, self.sharpa_deployable_object_pose_obs_queue_len, 7),
            device=self.device,
            dtype=torch.float32,
        )
        self.sharpa_deployable_object_pose_obs_queue[:, :, 3] = 1.0
        self.sharpa_deployable_object_pose_obs_delay_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.sharpa_deployable_object_pose_obs_prev_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.sharpa_deployable_object_pose_obs_palm = torch.zeros((self.num_envs, 7), device=self.device)
        self.sharpa_deployable_object_pose_obs_palm[:, 3] = 1.0
        self.sharpa_deployable_object_pose_obs_last_valid_palm = self.sharpa_deployable_object_pose_obs_palm.clone()
        self.sharpa_deployable_object_pose_obs_confidence = torch.full(
            (self.num_envs, 1),
            float(getattr(self.cfg, "sharpa_deployable_pose_tracker_confidence", 1.0)),
            device=self.device,
            dtype=torch.float32,
        )
        self.sharpa_deployable_object_pose_obs_dropout = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.sharpa_deployable_object_pose_obs_pos_noise_norm_m = torch.zeros_like(
            self.sharpa_deployable_object_pose_obs_dropout
        )
        self.sharpa_deployable_object_pose_obs_rot_noise_abs_deg = torch.zeros_like(
            self.sharpa_deployable_object_pose_obs_dropout
        )
        self.hora_axis_angvel = torch.zeros(self.num_envs, device=self.device)
        self.hora_orthogonal_angvel = torch.zeros_like(self.hora_axis_angvel)
        self.hora_object_linvel = torch.zeros_like(self.hora_axis_angvel)
        self.hora_cumulative_axis_rotation = torch.zeros_like(self.hora_axis_angvel)
        self.hora_joint_pose_deviation = torch.zeros_like(self.hora_axis_angvel)
        self.dexndm_prev_cube_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.dexndm_init_cube_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.dexndm_prev_cube_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.dexndm_prev_cube_quat_w[:, 0] = 1.0
        self.dexndm_prev_q_action = torch.zeros_like(self.actions)
        self.dexndm_dof_vel_finite_diff = torch.zeros_like(self.actions)
        self.dexndm_rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device)
        self.dexndm_rot_axis_buf[:, 2] = 1.0
        self.dexndm_wrist_rnd_rot_buf = torch.zeros((self.num_envs, 4), device=self.device)
        self.dexndm_wrist_rnd_rot_buf[:, 0] = 1.0
        self.hand_root_rot_buf = torch.tensor(self.hand_root_rot, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self.active_cube_size_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.active_cube_side_length_m = torch.full(
            (self.num_envs,),
            self.object_keypoint_side_length_m,
            device=self.device,
            dtype=torch.float32,
        )
        self.active_cube_size_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.active_object_dims_m = torch.full(
            (self.num_envs, 3),
            self.object_keypoint_side_length_m,
            device=self.device,
            dtype=torch.float32,
        )
        self.active_object_dim_norm = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.active_object_is_cylinder = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.active_object_shape_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.active_object_shape_onehot = torch.zeros(
            (self.num_envs, len(self.primitive_shape_names)),
            device=self.device,
            dtype=torch.float32,
        )
        self.active_object_geometry_latent = torch.zeros(
            (self.num_envs, max(self.primitive_geometry_latent_dim, 0)),
            device=self.device,
            dtype=torch.float32,
        )
        self._initialize_object_domain_randomization_buffers()
        self.dexndm_torque_penalty = torch.zeros_like(self.hora_axis_angvel)
        self.dexndm_work_penalty = torch.zeros_like(self.hora_axis_angvel)
        self.grasp_wrench_quality = torch.zeros(self.num_envs, device=self.device)
        self.grasp_wrench_force_margin_min = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_torque_margin_min = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_directional_margins = torch.zeros((self.num_envs, 6), device=self.device)
        self.grasp_wrench_contact_confidence = torch.zeros(
            (self.num_envs, len(self.dexndm_grasp_wrench_contact_bodies)), device=self.device
        )
        self.grasp_wrench_target_quality = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_deficit = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_warmup_quality_sum = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_warmup_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Reward-time snapshots are intentionally not reset in _reset_idx. DirectRLEnv resets
        # terminated environments before returning from step(), so diagnostics tools need these
        # copies to retain the exact pre-reset sample associated with reward/done.
        self.grasp_wrench_reward_quality = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_force_margin_min = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_torque_margin_min = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_directional_margins = torch.zeros_like(
            self.grasp_wrench_directional_margins
        )
        self.grasp_wrench_reward_contact_confidence = torch.zeros_like(
            self.grasp_wrench_contact_confidence
        )
        self.grasp_wrench_reward_target_quality = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_deficit = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_maintenance_reward = torch.zeros_like(
            self.grasp_wrench_quality
        )
        self.grasp_wrench_reward_warmup_count = torch.zeros_like(self.grasp_wrench_warmup_count)
        self.grasp_wrench_reward_pose_error = torch.zeros_like(self.grasp_wrench_quality)
        self.grasp_wrench_reward_drop = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.grasp_wrench_reward_success = torch.zeros_like(
            self.grasp_wrench_reward_drop
        )
        self.grasp_wrench_reward_reset_grasp_index = torch.zeros_like(self.reset_grasp_indices)
        self.grasp_wrench_reward_q_cmd_action_rad = torch.zeros_like(self.actions)
        self.grasp_wrench_reward_q_action_rad = torch.zeros_like(self.actions)
        self.grasp_wrench_reward_qdot_action_rad_s = torch.zeros_like(self.actions)
        self.grasp_wrench_reward_raw_commanded_torque_action_nm = torch.zeros_like(
            self.actions
        )
        self.grasp_wrench_reward_commanded_torque_action_nm = torch.zeros_like(
            self.actions
        )
        self.grasp_wrench_reward_applied_torque_action_nm = torch.zeros_like(
            self.actions
        )
        self.grasp_wrench_reward_contact_force_w = torch.zeros(
            (
                self.num_envs,
                len(self.dexndm_grasp_wrench_contact_bodies),
                3,
            ),
            device=self.device,
        )
        self.grasp_wrench_reward_object_external_force_w = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.grasp_wrench_reward_object_external_torque_w = torch.zeros_like(
            self.grasp_wrench_reward_object_external_force_w
        )
        self.grasp_wrench_reward_current_pose_palm = torch.zeros(
            (self.num_envs, 7), device=self.device
        )
        self.grasp_wrench_reward_current_pose_palm[:, 3] = 1.0
        self.grasp_wrench_reward_target_pose_palm = self.grasp_wrench_reward_current_pose_palm.clone()
        self.dexndm_rotp = torch.zeros_like(self.hora_axis_angvel)
        self.dexndm_rotp_step = 0
        self.dexndm_cur_rotp_coef = 0.0
        self.near_success_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.achieved_goals = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_refresh_fallback_level = torch.zeros_like(self.reset_node_ids)
        self.target_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_new_goal = torch.zeros_like(self.last_success)
        self.last_drop = torch.zeros_like(self.last_success)
        self.last_pose_error = torch.zeros(self.num_envs, device=self.device)
        self.last_best_pose_error = torch.zeros_like(self.last_pose_error)
        self.local_random_target_dist = torch.zeros_like(self.edge_dist)
        self.local_random_target_translation = torch.zeros_like(self.edge_dist)
        self.local_random_target_rotation = torch.zeros_like(self.edge_dist)
        self.local_random_resample_fail = torch.zeros_like(self.edge_dist)
        self.local_random_workspace_reject = torch.zeros_like(self.edge_dist)
        self.mixed_target_is_scripted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.scripted_rotation_base_pose_palm = torch.zeros((self.num_envs, 7), device=self.device)
        self.scripted_rotation_base_pose_palm[:, 3] = 1.0
        self.scripted_rotation_target_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.scripted_rotation_direction = torch.ones(self.num_envs, device=self.device)
        self.scripted_rotation_step_deg = torch.zeros_like(self.edge_dist)
        self.dexndm_guided_rotation_base_pose_palm = torch.zeros((self.num_envs, 7), device=self.device)
        self.dexndm_guided_rotation_base_pose_palm[:, 3] = 1.0
        self.dexndm_guided_rotation_guide_pose_palm = torch.zeros_like(self.dexndm_guided_rotation_base_pose_palm)
        self.dexndm_guided_rotation_guide_pose_palm[:, 3] = 1.0
        self.dexndm_guided_rotation_index = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.dexndm_guided_rotation_advances = torch.zeros_like(self.dexndm_guided_rotation_index)
        self.dexndm_guided_rotation_rot_dist = torch.zeros_like(self.edge_dist)
        self.dexndm_guided_rotation_prev_rot_dist = torch.zeros_like(self.edge_dist)
        self.dexndm_guided_rotation_progress = torch.zeros_like(self.edge_dist)
        self.dexndm_guided_rotation_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.local_random_workspace_lo, self.local_random_workspace_hi = self._compute_local_random_workspace_bounds()
        self.reset_score_workspace_lo, self.reset_score_workspace_hi = self._compute_quantile_local_random_workspace_bounds()
        self.reset_bad_score = torch.zeros(self.num_stable_grasps, device=self.device)
        self.reset_grasp_candidate_indices = self._build_reset_grasp_candidate_indices()
        self.contact_force_norm = torch.zeros((self.num_envs, len(self.hand_spec.force_sensor_bodies)), device=self.device)
        self.contact_bit_force_norm = torch.zeros((self.num_envs, len(self.hand_spec.contact_bit_bodies)), device=self.device)
        self.contact_bit_obs = torch.zeros_like(self.contact_bit_force_norm)
        self.all_body_contact_force_norm = torch.zeros((self.num_envs, self.all_body_contact_obs_count), device=self.device)

        self._compute_intermediate_values()

    def _resolve_reset_source(self) -> str:
        if self.cfg.reset_source != "stable_cache":
            raise ValueError("POISE requires reset_source='stable_cache'.")
        return "stable_cache"




    def _uses_pose6d_feasible_target_filter(self) -> bool:
        return bool(getattr(self.cfg, 'pose6d_feasible_target_filter_enabled', False))

    def _uses_pose6d_anchor_relaxed_targets(self) -> bool:
        return (
            bool(getattr(self.cfg, 'pose6d_anchor_relaxed_enabled', False)) and (not self._uses_pose6d_feasible_target_filter())
        )

    def _active_pose6d_anchor_success_threshold(self) -> torch.Tensor:
        if self.pose6d_anchor_success_threshold_by_size.numel() > 0:
            return self.pose6d_anchor_success_threshold_by_size[self.active_cube_size_id]
        return torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "pose6d_anchor_success_threshold_m", 0.010)),
            device=self.device,
            dtype=torch.float32,
        )

    def _active_twist_swing_success_threshold(self) -> torch.Tensor:
        if self.pose6d_anchor_swing_success_threshold_by_size.numel() > 0:
            return self.pose6d_anchor_swing_success_threshold_by_size[self.active_cube_size_id]
        return torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "twist_swing_success_threshold_rad", 0.174533)),
            device=self.device,
            dtype=torch.float32,
        )









    def _twist_target_sampling_mode(self) -> str:
        mode = str(getattr(self.cfg, "twist_target_sampling_mode", "curriculum")).lower()
        if mode not in {"curriculum", "full_range_uniform"}:
            raise ValueError(
                "twist_target_sampling_mode must be 'curriculum' or "
                f"'full_range_uniform'; got {getattr(self.cfg, 'twist_target_sampling_mode', None)!r}."
            )
        return mode

    def _twist_target_curriculum_mode(self) -> str:
        return str(getattr(self.cfg, "twist_target_curriculum_mode", "adaptive_frontier")).lower()

    def _adaptive_twist_target_curriculum_enabled(self) -> bool:
        return (
            self._twist_target_sampling_mode() == "curriculum"
            and bool(getattr(self.cfg, "twist_target_curriculum_enabled", False))
            and self._twist_target_curriculum_mode() in {"adaptive", "adaptive_frontier"}
        )

    def _current_twist_target_delta_rad(self) -> float:
        if not bool(getattr(self.cfg, "twist_target_curriculum_enabled", False)):
            return float(getattr(self.cfg, "target_twist_angle_rad", 0.0872665))
        mode = self._twist_target_curriculum_mode()
        if mode in {"adaptive", "adaptive_frontier"}:
            return float(self.adaptive_target_delta_max.item())
        if mode not in {"linear", "time_linear"}:
            raise ValueError(
                "twist_target_curriculum_mode must be one of 'linear' or 'adaptive_frontier'; "
                f"got {getattr(self.cfg, 'twist_target_curriculum_mode', None)!r}."
            )
        start = float(getattr(self.cfg, "twist_target_curriculum_start_rad", 0.0872665))
        end = float(getattr(self.cfg, "twist_target_curriculum_end_rad", start))
        steps = max(int(getattr(self.cfg, "twist_target_curriculum_steps", 1)), 1)
        alpha = min(float(self.common_step_counter) / float(steps), 1.0)
        return start + alpha * (end - start)

    def _chain_post_success_target_curriculum_enabled(self) -> bool:
        return bool(getattr(self.cfg, "chain_post_success_target_curriculum_enabled", False))

    def _primary_chain_shared_rotation_curriculum_enabled(self) -> bool:
        """Whether Primary and Chain jointly use the Primary Rotation frontier."""
        return bool(getattr(self.cfg, "primary_chain_shared_rotation_curriculum_enabled", False))

    def _multiobject_per_object_curriculum_enabled(self) -> bool:
        return bool(getattr(self, "multiobject_per_object_curriculum_enabled", False))

    def _sync_shared_rotation_curriculum_frontier_aliases(self, object_id: int | None = None) -> None:
        """Keep legacy Chain frontier-max fields aligned in shared Rotation mode.

        Sampling and promotion use only the Primary state. The Chain max remains
        a read-only compatibility alias so existing dashboards do not report a
        stale 30-degree Chain angle when the shared frontier has advanced.
        """
        if not self._primary_chain_shared_rotation_curriculum_enabled():
            return
        if object_id is not None and self._multiobject_per_object_curriculum_enabled():
            self.adaptive_chain_target_delta_max_by_object[object_id].copy_(
                self.adaptive_target_delta_max_by_object[object_id]
            )
            return
        self.adaptive_chain_target_delta_max.copy_(self.adaptive_target_delta_max)
        if self._multiobject_per_object_curriculum_enabled():
            self.adaptive_chain_target_delta_max_by_object.copy_(self.adaptive_target_delta_max_by_object)

    def _all_reduce_curriculum_stats(
        self, statistics: torch.Tensor, *, multiobject: bool = False
    ) -> tuple[torch.Tensor, int]:
        config_name = (
            "multiobject_curriculum_distributed_sync_enabled"
            if multiobject
            else "curriculum_distributed_sync_enabled"
        )
        if not bool(getattr(self.cfg, config_name, False)):
            return statistics, 1
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return statistics, 1
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return statistics, 1
        if str(torch.distributed.get_backend()).lower() == "gloo":
            reduced = statistics.detach().cpu()
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            return reduced.to(device=self.device), world_size
        reduced = statistics.detach().clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        return reduced, world_size

    def _curriculum_min_frontier_done_episodes(
        self,
        config_name: str,
        default: int,
        object_id: int | None,
        min_done_scale: int,
    ) -> int:
        base = max(int(getattr(self.cfg, config_name, default)), 0)
        if object_id is None:
            global_min = int(getattr(self.cfg, "curriculum_global_min_frontier_done_episodes", -1))
            if global_min >= 0:
                return global_min
        else:
            global_per_object = int(
                getattr(self.cfg, "multiobject_curriculum_global_min_frontier_done_episodes", -1)
            )
            if global_per_object >= 0:
                return global_per_object
        return base * max(int(min_done_scale), 1)

    def _curriculum_object_ids_for_envs(self, env_ids: torch.Tensor | None) -> torch.Tensor | None:
        if env_ids is None or not self._multiobject_per_object_curriculum_enabled():
            return None
        return self.active_cube_size_id[env_ids].to(dtype=torch.long)

    def _sync_multiobject_curriculum_summaries(self) -> None:
        if not self._multiobject_per_object_curriculum_enabled():
            return
        self.adaptive_target_delta_max.copy_(self.adaptive_target_delta_max_by_object.mean())
        self.adaptive_frontier_success_rate.copy_(self.adaptive_frontier_success_rate_by_object.mean())
        self.adaptive_frontier_done_count.copy_(self.adaptive_frontier_done_count_by_object.sum())
        self.adaptive_frontier_done_total.copy_(self.adaptive_frontier_done_total_by_object.sum())
        self.adaptive_curriculum_advances.copy_(self.adaptive_curriculum_advances_by_object.sum())
        self.adaptive_soft_promotion_counts.copy_(self.adaptive_soft_promotion_counts_by_object.sum(dim=0))
        self.adaptive_twist_chain_sync_limited_count.copy_(
            self.adaptive_twist_chain_sync_limited_count_by_object.sum()
        )
        self.adaptive_chain_target_delta_max.copy_(self.adaptive_chain_target_delta_max_by_object.mean())
        self.adaptive_chain_frontier_success_rate.copy_(self.adaptive_chain_frontier_success_rate_by_object.mean())
        self.adaptive_chain_frontier_done_count.copy_(self.adaptive_chain_frontier_done_count_by_object.sum())
        self.adaptive_chain_frontier_done_total.copy_(self.adaptive_chain_frontier_done_total_by_object.sum())
        self.adaptive_chain_curriculum_advances.copy_(self.adaptive_chain_curriculum_advances_by_object.sum())
        self.adaptive_translation_delta_max.copy_(self.adaptive_translation_delta_max_by_object.mean())
        self.adaptive_translation_frontier_success_rate.copy_(
            self.adaptive_translation_frontier_success_rate_by_object.mean()
        )
        self.adaptive_translation_frontier_done_count.copy_(
            self.adaptive_translation_frontier_done_count_by_object.sum()
        )
        self.adaptive_translation_frontier_done_total.copy_(
            self.adaptive_translation_frontier_done_total_by_object.sum()
        )
        self.adaptive_translation_curriculum_advances.copy_(
            self.adaptive_translation_curriculum_advances_by_object.sum()
        )
        self.adaptive_translation_chain_sync_limited_count.copy_(
            self.adaptive_translation_chain_sync_limited_count_by_object.sum()
        )
        self.adaptive_chain_translation_delta_max.copy_(self.adaptive_chain_translation_delta_max_by_object.mean())
        self.adaptive_chain_translation_frontier_success_rate.copy_(
            self.adaptive_chain_translation_frontier_success_rate_by_object.mean()
        )
        self.adaptive_chain_translation_frontier_done_count.copy_(
            self.adaptive_chain_translation_frontier_done_count_by_object.sum()
        )
        self.adaptive_chain_translation_frontier_done_total.copy_(
            self.adaptive_chain_translation_frontier_done_total_by_object.sum()
        )
        self.adaptive_chain_translation_curriculum_advances.copy_(
            self.adaptive_chain_translation_curriculum_advances_by_object.sum()
        )

    def _sample_adaptive_twist_target_delta_rad(
        self,
        count: int,
        start: float,
        cap: float,
        adaptive_max: torch.Tensor,
        object_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_modes = torch.zeros(count, device=self.device, dtype=torch.long)
        if object_ids is None:
            current_max = min(max(float(adaptive_max.item()), start), cap)
            adaptive_max.fill_(current_max)
            current_max_tensor = torch.full((count,), current_max, device=self.device)
        else:
            if int(object_ids.numel()) != count:
                raise ValueError("object_ids must match sampled target count.")
            adaptive_max.clamp_(min=start, max=cap)
            current_max_tensor = adaptive_max[object_ids].to(device=self.device)
        target_delta = torch.empty(count, device=self.device)

        frontier_prob = max(float(getattr(self.cfg, "twist_target_adaptive_frontier_sample_prob", 0.60)), 0.0)
        replay_prob = max(float(getattr(self.cfg, "twist_target_adaptive_replay_sample_prob", 0.30)), 0.0)
        anchor_prob = max(float(getattr(self.cfg, "twist_target_adaptive_anchor_sample_prob", 0.10)), 0.0)
        prob_total = frontier_prob + replay_prob + anchor_prob
        if prob_total <= 0.0:
            frontier_prob, replay_prob, anchor_prob = 0.60, 0.30, 0.10
            prob_total = 1.0
        frontier_prob /= prob_total
        replay_prob /= prob_total
        anchor_prob /= prob_total

        u = torch.rand(count, device=self.device, generator=self.target_rng)
        frontier_mask = u < frontier_prob
        replay_mask = (u >= frontier_prob) & (u < frontier_prob + replay_prob)
        anchor_mask = ~(frontier_mask | replay_mask)

        width = max(float(getattr(self.cfg, "twist_target_adaptive_frontier_width_rad", 0.174533)), 0.0)
        frontier_low = torch.clamp(current_max_tensor - width, min=start)
        frontier_span = torch.clamp(current_max_tensor - frontier_low, min=0.0)
        replay_span = torch.clamp(current_max_tensor - start, min=0.0)

        target_delta[replay_mask] = start + replay_span[replay_mask] * torch.rand(
            int(replay_mask.sum().item()),
            device=self.device,
            generator=self.target_rng,
        )
        target_delta[frontier_mask] = frontier_low[frontier_mask] + frontier_span[frontier_mask] * torch.rand(
            int(frontier_mask.sum().item()),
            device=self.device,
            generator=self.target_rng,
        )
        target_delta[anchor_mask] = current_max_tensor[anchor_mask]

        sample_modes[frontier_mask] = 1
        sample_modes[replay_mask] = 2
        sample_modes[anchor_mask] = 3
        target_delta = torch.clamp(target_delta, min=start, max=cap)
        max_at_reset = current_max_tensor.clone()
        return target_delta, sample_modes, max_at_reset

    def _sample_twist_target_delta_rad(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_modes = torch.zeros(count, device=self.device, dtype=torch.long)
        if self._twist_target_sampling_mode() == "full_range_uniform":
            low = float(getattr(self.cfg, "twist_target_uniform_min_rad", 0.0872665))
            high = float(getattr(self.cfg, "twist_target_uniform_max_rad", torch.pi))
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low < 0.0
                or high < low
                or high > math.pi + 1.0e-6
            ):
                raise ValueError(
                    "full-range uniform twist bounds must be finite and satisfy "
                    f"0 <= min <= max <= pi; got min={low}, max={high}."
                )
            target_delta = low + (high - low) * torch.rand(
                count, device=self.device, generator=self.target_rng
            )
            sample_modes.fill_(4)
            maximum = torch.full((count,), high, device=self.device)
            return target_delta, sample_modes, maximum
        if not bool(getattr(self.cfg, "twist_target_curriculum_enabled", False)):
            target_delta = torch.full(
                (count,),
                float(getattr(self.cfg, "target_twist_angle_rad", 0.0872665)),
                device=self.device,
            )
            return target_delta, sample_modes, target_delta

        mode = self._twist_target_curriculum_mode()
        if mode in {"linear", "time_linear"}:
            target_delta = torch.full((count,), self._current_twist_target_delta_rad(), device=self.device)
            return target_delta, sample_modes, target_delta
        if mode not in {"adaptive", "adaptive_frontier"}:
            raise ValueError(
                "twist_target_curriculum_mode must be one of 'linear' or 'adaptive_frontier'; "
                f"got {getattr(self.cfg, 'twist_target_curriculum_mode', None)!r}."
            )

        start = float(getattr(self.cfg, "twist_target_curriculum_start_rad", 0.0872665))
        cap = float(getattr(self.cfg, "twist_target_curriculum_end_rad", start))
        object_ids = self._curriculum_object_ids_for_envs(env_ids)
        adaptive_max = self.adaptive_target_delta_max
        if object_ids is not None:
            adaptive_max = self.adaptive_target_delta_max_by_object
        return self._sample_adaptive_twist_target_delta_rad(count, start, cap, adaptive_max, object_ids)

    def _sample_chain_twist_target_delta_rad(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._chain_post_success_target_curriculum_enabled():
            return self._sample_twist_target_delta_rad(count, env_ids)
        if self._primary_chain_shared_rotation_curriculum_enabled():
            return self._sample_twist_target_delta_rad(count, env_ids)
        if not self._adaptive_twist_target_curriculum_enabled():
            return self._sample_twist_target_delta_rad(count, env_ids)

        start = float(getattr(self.cfg, "chain_post_success_target_curriculum_start_rad", 0.0872665))
        cap = float(getattr(self.cfg, "chain_post_success_target_curriculum_end_rad", start))
        object_ids = self._curriculum_object_ids_for_envs(env_ids)
        adaptive_max = self.adaptive_chain_target_delta_max
        if object_ids is not None:
            adaptive_max = self.adaptive_chain_target_delta_max_by_object
        return self._sample_adaptive_twist_target_delta_rad(count, start, cap, adaptive_max, object_ids)

    def _adaptive_soft_promotion_candidate(
        self,
        *,
        axis: str,
        current_max: float,
        cap: float,
        soft_step: float,
        success_rate: float,
        history_batches: int,
        min_frontier_done: int,
        frontier_done_total: float,
        object_id: int | None,
    ) -> float | None:
        if not bool(getattr(self.cfg, "adaptive_frontier_soft_promotion_enabled", False)):
            return None
        if current_max >= cap - 1.0e-9 or soft_step <= 0.0:
            return None

        if object_id is None:
            last_level_change_step = self.adaptive_soft_promotion_last_level_change_step[axis]
            last_global_soft_promotion_step = self.adaptive_soft_promotion_last_global_step
        else:
            last_level_change_step = int(
                self.adaptive_soft_promotion_last_level_change_step_by_object[axis][object_id].item()
            )
            last_global_soft_promotion_step = int(
                self.adaptive_soft_promotion_last_global_step_by_object[object_id].item()
            )

        steps_per_iteration = max(
            int(getattr(self.cfg, "adaptive_frontier_soft_promotion_steps_per_iteration", 8)),
            1,
        )
        plateau_iterations = max(
            int(getattr(self.cfg, "adaptive_frontier_soft_promotion_plateau_iterations", 2_000)),
            0,
        )
        plateau_steps = plateau_iterations * steps_per_iteration
        global_cooldown_steps = max(
            int(getattr(self.cfg, "adaptive_frontier_soft_promotion_global_cooldown_iterations", 1_000)),
            0,
        ) * steps_per_iteration
        min_history = max(
            int(getattr(self.cfg, "adaptive_frontier_soft_promotion_min_history_batches", 50)),
            1,
        )
        current_step = int(self.common_step_counter)
        if not adaptive_soft_promotion_is_eligible(
            current_step=current_step,
            last_level_change_step=last_level_change_step,
            last_global_soft_promotion_step=last_global_soft_promotion_step,
            plateau_steps=plateau_steps,
            global_cooldown_steps=global_cooldown_steps,
            success_rate=success_rate,
            success_threshold=float(
                getattr(self.cfg, "adaptive_frontier_soft_promotion_success_threshold", 0.30)
            ),
            history_batches=history_batches,
            min_history_batches=min_history,
            frontier_done_total=frontier_done_total,
            min_frontier_done=min_frontier_done,
        ):
            return None
        return min(current_max + soft_step, cap)

    def _record_adaptive_curriculum_level_change(
        self, axis: str, *, soft: bool, object_id: int | None = None
    ) -> None:
        current_step = int(self.common_step_counter)
        if object_id is None:
            self.adaptive_soft_promotion_last_level_change_step[axis] = current_step
        else:
            self.adaptive_soft_promotion_last_level_change_step_by_object[axis][object_id] = current_step
        if not soft:
            return
        axis_index = {"twist": 0, "chain_twist": 1, "translation": 2}[axis]
        if object_id is None:
            self.adaptive_soft_promotion_counts[axis_index].add_(1.0)
            self.adaptive_soft_promotion_last_global_step = current_step
        else:
            self.adaptive_soft_promotion_counts_by_object[object_id, axis_index].add_(1.0)
            self.adaptive_soft_promotion_last_global_step_by_object[object_id] = current_step

    def _adaptive_soft_promotion_dwell_iterations(
        self, axis: str, object_id: int | None = None
    ) -> torch.Tensor:
        steps_per_iteration = max(
            int(getattr(self.cfg, "adaptive_frontier_soft_promotion_steps_per_iteration", 8)),
            1,
        )
        if object_id is None and self._multiobject_per_object_curriculum_enabled():
            dwell_steps = (
                int(self.common_step_counter)
                - self.adaptive_soft_promotion_last_level_change_step_by_object[axis]
            )
            return dwell_steps.float().mean() / float(steps_per_iteration)
        if object_id is None:
            last_level_change_step = self.adaptive_soft_promotion_last_level_change_step[axis]
        else:
            last_level_change_step = int(
                self.adaptive_soft_promotion_last_level_change_step_by_object[axis][object_id].item()
            )
        dwell_steps = int(self.common_step_counter) - last_level_change_step
        return torch.tensor(float(dwell_steps) / float(steps_per_iteration), device=self.device)

    def _reset_adaptive_twist_target_history(self, object_id: int | None = None) -> None:
        if object_id is not None:
            self.adaptive_frontier_success_history_by_object[object_id].zero_()
            self.adaptive_frontier_count_history_by_object[object_id].zero_()
            self.adaptive_frontier_success_history_count_by_object[object_id] = 0
            self.adaptive_frontier_success_history_index_by_object[object_id] = 0
            self.adaptive_frontier_success_rate_by_object[object_id].zero_()
            self.adaptive_frontier_done_count_by_object[object_id].zero_()
            self.adaptive_frontier_done_total_by_object[object_id].zero_()
            return
        self.adaptive_frontier_success_history.zero_()
        self.adaptive_frontier_count_history.zero_()
        self.adaptive_frontier_success_history_count = 0
        self.adaptive_frontier_success_history_index = 0
        self.adaptive_frontier_success_rate.zero_()
        self.adaptive_frontier_done_count.zero_()
        self.adaptive_frontier_done_total.zero_()

    def _reset_adaptive_chain_twist_target_history(self, object_id: int | None = None) -> None:
        if object_id is not None:
            self.adaptive_chain_frontier_success_history_by_object[object_id].zero_()
            self.adaptive_chain_frontier_count_history_by_object[object_id].zero_()
            self.adaptive_chain_frontier_success_history_count_by_object[object_id] = 0
            self.adaptive_chain_frontier_success_history_index_by_object[object_id] = 0
            self.adaptive_chain_frontier_success_rate_by_object[object_id].zero_()
            self.adaptive_chain_frontier_done_count_by_object[object_id].zero_()
            self.adaptive_chain_frontier_done_total_by_object[object_id].zero_()
            return
        self.adaptive_chain_frontier_success_history.zero_()
        self.adaptive_chain_frontier_count_history.zero_()
        self.adaptive_chain_frontier_success_history_count = 0
        self.adaptive_chain_frontier_success_history_index = 0
        self.adaptive_chain_frontier_success_rate.zero_()
        self.adaptive_chain_frontier_done_count.zero_()
        self.adaptive_chain_frontier_done_total.zero_()

    def _chain_curriculum_sync_enabled(self) -> bool:
        return bool(getattr(self.cfg, "chain_curriculum_sync_enabled", False))

    def _limit_first_twist_curriculum_by_chain(
        self,
        candidate_max: float,
        current_max: float,
        cap: float,
        object_id: int | None = None,
    ) -> tuple[float, bool]:
        if not self._chain_curriculum_sync_enabled():
            return candidate_max, False
        if not self._chain_post_success_target_curriculum_enabled():
            return candidate_max, False
        if self._primary_chain_shared_rotation_curriculum_enabled():
            return candidate_max, False
        max_lead = float(getattr(self.cfg, "twist_target_chain_sync_max_lead_rad", -1.0))
        if max_lead < 0.0:
            return candidate_max, False

        if object_id is not None and self._multiobject_per_object_curriculum_enabled():
            chain_max = float(self.adaptive_chain_target_delta_max_by_object[object_id].item())
        else:
            chain_max = float(self.adaptive_chain_target_delta_max.item())
        sync_cap = max(current_max, min(cap, chain_max + max_lead))
        limited_max = min(candidate_max, sync_cap)
        return limited_max, limited_max < candidate_max - 1.0e-6

    def _update_adaptive_twist_target_curriculum(
        self,
        event_envs: torch.Tensor,
        success_mask: torch.Tensor | None = None,
        object_id: int | None = None,
        batch_stats: torch.Tensor | None = None,
        min_done_scale: int = 1,
    ) -> None:
        if not self._adaptive_twist_target_curriculum_enabled():
            return

        if success_mask is None:
            success_mask = self.target_reached

        if self._multiobject_per_object_curriculum_enabled() and object_id is None:
            object_count = len(self.multisize_object_ids)
            statistics = torch.zeros((object_count, 2), device=self.device)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                current_max = self.adaptive_target_delta_max_by_object[size_id]
                level_match = torch.abs(self.target_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
                frontier_events = object_envs & (
                    (self.target_delta_sample_mode == 1) | (self.target_delta_sample_mode == 3)
                ) & level_match
                if (
                    self._chain_post_success_target_curriculum_enabled()
                    and not self._primary_chain_shared_rotation_curriculum_enabled()
                ):
                    frontier_events &= self.target_chain_phase == 0
                statistics[size_id, 0] = success_mask[frontier_events].float().sum()
                statistics[size_id, 1] = frontier_events.sum()
            statistics, world_size = self._all_reduce_curriculum_stats(statistics, multiobject=True)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                self._update_adaptive_twist_target_curriculum(
                    object_envs,
                    success_mask,
                    object_id=size_id,
                    batch_stats=statistics[size_id],
                    min_done_scale=world_size,
                )
            self._sync_multiobject_curriculum_summaries()
            return

        if object_id is None:
            adaptive_max = self.adaptive_target_delta_max
            success_history = self.adaptive_frontier_success_history
            count_history = self.adaptive_frontier_count_history
            history_count = self.adaptive_frontier_success_history_count
            history_index = self.adaptive_frontier_success_history_index
            success_rate = self.adaptive_frontier_success_rate
            done_count = self.adaptive_frontier_done_count
            done_total = self.adaptive_frontier_done_total
            advances = self.adaptive_curriculum_advances
        else:
            adaptive_max = self.adaptive_target_delta_max_by_object[object_id]
            success_history = self.adaptive_frontier_success_history_by_object[object_id]
            count_history = self.adaptive_frontier_count_history_by_object[object_id]
            history_count = int(self.adaptive_frontier_success_history_count_by_object[object_id].item())
            history_index = int(self.adaptive_frontier_success_history_index_by_object[object_id].item())
            success_rate = self.adaptive_frontier_success_rate_by_object[object_id]
            done_count = self.adaptive_frontier_done_count_by_object[object_id]
            done_total = self.adaptive_frontier_done_total_by_object[object_id]
            advances = self.adaptive_curriculum_advances_by_object[object_id]

        current_max = float(adaptive_max.item())
        if batch_stats is None:
            level_match = torch.abs(self.target_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
            frontier_or_anchor = (self.target_delta_sample_mode == 1) | (self.target_delta_sample_mode == 3)
            frontier_events = event_envs & frontier_or_anchor & level_match
            if (
                self._chain_post_success_target_curriculum_enabled()
                and not self._primary_chain_shared_rotation_curriculum_enabled()
            ):
                frontier_events = frontier_events & (self.target_chain_phase == 0)
            batch_success_count = success_mask[frontier_events].float().sum()
            frontier_count_tensor = frontier_events.sum().to(dtype=batch_success_count.dtype)
            if object_id is None:
                statistics = torch.stack((batch_success_count, frontier_count_tensor))
                statistics, min_done_scale = self._all_reduce_curriculum_stats(statistics)
                batch_success_count = statistics[0]
                frontier_count_tensor = statistics[1]
            frontier_count = int(frontier_count_tensor.item())
        else:
            batch_success_count = batch_stats[0]
            frontier_count = int(batch_stats[1].item())
        done_count.fill_(float(frontier_count))
        if frontier_count <= 0:
            return

        done_total.add_(float(frontier_count))
        history_len = int(success_history.numel())
        # Weight done/refresh batches by event count; refresh batches can be much smaller.
        success_history[history_index] = batch_success_count.detach()
        count_history[history_index] = float(frontier_count)
        history_index = (history_index + 1) % history_len
        history_count = min(history_count + 1, history_len)
        if object_id is None:
            self.adaptive_frontier_success_history_index = history_index
            self.adaptive_frontier_success_history_count = history_count
        else:
            self.adaptive_frontier_success_history_index_by_object[object_id] = history_index
            self.adaptive_frontier_success_history_count_by_object[object_id] = history_count

        valid_count = history_count
        valid_success_count = success_history[:valid_count].sum()
        valid_event_count = torch.clamp(count_history[:valid_count].sum(), min=1.0)
        frontier_success_rate = valid_success_count / valid_event_count
        success_rate.copy_(frontier_success_rate)

        min_history = max(int(getattr(self.cfg, "twist_target_adaptive_min_history_batches", 5)), 1)
        if valid_count < min_history:
            return

        min_done_episodes = self._curriculum_min_frontier_done_episodes(
            "twist_target_adaptive_min_frontier_done_episodes", 20_000, object_id, min_done_scale
        )
        if float(done_total.item()) < float(min_done_episodes):
            return

        freeze_threshold = float(getattr(self.cfg, "twist_target_adaptive_freeze_threshold", 0.20))
        if float(frontier_success_rate.item()) < freeze_threshold:
            return

        cap = float(getattr(self.cfg, "twist_target_curriculum_end_rad", current_max))
        step = max(float(getattr(self.cfg, "twist_target_adaptive_step_rad", 0.0872665)), 0.0)
        threshold = float(getattr(self.cfg, "twist_target_adaptive_success_threshold", 0.80))
        soft_promotion = False
        next_max = None
        if step > 0.0 and current_max < cap - 1.0e-6 and float(frontier_success_rate.item()) >= threshold:
            next_max = min(current_max + step, cap)
        else:
            next_max = self._adaptive_soft_promotion_candidate(
                axis="twist",
                current_max=current_max,
                cap=cap,
                soft_step=max(
                    float(getattr(self.cfg, "adaptive_frontier_soft_promotion_twist_step_rad", 0.0872665)),
                    0.0,
                ),
                success_rate=float(frontier_success_rate.item()),
                history_batches=valid_count,
                min_frontier_done=min_done_episodes,
                frontier_done_total=float(done_total.item()),
                object_id=object_id,
            )
            soft_promotion = next_max is not None
        if next_max is not None:
            next_max, sync_limited = self._limit_first_twist_curriculum_by_chain(
                next_max,
                current_max,
                cap,
                object_id=object_id,
            )
            if sync_limited:
                if object_id is None:
                    self.adaptive_twist_chain_sync_limited_count.add_(1.0)
                else:
                    self.adaptive_twist_chain_sync_limited_count_by_object[object_id].add_(1.0)
            if next_max <= current_max + 1.0e-6:
                return
            adaptive_max.fill_(next_max)
            self._sync_shared_rotation_curriculum_frontier_aliases(object_id=object_id)
            advances.add_(1.0)
            self._record_adaptive_curriculum_level_change("twist", soft=soft_promotion, object_id=object_id)
            self._reset_adaptive_twist_target_history(object_id=object_id)

    def _update_adaptive_chain_twist_target_curriculum(
        self,
        event_envs: torch.Tensor,
        success_mask: torch.Tensor | None = None,
        object_id: int | None = None,
        batch_stats: torch.Tensor | None = None,
        min_done_scale: int = 1,
    ) -> None:
        if not self._chain_post_success_target_curriculum_enabled():
            return
        # In shared mode the Primary updater consumes both Primary and Chain
        # frontier events. Updating this legacy Chain state as well would count
        # every Chain event twice and could advance two independent frontiers.
        if self._primary_chain_shared_rotation_curriculum_enabled():
            return
        if not self._adaptive_twist_target_curriculum_enabled():
            return

        if success_mask is None:
            success_mask = self.target_reached

        if self._multiobject_per_object_curriculum_enabled() and object_id is None:
            object_count = len(self.multisize_object_ids)
            statistics = torch.zeros((object_count, 2), device=self.device)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                current_max = self.adaptive_chain_target_delta_max_by_object[size_id]
                level_match = torch.abs(self.target_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
                frontier_events = object_envs & (
                    (self.target_delta_sample_mode == 1) | (self.target_delta_sample_mode == 3)
                ) & level_match & (self.target_chain_phase > 0)
                statistics[size_id, 0] = success_mask[frontier_events].float().sum()
                statistics[size_id, 1] = frontier_events.sum()
            statistics, world_size = self._all_reduce_curriculum_stats(statistics, multiobject=True)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                self._update_adaptive_chain_twist_target_curriculum(
                    object_envs,
                    success_mask,
                    object_id=size_id,
                    batch_stats=statistics[size_id],
                    min_done_scale=world_size,
                )
            self._sync_multiobject_curriculum_summaries()
            return

        if object_id is None:
            adaptive_max = self.adaptive_chain_target_delta_max
            success_history = self.adaptive_chain_frontier_success_history
            count_history = self.adaptive_chain_frontier_count_history
            history_count = self.adaptive_chain_frontier_success_history_count
            history_index = self.adaptive_chain_frontier_success_history_index
            success_rate = self.adaptive_chain_frontier_success_rate
            done_count = self.adaptive_chain_frontier_done_count
            done_total = self.adaptive_chain_frontier_done_total
            advances = self.adaptive_chain_curriculum_advances
        else:
            adaptive_max = self.adaptive_chain_target_delta_max_by_object[object_id]
            success_history = self.adaptive_chain_frontier_success_history_by_object[object_id]
            count_history = self.adaptive_chain_frontier_count_history_by_object[object_id]
            history_count = int(self.adaptive_chain_frontier_success_history_count_by_object[object_id].item())
            history_index = int(self.adaptive_chain_frontier_success_history_index_by_object[object_id].item())
            success_rate = self.adaptive_chain_frontier_success_rate_by_object[object_id]
            done_count = self.adaptive_chain_frontier_done_count_by_object[object_id]
            done_total = self.adaptive_chain_frontier_done_total_by_object[object_id]
            advances = self.adaptive_chain_curriculum_advances_by_object[object_id]

        current_max = float(adaptive_max.item())
        if batch_stats is None:
            level_match = torch.abs(self.target_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
            frontier_or_anchor = (self.target_delta_sample_mode == 1) | (self.target_delta_sample_mode == 3)
            frontier_events = event_envs & frontier_or_anchor & level_match & (self.target_chain_phase > 0)
            batch_success_count = success_mask[frontier_events].float().sum()
            frontier_count_tensor = frontier_events.sum().to(dtype=batch_success_count.dtype)
            if object_id is None:
                statistics = torch.stack((batch_success_count, frontier_count_tensor))
                statistics, min_done_scale = self._all_reduce_curriculum_stats(statistics)
                batch_success_count = statistics[0]
                frontier_count_tensor = statistics[1]
            frontier_count = int(frontier_count_tensor.item())
        else:
            batch_success_count = batch_stats[0]
            frontier_count = int(batch_stats[1].item())
        done_count.fill_(float(frontier_count))
        if frontier_count <= 0:
            return

        done_total.add_(float(frontier_count))
        history_len = int(success_history.numel())
        success_history[history_index] = batch_success_count.detach()
        count_history[history_index] = float(frontier_count)
        history_index = (history_index + 1) % history_len
        history_count = min(history_count + 1, history_len)
        if object_id is None:
            self.adaptive_chain_frontier_success_history_index = history_index
            self.adaptive_chain_frontier_success_history_count = history_count
        else:
            self.adaptive_chain_frontier_success_history_index_by_object[object_id] = history_index
            self.adaptive_chain_frontier_success_history_count_by_object[object_id] = history_count

        valid_count = history_count
        valid_success_count = success_history[:valid_count].sum()
        valid_event_count = torch.clamp(count_history[:valid_count].sum(), min=1.0)
        frontier_success_rate = valid_success_count / valid_event_count
        success_rate.copy_(frontier_success_rate)

        min_history = max(int(getattr(self.cfg, "twist_target_adaptive_min_history_batches", 5)), 1)
        if valid_count < min_history:
            return

        min_done_episodes = self._curriculum_min_frontier_done_episodes(
            "twist_target_adaptive_min_frontier_done_episodes", 20_000, object_id, min_done_scale
        )
        if float(done_total.item()) < float(min_done_episodes):
            return

        freeze_threshold = float(getattr(self.cfg, "twist_target_adaptive_freeze_threshold", 0.20))
        if float(frontier_success_rate.item()) < freeze_threshold:
            return

        cap = float(getattr(self.cfg, "chain_post_success_target_curriculum_end_rad", current_max))
        step = max(float(getattr(self.cfg, "twist_target_adaptive_step_rad", 0.0872665)), 0.0)
        threshold = float(
            getattr(
                self.cfg,
                "chain_twist_target_adaptive_success_threshold",
                getattr(self.cfg, "twist_target_adaptive_success_threshold", 0.80),
            )
        )
        if threshold < 0.0:
            threshold = float(getattr(self.cfg, "twist_target_adaptive_success_threshold", 0.80))
        soft_promotion = False
        next_max = None
        if step > 0.0 and current_max < cap - 1.0e-6 and float(frontier_success_rate.item()) >= threshold:
            next_max = min(current_max + step, cap)
        else:
            next_max = self._adaptive_soft_promotion_candidate(
                axis="chain_twist",
                current_max=current_max,
                cap=cap,
                soft_step=max(
                    float(getattr(self.cfg, "adaptive_frontier_soft_promotion_twist_step_rad", 0.0872665)),
                    0.0,
                ),
                success_rate=float(frontier_success_rate.item()),
                history_batches=valid_count,
                min_frontier_done=min_done_episodes,
                frontier_done_total=float(done_total.item()),
                object_id=object_id,
            )
            soft_promotion = next_max is not None
        if next_max is not None:
            adaptive_max.fill_(next_max)
            advances.add_(1.0)
            self._record_adaptive_curriculum_level_change(
                "chain_twist", soft=soft_promotion, object_id=object_id
            )
            self._reset_adaptive_chain_twist_target_history(object_id=object_id)

    def _pose6d_translation_sampling_mode(self) -> str:
        mode = str(getattr(self.cfg, "pose6d_translation_sampling_mode", "curriculum")).lower()
        if mode not in {"curriculum", "full_range_uniform"}:
            raise ValueError(
                "pose6d_translation_sampling_mode must be 'curriculum' or "
                f"'full_range_uniform'; got {getattr(self.cfg, 'pose6d_translation_sampling_mode', None)!r}."
            )
        return mode

    def _pose6d_translation_curriculum_mode(self) -> str:
        return str(getattr(self.cfg, "pose6d_translation_curriculum_mode", "adaptive_frontier")).lower()

    def _adaptive_pose6d_translation_curriculum_enabled(self) -> bool:
        return (
            self._pose6d_translation_sampling_mode() == "curriculum"
            and bool(getattr(self.cfg, "pose6d_translation_curriculum_enabled", False))
            and self._pose6d_translation_curriculum_mode() in {"adaptive", "adaptive_frontier"}
        )

    def _current_pose6d_translation_delta_m(self) -> float:
        if not bool(getattr(self.cfg, "pose6d_translation_curriculum_enabled", False)):
            return float(getattr(self.cfg, "pose6d_translation_target_delta_m", 0.002))
        mode = self._pose6d_translation_curriculum_mode()
        if mode in {"adaptive", "adaptive_frontier"}:
            return float(self.adaptive_translation_delta_max.item())
        if mode not in {"linear", "time_linear"}:
            raise ValueError(
                "pose6d_translation_curriculum_mode must be one of 'linear' or 'adaptive_frontier'; "
                f"got {getattr(self.cfg, 'pose6d_translation_curriculum_mode', None)!r}."
            )
        start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002))
        end = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", start))
        steps = max(int(getattr(self.cfg, "pose6d_translation_curriculum_steps", 1)), 1)
        alpha = min(float(self.common_step_counter) / float(steps), 1.0)
        return start + alpha * (end - start)

    def _chain_pose6d_translation_curriculum_enabled(self) -> bool:
        return bool(getattr(self.cfg, "chain_pose6d_translation_curriculum_enabled", False)) and (
            self._adaptive_pose6d_translation_curriculum_enabled()
        )

    def _sample_adaptive_pose6d_translation_delta_m(
        self,
        count: int,
        start: float,
        cap: float,
        adaptive_max: torch.Tensor,
        object_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_modes = torch.zeros(count, device=self.device, dtype=torch.long)
        if object_ids is None:
            current_max = min(max(float(adaptive_max.item()), start), cap)
            adaptive_max.fill_(current_max)
            current_max_tensor = torch.full((count,), current_max, device=self.device)
        else:
            if int(object_ids.numel()) != count:
                raise ValueError("object_ids must match sampled translation count.")
            adaptive_max.clamp_(min=start, max=cap)
            current_max_tensor = adaptive_max[object_ids].to(device=self.device)
        target_delta = torch.empty(count, device=self.device)

        frontier_prob = max(float(getattr(self.cfg, "pose6d_translation_adaptive_frontier_sample_prob", 0.60)), 0.0)
        replay_prob = max(float(getattr(self.cfg, "pose6d_translation_adaptive_replay_sample_prob", 0.30)), 0.0)
        anchor_prob = max(float(getattr(self.cfg, "pose6d_translation_adaptive_anchor_sample_prob", 0.10)), 0.0)
        prob_total = frontier_prob + replay_prob + anchor_prob
        if prob_total <= 0.0:
            frontier_prob, replay_prob, anchor_prob = 0.60, 0.30, 0.10
            prob_total = 1.0
        frontier_prob /= prob_total
        replay_prob /= prob_total
        anchor_prob /= prob_total

        u = torch.rand(count, device=self.device, generator=self.target_rng)
        frontier_mask = u < frontier_prob
        replay_mask = (u >= frontier_prob) & (u < frontier_prob + replay_prob)
        anchor_mask = ~(frontier_mask | replay_mask)

        width = max(float(getattr(self.cfg, "pose6d_translation_adaptive_frontier_width_m", 0.005)), 0.0)
        frontier_low = torch.clamp(current_max_tensor - width, min=start)
        frontier_span = torch.clamp(current_max_tensor - frontier_low, min=0.0)
        replay_span = torch.clamp(current_max_tensor - start, min=0.0)

        target_delta[frontier_mask] = frontier_low[frontier_mask] + frontier_span[frontier_mask] * torch.rand(
            int(frontier_mask.sum().item()), device=self.device, generator=self.target_rng
        )
        target_delta[replay_mask] = start + replay_span[replay_mask] * torch.rand(
            int(replay_mask.sum().item()), device=self.device, generator=self.target_rng
        )
        target_delta[anchor_mask] = current_max_tensor[anchor_mask]

        sample_modes[frontier_mask] = 1
        sample_modes[replay_mask] = 2
        sample_modes[anchor_mask] = 3
        target_delta = torch.clamp(target_delta, min=start, max=cap)
        max_at_reset = current_max_tensor.clone()
        return target_delta, sample_modes, max_at_reset

    def _sample_pose6d_translation_delta_m(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_modes = torch.zeros(count, device=self.device, dtype=torch.long)
        if self._pose6d_translation_sampling_mode() == "full_range_uniform":
            low = float(getattr(self.cfg, "pose6d_translation_uniform_min_m", 0.002))
            high = float(getattr(self.cfg, "pose6d_translation_uniform_max_m", 0.015))
            if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or high < low:
                raise ValueError(
                    "full-range uniform translation bounds must be finite and satisfy "
                    f"0 <= min <= max; got min={low}, max={high}."
                )
            target_delta = low + (high - low) * torch.rand(
                count, device=self.device, generator=self.target_rng
            )
            sample_modes.fill_(4)
            maximum = torch.full((count,), high, device=self.device)
            return target_delta, sample_modes, maximum
        if not bool(getattr(self.cfg, "pose6d_translation_curriculum_enabled", False)):
            target_delta = torch.full(
                (count,),
                float(getattr(self.cfg, "pose6d_translation_target_delta_m", 0.002)),
                device=self.device,
            )
            return target_delta, sample_modes, target_delta

        mode = self._pose6d_translation_curriculum_mode()
        if mode in {"linear", "time_linear"}:
            target_delta = torch.full((count,), self._current_pose6d_translation_delta_m(), device=self.device)
            return target_delta, sample_modes, target_delta
        if mode not in {"adaptive", "adaptive_frontier"}:
            raise ValueError(
                "pose6d_translation_curriculum_mode must be one of 'linear' or 'adaptive_frontier'; "
                f"got {getattr(self.cfg, 'pose6d_translation_curriculum_mode', None)!r}."
            )

        start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002))
        cap = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", start))
        object_ids = self._curriculum_object_ids_for_envs(env_ids)
        adaptive_max = self.adaptive_translation_delta_max
        if object_ids is not None:
            adaptive_max = self.adaptive_translation_delta_max_by_object
        return self._sample_adaptive_pose6d_translation_delta_m(
            count,
            start,
            cap,
            adaptive_max,
            object_ids,
        )

    def _sample_chain_pose6d_translation_delta_m(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._chain_pose6d_translation_curriculum_enabled():
            return self._sample_pose6d_translation_delta_m(count, env_ids)

        start = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_start_m", 0.002))
        cap = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_end_m", start))
        object_ids = self._curriculum_object_ids_for_envs(env_ids)
        adaptive_max = self.adaptive_chain_translation_delta_max
        if object_ids is not None:
            adaptive_max = self.adaptive_chain_translation_delta_max_by_object
        return self._sample_adaptive_pose6d_translation_delta_m(
            count,
            start,
            cap,
            adaptive_max,
            object_ids,
        )

    def _sample_pose6d_translation_axes_palm(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        axis_name = str(getattr(self.cfg, "pose6d_translation_target_axis", "random")).lower()
        axis_map = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "neg_x": (-1.0, 0.0, 0.0),
            "neg_y": (0.0, -1.0, 0.0),
            "neg_z": (0.0, 0.0, -1.0),
        }
        if axis_name in axis_map:
            axis = torch.tensor(axis_map[axis_name], device=self.device, dtype=dtype)
            return axis.unsqueeze(0).expand(count, -1)
        if axis_name != "random":
            raise ValueError(
                "pose6d_translation_target_axis must be one of "
                "'random', 'x', 'y', 'z', 'neg_x', 'neg_y', or 'neg_z'; "
                f"got {getattr(self.cfg, 'pose6d_translation_target_axis', None)!r}."
            )
        translation_dir = torch.randn(
            (count, 3),
            device=self.device,
            dtype=dtype,
            generator=self.target_rng,
        )
        return translation_dir / torch.clamp(torch.linalg.norm(translation_dir, dim=-1, keepdim=True), min=1.0e-6)

    def _pose6d_anchor_forbidden_block_bounds(self, dtype: torch.dtype) -> torch.Tensor:
        blocks_raw = tuple(getattr(self.cfg, "pose6d_anchor_forbidden_block_bounds_m", ()))
        if len(blocks_raw) == 0:
            return torch.empty((0, 6), device=self.device, dtype=dtype)
        blocks = torch.tensor(
            tuple(tuple(float(v) for v in block) for block in blocks_raw),
            device=self.device,
            dtype=dtype,
        )
        if blocks.ndim != 2 or tuple(blocks.shape[1:]) != (6,):
            raise ValueError("pose6d_anchor_forbidden_block_bounds_m must be a sequence of 6D bounds.")
        names = tuple(str(v) for v in getattr(self.cfg, "pose6d_anchor_forbidden_block_names", ()))
        if names and len(names) != int(blocks.shape[0]):
            raise ValueError(
                "pose6d_anchor_forbidden_block_names must be empty or match "
                "pose6d_anchor_forbidden_block_bounds_m length."
            )
        if bool(
            (
                (blocks[:, 0] > blocks[:, 1])
                | (blocks[:, 2] > blocks[:, 3])
                | (blocks[:, 4] > blocks[:, 5])
            ).any()
        ):
            raise ValueError("Each forbidden block must be ordered as xmin,xmax,ymin,ymax,zmin,zmax.")
        return blocks

    def _pose6d_anchor_forbidden_block_hits(
        self,
        points: torch.Tensor,
        dims_m: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        blocks = self._pose6d_anchor_forbidden_block_bounds(dtype)
        if int(blocks.shape[0]) == 0:
            return torch.zeros(points.shape[0], device=self.device, dtype=torch.bool)
        margin = max(float(getattr(self.cfg, "pose6d_anchor_forbidden_block_margin_m", 0.0)), 0.0)
        inflate = torch.full((points.shape[0],), margin, device=self.device, dtype=dtype)
        if bool(getattr(self.cfg, "pose6d_anchor_forbidden_block_inflate_by_object_radius", False)):
            inflate = inflate + 0.5 * dims_m[:, 0].to(dtype=dtype)
        mins = torch.stack((blocks[:, 0], blocks[:, 2], blocks[:, 4]), dim=-1)
        maxs = torch.stack((blocks[:, 1], blocks[:, 3], blocks[:, 5]), dim=-1)
        inside = (
            (points[:, None, :, :] >= mins[None, :, None, :] - inflate[:, None, None, None])
            & (points[:, None, :, :] <= maxs[None, :, None, :] + inflate[:, None, None, None])
        ).all(dim=-1)
        return inside.any(dim=(1, 2))

    def _estimate_dynamic_anchor_local_pos(
        self,
        env_ids: torch.Tensor,
        source_pose_palm: torch.Tensor,
        previous_anchor_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(env_ids.numel())
        if count == 0 or not bool(getattr(self.cfg, "pose6d_anchor_dynamic_enabled", False)):
            return previous_anchor_local, torch.zeros(count, device=self.device, dtype=source_pose_palm.dtype)
        body_names = tuple(str(v) for v in self.hand_spec.force_sensor_bodies)
        available_body_names = tuple(name for name in body_names if name in self.hand_body_name_to_idx)
        body_ids = [self.hand_body_name_to_idx[name] for name in available_body_names]
        if len(body_ids) == 0:
            return previous_anchor_local, torch.zeros(count, device=self.device, dtype=source_pose_palm.dtype)

        body_pos_w = self.hand.data.body_pos_w[env_ids][:, body_ids, :]
        body_pos_palm = self._world_points_to_hand(body_pos_w, env_ids).to(dtype=source_pose_palm.dtype)
        inv_object_quat = quat_conjugate_wxyz(source_pose_palm[:, 3:7]).unsqueeze(1).expand(count, len(body_ids), 4)
        body_pos_obj = quat_apply_wxyz(
            inv_object_quat,
            body_pos_palm - source_pose_palm[:, None, :3],
        )
        half_len = 0.5 * self.active_object_dims_m[env_ids, 1].to(dtype=source_pose_palm.dtype)
        anchor_y = torch.clamp(body_pos_obj[..., 1], -half_len.unsqueeze(-1), half_len.unsqueeze(-1))

        force_norm = torch.zeros((self.num_envs, len(body_ids)), device=self.device, dtype=source_pose_palm.dtype)
        if self.cube_contact_sensor is not None:
            forces_w = self._get_cube_contact_forces_w()
            selected = self._select_contact_forces_w(forces_w, available_body_names)
            force_norm = torch.linalg.norm(selected, dim=-1).to(dtype=source_pose_palm.dtype)
        force_norm = force_norm[env_ids]
        threshold = max(float(getattr(self.cfg, "pose6d_anchor_dynamic_force_threshold_n", 0.05)), 0.0)
        contact_weights = torch.where(force_norm > threshold, force_norm, torch.zeros_like(force_norm))
        contact_weight_sum = contact_weights.sum(dim=-1)
        contact_count = (contact_weights > 0.0).to(dtype=source_pose_palm.dtype).sum(dim=-1)

        mode = str(getattr(self.cfg, "pose6d_anchor_dynamic_source", "contact_or_nearest")).lower()
        if mode not in {"contact", "nearest", "contact_or_nearest"}:
            raise ValueError("pose6d_anchor_dynamic_source must be 'contact', 'nearest', or 'contact_or_nearest'.")
        radius = 0.5 * self.active_object_dims_m[env_ids, 0].to(dtype=source_pose_palm.dtype).unsqueeze(-1)
        radial_dist = torch.linalg.norm(body_pos_obj[..., (0, 2)], dim=-1)
        surface_dist = torch.abs(radial_dist - radius)
        max_dist = max(float(getattr(self.cfg, "pose6d_anchor_dynamic_nearest_max_dist_m", 0.080)), 1.0e-6)
        within_len = torch.abs(body_pos_obj[..., 1]) <= (half_len.unsqueeze(-1) + max_dist)
        nearest_weights = torch.clamp(max_dist - surface_dist, min=0.0)
        nearest_weights = nearest_weights * nearest_weights * within_len.to(dtype=nearest_weights.dtype)

        if mode == "contact":
            weights = contact_weights
        elif mode == "nearest":
            weights = nearest_weights
        else:
            weights = torch.where(contact_weight_sum.unsqueeze(-1) > 1.0e-6, contact_weights, nearest_weights)
        weight_sum = weights.sum(dim=-1)
        estimated_y = torch.sum(anchor_y * weights, dim=-1) / torch.clamp(weight_sum, min=1.0e-6)
        fallback = weight_sum <= 1.0e-6
        if bool(getattr(self.cfg, "pose6d_anchor_dynamic_fallback_to_previous", True)):
            estimated_y = torch.where(fallback, previous_anchor_local[:, 1].to(dtype=estimated_y.dtype), estimated_y)
        else:
            estimated_y = torch.where(fallback, torch.zeros_like(estimated_y), estimated_y)

        smoothing = min(max(float(getattr(self.cfg, "pose6d_anchor_dynamic_smoothing", 0.0)), 0.0), 1.0)
        if smoothing > 0.0:
            estimated_y = smoothing * previous_anchor_local[:, 1].to(dtype=estimated_y.dtype) + (1.0 - smoothing) * estimated_y

        anchor_local = previous_anchor_local.clone()
        anchor_local[:, 1] = torch.clamp(estimated_y, -half_len, half_len)
        return anchor_local, contact_count

    def _evaluate_pose6d_anchor_candidates(
        self,
        target_pose: torch.Tensor,
        target_anchor_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = int(target_pose.shape[0])
        dtype = target_pose.dtype
        if env_ids is None:
            dims_m = self.active_object_dims_m[:count].to(dtype=dtype)
        else:
            dims_m = self.active_object_dims_m[env_ids].to(dtype=dtype)
        workspace_bounds = self._bounds_tensor(
            getattr(self.cfg, "pose6d_anchor_workspace_bounds_m", (0.020, 0.070, -0.040, 0.040, 0.070, 0.130)),
            name="pose6d_anchor_workspace_bounds_m",
            dtype=dtype,
        )
        segment_bounds = self._bounds_tensor(
            getattr(
                self.cfg,
                "pose6d_anchor_segment_centerline_bounds_m",
                (-0.040, 0.190, -0.150, 0.150, 0.020, 0.210),
            ),
            name="pose6d_anchor_segment_centerline_bounds_m",
            dtype=dtype,
        )
        anchor_inside = self._points_inside_bounds(target_anchor_pos, workspace_bounds)

        samples = max(3, int(getattr(self.cfg, "pose6d_anchor_segment_samples", 11)))
        y_unit = torch.linspace(-1.0, 1.0, samples, device=self.device, dtype=dtype)
        local_points = torch.zeros((count, samples, 3), device=self.device, dtype=dtype)
        local_points[..., 1] = y_unit.unsqueeze(0) * (0.5 * dims_m[:, 1]).unsqueeze(-1)
        points = target_pose[:, None, :3] + quat_apply_wxyz(
            target_pose[:, None, 3:7].expand(count, samples, 4),
            local_points,
        )

        overlap_count = self._points_inside_bounds(points.reshape(-1, 3), workspace_bounds).view(count, samples).sum(dim=-1)
        workspace_overlap = overlap_count >= int(getattr(self.cfg, "pose6d_anchor_min_segment_points_inside_workspace", 1))
        feasibility_mode = str(getattr(self.cfg, "pose6d_anchor_feasibility_mode", "legacy")).lower()
        if feasibility_mode in {"cylinder_anchor_sharpa_specific", "sharpa_cylinder_centerline"}:
            forbidden_hit = self._pose6d_anchor_forbidden_block_hits(points, dims_m, dtype)
            ok = anchor_inside & workspace_overlap & (~forbidden_hit)
            min_centerline_z = points[..., 2].min(dim=-1).values
            return ok, {
                "workspace_reject": (~ok).to(dtype=dtype),
                "forbidden_reject": forbidden_hit.to(dtype=dtype),
                "overlap_count": overlap_count.to(dtype=dtype),
                "min_centerline_z": min_centerline_z,
            }
        if feasibility_mode != "legacy":
            raise ValueError(
                "pose6d_anchor_feasibility_mode must be 'legacy' or 'cylinder_anchor_sharpa_specific', "
                f"got {getattr(self.cfg, 'pose6d_anchor_feasibility_mode', None)!r}."
            )
        segment_inside = self._points_inside_bounds(points.reshape(-1, 3), segment_bounds).view(count, samples).all(dim=-1)

        radius = 0.5 * dims_m[:, 0]
        palm_min_x = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_min_x_m", -0.020)) - radius
        palm_max_x = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_x_m", 0.080)) + radius
        palm_max_abs_y = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_abs_y_m", 0.070)) + radius
        palm_max_z = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_z_m", 0.030)) + radius
        in_palm_block = (
            (points[..., 0] >= palm_min_x.unsqueeze(-1))
            & (points[..., 0] <= palm_max_x.unsqueeze(-1))
            & (torch.abs(points[..., 1]) <= palm_max_abs_y.unsqueeze(-1))
            & (points[..., 2] <= palm_max_z.unsqueeze(-1))
        ).any(dim=-1)
        palm_clear = ~in_palm_block

        wrist_min_x = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_min_x_m", -0.080)) - radius
        wrist_max_x = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_x_m", 0.005)) + radius
        wrist_max_abs_y = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_abs_y_m", 0.090)) + radius
        wrist_max_z = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_z_m", 0.090)) + radius
        in_wrist_block = (
            (points[..., 0] >= wrist_min_x.unsqueeze(-1))
            & (points[..., 0] <= wrist_max_x.unsqueeze(-1))
            & (torch.abs(points[..., 1]) <= wrist_max_abs_y.unsqueeze(-1))
            & (points[..., 2] <= wrist_max_z.unsqueeze(-1))
        ).any(dim=-1)
        wrist_clear = ~in_wrist_block

        min_centerline_z = points[..., 2].min(dim=-1).values
        height_ok = min_centerline_z >= float(getattr(self.cfg, "pose6d_anchor_min_centerline_z_m", 0.020))
        ok = anchor_inside & workspace_overlap & segment_inside & palm_clear & wrist_clear & height_ok
        return ok, {
            "workspace_reject": (~ok).to(dtype=dtype),
            "forbidden_reject": (in_palm_block | in_wrist_block).to(dtype=dtype),
            "overlap_count": overlap_count.to(dtype=dtype),
            "min_centerline_z": min_centerline_z,
        }

    def _sample_pose6d_anchor_relaxed_targets(
        self,
        env_ids: torch.Tensor,
        source_pose_palm: torch.Tensor,
        target_quat_palm: torch.Tensor,
        parallel_delta_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = int(env_ids.numel())
        dtype = source_pose_palm.dtype
        source_anchor_local = self.source_anchor_local_pos_m[env_ids].to(dtype=dtype)
        dynamic_contact_count = torch.zeros(count, device=self.device, dtype=dtype)
        if bool(getattr(self.cfg, "pose6d_anchor_dynamic_enabled", False)):
            source_anchor_local, dynamic_contact_count = self._estimate_dynamic_anchor_local_pos(
                env_ids,
                source_pose_palm,
                source_anchor_local,
            )
            self.source_anchor_local_pos_m[env_ids] = source_anchor_local.to(dtype=torch.float32)
            self.pose6d_anchor_dynamic_contact_count[env_ids] = dynamic_contact_count.to(dtype=torch.float32)
            self.pose6d_anchor_dynamic_y_local_m[env_ids] = source_anchor_local[:, 1].to(dtype=torch.float32)
        target_anchor_local = source_anchor_local.clone()

        min_shift = float(getattr(self.cfg, "pose6d_anchor_local_y_shift_min_m", 0.0))
        max_shift = float(getattr(self.cfg, "pose6d_anchor_local_y_shift_max_m", 0.0))
        if max_shift > 0.0 or min_shift > 0.0:
            shift_abs = min_shift + (max_shift - min_shift) * torch.rand(
                (count,), device=self.device, dtype=dtype, generator=self.target_rng
            )
            shift_sign = torch.randint(0, 2, (count,), device=self.device, generator=self.target_rng).to(dtype=dtype)
            shift_sign = shift_sign * 2.0 - 1.0
            target_anchor_local[:, 1] = target_anchor_local[:, 1] + shift_sign * shift_abs
            if bool(getattr(self.cfg, "pose6d_anchor_clip_local_to_object_ends", True)):
                half_len = 0.5 * self.active_object_dims_m[env_ids, 1].to(dtype=dtype)
                target_anchor_local[:, 1] = torch.minimum(torch.maximum(target_anchor_local[:, 1], -half_len), half_len)

        source_anchor_pos = self._anchor_pos_from_pose(source_pose_palm, source_anchor_local)
        accepted = torch.zeros(count, device=self.device, dtype=torch.bool)
        target_pose = source_pose_palm.clone()
        target_pose[:, 3:7] = target_quat_palm
        target_anchor_pos = source_anchor_pos.clone()
        translation_axis = self.target_translation_axis_palm[env_ids].to(dtype=dtype)
        actual_delta = torch.zeros(count, device=self.device, dtype=dtype)
        fail = torch.ones(count, device=self.device, dtype=dtype)
        workspace_reject_count = torch.zeros(count, device=self.device, dtype=dtype)
        forbidden_reject_count = torch.zeros(count, device=self.device, dtype=dtype)
        attempt_count = torch.zeros(count, device=self.device, dtype=dtype)
        overlap_count = torch.zeros(count, device=self.device, dtype=dtype)
        min_centerline_z = torch.zeros(count, device=self.device, dtype=dtype)

        local_y = torch.zeros((count, 3), device=self.device, dtype=dtype)
        local_y[:, 1] = 1.0
        long_axis_all = quat_apply_wxyz(source_pose_palm[:, 3:7], local_y)
        long_axis_all = long_axis_all / torch.clamp(torch.linalg.norm(long_axis_all, dim=-1, keepdim=True), min=1.0e-6)
        perp1_all, perp2_all = self._perpendicular_basis_from_axis(long_axis_all)
        mode = str(getattr(self.cfg, "pose6d_anchor_translation_mode", "long_axis_biased")).lower()
        max_attempts = max(1, int(getattr(self.cfg, "pose6d_anchor_feasible_max_attempts", 16)))
        perp_min = max(float(getattr(self.cfg, "pose6d_anchor_perp_translation_curriculum_start_m", 0.0)), 0.0)
        perp_max_all = torch.clamp(
            self._pose6d_anchor_perp_delta_max_m_for_envs(env_ids, dtype),
            min=perp_min,
        )

        for _ in range(max_attempts):
            unresolved = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            n = int(unresolved.numel())
            if mode == "isotropic_sphere":
                anchor_delta = self._sample_pose6d_translation_axes_palm(n, dtype) * parallel_delta_m[unresolved].unsqueeze(-1)
            elif mode == "long_axis_biased":
                parallel_sign = torch.randint(
                    0, 2, (n,), device=self.device, dtype=torch.long, generator=self.target_rng
                ).to(dtype=dtype)
                parallel_sign = parallel_sign * 2.0 - 1.0
                perp_span = torch.clamp(perp_max_all[unresolved] - perp_min, min=0.0)
                perp_abs_1 = perp_min + perp_span * torch.rand(
                    (n,), device=self.device, dtype=dtype, generator=self.target_rng
                )
                perp_abs_2 = perp_min + perp_span * torch.rand(
                    (n,), device=self.device, dtype=dtype, generator=self.target_rng
                )
                perp_sign_1 = torch.randint(
                    0, 2, (n,), device=self.device, dtype=torch.long, generator=self.target_rng
                ).to(dtype=dtype)
                perp_sign_2 = torch.randint(
                    0, 2, (n,), device=self.device, dtype=torch.long, generator=self.target_rng
                ).to(dtype=dtype)
                perp_delta_1 = (perp_sign_1 * 2.0 - 1.0) * perp_abs_1
                perp_delta_2 = (perp_sign_2 * 2.0 - 1.0) * perp_abs_2
                anchor_delta = (
                    long_axis_all[unresolved] * (parallel_sign * parallel_delta_m[unresolved]).unsqueeze(-1)
                    + perp1_all[unresolved] * perp_delta_1.unsqueeze(-1)
                    + perp2_all[unresolved] * perp_delta_2.unsqueeze(-1)
                )
            else:
                raise ValueError(
                    "pose6d_anchor_translation_mode must be 'long_axis_biased' or 'isotropic_sphere', "
                    f"got {getattr(self.cfg, 'pose6d_anchor_translation_mode', None)!r}."
                )

            cand_anchor = source_anchor_pos[unresolved] + anchor_delta
            cand_pose = source_pose_palm[unresolved].clone()
            cand_pose[:, 3:7] = target_quat_palm[unresolved]
            cand_pose[:, :3] = cand_anchor - quat_apply_wxyz(cand_pose[:, 3:7], target_anchor_local[unresolved])
            ok, info = self._evaluate_pose6d_anchor_candidates(cand_pose, cand_anchor, env_ids[unresolved])
            attempt_count[unresolved] += 1.0
            workspace_reject_count[unresolved] += info["workspace_reject"]
            forbidden_reject_count[unresolved] += info.get("forbidden_reject", torch.zeros_like(info["workspace_reject"]))

            if bool(ok.any()):
                slots = unresolved[ok]
                target_pose[slots] = cand_pose[ok]
                target_anchor_pos[slots] = cand_anchor[ok]
                axis_delta = target_anchor_pos[slots] - source_anchor_pos[slots]
                delta_norm = torch.linalg.norm(axis_delta, dim=-1)
                safe_norm = torch.clamp(delta_norm, min=1.0e-6)
                translation_axis[slots] = axis_delta / safe_norm.unsqueeze(-1)
                actual_delta[slots] = delta_norm
                accepted[slots] = True
                fail[slots] = 0.0
                overlap_count[slots] = info["overlap_count"][ok]
                min_centerline_z[slots] = info["min_centerline_z"][ok]

        failed = ~accepted
        if bool(failed.any()):
            target_anchor_pos[failed] = source_anchor_pos[failed]
            target_pose[failed, :3] = target_anchor_pos[failed] - quat_apply_wxyz(
                target_quat_palm[failed], target_anchor_local[failed]
            )
            target_pose[failed, 3:7] = target_quat_palm[failed]
            translation_axis[failed] = long_axis_all[failed]
            ok, info = self._evaluate_pose6d_anchor_candidates(
                target_pose[failed],
                target_anchor_pos[failed],
                env_ids[failed],
            )
            overlap_count[failed] = info["overlap_count"]
            min_centerline_z[failed] = info["min_centerline_z"]

        workspace_reject = workspace_reject_count / torch.clamp(attempt_count, min=1.0)
        forbidden_reject = forbidden_reject_count / torch.clamp(attempt_count, min=1.0)
        return {
            "target_pose": target_pose,
            "source_anchor_pos": source_anchor_pos,
            "target_anchor_pos": target_anchor_pos,
            "target_anchor_local": target_anchor_local,
            "translation_axis": translation_axis,
            "actual_delta": actual_delta,
            "fail": fail,
            "workspace_reject": workspace_reject,
            "forbidden_reject": forbidden_reject,
            "overlap_count": overlap_count,
            "min_centerline_z": min_centerline_z,
        }

    def _reset_adaptive_pose6d_translation_history(self, object_id: int | None = None) -> None:
        if object_id is not None:
            self.adaptive_translation_frontier_success_history_by_object[object_id].zero_()
            self.adaptive_translation_frontier_count_history_by_object[object_id].zero_()
            self.adaptive_translation_frontier_success_history_count_by_object[object_id] = 0
            self.adaptive_translation_frontier_success_history_index_by_object[object_id] = 0
            self.adaptive_translation_frontier_success_rate_by_object[object_id].zero_()
            self.adaptive_translation_frontier_done_count_by_object[object_id].zero_()
            self.adaptive_translation_frontier_done_total_by_object[object_id].zero_()
            return
        self.adaptive_translation_frontier_success_history.zero_()
        self.adaptive_translation_frontier_count_history.zero_()
        self.adaptive_translation_frontier_success_history_count = 0
        self.adaptive_translation_frontier_success_history_index = 0
        self.adaptive_translation_frontier_success_rate.zero_()
        self.adaptive_translation_frontier_done_count.zero_()
        self.adaptive_translation_frontier_done_total.zero_()

    def _reset_adaptive_chain_pose6d_translation_history(self, object_id: int | None = None) -> None:
        if object_id is not None:
            self.adaptive_chain_translation_frontier_success_history_by_object[object_id].zero_()
            self.adaptive_chain_translation_frontier_count_history_by_object[object_id].zero_()
            self.adaptive_chain_translation_frontier_success_history_count_by_object[object_id] = 0
            self.adaptive_chain_translation_frontier_success_history_index_by_object[object_id] = 0
            self.adaptive_chain_translation_frontier_success_rate_by_object[object_id].zero_()
            self.adaptive_chain_translation_frontier_done_count_by_object[object_id].zero_()
            self.adaptive_chain_translation_frontier_done_total_by_object[object_id].zero_()
            return
        self.adaptive_chain_translation_frontier_success_history.zero_()
        self.adaptive_chain_translation_frontier_count_history.zero_()
        self.adaptive_chain_translation_frontier_success_history_count = 0
        self.adaptive_chain_translation_frontier_success_history_index = 0
        self.adaptive_chain_translation_frontier_success_rate.zero_()
        self.adaptive_chain_translation_frontier_done_count.zero_()
        self.adaptive_chain_translation_frontier_done_total.zero_()

    @staticmethod
    def _curriculum_progress(value: float, start: float, end: float) -> float:
        span = end - start
        if abs(span) <= 1.0e-9:
            return 1.0
        return min(max((value - start) / span, 0.0), 1.0)

    def _first_chain_translation_progress_gap(self) -> torch.Tensor:
        first_start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002))
        first_end = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", first_start))
        chain_start = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_start_m", first_start))
        chain_end = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_end_m", first_end))
        if self._multiobject_per_object_curriculum_enabled():
            first_span = first_end - first_start
            chain_span = chain_end - chain_start
            if abs(first_span) <= 1.0e-9:
                first_progress = torch.ones_like(self.adaptive_translation_delta_max_by_object)
            else:
                first_progress = torch.clamp(
                    (self.adaptive_translation_delta_max_by_object - first_start) / first_span,
                    min=0.0,
                    max=1.0,
                )
            if abs(chain_span) <= 1.0e-9:
                chain_progress = torch.ones_like(self.adaptive_chain_translation_delta_max_by_object)
            else:
                chain_progress = torch.clamp(
                    (self.adaptive_chain_translation_delta_max_by_object - chain_start) / chain_span,
                    min=0.0,
                    max=1.0,
                )
            return (first_progress - chain_progress).mean()
        first_progress = self._curriculum_progress(
            float(self.adaptive_translation_delta_max.item()),
            first_start,
            first_end,
        )
        chain_progress = self._curriculum_progress(
            float(self.adaptive_chain_translation_delta_max.item()),
            chain_start,
            chain_end,
        )
        return torch.tensor(first_progress - chain_progress, device=self.device)

    def _limit_first_translation_curriculum_by_chain(
        self,
        candidate_max: float,
        current_max: float,
        cap: float,
        object_id: int | None = None,
    ) -> tuple[float, bool]:
        if not self._chain_curriculum_sync_enabled():
            return candidate_max, False
        if not self._chain_pose6d_translation_curriculum_enabled():
            return candidate_max, False
        max_lead_fraction = float(getattr(self.cfg, "pose6d_translation_chain_sync_max_lead_fraction", -1.0))
        if max_lead_fraction < 0.0:
            return candidate_max, False

        first_start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002))
        first_end = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", first_start))
        chain_start = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_start_m", first_start))
        chain_end = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_end_m", first_end))
        first_span = first_end - first_start
        if abs(first_span) <= 1.0e-9:
            return candidate_max, False

        if object_id is not None and self._multiobject_per_object_curriculum_enabled():
            chain_current = float(self.adaptive_chain_translation_delta_max_by_object[object_id].item())
        else:
            chain_current = float(self.adaptive_chain_translation_delta_max.item())
        chain_progress = self._curriculum_progress(chain_current, chain_start, chain_end)
        allowed_progress = min(chain_progress + max_lead_fraction, 1.0)
        allowed_max = first_start + allowed_progress * first_span
        sync_cap = max(current_max, min(cap, allowed_max))
        limited_max = min(candidate_max, sync_cap)
        return limited_max, limited_max < candidate_max - 1.0e-9

    def _update_adaptive_pose6d_translation_curriculum(
        self,
        event_envs: torch.Tensor,
        success_mask: torch.Tensor | None = None,
        object_id: int | None = None,
        batch_stats: torch.Tensor | None = None,
        min_done_scale: int = 1,
    ) -> None:
        if not self._adaptive_pose6d_translation_curriculum_enabled():
            return

        if success_mask is None:
            success_mask = self.target_reached

        if self._multiobject_per_object_curriculum_enabled() and object_id is None:
            object_count = len(self.multisize_object_ids)
            statistics = torch.zeros((object_count, 2), device=self.device)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                current_max = self.adaptive_translation_delta_max_by_object[size_id]
                level_match = torch.abs(self.translation_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
                frontier_events = object_envs & (
                    (self.translation_delta_sample_mode == 1) | (self.translation_delta_sample_mode == 3)
                ) & level_match
                if self._chain_pose6d_translation_curriculum_enabled():
                    frontier_events &= self.target_chain_phase == 0
                statistics[size_id, 0] = success_mask[frontier_events].float().sum()
                statistics[size_id, 1] = frontier_events.sum()
            statistics, world_size = self._all_reduce_curriculum_stats(statistics, multiobject=True)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                self._update_adaptive_pose6d_translation_curriculum(
                    object_envs,
                    success_mask,
                    object_id=size_id,
                    batch_stats=statistics[size_id],
                    min_done_scale=world_size,
                )
            self._sync_multiobject_curriculum_summaries()
            return

        if object_id is None:
            adaptive_max = self.adaptive_translation_delta_max
            success_history = self.adaptive_translation_frontier_success_history
            count_history = self.adaptive_translation_frontier_count_history
            history_count = self.adaptive_translation_frontier_success_history_count
            history_index = self.adaptive_translation_frontier_success_history_index
            success_rate = self.adaptive_translation_frontier_success_rate
            done_count = self.adaptive_translation_frontier_done_count
            done_total = self.adaptive_translation_frontier_done_total
            advances = self.adaptive_translation_curriculum_advances
        else:
            adaptive_max = self.adaptive_translation_delta_max_by_object[object_id]
            success_history = self.adaptive_translation_frontier_success_history_by_object[object_id]
            count_history = self.adaptive_translation_frontier_count_history_by_object[object_id]
            history_count = int(
                self.adaptive_translation_frontier_success_history_count_by_object[object_id].item()
            )
            history_index = int(
                self.adaptive_translation_frontier_success_history_index_by_object[object_id].item()
            )
            success_rate = self.adaptive_translation_frontier_success_rate_by_object[object_id]
            done_count = self.adaptive_translation_frontier_done_count_by_object[object_id]
            done_total = self.adaptive_translation_frontier_done_total_by_object[object_id]
            advances = self.adaptive_translation_curriculum_advances_by_object[object_id]

        current_max = float(adaptive_max.item())
        if batch_stats is None:
            level_match = torch.abs(self.translation_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
            frontier_or_anchor = (self.translation_delta_sample_mode == 1) | (self.translation_delta_sample_mode == 3)
            frontier_events = event_envs & frontier_or_anchor & level_match
            if self._chain_pose6d_translation_curriculum_enabled():
                frontier_events &= self.target_chain_phase == 0
            batch_success_count = success_mask[frontier_events].float().sum()
            frontier_count_tensor = frontier_events.sum().to(dtype=batch_success_count.dtype)
            if object_id is None:
                statistics = torch.stack((batch_success_count, frontier_count_tensor))
                statistics, min_done_scale = self._all_reduce_curriculum_stats(statistics)
                batch_success_count = statistics[0]
                frontier_count_tensor = statistics[1]
            frontier_count = int(frontier_count_tensor.item())
        else:
            batch_success_count = batch_stats[0]
            frontier_count = int(batch_stats[1].item())
        done_count.fill_(float(frontier_count))
        if frontier_count <= 0:
            return

        done_total.add_(float(frontier_count))
        history_len = int(success_history.numel())
        success_history[history_index] = batch_success_count.detach()
        count_history[history_index] = float(frontier_count)
        history_index = (history_index + 1) % history_len
        history_count = min(history_count + 1, history_len)
        if object_id is None:
            self.adaptive_translation_frontier_success_history_index = history_index
            self.adaptive_translation_frontier_success_history_count = history_count
        else:
            self.adaptive_translation_frontier_success_history_index_by_object[object_id] = history_index
            self.adaptive_translation_frontier_success_history_count_by_object[object_id] = history_count

        valid_count = history_count
        valid_success_count = success_history[:valid_count].sum()
        valid_event_count = torch.clamp(count_history[:valid_count].sum(), min=1.0)
        frontier_success_rate = valid_success_count / valid_event_count
        success_rate.copy_(frontier_success_rate)

        min_history = max(int(getattr(self.cfg, "pose6d_translation_adaptive_min_history_batches", 5)), 1)
        if valid_count < min_history:
            return

        min_done_episodes = self._curriculum_min_frontier_done_episodes(
            "pose6d_translation_adaptive_min_frontier_done_episodes", 20_000, object_id, min_done_scale
        )
        if float(done_total.item()) < float(min_done_episodes):
            return

        freeze_threshold = float(getattr(self.cfg, "pose6d_translation_adaptive_freeze_threshold", 0.20))
        if float(frontier_success_rate.item()) < freeze_threshold:
            return

        cap = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", current_max))
        step = max(float(getattr(self.cfg, "pose6d_translation_adaptive_step_m", 0.0025)), 0.0)
        threshold = float(getattr(self.cfg, "pose6d_translation_adaptive_success_threshold", 0.50))
        soft_promotion = False
        next_max = None
        if step > 0.0 and current_max < cap - 1.0e-6 and float(frontier_success_rate.item()) >= threshold:
            next_max = min(current_max + step, cap)
        else:
            next_max = self._adaptive_soft_promotion_candidate(
                axis="translation",
                current_max=current_max,
                cap=cap,
                soft_step=max(
                    float(getattr(self.cfg, "adaptive_frontier_soft_promotion_translation_step_m", 0.005)),
                    0.0,
                ),
                success_rate=float(frontier_success_rate.item()),
                history_batches=valid_count,
                min_frontier_done=min_done_episodes,
                frontier_done_total=float(done_total.item()),
                object_id=object_id,
            )
            soft_promotion = next_max is not None
        if next_max is not None:
            next_max, sync_limited = self._limit_first_translation_curriculum_by_chain(
                next_max,
                current_max,
                cap,
                object_id=object_id,
            )
            if sync_limited:
                if object_id is None:
                    self.adaptive_translation_chain_sync_limited_count.add_(1.0)
                else:
                    self.adaptive_translation_chain_sync_limited_count_by_object[object_id].add_(1.0)
            if next_max <= current_max + 1.0e-9:
                return
            adaptive_max.fill_(next_max)
            advances.add_(1.0)
            self._record_adaptive_curriculum_level_change(
                "translation", soft=soft_promotion, object_id=object_id
            )
            self._reset_adaptive_pose6d_translation_history(object_id=object_id)

    def _update_adaptive_chain_pose6d_translation_curriculum(
        self,
        event_envs: torch.Tensor,
        success_mask: torch.Tensor | None = None,
        object_id: int | None = None,
        batch_stats: torch.Tensor | None = None,
        min_done_scale: int = 1,
    ) -> None:
        if not self._chain_pose6d_translation_curriculum_enabled():
            return

        if success_mask is None:
            success_mask = self.target_reached

        if self._multiobject_per_object_curriculum_enabled() and object_id is None:
            object_count = len(self.multisize_object_ids)
            statistics = torch.zeros((object_count, 2), device=self.device)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                current_max = self.adaptive_chain_translation_delta_max_by_object[size_id]
                level_match = torch.abs(self.translation_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
                frontier_events = object_envs & (
                    (self.translation_delta_sample_mode == 1) | (self.translation_delta_sample_mode == 3)
                ) & level_match & (self.target_chain_phase > 0)
                statistics[size_id, 0] = success_mask[frontier_events].float().sum()
                statistics[size_id, 1] = frontier_events.sum()
            statistics, world_size = self._all_reduce_curriculum_stats(statistics, multiobject=True)
            for size_id in range(object_count):
                object_envs = event_envs & (self.active_cube_size_id == size_id)
                self._update_adaptive_chain_pose6d_translation_curriculum(
                    object_envs,
                    success_mask,
                    object_id=size_id,
                    batch_stats=statistics[size_id],
                    min_done_scale=world_size,
                )
            self._sync_multiobject_curriculum_summaries()
            return

        if object_id is None:
            adaptive_max = self.adaptive_chain_translation_delta_max
            success_history = self.adaptive_chain_translation_frontier_success_history
            count_history = self.adaptive_chain_translation_frontier_count_history
            history_count = self.adaptive_chain_translation_frontier_success_history_count
            history_index = self.adaptive_chain_translation_frontier_success_history_index
            success_rate = self.adaptive_chain_translation_frontier_success_rate
            done_count = self.adaptive_chain_translation_frontier_done_count
            done_total = self.adaptive_chain_translation_frontier_done_total
            advances = self.adaptive_chain_translation_curriculum_advances
        else:
            adaptive_max = self.adaptive_chain_translation_delta_max_by_object[object_id]
            success_history = self.adaptive_chain_translation_frontier_success_history_by_object[object_id]
            count_history = self.adaptive_chain_translation_frontier_count_history_by_object[object_id]
            history_count = int(
                self.adaptive_chain_translation_frontier_success_history_count_by_object[object_id].item()
            )
            history_index = int(
                self.adaptive_chain_translation_frontier_success_history_index_by_object[object_id].item()
            )
            success_rate = self.adaptive_chain_translation_frontier_success_rate_by_object[object_id]
            done_count = self.adaptive_chain_translation_frontier_done_count_by_object[object_id]
            done_total = self.adaptive_chain_translation_frontier_done_total_by_object[object_id]
            advances = self.adaptive_chain_translation_curriculum_advances_by_object[object_id]

        current_max = float(adaptive_max.item())
        if batch_stats is None:
            level_match = torch.abs(self.translation_delta_curriculum_max_at_reset - current_max) <= 1.0e-6
            frontier_or_anchor = (self.translation_delta_sample_mode == 1) | (self.translation_delta_sample_mode == 3)
            frontier_events = event_envs & frontier_or_anchor & level_match & (self.target_chain_phase > 0)
            batch_success_count = success_mask[frontier_events].float().sum()
            frontier_count_tensor = frontier_events.sum().to(dtype=batch_success_count.dtype)
            if object_id is None:
                statistics = torch.stack((batch_success_count, frontier_count_tensor))
                statistics, min_done_scale = self._all_reduce_curriculum_stats(statistics)
                batch_success_count = statistics[0]
                frontier_count_tensor = statistics[1]
            frontier_count = int(frontier_count_tensor.item())
        else:
            batch_success_count = batch_stats[0]
            frontier_count = int(batch_stats[1].item())
        done_count.fill_(float(frontier_count))
        if frontier_count <= 0:
            return

        done_total.add_(float(frontier_count))
        history_len = int(success_history.numel())
        success_history[history_index] = batch_success_count.detach()
        count_history[history_index] = float(frontier_count)
        history_index = (history_index + 1) % history_len
        history_count = min(history_count + 1, history_len)
        if object_id is None:
            self.adaptive_chain_translation_frontier_success_history_index = history_index
            self.adaptive_chain_translation_frontier_success_history_count = history_count
        else:
            self.adaptive_chain_translation_frontier_success_history_index_by_object[object_id] = history_index
            self.adaptive_chain_translation_frontier_success_history_count_by_object[object_id] = history_count

        valid_count = history_count
        valid_success_count = success_history[:valid_count].sum()
        valid_event_count = torch.clamp(count_history[:valid_count].sum(), min=1.0)
        frontier_success_rate = valid_success_count / valid_event_count
        success_rate.copy_(frontier_success_rate)

        min_history = max(int(getattr(self.cfg, "pose6d_translation_adaptive_min_history_batches", 5)), 1)
        if valid_count < min_history:
            return

        min_done_episodes = self._curriculum_min_frontier_done_episodes(
            "pose6d_translation_adaptive_min_frontier_done_episodes", 20_000, object_id, min_done_scale
        )
        if float(done_total.item()) < float(min_done_episodes):
            return

        freeze_threshold = float(getattr(self.cfg, "pose6d_translation_adaptive_freeze_threshold", 0.20))
        if float(frontier_success_rate.item()) < freeze_threshold:
            return

        cap = float(getattr(self.cfg, "chain_pose6d_translation_curriculum_end_m", current_max))
        step = max(float(getattr(self.cfg, "pose6d_translation_adaptive_step_m", 0.0025)), 0.0)
        threshold = float(
            getattr(
                self.cfg,
                "chain_pose6d_translation_adaptive_success_threshold",
                getattr(self.cfg, "pose6d_translation_adaptive_success_threshold", 0.50),
            )
        )
        if threshold < 0.0:
            threshold = float(getattr(self.cfg, "pose6d_translation_adaptive_success_threshold", 0.50))
        if step > 0.0 and current_max < cap - 1.0e-6 and float(frontier_success_rate.item()) >= threshold:
            adaptive_max.fill_(min(current_max + step, cap))
            advances.add_(1.0)
            self._reset_adaptive_chain_pose6d_translation_history(object_id=object_id)

    def _hand_root_pos_w(self, dtype: torch.dtype, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        base_pos = torch.tensor(self.hand_root_pos, device=self.device, dtype=dtype).view(1, 3)
        return base_pos.expand(env_ids.numel(), 3) + self.scene.env_origins[env_ids].to(dtype=dtype)

    def _hand_root_quat_w(self, dtype: torch.dtype, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.hand_root_rot_buf.to(dtype=dtype)
        return self.hand_root_rot_buf[env_ids].to(dtype=dtype)

    def _fixed_twist_axis_palm(self, dtype: torch.dtype) -> torch.Tensor:
        axis_name = str(getattr(self.cfg, "dexndm_twist_target_axis", "x")).lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError(
                f"Unsupported fixed dexndm_twist_target_axis={axis_name!r}. Expected 'x', 'y', or 'z'."
            )
        axis = torch.zeros((3,), device=self.device, dtype=dtype)
        axis[{"x": 0, "y": 1, "z": 2}[axis_name]] = 1.0
        return axis

    def _sample_twist_axes_palm(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        axis_name = str(getattr(self.cfg, "dexndm_twist_target_axis", "x")).lower()
        if axis_name in {"random", "any", "any_axis"}:
            return self._random_unit_vectors(count, dtype)
        if axis_name in {"x", "y", "z"}:
            return self._fixed_twist_axis_palm(dtype).view(1, 3).repeat(count, 1)
        raise ValueError(
            f"Unsupported dexndm_twist_target_axis={axis_name!r}. "
            "Expected 'x', 'y', 'z', 'random', or 'any'."
        )

    def _reset_dexndm_wrist_orientation(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        count = int(env_ids.numel())
        base_quat = torch.tensor(self.hand_root_rot, device=self.device, dtype=torch.float32).view(1, 4).expand(count, 4)
        if self.dexndm_omni_wrist_ornt:
            unit_axes = torch.eye(3, device=self.device, dtype=torch.float32)
            x_unit = unit_axes[0].view(1, 3).expand(count, 3)
            y_unit = unit_axes[1].view(1, 3).expand(count, 3)
            z_unit = unit_axes[2].view(1, 3).expand(count, 3)
            fixed_rpy_deg = getattr(self.cfg, "dexndm_fixed_wrist_rpy_deg", None)
            if fixed_rpy_deg is None:
                rpy_over_pi = 2.0 * torch.rand(
                    (count, 3), device=self.device, generator=self.target_rng
                ) - 1.0
            else:
                if len(fixed_rpy_deg) != 3:
                    raise ValueError(
                        "dexndm_fixed_wrist_rpy_deg must contain exactly "
                        f"(roll, pitch, yaw), got {fixed_rpy_deg!r}."
                    )
                fixed_rpy = torch.tensor(
                    fixed_rpy_deg, device=self.device, dtype=torch.float32
                )
                if not bool(torch.isfinite(fixed_rpy).all()):
                    raise ValueError(
                        "dexndm_fixed_wrist_rpy_deg must contain finite values, "
                        f"got {fixed_rpy_deg!r}."
                    )
                rpy_over_pi = (fixed_rpy / 180.0).view(1, 3).expand(count, 3)
            rnd_rot = randomize_rotation_rpy_wxyz(
                rpy_over_pi[:, 0],
                rpy_over_pi[:, 1],
                rpy_over_pi[:, 2],
                x_unit,
                y_unit,
                z_unit,
            )
        else:
            rnd_rot = torch.zeros((count, 4), device=self.device, dtype=torch.float32)
            rnd_rot[:, 0] = 1.0

        self.dexndm_wrist_rnd_rot_buf[env_ids] = rnd_rot
        self.hand_root_rot_buf[env_ids] = normalize_quat_wxyz(quat_mul_wxyz(rnd_rot, base_quat))


    def _hora_rotation_axis_world(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        axis_name = str(self.cfg.hora_rotation_axis).lower()
        if axis_name not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported hora_rotation_axis={axis_name!r}. Expected 'x', 'y', or 'z'.")
        axis_frame = str(self.cfg.hora_rotation_axis_frame)
        if axis_frame not in {"world", "palm"}:
            raise ValueError(
                f"Unsupported hora_rotation_axis_frame={axis_frame!r}. Expected 'world' or 'palm'."
            )
        axis = torch.zeros(3, device=self.device, dtype=dtype)
        axis["xyz".index(axis_name)] = float(self.cfg.hora_rotation_axis_sign)
        if axis_frame == "palm":
            hand_quat = torch.tensor(self.hand_root_rot, device=self.device, dtype=dtype).view(1, 4)
            axis = quat_apply_wxyz(hand_quat, axis.unsqueeze(0)).squeeze(0)
        return axis / torch.clamp(torch.linalg.norm(axis), min=1.0e-6)


    def _dexndm_axis_env_count(self, env_ids: torch.Tensor | None) -> int:
        return self.num_envs if env_ids is None else int(env_ids.numel())

    def _dexndm_rotation_axis_canonical(
        self,
        dtype: torch.dtype = torch.float32,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bool(getattr(self.cfg, "dexndm_randomize_rot_dir", False)):
            axis = self.dexndm_rot_axis_buf if env_ids is None else self.dexndm_rot_axis_buf[env_ids]
            axis = axis.to(dtype=dtype)
        else:
            axis = self._hora_rotation_axis_world(dtype=dtype).view(1, 3).expand(
                self._dexndm_axis_env_count(env_ids), 3
            )
        return axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)

    def _dexndm_rotation_axis_world(
        self,
        dtype: torch.dtype = torch.float32,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        axis = self._dexndm_rotation_axis_canonical(dtype=dtype, env_ids=env_ids)
        if self.dexndm_omni_wrist_ornt:
            rnd_rot = (
                self.dexndm_wrist_rnd_rot_buf if env_ids is None else self.dexndm_wrist_rnd_rot_buf[env_ids]
            ).to(dtype=dtype)
            axis = quat_apply_wxyz(rnd_rot, axis)
        return axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)

    def _dexndm_rotation_axis_palm(
        self,
        dtype: torch.dtype = torch.float32,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        axis_w = self._dexndm_rotation_axis_world(dtype=dtype, env_ids=env_ids)
        hand_quat = self._hand_root_quat_w(dtype=dtype, env_ids=env_ids)
        inv_hand_quat = quat_conjugate_wxyz(hand_quat)
        axis = quat_apply_wxyz(inv_hand_quat, axis_w)
        return axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)

    def _reset_dexndm_rot_axis(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        fixed_axis = self._hora_rotation_axis_world(dtype=torch.float32).view(1, 3).expand(env_ids.numel(), 3)
        if bool(getattr(self.cfg, "dexndm_force_rot_axis", False)):
            fixed_axis = torch.tensor(
                (
                    float(getattr(self.cfg, "dexndm_forced_rot_axis_x", 0.0)),
                    float(getattr(self.cfg, "dexndm_forced_rot_axis_y", 0.0)),
                    float(getattr(self.cfg, "dexndm_forced_rot_axis_z", 1.0)),
                ),
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3).expand(env_ids.numel(), 3)
            if bool(getattr(self.cfg, "dexndm_forced_rot_axis_random_sign", False)):
                sign = torch.where(
                    torch.rand((env_ids.numel(), 1), device=self.device, generator=self.target_rng) < 0.5,
                    -1.0,
                    1.0,
                )
                fixed_axis = fixed_axis * sign
        elif bool(getattr(self.cfg, "dexndm_randomize_rot_dir", False)):
            rnd = 2.0 * torch.rand((env_ids.numel(), 3), device=self.device, generator=self.target_rng) - 1.0
            unit_axes = torch.eye(3, device=self.device, dtype=torch.float32)
            x_unit = unit_axes[0].view(1, 3).expand(env_ids.numel(), 3)
            y_unit = unit_axes[1].view(1, 3).expand(env_ids.numel(), 3)
            z_unit = unit_axes[2].view(1, 3).expand(env_ids.numel(), 3)
            rnd_rot = randomize_rotation_rpy_wxyz(rnd[:, 0], rnd[:, 1], rnd[:, 2], x_unit, y_unit, z_unit)
            fixed_axis = quat_apply_wxyz(rnd_rot, fixed_axis)
        self.dexndm_rot_axis_buf[env_ids] = fixed_axis / torch.clamp(
            torch.linalg.norm(fixed_axis, dim=-1, keepdim=True), min=1.0e-6
        )

    def _dexndm_guided_rotation_delta_quat(
        self,
        env_ids: torch.Tensor,
        guide_index: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        angle = guide_index.to(device=self.device, dtype=dtype) * float(self.cfg.dexndm_guided_rotation_delta_deg)
        angle = angle * torch.pi / 180.0
        axis = self._dexndm_rotation_axis_palm(dtype=dtype, env_ids=env_ids)
        return quat_from_angle_axis_wxyz(angle, axis)

    def _set_dexndm_guided_rotation_pose(self, env_ids: torch.Tensor) -> None:
        base_pose = self.dexndm_guided_rotation_base_pose_palm[env_ids].to(dtype=torch.float32)
        delta = self._dexndm_guided_rotation_delta_quat(
            env_ids,
            self.dexndm_guided_rotation_index[env_ids],
            base_pose.dtype,
        )
        guide_pose = base_pose.clone()
        guide_pose[:, 3:7] = normalize_quat_wxyz(quat_mul_wxyz(delta, base_pose[:, 3:7]))
        self.dexndm_guided_rotation_guide_pose_palm[env_ids] = guide_pose

    def _dexndm_guided_rotation_distance(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            guide_quat = self.dexndm_guided_rotation_guide_pose_palm[:, 3:7]
            current_quat = self.current_pose_palm[:, 3:7]
        else:
            guide_quat = self.dexndm_guided_rotation_guide_pose_palm[env_ids, 3:7]
            current_quat = self.current_pose_palm[env_ids, 3:7]
        return quat_angle_error_wxyz(current_quat, guide_quat)

    def _reset_dexndm_guided_rotation(self, env_ids: torch.Tensor, source_pose_palm: torch.Tensor) -> None:
        self.dexndm_guided_rotation_base_pose_palm[env_ids] = source_pose_palm.to(device=self.device, dtype=torch.float32)
        self.dexndm_guided_rotation_index[env_ids] = 1
        self.dexndm_guided_rotation_advances[env_ids] = 0
        self._set_dexndm_guided_rotation_pose(env_ids)
        rot_dist = self._dexndm_guided_rotation_distance(env_ids).detach()
        self.dexndm_guided_rotation_rot_dist[env_ids] = rot_dist
        self.dexndm_guided_rotation_prev_rot_dist[env_ids] = rot_dist
        self.dexndm_guided_rotation_progress[env_ids] = 0.0
        self.dexndm_guided_rotation_reached[env_ids] = False




    def _update_dexndm_rotp_coef_on_reset(self, env_ids: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "dexndm_rotp_enabled", False)) or env_ids.numel() == 0:
            self.dexndm_cur_rotp_coef = 0.0
            return
        warmup = max(0, int(self.cfg.dexndm_rotp_warmup_resets))
        increase = max(1, int(self.cfg.dexndm_rotp_increase_resets))
        final_coef = float(self.cfg.dexndm_rotp_coef)
        if self.dexndm_rotp_step < warmup:
            self.dexndm_cur_rotp_coef = 0.0
        elif self.dexndm_rotp_step < warmup + increase:
            self.dexndm_cur_rotp_coef = final_coef * float(self.dexndm_rotp_step - warmup) / float(increase)
        else:
            self.dexndm_cur_rotp_coef = final_coef
        self.dexndm_rotp_step += 1

    def _directional_rotation_terms(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            source_pose = self.target_source_pose_palm
            current_pose = self.current_pose_palm
            target_pose = self.target_pose_palm
        else:
            source_pose = self.target_source_pose_palm[env_ids]
            current_pose = self.current_pose_palm[env_ids]
            target_pose = self.target_pose_palm[env_ids]

        goal_delta = quat_mul_wxyz(target_pose[:, 3:7], quat_conjugate_wxyz(source_pose[:, 3:7]))
        goal_rotvec = quat_to_rotvec_wxyz(goal_delta)
        target_angle = torch.linalg.norm(goal_rotvec, dim=-1)
        valid = target_angle > 1.0e-6
        axis = goal_rotvec / torch.clamp(target_angle.unsqueeze(-1), min=1.0e-6)

        current_delta = quat_mul_wxyz(current_pose[:, 3:7], quat_conjugate_wxyz(source_pose[:, 3:7]))
        current_rotvec = quat_to_rotvec_wxyz(current_delta)
        raw_coord = torch.sum(current_rotvec * axis, dim=-1)
        coord = torch.clamp(raw_coord, min=0.0)
        coord = torch.minimum(coord, target_angle)
        coord = torch.where(valid, coord, torch.zeros_like(coord))
        target_angle = torch.where(valid, target_angle, torch.zeros_like(target_angle))
        remaining = torch.clamp(target_angle - coord, min=0.0)
        orthogonal = current_rotvec - raw_coord.unsqueeze(-1) * axis
        orthogonal_error = torch.where(valid, torch.linalg.norm(orthogonal, dim=-1), torch.zeros_like(coord))
        return {
            "coord": coord,
            "target_angle": target_angle,
            "remaining": remaining,
            "orthogonal_error": orthogonal_error,
        }

    def _twist_target_terms(self) -> dict[str, torch.Tensor]:
        axis = self.target_twist_axis_palm.to(dtype=self.current_pose_palm.dtype)
        source_pose = self.target_source_pose_palm
        current_pose = self.current_pose_palm
        target_pose = self.target_pose_palm

        q_source_to_target = quat_mul_wxyz(target_pose[:, 3:7], quat_conjugate_wxyz(source_pose[:, 3:7]))
        target_twist_angle = signed_twist_angle_wxyz(q_source_to_target, axis)

        q_source_to_current = quat_mul_wxyz(current_pose[:, 3:7], quat_conjugate_wxyz(source_pose[:, 3:7]))
        current_twist_angle = signed_twist_angle_wxyz(q_source_to_current, axis)

        twist_remaining_signed = wrap_to_pi(target_twist_angle - current_twist_angle)
        q_target_error = quat_mul_wxyz(target_pose[:, 3:7], quat_conjugate_wxyz(current_pose[:, 3:7]))
        if self._uses_pose6d_anchor_relaxed_targets():
            position_drift = torch.linalg.norm(self.current_anchor_pos_palm - self.target_anchor_pos_palm, dim=-1)
        else:
            position_drift = torch.linalg.norm(current_pose[:, :3] - target_pose[:, :3], dim=-1)
        return {
            "target_twist_angle": target_twist_angle,
            "current_twist_angle": current_twist_angle,
            "twist_remaining": torch.abs(twist_remaining_signed),
            "twist_swing_drift": swing_angle_wxyz(q_target_error, axis),
            "twist_position_drift": position_drift,
        }

    def _pose6d_translation_target_terms(self) -> dict[str, torch.Tensor]:
        if self._uses_pose6d_anchor_relaxed_targets():
            source_pos = self.source_anchor_pos_palm
            current_pos = self.current_anchor_pos_palm
            target_pos = self.target_anchor_pos_palm
        else:
            source_pos = self.target_source_pose_palm[:, :3]
            current_pos = self.current_pose_palm[:, :3]
            target_pos = self.target_pose_palm[:, :3]

        target_offset = target_pos - source_pos
        target_delta = torch.linalg.norm(target_offset, dim=-1)
        safe_delta = torch.clamp(target_delta, min=1.0e-6)
        fallback_axis = self.target_translation_axis_palm.to(dtype=current_pos.dtype)
        axis = target_offset / safe_delta.unsqueeze(-1)
        axis = torch.where((target_delta > 1.0e-6).unsqueeze(-1), axis, fallback_axis)

        current_offset = current_pos - source_pos
        progress_raw = torch.sum(current_offset * axis, dim=-1)
        progress = torch.minimum(torch.clamp(progress_raw, min=0.0), target_delta)
        remaining = target_delta - progress_raw
        orthogonal_vec = current_offset - progress_raw.unsqueeze(-1) * axis
        return {
            "axis": axis,
            "target_delta": target_delta,
            "progress_raw": progress_raw,
            "progress": progress,
            "remaining": remaining,
            "orthogonal_drift": torch.linalg.norm(orthogonal_vec, dim=-1),
            "target_error": torch.linalg.norm(current_pos - target_pos, dim=-1),
        }

    def _update_twist_target_terms(self) -> None:
        terms = self._twist_target_terms()
        self.target_twist_angle[:] = terms["target_twist_angle"].detach()
        self.current_twist_angle[:] = terms["current_twist_angle"].detach()
        self.twist_remaining[:] = terms["twist_remaining"].detach()
        self.twist_swing_drift[:] = terms["twist_swing_drift"].detach()
        self.twist_position_drift[:] = terms["twist_position_drift"].detach()
        translation_terms = self._pose6d_translation_target_terms()
        self.target_translation_axis_palm[:] = translation_terms["axis"].detach()
        self.target_translation_delta[:] = translation_terms["target_delta"].detach()
        self.translation_progress_raw[:] = translation_terms["progress_raw"].detach()
        self.translation_progress[:] = translation_terms["progress"].detach()
        self.translation_remaining[:] = translation_terms["remaining"].detach()
        self.translation_orthogonal_drift[:] = translation_terms["orthogonal_drift"].detach()
        self.pose6d_anchor_error_m[:] = translation_terms["target_error"].detach()

    def _get_pose6d_anchor_obs(self) -> torch.Tensor:
        self._update_twist_target_terms()
        return torch.cat((self.current_anchor_pos_palm, self.target_anchor_pos_palm), dim=-1).to(dtype=torch.float32)

    def _get_dexndm_twist_target_obs(self) -> torch.Tensor:
        self._update_twist_target_terms()
        target_pose_obs = self.target_pose_palm.clone()
        target_pose_obs[:, 3:7] = positive_quat_wxyz(target_pose_obs[:, 3:7])
        twist_scalars = torch.stack(
            (
                self.target_twist_angle,
                self.current_twist_angle,
                self.twist_remaining,
                self.twist_swing_drift,
                self.twist_position_drift,
            ),
            dim=-1,
        )
        obs_parts = [target_pose_obs]
        if bool(getattr(self.cfg, "dexndm_zpos_twist_target_axis_obs_enabled", False)):
            obs_parts.append(self.target_twist_axis_palm.to(dtype=torch.float32))
        obs_parts.append(twist_scalars)
        return torch.cat(obs_parts, dim=-1).to(dtype=torch.float32)

    def _reset_directional_rotation_progress(
        self,
        env_ids: torch.Tensor,
        source_pose_palm: torch.Tensor | None = None,
    ) -> None:
        if source_pose_palm is not None:
            self.target_source_pose_palm[env_ids] = source_pose_palm.to(device=self.device, dtype=torch.float32)
        terms = self._directional_rotation_terms(env_ids)
        self.directional_rotation_coord[env_ids] = terms["coord"]
        self.best_directional_rotation_coord[env_ids] = terms["coord"]
        self.directional_rotation_progress[env_ids] = 0.0
        self.directional_rotation_target_angle[env_ids] = terms["target_angle"]
        self.directional_rotation_remaining[env_ids] = terms["remaining"]
        self.directional_rotation_orthogonal_error[env_ids] = terms["orthogonal_error"]




    def _active_dims_for_pose_batch(self, pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is not None:
            return self.active_object_dims_m[env_ids].to(device=pose.device, dtype=pose.dtype)
        if pose.shape[0] == self.num_envs:
            return self.active_object_dims_m.to(device=pose.device, dtype=pose.dtype)
        raise ValueError("env_ids must be provided for variable-dimension object metrics on sub-batches.")

    def _variable_cuboid_keypoints_from_pose(self, pose: torch.Tensor, dims_m: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor(
            [
                [x, y, z]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            device=pose.device,
            dtype=pose.dtype,
        )
        local = signs.unsqueeze(0) * (0.5 * dims_m).unsqueeze(1)
        quat = pose[:, 3:7].unsqueeze(1).expand(-1, signs.shape[0], -1)
        return quat_apply_wxyz(quat, local) + pose[:, None, :3]

    def _single_object_variable_dims_enabled(self) -> bool:
        return bool(getattr(self, "object_scale_randomization_enabled", False)) and not bool(
            getattr(self, "multiobject_enabled", False)
        )

    def _object_keypoints_from_pose(self, pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self._single_object_variable_dims_enabled() or bool(getattr(self, "multiobject_enabled", False)):
            dims = self._active_dims_for_pose_batch(pose, env_ids)
            return self._variable_cuboid_keypoints_from_pose(pose, dims)
        return cube_keypoints_from_pose(pose, side_length=self.object_keypoint_side_length_m)

    def _object_pose_distance(
        self,
        current_pose: torch.Tensor,
        goal_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._single_object_variable_dims_enabled() or bool(getattr(self, "multiobject_enabled", False)):
            dims = self._active_dims_for_pose_batch(current_pose, env_ids)
            current_keypoints = self._variable_cuboid_keypoints_from_pose(current_pose, dims)
            goal_keypoints = self._variable_cuboid_keypoints_from_pose(
                goal_pose.to(device=current_pose.device, dtype=current_pose.dtype),
                dims,
            )
            return torch.max(torch.linalg.norm(current_keypoints - goal_keypoints, dim=-1), dim=-1).values
        return cube_pose_distance(current_pose, goal_pose, side_length=self.object_keypoint_side_length_m)




    def _contact_sensor_body_names_from_cfg(self) -> tuple[str, ...]:
        if (
            bool(
                getattr(
                    self.cfg,
                    "dexndm_grasp_wrench_deficit_penalty_enabled",
                    getattr(self.cfg, "dexndm_grasp_wrench_maintenance_reward_enabled", False),
                )
            )
            and str(getattr(self.cfg, "dexndm_grasp_wrench_body_mode", "primary_10")).lower()
            == "all_collision_capped_secondary"
        ):
            if self.hand_spec.hand_id != "sharpa_wave_right":
                raise ValueError("All-collision CSGWCD sensor expansion is Sharpa-only.")
            return SHARPA_CSGWCD_COLLISION_BODIES
        observation_profile = str(self.cfg.observation_profile).lower()
        if observation_profile in {"hora_aligned", "hora_strict", "dexndm_zpos", "dexndm_omni"} or bool(
            self.cfg.enable_all_body_contact_force_obs
        ):
            return self.hand_spec.all_contact_bodies
        return self.hand_spec.contact_sensor_bodies

    def _multiobject_spawn_pos_m(
        self,
        size_id: int,
        object_count: int,
        dims_m: tuple[float, float, float],
        mode: str,
    ) -> tuple[float, float, float]:
        y_center = 0.5 * float(object_count - 1)
        if mode == "primitive":
            park_y_attr = "primitive_park_y_spacing_m"
            park_x_attr = "primitive_park_x_m"
        elif mode == "multicuboid":
            park_y_attr = "multicuboid_park_y_spacing_m"
            park_x_attr = "multicuboid_park_x_m"
        else:
            park_y_attr = "multisize_cube_park_y_spacing_m"
            park_x_attr = "multisize_cube_park_x_m"
        return (
            float(getattr(self.cfg, park_x_attr, -0.22)),
            (float(size_id) - y_center) * float(getattr(self.cfg, park_y_attr, 0.22)),
            0.5 * float(dims_m[2]),
        )

    def _multiobject_pool_size(self) -> int:
        """Return the number of configured objects in the active object pool (1 if single-object)."""

        for enabled_key, ids_key in (
            ("primitive_multi_shape_enabled", "primitive_object_ids"),
            ("multicuboid_enabled", "multicuboid_object_ids"),
            ("multisize_cube_enabled", "multisize_cube_object_ids"),
        ):
            if bool(getattr(self.cfg, enabled_key, False)):
                return len(tuple(getattr(self.cfg, ids_key, ()) or ()))
        return 1

    def _scale_randomization_prim_path(self) -> str:
        """Prim path whose per-env USD scale is randomized (supports single-object pools)."""

        if bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)):
            return "/World/envs/env_.*/Object"

        for enabled_key, ids_key in (
            ("primitive_multi_shape_enabled", "primitive_object_ids"),
            ("multicuboid_enabled", "multicuboid_object_ids"),
            ("multisize_cube_enabled", "multisize_cube_object_ids"),
        ):
            if bool(getattr(self.cfg, enabled_key, False)):
                ids = tuple(str(v) for v in (getattr(self.cfg, ids_key, ()) or ()))
                if len(ids) != 1:
                    return self.cfg.cube_cfg.prim_path
                object_id = ids[0]
                prim_name = (
                    MULTISIZE_CUBE_SPECS[object_id][2]
                    if object_id in MULTISIZE_CUBE_SPECS
                    else multiobject_prim_name(object_id)
                )
                return f"/World/envs/env_.*/{prim_name}"
        return self.cfg.cube_cfg.prim_path

    def _sample_startup_object_scales_cpu(self, num_envs: int) -> torch.Tensor:
        """Sample one fixed object scale per env for startup-only USD scale randomization."""

        if not bool(getattr(self.cfg, "object_scale_randomization_enabled", False)):
            return torch.ones(num_envs, dtype=torch.float32)
        scale_range = tuple(float(v) for v in getattr(self.cfg, "object_scale_range", (1.0, 1.0)))
        if len(scale_range) != 2 or scale_range[0] <= 0.0 or scale_range[1] < scale_range[0]:
            raise ValueError(f"Invalid object_scale_range={scale_range!r}.")
        if self._multiobject_pool_size() > 1 and not bool(
            getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)
        ):
            raise ValueError(
                "object_scale_randomization_enabled with a multi-object pool requires "
                "primitive_single_asset_per_env_enabled=True so each environment owns one fixed asset; "
                f"got pool size {self._multiobject_pool_size()}."
            )
        if bool(getattr(self.cfg.scene, "replicate_physics", True)):
            raise ValueError(
                "object_scale_randomization_enabled requires cfg.scene.replicate_physics=False because "
                "rigid-body scale randomization is parsed by PhysX before simulation startup."
            )
        generator = torch.Generator(device="cpu").manual_seed(int(getattr(self.cfg, "seed", 0) or 0) + 707)
        return scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(
            (num_envs,), generator=generator, dtype=torch.float32
        )

    def _sample_startup_nearcube_geometry_cpu(
        self, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample fixed XYZ dimensions/scales and branch IDs for near-cube GeoDR."""

        dims, branch_ids = sample_nearcube_geometry_cpu(
            num_envs,
            mixture_weights=tuple(
                float(v) for v in getattr(self.cfg, "object_nearcube_geometry_mixture_weights", (0.6, 0.2, 0.2))
            ),
            nominal_side_range_m=tuple(
                float(v) for v in getattr(self.cfg, "object_nearcube_nominal_side_range_m", (0.039, 0.041))
            ),
            nominal_side_mode_m=float(getattr(self.cfg, "object_nearcube_nominal_side_mode_m", 0.040)),
            wide_side_range_m=tuple(
                float(v) for v in getattr(self.cfg, "object_nearcube_wide_side_range_m", (0.038, 0.042))
            ),
            cuboid_dim_range_m=tuple(
                float(v) for v in getattr(self.cfg, "object_nearcube_cuboid_dim_range_m", (0.038, 0.042))
            ),
            maximum_aspect_ratio=float(
                getattr(self.cfg, "object_nearcube_maximum_aspect_ratio", 1.08)
            ),
            seed=int(getattr(self.cfg, "seed", 0) or 0) + 1707,
        )
        base_dims = torch.tensor(
            tuple(float(v) for v in getattr(self.cfg, "object_nearcube_base_dims_m", (0.04, 0.04, 0.04))),
            dtype=torch.float32,
        )
        if base_dims.shape != (3,) or bool((base_dims <= 0.0).any()):
            raise ValueError("object_nearcube_base_dims_m must contain three positive values.")
        return dims / base_dims, dims, branch_ids

    def _apply_prestartup_single_object_scale_randomization(self) -> None:
        """Apply per-env uniform object scale before PhysX starts parsing the scene."""

        scalar_enabled = bool(getattr(self.cfg, "object_scale_randomization_enabled", False))
        geometry_enabled = bool(getattr(self.cfg, "object_nearcube_geometry_randomization_enabled", False))
        # This hook runs from _setup_scene() inside DirectRLEnv.__init__, before
        # post-super runtime attributes such as object_keypoint_side_length_m exist.
        base_side_length_m = float(getattr(self.cfg, "object_keypoint_side_length_m", 0.04))
        num_envs = int(self.cfg.scene.num_envs)
        if bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)):
            configured_dims = torch.as_tensor(
                tuple(getattr(self.cfg, "primitive_dims_m", ()) or ()), dtype=torch.float32
            )
            if configured_dims.ndim != 2 or configured_dims.shape[1] != 3 or configured_dims.shape[0] == 0:
                raise ValueError(
                    "primitive_single_asset_per_env_enabled scale DR requires non-empty primitive_dims_m [object,3]."
                )
            fixed_ids = torch.arange(num_envs, dtype=torch.long).remainder(configured_dims.shape[0])
            base_dims_cpu = configured_dims[fixed_ids]
        else:
            base_dims_cpu = torch.full((num_envs, 3), base_side_length_m, dtype=torch.float32)
        if scalar_enabled and geometry_enabled:
            raise ValueError(
                "object_scale_randomization_enabled and "
                "object_nearcube_geometry_randomization_enabled are mutually exclusive."
            )
        if not scalar_enabled and not geometry_enabled:
            self._prestartup_object_scale_values_cpu = torch.ones(num_envs, dtype=torch.float32)
            self._prestartup_object_scale_xyz_values_cpu = torch.ones((num_envs, 3), dtype=torch.float32)
            self._prestartup_object_dims_m_cpu = base_dims_cpu.clone()
            self._prestartup_object_geometry_branch_ids_cpu = torch.zeros(num_envs, dtype=torch.long)
            return

        if geometry_enabled:
            scale_xyz, dims, branch_ids = self._sample_startup_nearcube_geometry_cpu(num_envs)
            scales = scale_xyz.prod(dim=-1).pow(1.0 / 3.0)
        else:
            scales = self._sample_startup_object_scales_cpu(num_envs)
            scale_xyz = scales.unsqueeze(-1).repeat(1, 3)
            dims = base_dims_cpu * scale_xyz
            branch_ids = torch.zeros(num_envs, dtype=torch.long)
        self._prestartup_object_scale_values_cpu = scales
        self._prestartup_object_scale_xyz_values_cpu = scale_xyz
        self._prestartup_object_dims_m_cpu = dims
        self._prestartup_object_geometry_branch_ids_cpu = branch_ids
        object_prim_path = self._scale_randomization_prim_path()
        prim_paths = sim_utils.find_matching_prim_paths(object_prim_path)
        if len(prim_paths) != num_envs:
            raise RuntimeError(
                f"Expected {num_envs} object prims for object scale randomization, found {len(prim_paths)} "
                f"for prim_path={object_prim_path!r}."
            )

        stage = get_current_stage()
        with Sdf.ChangeBlock():
            for prim_path, scale in zip(prim_paths, scale_xyz.tolist(), strict=True):
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + ".xformOp:scale",
                        Sdf.ValueTypeNames.Double3,
                    )
                scale_spec.default = Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
                if not has_scale_attr:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec,
                            UsdGeom.Tokens.xformOpOrder,
                            Sdf.ValueTypeNames.TokenArray,
                        )
                    op_order_spec.default = Vt.TokenArray(
                        ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                    )

    def _setup_scene(self):
        contact_sensor_required = True
        self.cfg.cube_cfg.spawn.activate_contact_sensors = True
        if os.environ.get("POISE_SHARED_HAND_GEOMETRY", "1") != "0":
            from poise.sim.shared_geometry import reference_hand_definition
            # Spawn the first hand through the unchanged Isaac Lab spawner so
            # collision/rigid-body/articulation overrides are fully applied.
            # Share the resolved USD definition before the standard copy
            # operation; no live environment is used as an inheritance source.
            original_spawn = self.cfg.robot_cfg.spawn.func

            def spawn_shared_hand(path, cfg, **kwargs):
                from isaaclab.sim.utils import clone

                @clone
                def spawn_one(source_path, source_cfg, **source_kwargs):
                    prim = original_spawn(source_path, source_cfg, **source_kwargs)
                    self._hand_geometry_layers = reference_hand_definition(get_current_stage(), source_path)
                    return prim

                return spawn_one(path, cfg, **kwargs)

            hand_cfg = self.cfg.robot_cfg.copy()
            hand_cfg.spawn.func = spawn_shared_hand
        else:
            hand_cfg = self.cfg.robot_cfg
        self.hand = Articulation(hand_cfg)
        self.multisize_cubes: dict[str, RigidObject] = {}
        self.multisize_cube_contact_sensors: dict[str, ContactSensor] = {}
        multiobject_enabled = bool(
            getattr(self.cfg, "multisize_cube_enabled", False)
            or getattr(self.cfg, "multicuboid_enabled", False)
            or getattr(self.cfg, "primitive_multi_shape_enabled", False)
        )
        if bool(getattr(self.cfg, "multisize_cube_enabled", False)):
            configured_object_ids = tuple(str(item) for item in getattr(self.cfg, "multisize_cube_object_ids", ()))
            configured_side_lengths = tuple(float(item) for item in getattr(self.cfg, "multisize_cube_side_lengths_m", ()))
            for size_id, object_id in enumerate(configured_object_ids):
                if object_id not in MULTISIZE_CUBE_SPECS:
                    raise ValueError(f"Unsupported multisize cube object_id={object_id!r}.")
                _side_length, cfg_factory, prim_name = MULTISIZE_CUBE_SPECS[object_id]
                cube_cfg = cfg_factory(
                    prim_path=f"/World/envs/env_.*/{prim_name}",
                    init_pos=self._multiobject_spawn_pos_m(
                        size_id,
                        len(configured_object_ids),
                        (_side_length, _side_length, _side_length),
                        "multisize_cube",
                    ),
                    disable_gravity=False,
                )
                cube_cfg.spawn.activate_contact_sensors = contact_sensor_required
                self.multisize_cubes[object_id] = RigidObject(cube_cfg)
            self.cube = self.multisize_cubes.get("cube_4cm", next(iter(self.multisize_cubes.values())))
            self._multisize_configured_side_lengths = configured_side_lengths
        elif bool(getattr(self.cfg, "multicuboid_enabled", False)):
            configured_object_ids = tuple(str(item) for item in getattr(self.cfg, "multicuboid_object_ids", ()))
            configured_dims = tuple(
                tuple(float(v) for v in item) for item in getattr(self.cfg, "multicuboid_dims_m", ())
            )
            if len(configured_object_ids) != len(configured_dims):
                raise ValueError("multicuboid_object_ids and multicuboid_dims_m must have the same length.")
            shared_materials_enabled = bool(getattr(self.cfg, "multicuboid_shared_materials_enabled", True))
            shared_physics_material_path = str(
                getattr(self.cfg, "multicuboid_shared_physics_material_path", "/Looks/poise_multicuboid_shared_physics")
            )
            visual_material_path_prefix = str(
                getattr(self.cfg, "multicuboid_visual_material_path_prefix", "/Looks/poise_multicuboid_visual_")
            )
            for size_id, (object_id, dims_m) in enumerate(zip(configured_object_ids, configured_dims, strict=True)):
                prim_name = multiobject_prim_name(object_id)
                color = MULTICUBOID_VISUAL_COLORS[size_id % len(MULTICUBOID_VISUAL_COLORS)]
                visual_material_path = "material"
                physics_material_path = "material"
                if shared_materials_enabled:
                    visual_material_path = f"{visual_material_path_prefix}{object_id}"
                    physics_material_path = shared_physics_material_path
                cuboid_cfg = get_cuboid_cfg(
                    prim_path=f"/World/envs/env_.*/{prim_name}",
                    dims_m=dims_m,
                    init_pos=self._multiobject_spawn_pos_m(
                        size_id,
                        len(configured_object_ids),
                        dims_m,
                        "multicuboid",
                    ),
                    disable_gravity=False,
                    visual_color=color,
                    visual_material_path=visual_material_path,
                    physics_material_path=physics_material_path,
                )
                cuboid_cfg.spawn.activate_contact_sensors = contact_sensor_required
                self.multisize_cubes[object_id] = RigidObject(cuboid_cfg)
            self.cube = next(iter(self.multisize_cubes.values()))
        elif bool(getattr(self.cfg, "primitive_multi_shape_enabled", False)):
            configured_object_ids = tuple(str(item) for item in getattr(self.cfg, "primitive_object_ids", ()))
            configured_dims = tuple(
                tuple(float(v) for v in item) for item in getattr(self.cfg, "primitive_dims_m", ())
            )
            if len(configured_object_ids) != len(configured_dims):
                raise ValueError("primitive_object_ids and primitive_dims_m must have the same length.")
            shared_materials_enabled = bool(getattr(self.cfg, "primitive_shared_materials_enabled", True))
            shared_physics_material_path = str(
                getattr(self.cfg, "primitive_shared_physics_material_path", "/Looks/poise_primitive_shared_physics")
            )
            visual_material_path_prefix = str(
                getattr(self.cfg, "primitive_visual_material_path_prefix", "/Looks/poise_primitive_visual_")
            )
            single_asset_per_env = bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False))
            single_asset_spawn_cfgs = []
            for size_id, (object_id, dims_m) in enumerate(zip(configured_object_ids, configured_dims, strict=True)):
                prim_name = multiobject_prim_name(object_id)
                color = MULTICUBOID_VISUAL_COLORS[size_id % len(MULTICUBOID_VISUAL_COLORS)]
                visual_material_path = "material"
                physics_material_path = "material"
                if shared_materials_enabled:
                    visual_material_path = f"{visual_material_path_prefix}{object_id}"
                    physics_material_path = shared_physics_material_path
                if object_id in MULTISIZE_CUBE_SPECS:
                    _side_length, cfg_factory, _prim_name = MULTISIZE_CUBE_SPECS[object_id]
                    object_cfg = cfg_factory(
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                    )
                elif object_id.startswith(("cube_", "cuboid_")):
                    object_cfg = get_cuboid_cfg(
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        dims_m=dims_m,
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                        visual_color=color,
                        visual_material_path=visual_material_path,
                        physics_material_path=physics_material_path,
                    )
                elif object_id.startswith("cylinder_"):
                    if is_local_y_cylinder_object_id(object_id):
                        object_cfg = get_cylinder_cfg(
                            prim_path=f"/World/envs/env_.*/{prim_name}",
                            radius_m=0.5 * float(dims_m[0]),
                            height_m=float(dims_m[1]),
                            axis="Y",
                            init_pos=self._multiobject_spawn_pos_m(
                                size_id,
                                len(configured_object_ids),
                                dims_m,
                                "primitive",
                            ),
                            disable_gravity=False,
                            visual_color=color,
                            visual_material_path=visual_material_path,
                            physics_material_path=physics_material_path,
                        )
                    else:
                        if abs(float(dims_m[0]) - float(dims_m[1])) > 1.0e-6:
                            raise ValueError(f"Cylinder dimensions must be diameter, diameter, height; got {dims_m!r}.")
                        object_cfg = get_cylinder_cfg(
                            prim_path=f"/World/envs/env_.*/{prim_name}",
                            radius_m=0.5 * float(dims_m[0]),
                            height_m=float(dims_m[2]),
                            init_pos=self._multiobject_spawn_pos_m(
                                size_id,
                                len(configured_object_ids),
                                dims_m,
                                "primitive",
                            ),
                            disable_gravity=False,
                            visual_color=color,
                            visual_material_path=visual_material_path,
                            physics_material_path=physics_material_path,
                        )
                elif object_id.startswith("sphere_"):
                    if abs(float(dims_m[0]) - float(dims_m[1])) > 1.0e-6 or abs(float(dims_m[0]) - float(dims_m[2])) > 1.0e-6:
                        raise ValueError(f"Sphere dimensions must be diameter, diameter, diameter; got {dims_m!r}.")
                    object_cfg = get_sphere_cfg(
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        radius_m=0.5 * float(dims_m[0]),
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                        visual_color=color,
                        visual_material_path=visual_material_path,
                        physics_material_path=physics_material_path,
                    )
                elif object_id.startswith("capsule_"):
                    if abs(float(dims_m[0]) - float(dims_m[1])) > 1.0e-6:
                        raise ValueError(f"Capsule dimensions must be diameter, diameter, height; got {dims_m!r}.")
                    object_cfg = get_capsule_cfg(
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        radius_m=0.5 * float(dims_m[0]),
                        height_m=capsule_shaft_height_from_bbox(dims_m),
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                        visual_color=color,
                        visual_material_path=visual_material_path,
                        physics_material_path=physics_material_path,
                    )
                elif object_id in SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS:
                    object_cfg = get_sharpa_manual_hex_prism_cfg(
                        object_id=object_id,
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                    )
                elif object_id in SHARPA_MANUAL_CONVEX_OBJECT_IDS:
                    object_cfg = get_sharpa_manual_convex_object_cfg(
                        object_id=object_id,
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                    )
                elif is_irregular_object_id(object_id):
                    object_cfg = get_irregular_object_cfg(
                        object_id=object_id,
                        prim_path=f"/World/envs/env_.*/{prim_name}",
                        dims_m=dims_m,
                        init_pos=self._multiobject_spawn_pos_m(
                            size_id,
                            len(configured_object_ids),
                            dims_m,
                            "primitive",
                        ),
                        disable_gravity=False,
                    )
                else:
                    raise ValueError(f"Unsupported primitive object_id={object_id!r}.")
                object_cfg.spawn.activate_contact_sensors = contact_sensor_required
                if single_asset_per_env:
                    single_asset_spawn_cfgs.append(object_cfg.spawn)
                else:
                    self.multisize_cubes[object_id] = RigidObject(object_cfg)
            if single_asset_per_env:
                mixed_object_cfg = RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Object",
                    spawn=sim_utils.MultiAssetSpawnerCfg(
                        assets_cfg=single_asset_spawn_cfgs,
                        random_choice=False,
                        activate_contact_sensors=contact_sensor_required,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),
                )
                self.cube = RigidObject(mixed_object_cfg)
                self.multisize_cubes = {object_id: self.cube for object_id in configured_object_ids}
            else:
                self.cube = next(iter(self.multisize_cubes.values()))
        else:
            self.cube = RigidObject(self.cfg.cube_cfg)
        floor = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=self.cfg.ground_cfg.physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12)),
        )
        floor.func("/World/ground", floor, translation=(0.0, 0.0, -0.05))
        self.cube_contact_sensor = None
        if contact_sensor_required:
            body_names = self._contact_sensor_body_names_from_cfg()
            max_contact_data_count_per_prim = max(
                1, int(getattr(self.cfg, "contact_sensor_max_contact_data_count_per_prim", 4))
            )
            if multiobject_enabled:
                if bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)):
                    sensor_cfg = ContactSensorCfg(
                        prim_path="/World/envs/env_.*/Object",
                        filter_prim_paths_expr=[
                            f"/World/envs/env_.*/Robot/{body_name}" for body_name in body_names
                        ],
                        max_contact_data_count_per_prim=max_contact_data_count_per_prim,
                    )
                    sensor = ContactSensor(sensor_cfg)
                    self.multisize_cube_contact_sensors = {
                        object_id: sensor for object_id in self.multisize_cubes
                    }
                    self.scene.sensors["object_hand_contact_sensor"] = sensor
                    self.cube_contact_sensor = sensor
                else:
                    for object_id in self.multisize_cubes:
                        prim_name = (
                            MULTISIZE_CUBE_SPECS[object_id][2]
                            if object_id in MULTISIZE_CUBE_SPECS
                            else multiobject_prim_name(object_id)
                        )
                        sensor_cfg = ContactSensorCfg(
                            prim_path=f"/World/envs/env_.*/{prim_name}",
                            filter_prim_paths_expr=[
                                f"/World/envs/env_.*/Robot/{body_name}" for body_name in body_names
                            ],
                            max_contact_data_count_per_prim=max_contact_data_count_per_prim,
                        )
                        sensor = ContactSensor(sensor_cfg)
                        self.multisize_cube_contact_sensors[object_id] = sensor
                        self.scene.sensors[f"{object_id}_hand_contact_sensor"] = sensor
                    self.cube_contact_sensor = self.multisize_cube_contact_sensors.get(
                        "cube_4cm", next(iter(self.multisize_cube_contact_sensors.values()))
                    )
            else:
                sensor_cfg = ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Cube",
                    filter_prim_paths_expr=[
                        f"/World/envs/env_.*/Robot/{body_name}" for body_name in body_names
                    ],
                    max_contact_data_count_per_prim=max_contact_data_count_per_prim,
                )
                self.cube_contact_sensor = ContactSensor(sensor_cfg)
                self.scene.sensors["cube_hand_contact_sensor"] = self.cube_contact_sensor
        if self.tactile_enabled:
            # A single object sensor already filters every relevant hand link.
            # Enable points and two physics samples without changing collisions.
            sensors = set(self.multisize_cube_contact_sensors.values()) | {self.cube_contact_sensor}
            for sensor in sensors:
                sensor.cfg.track_contact_points = True
                sensor.cfg.history_length = 2
                sensor.cfg.max_contact_data_count_per_prim = int(
                    getattr(self.cfg, "tactile_contact_capacity", 64))
        # InteractiveScene already creates independent envs for heterogeneous physics scenes.
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.hand
        if multiobject_enabled:
            if bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)):
                self.scene.rigid_objects["object"] = self.cube
            else:
                for object_id, cube in self.multisize_cubes.items():
                    self.scene.rigid_objects[object_id] = cube
        else:
            self.scene.rigid_objects["cube"] = self.cube
        if self._multiobject_pool_size() == 1 or bool(
            getattr(self.cfg, "primitive_single_asset_per_env_enabled", False)
        ):
            self._apply_prestartup_single_object_scale_randomization()
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_cube_inertia_override(self) -> None:
        """Optionally set a DexNDM-style cube inertia tensor in PhysX."""

        if not bool(getattr(self.cfg, "cube_inertia_override_enabled", False)):
            return
        diag_value = float(getattr(self.cfg, "cube_inertia_diag_kg_m2", 0.0))
        if diag_value <= 0.0:
            return

        cubes = (
            (self.cube,)
            if bool(getattr(self.cfg, "primitive_single_asset_per_env_enabled", False))
            else self.multisize_cubes.values()
            if bool(
                getattr(self.cfg, "multisize_cube_enabled", False)
                or getattr(self.cfg, "multicuboid_enabled", False)
                or getattr(self.cfg, "primitive_multi_shape_enabled", False)
            )
            and hasattr(self, "multisize_cubes")
            else (self.cube,)
        )
        for cube in cubes:
            inertias = cube.root_physx_view.get_inertias()
            new_inertias = torch.zeros_like(inertias)
            diag = torch.tensor((diag_value, diag_value, diag_value), device=self.device, dtype=new_inertias.dtype)
            if new_inertias.ndim == 2:
                new_inertias[:, 0] = diag[0]
                new_inertias[:, 4] = diag[1]
                new_inertias[:, 8] = diag[2]
                env_ids = torch.arange(new_inertias.shape[0], device="cpu", dtype=torch.int32)
            elif new_inertias.ndim == 3:
                new_inertias[:, :, 0] = diag[0]
                new_inertias[:, :, 4] = diag[1]
                new_inertias[:, :, 8] = diag[2]
                env_ids = torch.arange(new_inertias.shape[0], device="cpu", dtype=torch.int32)
            else:
                raise RuntimeError(f"Unexpected cube inertia tensor shape: {tuple(new_inertias.shape)}")
            cube.root_physx_view.set_inertias(new_inertias, env_ids)
            cube.data.default_inertia = new_inertias.clone()

    @staticmethod
    def _validate_range_pair(values: Any, name: str, *, allow_zero: bool = True) -> tuple[float, float]:
        pair = tuple(float(v) for v in values)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly two values, got {pair!r}.")
        lo, hi = pair
        min_allowed = 0.0 if allow_zero else 1.0e-12
        if lo < min_allowed or hi < lo:
            raise ValueError(f"Invalid {name}={pair!r}.")
        return lo, hi

    def _initialize_object_domain_randomization_buffers(self) -> None:
        """Initialize per-env single-object sim2real randomization buffers."""

        self.object_scale_randomization_enabled = bool(
            getattr(self.cfg, "object_scale_randomization_enabled", False)
        )
        self.object_nearcube_geometry_randomization_enabled = bool(
            getattr(self.cfg, "object_nearcube_geometry_randomization_enabled", False)
        )
        if self.object_scale_randomization_enabled and self.object_nearcube_geometry_randomization_enabled:
            raise ValueError(
                "object_scale_randomization_enabled and "
                "object_nearcube_geometry_randomization_enabled are mutually exclusive."
            )
        self.object_mass_randomization_enabled = bool(
            getattr(self.cfg, "object_mass_randomization_enabled", False)
        )
        self.object_mass_cuboid_density_correlation_enabled = bool(
            getattr(self.cfg, "object_mass_cuboid_density_correlation_enabled", False)
        )
        self.object_mass_sampling_mode = str(
            getattr(self.cfg, "object_mass_sampling_mode", "uniform")
        ).strip().lower()
        if self.object_mass_sampling_mode not in {
            "uniform",
            "log_uniform",
            "nominal_log_uniform_mixture",
        }:
            raise ValueError(
                "object_mass_sampling_mode must be 'uniform', 'log_uniform', or "
                f"'nominal_log_uniform_mixture'; got {self.object_mass_sampling_mode!r}."
            )
        if self.object_mass_sampling_mode == "nominal_log_uniform_mixture":
            nominal_weight = float(getattr(self.cfg, "object_mass_nominal_mixture_weight", 0.5))
            if not 0.0 <= nominal_weight <= 1.0:
                raise ValueError("object_mass_nominal_mixture_weight must be in [0, 1].")
            wide_lo, wide_hi = self._validate_range_pair(
                getattr(self.cfg, "object_mass_range_kg", (0.020, 0.020)),
                "object_mass_range_kg",
                allow_zero=False,
            )
            nominal_lo, nominal_hi = self._validate_range_pair(
                getattr(self.cfg, "object_mass_nominal_range_kg", (0.016, 0.024)),
                "object_mass_nominal_range_kg",
                allow_zero=False,
            )
            if nominal_lo < wide_lo or nominal_hi > wide_hi:
                raise ValueError(
                    "object_mass_nominal_range_kg must lie within object_mass_range_kg."
                )
        self.object_mass_bucket_metrics_enabled = bool(
            getattr(self.cfg, "object_mass_bucket_metrics_enabled", False)
        )
        self.object_mass_bucket_edges_kg = tuple(
            float(value)
            for value in getattr(
                self.cfg,
                "object_mass_bucket_edges_kg",
                (0.010, 0.020, 0.040, 0.070, 0.100),
            )
        )
        if self.object_mass_bucket_metrics_enabled:
            # Reuse the pure helper so runtime metrics and unit tests share
            # exactly the same boundary and validation contract.
            mass_bucket_ids(
                torch.empty(0, device=self.device, dtype=torch.float32),
                self.object_mass_bucket_edges_kg,
            )
        if self.object_mass_cuboid_density_correlation_enabled and not self.object_mass_randomization_enabled:
            raise ValueError(
                "object_mass_cuboid_density_correlation_enabled requires "
                "object_mass_randomization_enabled=True."
            )
        self.object_com_randomization_enabled = bool(
            getattr(self.cfg, "object_com_randomization_enabled", False)
        )
        self.object_inertia_from_sampled_mass_dims_enabled = bool(
            getattr(self.cfg, "object_inertia_from_sampled_mass_dims_enabled", False)
        )
        self.object_mesh_mass_properties_enabled = bool(
            getattr(self.cfg, "object_mesh_mass_properties_enabled", False)
        )
        self.object_inertia_from_compound_cuboids_enabled = bool(
            getattr(self.cfg, "object_inertia_from_compound_cuboids_enabled", False)
        )
        self.object_inertia_compound_multiplier_randomization_enabled = bool(
            getattr(
                self.cfg,
                "object_inertia_compound_multiplier_randomization_enabled",
                False,
            )
        )
        self.object_inertia_real_to_legacy_randomization_enabled = bool(
            getattr(
                self.cfg,
                "object_inertia_real_to_legacy_randomization_enabled",
                False,
            )
        )
        self.object_inertia_mass_size_scaling_enabled = bool(
            getattr(self.cfg, "object_inertia_mass_size_scaling_enabled", False)
        )
        inertia_modes_enabled = sum(
            (
                self.object_inertia_from_sampled_mass_dims_enabled,
                self.object_mesh_mass_properties_enabled,
                self.object_inertia_from_compound_cuboids_enabled,
                self.object_inertia_real_to_legacy_randomization_enabled,
                self.object_inertia_mass_size_scaling_enabled,
            )
        )
        if inertia_modes_enabled > 1:
            raise ValueError(
                "object_inertia_from_sampled_mass_dims_enabled, "
                "object_mesh_mass_properties_enabled, "
                "object_inertia_from_compound_cuboids_enabled, and "
                "object_inertia_real_to_legacy_randomization_enabled, and "
                "object_inertia_mass_size_scaling_enabled are mutually exclusive."
            )
        if self.object_inertia_real_to_legacy_randomization_enabled:
            upper_inertia = float(
                getattr(
                    self.cfg,
                    "object_inertia_real_to_legacy_upper_diag_kg_m2",
                    1.0e-4,
                )
            )
            if not math.isfinite(upper_inertia) or upper_inertia <= 0.0:
                raise ValueError(
                    "object_inertia_real_to_legacy_upper_diag_kg_m2 must be "
                    "finite and positive."
                )
            inertia_distribution = str(
                getattr(
                    self.cfg,
                    "object_inertia_real_to_legacy_distribution",
                    "log_uniform",
                )
            ).strip().lower()
            if inertia_distribution not in {"uniform", "log_uniform"}:
                raise ValueError(
                    "object_inertia_real_to_legacy_distribution must be "
                    "'uniform' or 'log_uniform'."
                )
        if self.object_inertia_compound_multiplier_randomization_enabled:
            if not self.object_inertia_from_compound_cuboids_enabled:
                raise ValueError(
                    "object_inertia_compound_multiplier_randomization_enabled requires "
                    "object_inertia_from_compound_cuboids_enabled=True."
                )
            self._validate_range_pair(
                getattr(self.cfg, "object_inertia_compound_multiplier_range", (1.0, 1.0)),
                "object_inertia_compound_multiplier_range",
                allow_zero=False,
            )
            compound_multiplier_distribution = str(
                getattr(
                    self.cfg,
                    "object_inertia_compound_multiplier_distribution",
                    "log_uniform",
                )
            ).strip().lower()
            if compound_multiplier_distribution not in {"uniform", "log_uniform"}:
                raise ValueError(
                    "object_inertia_compound_multiplier_distribution must be "
                    "'uniform' or 'log_uniform'."
                )
        if (
            self.object_inertia_mass_size_scaling_enabled
            and not self.object_mass_randomization_enabled
        ):
            raise ValueError(
                "object_inertia_mass_size_scaling_enabled requires "
                "object_mass_randomization_enabled=True."
            )
        self.object_inertia_compound_cuboid_dims_m = torch.empty(
            (0, 3), device=self.device, dtype=torch.float32
        )
        self.object_inertia_compound_cuboid_centers_m = torch.empty(
            (0, 3), device=self.device, dtype=torch.float32
        )
        self.object_inertia_compound_cuboid_relative_density = torch.empty(
            (0,), device=self.device, dtype=torch.float32
        )
        if self.object_inertia_from_compound_cuboids_enabled:
            dims = torch.as_tensor(
                getattr(self.cfg, "object_inertia_compound_cuboid_dims_m", ()),
                device=self.device,
                dtype=torch.float32,
            )
            centers = torch.as_tensor(
                getattr(self.cfg, "object_inertia_compound_cuboid_centers_m", ()),
                device=self.device,
                dtype=torch.float32,
            )
            relative_density = torch.as_tensor(
                getattr(self.cfg, "object_inertia_compound_cuboid_relative_density", ()),
                device=self.device,
                dtype=torch.float32,
            )
            if dims.ndim != 2 or dims.shape[-1] != 3 or int(dims.shape[0]) == 0:
                raise ValueError(
                    "object_inertia_compound_cuboid_dims_m must have shape (P, 3)."
                )
            if centers.shape != dims.shape:
                raise ValueError(
                    "object_inertia_compound_cuboid_centers_m must match the part dimensions."
                )
            if relative_density.numel() == 0:
                relative_density = torch.ones(
                    dims.shape[0], device=self.device, dtype=torch.float32
                )
            if relative_density.shape != dims.shape[:-1]:
                raise ValueError(
                    "object_inertia_compound_cuboid_relative_density must have shape (P,)."
                )
            if bool((dims <= 0.0).any()) or bool((relative_density <= 0.0).any()):
                raise ValueError("Compound cuboid dimensions and relative densities must be positive.")
            self.object_inertia_compound_cuboid_dims_m = dims
            self.object_inertia_compound_cuboid_centers_m = centers
            self.object_inertia_compound_cuboid_relative_density = relative_density
        self.object_friction_randomization_enabled = bool(
            getattr(self.cfg, "object_friction_randomization_enabled", False)
        )
        self.random_external_force_enabled = bool(
            getattr(self.cfg, "random_external_force_enabled", False)
        )
        self.random_external_wrench_mode = str(
            getattr(self.cfg, "random_external_wrench_mode", "legacy_uniform_force")
        ).strip().lower()
        if self.random_external_wrench_mode not in {"legacy_uniform_force", "simtoolreal"}:
            raise ValueError(
                "random_external_wrench_mode must be 'legacy_uniform_force' or "
                f"'simtoolreal', got {self.random_external_wrench_mode!r}."
            )
        self.random_external_torque_sampling_mode = str(
            getattr(
                self.cfg,
                "random_external_torque_sampling_mode",
                "mass_scaled_gaussian",
            )
        ).strip().lower()
        if self.random_external_torque_sampling_mode not in {
            "mass_scaled_gaussian",
            "inertia_scaled_angular_acceleration",
        }:
            raise ValueError(
                "random_external_torque_sampling_mode must be "
                "'mass_scaled_gaussian' or "
                "'inertia_scaled_angular_acceleration', got "
                f"{self.random_external_torque_sampling_mode!r}."
            )
        angular_acceleration_scale = float(
            getattr(
                self.cfg,
                "random_external_angular_acceleration_scale_rad_s2",
                0.0,
            )
        )
        if angular_acceleration_scale < 0.0:
            raise ValueError(
                "random_external_angular_acceleration_scale_rad_s2 must be non-negative."
            )
        any_object_dr = bool(
            self.object_scale_randomization_enabled
            or self.object_nearcube_geometry_randomization_enabled
            or self.object_mass_randomization_enabled
            or self.object_com_randomization_enabled
            or self.object_friction_randomization_enabled
            or self.object_mesh_mass_properties_enabled
            or self.object_inertia_from_compound_cuboids_enabled
            or self.object_inertia_real_to_legacy_randomization_enabled
            or self.random_external_force_enabled
        )
        if (
            any_object_dr
            and self.multiobject_enabled
            and len(self.multisize_object_ids) != 1
            and not self.object_mesh_mass_properties_enabled
        ):
            raise ValueError("Multiobject domain randomization requires mesh mass properties.")

        prestartup_scales = getattr(self, "_prestartup_object_scale_values_cpu", None)
        if prestartup_scales is None:
            prestartup_scales = torch.ones(self.num_envs, dtype=torch.float32)
        if int(prestartup_scales.numel()) != self.num_envs:
            raise RuntimeError(
                f"Prestartup object scale buffer has {int(prestartup_scales.numel())} rows, expected {self.num_envs}."
            )
        self.object_scale_values = prestartup_scales.to(device=self.device, dtype=torch.float32)
        prestartup_scale_xyz = getattr(self, "_prestartup_object_scale_xyz_values_cpu", None)
        if prestartup_scale_xyz is None:
            prestartup_scale_xyz = self.object_scale_values.detach().cpu().unsqueeze(-1).repeat(1, 3)
        if tuple(prestartup_scale_xyz.shape) != (self.num_envs, 3):
            raise RuntimeError(
                "Prestartup object XYZ scale buffer must have shape "
                f"{(self.num_envs, 3)}, got {tuple(prestartup_scale_xyz.shape)}."
            )
        self.object_scale_xyz_values = prestartup_scale_xyz.to(device=self.device, dtype=torch.float32)
        prestartup_dims = getattr(self, "_prestartup_object_dims_m_cpu", None)
        if prestartup_dims is None:
            prestartup_dims = float(self.object_keypoint_side_length_m) * prestartup_scale_xyz
        self.object_geometry_branch_id = getattr(
            self,
            "_prestartup_object_geometry_branch_ids_cpu",
            torch.zeros(self.num_envs, dtype=torch.long),
        ).to(device=self.device, dtype=torch.long)
        base_side = float(self.object_keypoint_side_length_m)
        if (
            self.object_scale_randomization_enabled
            or self.object_nearcube_geometry_randomization_enabled
        ) and not self.multiobject_enabled:
            dims = prestartup_dims.to(device=self.device, dtype=torch.float32)
            equivalent_side = dims.prod(dim=-1).pow(1.0 / 3.0)
            self.active_cube_side_length_m[:] = equivalent_side
            self.active_cube_size_norm[:] = equivalent_side / CUBE_4CM_SIDE_LENGTH_M - 1.0
            self.active_object_dims_m[:] = dims
            self.active_object_dim_norm[:] = dims / CUBE_4CM_SIDE_LENGTH_M - 1.0

        self.scale_conditioned_cache_bucket_ids_by_env = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        if self.scale_conditioned_stable_cache_enabled:
            if self.multiobject_enabled:
                raise ValueError("Scale-conditioned stable-cache reset is incompatible with multiobject mode.")
            self.scale_conditioned_cache_bucket_ids_by_env = scale_cache_bucket_ids(
                self.active_cube_side_length_m,
                self.scale_conditioned_cache_interval_upper_m,
                minimum_size_m=float(self.scale_conditioned_cache_interval_lower_m[0]),
                maximum_size_m=float(self.scale_conditioned_cache_interval_upper_m[-1]),
            )
            self.active_cube_size_id[:] = self.scale_conditioned_cache_bucket_ids_by_env
        self.geometry_conditioned_cache_anchor_ids_by_env = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        if self.geometry_conditioned_stable_cache_enabled:
            if self.multiobject_enabled:
                raise ValueError("Geometry-conditioned stable-cache reset is incompatible with multiobject mode.")
            self.geometry_conditioned_cache_anchor_ids_by_env = geometry_cache_anchor_ids(
                self.active_object_dims_m,
                self.geometry_conditioned_cache_anchor_dims_m,
                metric_scale_m=self.geometry_conditioned_cache_metric_scale_m,
            )
            self.active_cube_size_id[:] = self.geometry_conditioned_cache_anchor_ids_by_env

        self.object_randomized_mass_kg = torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "object_mass_nominal_kg", 0.020)),
            device=self.device,
            dtype=torch.float32,
        )
        self.object_mass_sampling_branch_id = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.object_randomized_com_offset_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_friction_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.object_external_force_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_external_force_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.object_external_torque_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.object_external_target_angular_acceleration_w = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.object_external_torque_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.object_external_force_prob = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.object_external_torque_prob = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.object_external_force_reference_prob = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.object_external_torque_reference_prob = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.object_randomized_inertia_diag_kg_m2 = cuboid_inertia_diagonal(
            self.object_randomized_mass_kg,
            self.active_object_dims_m,
        )
        self.object_inertia_real_to_legacy_multiplier = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.object_inertia_compound_multiplier = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        initial_inertias = self.cube.root_physx_view.get_inertias()
        if initial_inertias.ndim == 2:
            initial_inertia_tensor_b = initial_inertias.reshape(-1, 3, 3)
        elif initial_inertias.ndim == 3:
            initial_inertia_tensor_b = initial_inertias[:, 0, :].reshape(-1, 3, 3)
        else:
            raise RuntimeError(
                f"Unexpected object inertia tensor shape: {tuple(initial_inertias.shape)}"
            )
        self.object_randomized_inertia_tensor_b_kg_m2 = initial_inertia_tensor_b.to(
            device=self.device,
            dtype=torch.float32,
        ).clone()
        self.object_randomized_base_com_b = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._object_default_coms = self.cube.root_physx_view.get_coms().clone()

        if self.object_friction_randomization_enabled:
            self._apply_startup_friction_randomization()

    def _articulation_body_shape_ranges(self, asset: Articulation) -> list[tuple[int, int]]:
        """Return material-shape index ranges for each articulation body."""

        ranges: list[tuple[int, int]] = []
        start = 0
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            stop = start + int(link_physx_view.max_shapes)
            ranges.append((start, stop))
            start = stop
        return ranges

    def _apply_startup_friction_randomization(self) -> None:
        """Assign object/hand friction buckets using Sharpa-RL-Lab-style material ratios."""

        lo, hi = self._validate_range_pair(
            getattr(self.cfg, "object_friction_scale_range", (1.0, 1.0)),
            "object_friction_scale_range",
            allow_zero=False,
        )
        num_buckets = max(1, int(getattr(self.cfg, "object_friction_randomization_buckets", 64)))
        if num_buckets == 1 or hi <= lo:
            self.object_friction_scale = torch.full(
                (self.num_envs,),
                0.5 * (lo + hi),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            bucket_values = torch.linspace(lo, hi, num_buckets, device=self.device, dtype=torch.float32)
            bucket_ids = torch.randint(
                0,
                num_buckets,
                (self.num_envs,),
                device=self.device,
                generator=self.target_rng,
            )
            self.object_friction_scale = bucket_values[bucket_ids]
        object_mu = self.object_friction_scale * float(getattr(self.cfg, "object_friction_base", 0.5))
        elastomer_mu = self.object_friction_scale * float(getattr(self.cfg, "elastomer_friction_base", 0.8))
        metal_mu = self.object_friction_scale * float(getattr(self.cfg, "metal_friction_base", 0.1))
        env_ids_cpu = torch.arange(self.num_envs, device="cpu", dtype=torch.long)

        object_assets = (
            (self.cube,)
            if not self.multiobject_enabled or self.primitive_single_asset_per_env_enabled
            else self.multisize_cubes.values()
        )
        for object_asset in object_assets:
            object_materials = object_asset.root_physx_view.get_material_properties()
            object_mu_m = object_mu.to(device=object_materials.device, dtype=object_materials.dtype)
            object_materials[:, :, 0] = object_mu_m.unsqueeze(-1)
            object_materials[:, :, 1] = object_mu_m.unsqueeze(-1)
            object_materials[:, :, 2] = 0.0
            object_asset.root_physx_view.set_material_properties(object_materials, env_ids_cpu)

        hand_materials = self.hand.root_physx_view.get_material_properties()
        elastomer_mu_m = elastomer_mu.to(device=hand_materials.device, dtype=hand_materials.dtype)
        metal_mu_m = metal_mu.to(device=hand_materials.device, dtype=hand_materials.dtype)
        body_ranges = self._articulation_body_shape_ranges(self.hand)
        body_name_to_idx = {name: idx for idx, name in enumerate(self.hand.body_names)}
        elastomer_body_ids = {
            body_name_to_idx[name] for name in self.hand_spec.force_sensor_bodies if name in body_name_to_idx
        }
        for body_id, (start, stop) in enumerate(body_ranges):
            if stop <= start:
                continue
            mu = elastomer_mu_m if body_id in elastomer_body_ids else metal_mu_m
            hand_materials[:, start:stop, 0] = mu.unsqueeze(-1)
            hand_materials[:, start:stop, 1] = mu.unsqueeze(-1)
            hand_materials[:, start:stop, 2] = 0.0
        self.hand.root_physx_view.set_material_properties(hand_materials, env_ids_cpu)

    def _randomize_single_object_mass_com_on_reset(
        self, env_ids: torch.Tensor, rigid_object: RigidObject | None = None
    ) -> None:
        """Apply reset-time mass, mesh inertia, and CoM randomization."""

        cube = self.cube if rigid_object is None else rigid_object

        if not (
            self.object_mass_randomization_enabled
            or self.object_com_randomization_enabled
            or self.object_inertia_from_sampled_mass_dims_enabled
            or self.object_mesh_mass_properties_enabled
            or self.object_inertia_from_compound_cuboids_enabled
            or self.object_inertia_real_to_legacy_randomization_enabled
        ):
            return
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        mass = self.object_randomized_mass_kg[env_ids].clone()
        if self.object_mass_randomization_enabled:
            lo, hi = self._validate_range_pair(
                getattr(self.cfg, "object_mass_range_kg", (0.020, 0.020)),
                "object_mass_range_kg",
                allow_zero=False,
            )
            if self.object_mass_cuboid_density_correlation_enabled:
                mass, density = sample_density_correlated_cuboid_masses(
                    self.active_object_dims_m[env_ids],
                    density_range_kg_m3=tuple(
                        float(v) for v in getattr(self.cfg, "object_mass_density_range_kg_m3", (250.0, 375.0))
                    ),
                    distribution=str(getattr(self.cfg, "object_mass_density_distribution", "uniform")),
                    generator=self.target_rng,
                )
                if not hasattr(self, "object_randomized_density_kg_m3"):
                    self.object_randomized_density_kg_m3 = torch.zeros(
                        self.num_envs, device=self.device, dtype=torch.float32
                    )
                self.object_randomized_density_kg_m3[env_ids] = density
            elif bool(getattr(self.cfg, "object_mass_scale_correlation_enabled", False)):
                scale_lo, scale_hi = self._validate_range_pair(
                    getattr(self.cfg, "object_scale_range", (1.0, 1.0)),
                    "object_scale_range",
                    allow_zero=False,
                )
                nominal_side = float(self.object_keypoint_side_length_m)
                mass, density = sample_density_correlated_cube_masses(
                    self.active_cube_side_length_m[env_ids],
                    minimum_mass_kg=lo,
                    maximum_mass_kg=hi,
                    minimum_side_length_m=nominal_side * scale_lo,
                    maximum_side_length_m=nominal_side * scale_hi,
                    distribution=str(
                        getattr(self.cfg, "object_mass_density_distribution", "log_uniform")
                    ),
                    generator=self.target_rng,
                )
                if not hasattr(self, "object_randomized_density_kg_m3"):
                    self.object_randomized_density_kg_m3 = torch.zeros(
                        self.num_envs, device=self.device, dtype=torch.float32
                    )
                self.object_randomized_density_kg_m3[env_ids] = density
            else:
                sampling_mode = self.object_mass_sampling_mode
                if sampling_mode == "nominal_log_uniform_mixture":
                    mass, branch_id = sample_nominal_log_uniform_masses(
                        env_ids.numel(),
                        wide_range_kg=(lo, hi),
                        nominal_range_kg=tuple(
                            float(v)
                            for v in getattr(
                                self.cfg,
                                "object_mass_nominal_range_kg",
                                (0.016, 0.024),
                            )
                        ),
                        nominal_weight=float(
                            getattr(self.cfg, "object_mass_nominal_mixture_weight", 0.5)
                        ),
                        device=self.device,
                        generator=self.target_rng,
                        dtype=torch.float32,
                    )
                    self.object_mass_sampling_branch_id[env_ids] = branch_id
                else:
                    uniform = torch.rand(
                        (env_ids.numel(),),
                        device=self.device,
                        generator=self.target_rng,
                        dtype=torch.float32,
                    )
                    if sampling_mode == "log_uniform":
                        mass = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * uniform)
                    else:
                        mass = lo + (hi - lo) * uniform
                    self.object_mass_sampling_branch_id[env_ids] = 0
            masses = cube.root_physx_view.get_masses()
            env_ids_mass = env_ids.detach().to(device=masses.device, dtype=torch.long)
            mass_m = mass.to(device=masses.device, dtype=masses.dtype)
            masses[env_ids_mass, 0] = mass_m
            cube.root_physx_view.set_masses(masses, env_ids_cpu)

            self.object_randomized_mass_kg[env_ids] = mass

        inertias = cube.root_physx_view.get_inertias()
        env_ids_inertia = env_ids.detach().to(device=inertias.device, dtype=torch.long)
        compound_com = None
        if self.object_mesh_mass_properties_enabled:
            object_indices = self.active_cube_size_id[env_ids]
            scale = self.object_scale_values[env_ids].to(dtype=mass.dtype)
            inertia_tensor = (
                self.multiobject_mesh_inertia_unit_mass_kg_m2[object_indices].to(dtype=mass.dtype)
                * mass.view(-1, 1, 1)
                * scale.square().view(-1, 1, 1)
            )
            low, high = self._validate_range_pair(
                getattr(self.cfg, "object_mesh_inertia_multiplier_range", (1.0, 1.0)),
                "object_mesh_inertia_multiplier_range",
                allow_zero=False,
            )
            inertia_multiplier = sample_inertia_multipliers(
                env_ids.numel(),
                low=low,
                high=high,
                distribution=str(getattr(self.cfg, "object_mesh_inertia_multiplier_distribution", "log_uniform")),
                device=self.device,
                dtype=mass.dtype,
                generator=self.target_rng,
            )
            inertia_tensor = inertia_tensor * inertia_multiplier.view(-1, 1, 1)
            tensor_m = inertia_tensor.to(device=inertias.device, dtype=inertias.dtype)
            if inertias.ndim == 2:
                inertias[env_ids_inertia] = tensor_m.reshape(-1, 9)
            elif inertias.ndim == 3:
                inertias[env_ids_inertia] = tensor_m.reshape(-1, 1, 9).expand(-1, inertias.shape[1], -1)
            else:
                raise RuntimeError(f"Unexpected object inertia tensor shape: {tuple(inertias.shape)}")
            self.object_randomized_inertia_diag_kg_m2[env_ids] = torch.diagonal(
                inertia_tensor, dim1=-2, dim2=-1
            )
            self.object_inertia_compound_multiplier[env_ids] = inertia_multiplier
            compound_com = self.multiobject_mesh_com_m[object_indices] * scale.unsqueeze(-1)
            self.object_randomized_base_com_b[env_ids] = compound_com
        elif self.object_inertia_from_compound_cuboids_enabled:
            scale = self.object_scale_values[env_ids].to(dtype=mass.dtype)
            part_dims = (
                self.object_inertia_compound_cuboid_dims_m.to(dtype=mass.dtype).unsqueeze(0)
                * scale.view(-1, 1, 1)
            )
            part_centers = (
                self.object_inertia_compound_cuboid_centers_m.to(dtype=mass.dtype).unsqueeze(0)
                * scale.view(-1, 1, 1)
            )
            relative_density = self.object_inertia_compound_cuboid_relative_density.to(
                dtype=mass.dtype
            ).unsqueeze(0).expand(env_ids.numel(), -1)
            compound_com, inertia_tensor = compound_cuboid_mass_properties(
                mass,
                part_dims,
                part_centers,
                relative_density=relative_density,
            )
            if self.object_inertia_compound_multiplier_randomization_enabled:
                low, high = self._validate_range_pair(
                    getattr(self.cfg, "object_inertia_compound_multiplier_range", (1.0, 1.0)),
                    "object_inertia_compound_multiplier_range",
                    allow_zero=False,
                )
                inertia_multiplier = sample_inertia_multipliers(
                    env_ids.numel(),
                    low=low,
                    high=high,
                    distribution=str(
                        getattr(
                            self.cfg,
                            "object_inertia_compound_multiplier_distribution",
                            "log_uniform",
                        )
                    ),
                    device=self.device,
                    dtype=mass.dtype,
                    generator=self.target_rng,
                )
                inertia_tensor = inertia_tensor * inertia_multiplier.view(-1, 1, 1)
                self.object_inertia_compound_multiplier[env_ids] = inertia_multiplier
            tensor_m = inertia_tensor.to(device=inertias.device, dtype=inertias.dtype)
            if inertias.ndim == 2:
                inertias[env_ids_inertia] = tensor_m.reshape(-1, 9)
            elif inertias.ndim == 3:
                inertias[env_ids_inertia] = tensor_m.reshape(-1, 1, 9).expand(
                    -1, inertias.shape[1], -1
                )
            else:
                raise RuntimeError(f"Unexpected object inertia tensor shape: {tuple(inertias.shape)}")
            inertia_diag = torch.diagonal(inertia_tensor, dim1=-2, dim2=-1)
            self.object_randomized_inertia_diag_kg_m2[env_ids] = inertia_diag
            self.object_randomized_base_com_b[env_ids] = compound_com
        elif self.object_inertia_real_to_legacy_randomization_enabled:
            inertia_diag, inertia_multiplier = sample_real_to_legacy_cuboid_inertia_diagonal(
                mass,
                self.active_object_dims_m[env_ids],
                upper_diag_kg_m2=float(
                    getattr(
                        self.cfg,
                        "object_inertia_real_to_legacy_upper_diag_kg_m2",
                        1.0e-4,
                    )
                ),
                distribution=str(
                    getattr(
                        self.cfg,
                        "object_inertia_real_to_legacy_distribution",
                        "log_uniform",
                    )
                ),
                generator=self.target_rng,
            )
            inertias[env_ids_inertia] = 0.0
            diag_m = inertia_diag.to(device=inertias.device, dtype=inertias.dtype)
            if inertias.ndim == 2:
                inertias[env_ids_inertia, 0] = diag_m[:, 0]
                inertias[env_ids_inertia, 4] = diag_m[:, 1]
                inertias[env_ids_inertia, 8] = diag_m[:, 2]
            elif inertias.ndim == 3:
                inertias[env_ids_inertia, :, 0] = diag_m[:, 0].unsqueeze(-1)
                inertias[env_ids_inertia, :, 4] = diag_m[:, 1].unsqueeze(-1)
                inertias[env_ids_inertia, :, 8] = diag_m[:, 2].unsqueeze(-1)
            else:
                raise RuntimeError(f"Unexpected object inertia tensor shape: {tuple(inertias.shape)}")
            self.object_randomized_inertia_diag_kg_m2[env_ids] = inertia_diag
            self.object_inertia_real_to_legacy_multiplier[env_ids] = inertia_multiplier
        elif self.object_inertia_from_sampled_mass_dims_enabled:
            inertia_diag = cuboid_inertia_diagonal(mass, self.active_object_dims_m[env_ids])
            inertias[env_ids_inertia] = 0.0
            diag_m = inertia_diag.to(device=inertias.device, dtype=inertias.dtype)
            if inertias.ndim == 2:
                inertias[env_ids_inertia, 0] = diag_m[:, 0]
                inertias[env_ids_inertia, 4] = diag_m[:, 1]
                inertias[env_ids_inertia, 8] = diag_m[:, 2]
            elif inertias.ndim == 3:
                inertias[env_ids_inertia, :, 0] = diag_m[:, 0].unsqueeze(-1)
                inertias[env_ids_inertia, :, 4] = diag_m[:, 1].unsqueeze(-1)
                inertias[env_ids_inertia, :, 8] = diag_m[:, 2].unsqueeze(-1)
            else:
                raise RuntimeError(f"Unexpected object inertia tensor shape: {tuple(inertias.shape)}")
            self.object_randomized_inertia_diag_kg_m2[env_ids] = inertia_diag
        elif self.object_mass_randomization_enabled:
            default_mass = cube.data.default_mass.to(device=self.device, dtype=torch.float32)[env_ids, 0]
            default_inertia = cube.data.default_inertia.to(device=inertias.device, dtype=inertias.dtype)
            default_inertia_rows = default_inertia[env_ids_inertia]
            if default_inertia_rows.ndim == 2:
                default_inertia_diag = default_inertia_rows[:, (0, 4, 8)]
            elif default_inertia_rows.ndim == 3:
                default_inertia_diag = default_inertia_rows[:, 0][:, (0, 4, 8)]
            else:
                raise RuntimeError(
                    "Unexpected default object inertia tensor shape: "
                    f"{tuple(default_inertia_rows.shape)}"
                )
            if self.object_inertia_mass_size_scaling_enabled:
                reference_mass = torch.full_like(
                    mass,
                    float(getattr(self.cfg, "object_mass_nominal_kg", 0.020)),
                )
                inertia_diag = mass_size_scaled_inertia_diagonal(
                    default_inertia_diag.to(device=self.device, dtype=torch.float32),
                    mass,
                    reference_mass,
                    self.active_cube_side_length_m[env_ids],
                    nominal_side_length_m=float(self.object_keypoint_side_length_m),
                )
                ratio = inertia_diag[:, 0] / torch.clamp(
                    default_inertia_diag[:, 0].to(device=self.device, dtype=torch.float32),
                    min=1.0e-12,
                )
            else:
                ratio = mass / torch.clamp(default_mass, min=1.0e-8)
                inertia_diag = (
                    default_inertia_diag.to(device=self.device, dtype=torch.float32)
                    * ratio.unsqueeze(-1)
                )
            ratio_m = ratio.to(device=inertias.device, dtype=inertias.dtype).reshape(
                (ratio.shape[0],) + (1,) * (default_inertia_rows.ndim - 1)
            )
            inertias[env_ids_inertia] = default_inertia_rows * ratio_m
            self.object_randomized_inertia_diag_kg_m2[env_ids] = inertia_diag
        if (
            self.object_inertia_from_sampled_mass_dims_enabled
            or self.object_mesh_mass_properties_enabled
            or self.object_inertia_from_compound_cuboids_enabled
            or self.object_inertia_real_to_legacy_randomization_enabled
            or self.object_mass_randomization_enabled
        ):
            cube.root_physx_view.set_inertias(inertias, env_ids_cpu)
        if inertias.ndim == 2:
            inertia_tensor_b = inertias[env_ids_inertia].reshape(-1, 3, 3)
        elif inertias.ndim == 3:
            inertia_tensor_b = inertias[env_ids_inertia, 0, :].reshape(-1, 3, 3)
        else:
            raise RuntimeError(f"Unexpected object inertia tensor shape: {tuple(inertias.shape)}")
        self.object_randomized_inertia_tensor_b_kg_m2[env_ids] = inertia_tensor_b.to(
            device=self.device,
            dtype=torch.float32,
        )

        if self.object_com_randomization_enabled or compound_com is not None:
            radius = max(float(getattr(self.cfg, "object_com_randomization_range_m", 0.0)), 0.0)
            if self.object_com_randomization_enabled and radius > 0.0:
                com_offset = (2.0 * torch.rand(
                    (env_ids.numel(), 3),
                    device=self.device,
                    generator=self.target_rng,
                    dtype=torch.float32,
                ) - 1.0) * radius
            else:
                com_offset = torch.zeros(
                    (env_ids.numel(), 3), device=self.device, dtype=torch.float32
                )
            coms = cube.root_physx_view.get_coms().clone()
            env_ids_com = env_ids.detach().to(device=coms.device, dtype=torch.long)
            default_coms = self._object_default_coms.to(device=coms.device, dtype=coms.dtype)
            com_offset_m = com_offset.to(device=coms.device, dtype=coms.dtype)
            if compound_com is not None:
                base_com_m = compound_com.to(device=coms.device, dtype=coms.dtype)
            elif default_coms.ndim == 3:
                base_com_m = default_coms[env_ids_com, 0, :3]
            else:
                base_com_m = default_coms[env_ids_com, :3]
            if coms.ndim == 3:
                coms[env_ids_com, 0, :3] = base_com_m + com_offset_m
            elif coms.ndim == 2:
                coms[env_ids_com, :3] = base_com_m + com_offset_m
            else:
                raise RuntimeError(f"Unexpected object COM tensor shape: {tuple(coms.shape)}")
            cube.root_physx_view.set_coms(coms, env_ids_cpu)
            self.object_randomized_com_offset_b[env_ids] = com_offset

    def _sample_log_uniform_probabilities(
        self,
        values: Any,
        count: int,
        name: str,
    ) -> torch.Tensor:
        """Sample positive trigger probabilities uniformly in log space."""

        lo, hi = self._validate_range_pair(values, name, allow_zero=False)
        if hi > 1.0:
            raise ValueError(f"{name} must be within (0, 1], got {(lo, hi)!r}.")
        uniform = torch.rand(
            (count,),
            device=self.device,
            generator=self.target_rng,
            dtype=torch.float32,
        )
        return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * uniform)

    def _reset_random_external_wrench_state(self, env_ids: torch.Tensor) -> None:
        """Reset wrench buffers and resample per-episode trigger probabilities."""

        if not self.random_external_force_enabled:
            return
        self.object_external_force_w[env_ids] = 0.0
        self.object_external_torque_w[env_ids] = 0.0
        self.object_external_target_angular_acceleration_w[env_ids] = 0.0
        self.object_external_force_active[env_ids] = 0.0
        self.object_external_torque_active[env_ids] = 0.0
        count = int(env_ids.numel())
        if self.random_external_wrench_mode == "simtoolreal":
            force_reference_prob = self._sample_log_uniform_probabilities(
                getattr(self.cfg, "random_external_force_prob_range", (0.001, 0.1)),
                count,
                "random_external_force_prob_range",
            )
            torque_reference_prob = self._sample_log_uniform_probabilities(
                getattr(self.cfg, "random_external_torque_prob_range", (0.001, 0.1)),
                count,
                "random_external_torque_prob_range",
            )
            self.object_external_force_reference_prob[env_ids] = force_reference_prob
            self.object_external_torque_reference_prob[env_ids] = torque_reference_prob
            if bool(getattr(self.cfg, "random_external_wrench_timebase_matching_enabled", False)):
                reference_hz = float(getattr(self.cfg, "random_external_wrench_reference_hz", 60.0))
                control_hz = 1.0 / float(self.step_dt)
                self.object_external_force_prob[env_ids] = timebase_matched_trigger_probability(
                    force_reference_prob,
                    reference_hz=reference_hz,
                    control_hz=control_hz,
                )
                self.object_external_torque_prob[env_ids] = timebase_matched_trigger_probability(
                    torque_reference_prob,
                    reference_hz=reference_hz,
                    control_hz=control_hz,
                )
            else:
                self.object_external_force_prob[env_ids] = force_reference_prob
                self.object_external_torque_prob[env_ids] = torque_reference_prob
        else:
            prob = min(max(float(getattr(self.cfg, "random_external_force_prob", 0.0)), 0.0), 1.0)
            self.object_external_force_prob[env_ids] = prob
            self.object_external_torque_prob[env_ids] = 0.0
            self.object_external_force_reference_prob[env_ids] = prob
            self.object_external_torque_reference_prob[env_ids] = 0.0

    def _set_active_object_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
    ) -> None:
        """Write per-env wrench buffers to the currently active object asset."""

        if not self.multiobject_enabled:
            self.cube.set_external_force_and_torque(forces, torques, is_global=True)
            return
        if self.primitive_single_asset_per_env_enabled:
            self.cube.set_external_force_and_torque(forces, torques, is_global=True)
            return
        for size_id, object_id in enumerate(self.multisize_object_ids):
            active = self.active_cube_size_id == size_id
            object_forces = torch.zeros_like(forces)
            object_torques = torch.zeros_like(torques)
            object_forces[active] = forces[active]
            object_torques[active] = torques[active]
            self.multisize_cubes[object_id].set_external_force_and_torque(
                object_forces,
                object_torques,
                is_global=True,
            )

    def _update_random_external_force(self) -> None:
        """Apply the legacy fixed-probability force at each physics substep."""

        if not self.random_external_force_enabled or self.random_external_wrench_mode != "legacy_uniform_force":
            return
        prob = min(max(float(getattr(self.cfg, "random_external_force_prob", 0.0)), 0.0), 1.0)
        force_scale = max(float(getattr(self.cfg, "random_external_force_scale", 0.0)), 0.0)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device, dtype=torch.float32)
        if prob > 0.0 and force_scale > 0.0:
            active = torch.rand((self.num_envs,), device=self.device, generator=self.target_rng) < prob
            if bool(active.any()):
                count = int(active.sum().item())
                direction = self._random_unit_vectors(count, torch.float32)
                magnitude = force_scale * torch.rand((count, 1), device=self.device, generator=self.target_rng)
                forces[active, 0, :] = direction * magnitude
                self.object_external_force_active[:] = active.to(dtype=torch.float32)
                self.object_external_force_w[:] = forces[:, 0, :]
            else:
                self.object_external_force_active.zero_()
                self.object_external_force_w.zero_()
        else:
            self.object_external_force_active.zero_()
            self.object_external_force_w.zero_()
        torques = torch.zeros_like(forces)
        self.object_external_torque_active.zero_()
        self.object_external_torque_w.zero_()
        self.object_external_target_angular_acceleration_w.zero_()
        self._set_active_object_external_force_and_torque(forces, torques)

    def _update_simtoolreal_external_wrench(self) -> None:
        """Apply Gaussian force and configured torque disturbance once per policy step."""

        if not self.random_external_force_enabled or self.random_external_wrench_mode != "simtoolreal":
            return

        policy_dt = float(self.step_dt)
        force_decay = max(float(getattr(self.cfg, "random_external_force_decay", 0.0)), 0.0)
        force_decay_interval = max(
            float(getattr(self.cfg, "random_external_force_decay_interval_s", 0.08)),
            1.0e-8,
        )
        if force_decay > 0.0:
            self.object_external_force_w *= force_decay ** (policy_dt / force_decay_interval)
        else:
            self.object_external_force_w.zero_()

        torque_decay = max(float(getattr(self.cfg, "random_external_torque_decay", 0.0)), 0.0)
        torque_decay_interval = max(
            float(getattr(self.cfg, "random_external_torque_decay_interval_s", 0.08)),
            1.0e-8,
        )
        if torque_decay > 0.0:
            torque_decay_factor = torque_decay ** (policy_dt / torque_decay_interval)
            self.object_external_torque_w *= torque_decay_factor
            self.object_external_target_angular_acceleration_w *= torque_decay_factor
        else:
            self.object_external_torque_w.zero_()
            self.object_external_target_angular_acceleration_w.zero_()

        force_active = (
            torch.rand(
                (self.num_envs,),
                device=self.device,
                generator=self.target_rng,
                dtype=torch.float32,
            )
            < self.object_external_force_prob
        )
        torque_active = (
            torch.rand(
                (self.num_envs,),
                device=self.device,
                generator=self.target_rng,
                dtype=torch.float32,
            )
            < self.object_external_torque_prob
        )
        self.object_external_force_active[:] = force_active.to(dtype=torch.float32)
        self.object_external_torque_active[:] = torque_active.to(dtype=torch.float32)

        force_scale = max(float(getattr(self.cfg, "random_external_force_scale", 0.0)), 0.0)
        if bool(getattr(self.cfg, "random_external_wrench_timebase_matching_enabled", False)):
            force_scale = timebase_matched_wrench_scale(
                force_scale,
                float(getattr(self.cfg, "random_external_wrench_reference_hz", 60.0)),
                1.0 / float(self.step_dt),
            )
        if force_scale > 0.0 and bool(force_active.any()):
            force_ids = torch.nonzero(force_active, as_tuple=False).squeeze(-1)
            force_noise = torch.randn(
                (force_ids.numel(), 3),
                device=self.device,
                generator=self.target_rng,
                dtype=torch.float32,
            )
            self.object_external_force_w[force_ids] = (
                force_noise
                * self.object_randomized_mass_kg[force_ids].unsqueeze(-1)
                * force_scale
            )

        if bool(torque_active.any()):
            torque_ids = torch.nonzero(torque_active, as_tuple=False).squeeze(-1)
            torque_noise = torch.randn(
                (torque_ids.numel(), 3),
                device=self.device,
                generator=self.target_rng,
                dtype=torch.float32,
            )
            if self.random_external_torque_sampling_mode == "mass_scaled_gaussian":
                torque_scale = max(
                    float(getattr(self.cfg, "random_external_torque_scale", 0.0)),
                    0.0,
                )
                if bool(getattr(self.cfg, "random_external_wrench_timebase_matching_enabled", False)):
                    torque_scale = timebase_matched_wrench_scale(
                        torque_scale,
                        float(getattr(self.cfg, "random_external_wrench_reference_hz", 60.0)),
                        1.0 / float(self.step_dt),
                    )
                self.object_external_torque_w[torque_ids] = (
                    torque_noise
                    * self.object_randomized_mass_kg[torque_ids].unsqueeze(-1)
                    * torque_scale
                )
            else:
                angular_acceleration_scale = max(
                    float(
                        getattr(
                            self.cfg,
                            "random_external_angular_acceleration_scale_rad_s2",
                            0.0,
                        )
                    ),
                    0.0,
                )
                if bool(getattr(self.cfg, "random_external_wrench_timebase_matching_enabled", False)):
                    angular_acceleration_scale = timebase_matched_wrench_scale(
                        angular_acceleration_scale,
                        float(getattr(self.cfg, "random_external_wrench_reference_hz", 60.0)),
                        1.0 / float(self.step_dt),
                    )
                target_alpha_w = torque_noise * angular_acceleration_scale
                self.object_external_target_angular_acceleration_w[torque_ids] = target_alpha_w
                self.object_external_torque_w[torque_ids] = (
                    world_torque_for_target_angular_acceleration(
                        self.object_randomized_inertia_tensor_b_kg_m2[torque_ids],
                        normalize_quat_wxyz(
                            self._active_cube_root_quat_w()[torque_ids].to(dtype=torch.float32)
                        ),
                        target_alpha_w,
                    )
                )

        self._set_active_object_external_force_and_torque(
            self.object_external_force_w.unsqueeze(1),
            self.object_external_torque_w.unsqueeze(1),
        )

    def _load_stable_cache_to_device(self, path: Path) -> dict[str, Any]:
        cache = load_stable_grasp_cache(path, map_location="cpu")
        device_cache: dict[str, Any] = {}
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                device_cache[key] = value.to(device=self.device)
            else:
                device_cache[key] = value
        return device_cache

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_scale_conditioned_stable_cache_bank(self, manifest_path: Path) -> dict[str, Any]:
        """Load and concatenate interval caches declared by a validated manifest."""

        if any(
            bool(getattr(self.cfg, key, False))
            for key in ("multisize_cube_enabled", "multicuboid_enabled", "primitive_multi_shape_enabled")
        ):
            raise ValueError("Scale-conditioned stable-cache manifests support only single-object tasks.")
        if not bool(getattr(self.cfg, "object_scale_randomization_enabled", False)):
            raise ValueError(
                "scale_conditioned_stable_cache_manifest_path requires object_scale_randomization_enabled=True."
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Scale-conditioned stable-cache manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "sharpa_hb1_cube_scale_cache_manifest_v1":
            raise ValueError(
                "Unsupported scale-conditioned stable-cache manifest schema: "
                f"{manifest.get('schema_version')!r}."
            )
        bucket_specs = manifest.get("buckets")
        if not isinstance(bucket_specs, list) or not bucket_specs:
            raise ValueError("Scale-conditioned stable-cache manifest must contain a non-empty buckets list.")

        caches: list[dict[str, Any]] = []
        cache_paths: list[Path] = []
        anchors: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        row_counts: list[int] = []
        identity: tuple[Any, ...] | None = None
        previous_upper: float | None = None
        manifest_asset_sha = str(manifest.get("asset_sha256", ""))
        for bucket_index, spec in enumerate(bucket_specs):
            if not isinstance(spec, dict):
                raise TypeError(f"Scale-cache bucket {bucket_index} must be an object, got {type(spec)}.")
            anchor = float(spec["anchor_size_m"])
            lower = float(spec["valid_size_min_m"])
            upper = float(spec["valid_size_max_m"])
            lower_bound = str(spec.get("valid_size_min_bound", ""))
            upper_bound = str(spec.get("valid_size_max_bound", ""))
            if not (0.0 < lower <= anchor <= upper):
                raise ValueError(
                    f"Invalid scale-cache interval/anchor at bucket {bucket_index}: "
                    f"lower={lower}, anchor={anchor}, upper={upper}."
                )
            if lower_bound != ("closed" if bucket_index == 0 else "open") or upper_bound != "closed":
                raise ValueError(
                    f"Bucket {bucket_index} must use the manifest convention "
                    "[min,max] for the first interval and (min,max] thereafter."
                )
            if previous_upper is not None and not math.isclose(lower, previous_upper, abs_tol=1.0e-9):
                raise ValueError(
                    f"Scale-cache intervals are not contiguous between {previous_upper} and {lower}."
                )
            previous_upper = upper

            cache_path = resolve_workspace_path(str(spec["cache_path"]))
            if not cache_path.is_file():
                raise FileNotFoundError(f"Scale-cache bucket file does not exist: {cache_path}")
            expected_sha = str(spec.get("cache_sha256", ""))
            actual_sha = self._sha256_file(cache_path)
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError(
                    f"Scale-cache SHA256 mismatch for bucket {bucket_index}: "
                    f"actual={actual_sha}, expected={expected_sha}, path={cache_path}."
                )
            cache = self._load_stable_cache_to_device(cache_path)
            cache_identity = (
                int(cache["schema_version"]),
                str(cache["hand"]),
                int(cache["action_dim"]),
                str(cache["object_id"]),
            )
            if identity is None:
                identity = cache_identity
            elif cache_identity != identity:
                raise ValueError(
                    f"Scale-cache bucket {bucket_index} identity={cache_identity!r}, expected {identity!r}."
                )
            cache_dims = tuple(float(value) for value in cache.get("object_dims_m", ()))
            if len(cache_dims) != 3 or not all(math.isclose(value, anchor, abs_tol=1.0e-7) for value in cache_dims):
                raise ValueError(
                    f"Scale-cache bucket {bucket_index} has object_dims_m={cache_dims!r}, "
                    f"expected {(anchor, anchor, anchor)!r}."
                )
            metadata = dict(cache.get("metadata", {}))
            if manifest_asset_sha and metadata.get("hb1_usda_sha256") != manifest_asset_sha:
                raise ValueError(
                    f"Scale-cache bucket {bucket_index} HB1 asset hash does not match its manifest."
                )
            rows = int(cache["q"].shape[0])
            if rows <= 0 or rows != int(spec.get("row_count", rows)):
                raise ValueError(
                    f"Scale-cache bucket {bucket_index} row count mismatch: cache={rows}, "
                    f"manifest={spec.get('row_count')}."
                )
            caches.append(cache)
            cache_paths.append(cache_path)
            anchors.append(anchor)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            row_counts.append(rows)

        if int(manifest.get("total_row_count", sum(row_counts))) != sum(row_counts):
            raise ValueError(
                f"Scale-cache manifest total_row_count={manifest.get('total_row_count')} "
                f"does not match loaded rows={sum(row_counts)}."
            )
        nominal_size = float(getattr(self.cfg, "object_keypoint_side_length_m", CUBE_4CM_SIDE_LENGTH_M))
        configured_scale_range = tuple(float(value) for value in getattr(self.cfg, "object_scale_range", (1.0, 1.0)))
        configured_size_range = (
            nominal_size * configured_scale_range[0],
            nominal_size * configured_scale_range[1],
        )
        if not (
            math.isclose(configured_size_range[0], lower_bounds[0], abs_tol=1.0e-7)
            and math.isclose(configured_size_range[1], upper_bounds[-1], abs_tol=1.0e-7)
        ):
            raise ValueError(
                f"Configured continuous object size range={configured_size_range} does not match "
                f"scale-cache coverage={(lower_bounds[0], upper_bounds[-1])}."
            )

        common_row_tensor_keys = set.intersection(
            *(
                {
                    key
                    for key, value in cache.items()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == rows
                }
                for cache, rows in zip(caches, row_counts, strict=True)
            )
        )
        required_row_keys = {"q", "qdot", "q_cmd", "object_pose_palm", "cube_keypoints_palm", "contact_bits", "stability_score"}
        if missing := sorted(required_row_keys - common_row_tensor_keys):
            raise ValueError(f"Scale-cache buckets do not share required row tensors: {missing}.")
        assert identity is not None
        merged: dict[str, Any] = {
            "schema_version": identity[0],
            "hand": identity[1],
            "action_dim": identity[2],
            "object_id": identity[3],
            "object_dims_m": (nominal_size, nominal_size, nominal_size),
            "metadata": {
                "schema_name": "scale_conditioned_stable_cache_bank_v1",
                "manifest_path": str(manifest_path),
                "bucket_cache_paths": [str(path) for path in cache_paths],
                "bucket_row_counts": row_counts,
            },
        }
        for key in sorted(common_row_tensor_keys):
            merged[key] = torch.cat([cache[key] for cache in caches], dim=0)
        merged["scale_cache_bucket_id"] = torch.cat(
            [torch.full((rows,), index, device=self.device, dtype=torch.long) for index, rows in enumerate(row_counts)]
        )

        offsets = [0]
        for rows in row_counts:
            offsets.append(offsets[-1] + rows)
        self.scale_conditioned_stable_cache_manifest = manifest
        self.scale_conditioned_stable_cache_paths = tuple(cache_paths)
        self.scale_conditioned_cache_anchor_sizes_m = torch.tensor(anchors, device=self.device, dtype=torch.float32)
        self.scale_conditioned_cache_interval_lower_m = torch.tensor(
            lower_bounds, device=self.device, dtype=torch.float32
        )
        self.scale_conditioned_cache_interval_upper_m = torch.tensor(
            upper_bounds, device=self.device, dtype=torch.float32
        )
        self.scale_conditioned_cache_offsets = torch.tensor(offsets, device=self.device, dtype=torch.long)
        self.scale_conditioned_cache_row_counts = torch.tensor(row_counts, device=self.device, dtype=torch.long)
        return merged

    def _load_geometry_conditioned_stable_cache_bank(self, manifest_path: Path) -> dict[str, Any]:
        """Load and concatenate near-cube geometry-anchor caches from a manifest."""

        if any(
            bool(getattr(self.cfg, key, False))
            for key in ("multisize_cube_enabled", "multicuboid_enabled", "primitive_multi_shape_enabled")
        ):
            raise ValueError("Geometry-conditioned stable-cache manifests support only single-object tasks.")
        if not bool(getattr(self.cfg, "object_nearcube_geometry_randomization_enabled", False)):
            raise ValueError(
                "geometry_conditioned_stable_cache_manifest_path requires "
                "object_nearcube_geometry_randomization_enabled=True."
            )
        if bool(getattr(self.cfg, "object_scale_randomization_enabled", False)):
            raise ValueError("Geometry-conditioned stable-cache manifests are incompatible with scalar scale DR.")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Geometry-conditioned stable-cache manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "sharpa_hb1_nearcube_geometry_cache_manifest_v1":
            raise ValueError(
                "Unsupported geometry-conditioned stable-cache manifest schema: "
                f"{manifest.get('schema_version')!r}."
            )
        anchor_specs = manifest.get("anchors")
        if not isinstance(anchor_specs, list) or not anchor_specs:
            raise ValueError("Geometry-conditioned stable-cache manifest must contain a non-empty anchors list.")
        metric_scale = torch.as_tensor(manifest.get("metric_scale_m", ()), dtype=torch.float64)
        if metric_scale.shape != (3,) or not bool(torch.isfinite(metric_scale).all()) or bool((metric_scale <= 0).any()):
            raise ValueError("Geometry-cache manifest metric_scale_m must contain three finite positive values.")

        caches: list[dict[str, Any]] = []
        cache_paths: list[Path] = []
        anchor_dims: list[tuple[float, float, float]] = []
        row_counts: list[int] = []
        identity: tuple[Any, ...] | None = None
        manifest_asset_sha = str(manifest.get("asset_sha256", ""))
        seen_anchor_ids: set[int] = set()
        for anchor_index, spec in enumerate(anchor_specs):
            if not isinstance(spec, dict):
                raise TypeError(f"Geometry-cache anchor {anchor_index} must be an object, got {type(spec)}.")
            anchor_id = int(spec.get("anchor_id", anchor_index))
            if anchor_id != anchor_index or anchor_id in seen_anchor_ids:
                raise ValueError(
                    "Geometry-cache anchor_id values must be unique, contiguous, and match list order; "
                    f"entry={anchor_index}, anchor_id={anchor_id}."
                )
            seen_anchor_ids.add(anchor_id)
            dims_tensor = torch.as_tensor(spec.get("anchor_dims_m", ()), dtype=torch.float64)
            if dims_tensor.shape != (3,) or not bool(torch.isfinite(dims_tensor).all()) or bool((dims_tensor <= 0).any()):
                raise ValueError(f"Geometry-cache anchor {anchor_index} has invalid anchor_dims_m.")
            dims = tuple(float(value) for value in dims_tensor.tolist())
            if any(
                all(math.isclose(value, previous_value, abs_tol=1.0e-9) for value, previous_value in zip(dims, previous, strict=True))
                for previous in anchor_dims
            ):
                raise ValueError(f"Geometry-cache anchor {anchor_index} duplicates anchor_dims_m={dims!r}.")

            cache_path = resolve_workspace_path(str(spec["cache_path"]))
            if not cache_path.is_file():
                raise FileNotFoundError(f"Geometry-cache anchor file does not exist: {cache_path}")
            expected_sha = str(spec.get("cache_sha256", ""))
            actual_sha = self._sha256_file(cache_path)
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError(
                    f"Geometry-cache SHA256 mismatch for anchor {anchor_index}: "
                    f"actual={actual_sha}, expected={expected_sha}, path={cache_path}."
                )
            cache = self._load_stable_cache_to_device(cache_path)
            cache_identity = (
                int(cache["schema_version"]),
                str(cache["hand"]),
                int(cache["action_dim"]),
                str(cache["object_id"]),
            )
            if identity is None:
                identity = cache_identity
            elif cache_identity != identity:
                raise ValueError(
                    f"Geometry-cache anchor {anchor_index} identity={cache_identity!r}, expected {identity!r}."
                )
            cache_dims = torch.as_tensor(cache.get("object_dims_m", ()), dtype=torch.float64)
            if cache_dims.shape != (3,) or not bool(torch.allclose(cache_dims, dims_tensor, atol=1.0e-7, rtol=0.0)):
                raise ValueError(
                    f"Geometry-cache anchor {anchor_index} has object_dims_m={cache.get('object_dims_m')!r}, "
                    f"expected {dims!r}."
                )
            metadata = dict(cache.get("metadata", {}))
            if manifest_asset_sha and metadata.get("hb1_usda_sha256") != manifest_asset_sha:
                raise ValueError(
                    f"Geometry-cache anchor {anchor_index} HB1 asset hash does not match its manifest."
                )
            rows = int(cache["q"].shape[0])
            if rows <= 0 or rows != int(spec.get("row_count", rows)):
                raise ValueError(
                    f"Geometry-cache anchor {anchor_index} row count mismatch: cache={rows}, "
                    f"manifest={spec.get('row_count')}."
                )
            caches.append(cache)
            cache_paths.append(cache_path)
            anchor_dims.append(dims)
            row_counts.append(rows)

        if int(manifest.get("total_row_count", sum(row_counts))) != sum(row_counts):
            raise ValueError(
                f"Geometry-cache manifest total_row_count={manifest.get('total_row_count')} "
                f"does not match loaded rows={sum(row_counts)}."
            )
        observation_contract = dict(manifest.get("observation_contract", {}))
        expected_contract = {
            "actor_observation_dim": 180,
            "privileged_state_dim": 428,
            "action_dim": 22,
            "actor_contains_geometry": False,
            "critic_contains_geometry": False,
        }
        for key, expected in expected_contract.items():
            if key in observation_contract and observation_contract[key] != expected:
                raise ValueError(
                    f"Geometry-cache manifest observation_contract[{key!r}]="
                    f"{observation_contract[key]!r}, expected {expected!r}."
                )

        common_row_tensor_keys = set.intersection(
            *(
                {
                    key
                    for key, value in cache.items()
                    if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == rows
                }
                for cache, rows in zip(caches, row_counts, strict=True)
            )
        )
        required_row_keys = {
            "q", "qdot", "q_cmd", "object_pose_palm", "cube_keypoints_palm", "contact_bits", "stability_score"
        }
        if missing := sorted(required_row_keys - common_row_tensor_keys):
            raise ValueError(f"Geometry-cache anchors do not share required row tensors: {missing}.")
        assert identity is not None
        merged: dict[str, Any] = {
            "schema_version": identity[0],
            "hand": identity[1],
            "action_dim": identity[2],
            "object_id": identity[3],
            "object_dims_m": tuple(float(v) for v in getattr(self.cfg, "object_nearcube_base_dims_m", (0.04, 0.04, 0.04))),
            "metadata": {
                "schema_name": "geometry_conditioned_stable_cache_bank_v1",
                "manifest_path": str(manifest_path),
                "anchor_cache_paths": [str(path) for path in cache_paths],
                "anchor_row_counts": row_counts,
            },
        }
        for key in sorted(common_row_tensor_keys):
            merged[key] = torch.cat([cache[key] for cache in caches], dim=0)
        merged["geometry_cache_anchor_id"] = torch.cat(
            [torch.full((rows,), index, device=self.device, dtype=torch.long) for index, rows in enumerate(row_counts)]
        )

        offsets = [0]
        for rows in row_counts:
            offsets.append(offsets[-1] + rows)
        self.geometry_conditioned_stable_cache_manifest = manifest
        self.geometry_conditioned_stable_cache_paths = tuple(cache_paths)
        self.geometry_conditioned_cache_anchor_dims_m = torch.tensor(
            anchor_dims, device=self.device, dtype=torch.float32
        )
        self.geometry_conditioned_cache_metric_scale_m = metric_scale.to(
            device=self.device, dtype=torch.float32
        )
        self.geometry_conditioned_cache_offsets = torch.tensor(offsets, device=self.device, dtype=torch.long)
        self.geometry_conditioned_cache_row_counts = torch.tensor(row_counts, device=self.device, dtype=torch.long)
        return merged

    @staticmethod
    def _validate_multiobject_cache_metadata(
        cache: dict[str, Any],
        object_id: str,
        expected_dims: Sequence[float],
        cache_path: str,
    ) -> None:
        cache_object_id = str(cache.get("object_id", ""))
        if cache_object_id != object_id:
            raise ValueError(
                f"Configured cache for {object_id!r} has object_id={cache_object_id!r}: {cache_path}"
            )
        cache_dims_raw = cache.get("object_dims_m", None)
        if cache_dims_raw is None and isinstance(cache.get("metadata"), dict):
            cache_dims_raw = cache["metadata"].get("object_dims_m", None)
        if cache_dims_raw is None:
            return
        cache_dims = torch.as_tensor(cache_dims_raw, dtype=torch.float32)
        expected = torch.tensor(tuple(float(v) for v in expected_dims), dtype=torch.float32)
        if tuple(cache_dims.shape) != (3,) or not bool(torch.allclose(cache_dims, expected, atol=1.0e-6)):
            raise ValueError(
                f"Configured cache for {object_id!r} has object_dims_m={cache_dims_raw!r}, "
                f"expected {tuple(float(v) for v in expected_dims)!r}: {cache_path}"
            )

    def _load_multiobject_geometry_latents(
        self,
        object_ids: tuple[str, ...],
        object_dims_m: torch.Tensor,
        path: Path,
        latent_dim: int,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Geometry latent file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_object_ids = tuple(str(v) for v in payload.get("object_ids", ()))
        if len(payload_object_ids) == 0:
            raise ValueError(f"Geometry latent file has no object_ids: {path}")
        if len(set(payload_object_ids)) != len(payload_object_ids):
            raise ValueError(f"Geometry latent file contains duplicate object_ids: {path}")
        latent_table = torch.as_tensor(payload.get("geometry_latents", None), dtype=torch.float32)
        if tuple(latent_table.shape) != (len(payload_object_ids), latent_dim):
            raise ValueError(
                f"Geometry latent table in {path} has shape {tuple(latent_table.shape)}, "
                f"expected ({len(payload_object_ids)}, {latent_dim})."
            )
        payload_id_set = set(payload_object_ids)
        missing = [object_id for object_id in object_ids if object_id not in payload_id_set]
        if missing:
            raise ValueError(f"Geometry latent file {path} is missing object ids: {missing}")

        payload_dims_raw = payload.get("object_dims_m", None)
        payload_dims = None
        if payload_dims_raw is not None:
            payload_dims = torch.as_tensor(payload_dims_raw, dtype=torch.float32)
            if tuple(payload_dims.shape) != (len(payload_object_ids), 3):
                raise ValueError(
                    f"Geometry latent object_dims_m in {path} has shape {tuple(payload_dims.shape)}, "
                    f"expected ({len(payload_object_ids)}, 3)."
                )

        id_to_index = {object_id: idx for idx, object_id in enumerate(payload_object_ids)}
        rows = []
        for local_idx, object_id in enumerate(object_ids):
            payload_idx = id_to_index[object_id]
            if payload_dims is not None:
                expected_dims = object_dims_m[local_idx].detach().cpu().to(dtype=torch.float32)
                got_dims = payload_dims[payload_idx]
                if not bool(torch.allclose(got_dims, expected_dims, atol=1.0e-6)):
                    raise ValueError(
                        f"Geometry latent dims mismatch for {object_id!r} in {path}: "
                        f"got {tuple(float(v) for v in got_dims.tolist())}, "
                        f"expected {tuple(float(v) for v in expected_dims.tolist())}."
                    )
            rows.append(latent_table[payload_idx])
        return torch.stack(rows, dim=0).to(device=self.device, dtype=torch.float32)

    def _load_multiobject_mesh_data(
        self,
        object_ids: tuple[str, ...],
        object_dims_m: torch.Tensor,
        path: Path,
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_ids = tuple(str(v) for v in payload["object_ids"])
        payload_dims = torch.as_tensor(payload["object_dims_m"], dtype=torch.float32)
        id_to_index = {object_id: index for index, object_id in enumerate(payload_ids)}
        indices = torch.tensor([id_to_index[object_id] for object_id in object_ids], dtype=torch.long)
        selected_dims = payload_dims[indices]
        if not bool(torch.allclose(selected_dims, object_dims_m.detach().cpu(), atol=1.0e-6)):
            raise ValueError(f"Mesh-data dimensions do not match configured objects: {path}")
        self.multiobject_mesh_com_m = torch.as_tensor(payload["mesh_com_m"], dtype=torch.float32)[indices].to(self.device)
        self.multiobject_mesh_inertia_unit_mass_kg_m2 = torch.as_tensor(
            payload["mesh_inertia_unit_mass_kg_m2"], dtype=torch.float32
        )[indices].to(self.device)
        self.multiobject_csgwcd_surface_delta_grid_m = torch.as_tensor(
            payload["csgwcd_surface_delta_grid_m"], dtype=torch.float32
        )[indices].to(self.device)
        self.multiobject_csgwcd_outward_normal_grid = torch.as_tensor(
            payload["csgwcd_outward_normal_grid"], dtype=torch.float32
        )[indices].to(self.device)
        self.csgwcd_mesh_grid_half_extent_m = float(payload["csgwcd_grid_half_extent_m"])

    def _resolve_body_indices(self, body_names: tuple[str, ...]) -> list[int]:
        body_name_to_idx = {name: idx for idx, name in enumerate(self.hand.body_names)}
        missing = tuple(name for name in body_names if name not in body_name_to_idx)
        if missing:
            raise RuntimeError(f"Wuji articulation is missing hand-surface-delta bodies: {missing}")
        return [body_name_to_idx[name] for name in body_names]

    def _load_multiobject_surface_points(
        self,
        object_ids: tuple[str, ...],
        object_dims_m: torch.Tensor,
        path: Path,
    ) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"Hand-surface point-cloud file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_object_ids = tuple(str(v) for v in payload.get("object_ids", ()))
        if len(payload_object_ids) == 0:
            raise ValueError(f"Hand-surface point-cloud file has no object_ids: {path}")
        if len(set(payload_object_ids)) != len(payload_object_ids):
            raise ValueError(f"Hand-surface point-cloud file contains duplicate object_ids: {path}")
        surface_points = torch.as_tensor(payload.get("surface_points", None), dtype=torch.float32)
        if surface_points.ndim != 3 or surface_points.shape[0] != len(payload_object_ids) or surface_points.shape[2] != 3:
            raise ValueError(
                f"surface_points in {path} must have shape ({len(payload_object_ids)}, N, 3), "
                f"got {tuple(surface_points.shape)}."
            )
        if surface_points.shape[1] <= 0:
            raise ValueError(f"surface_points in {path} must contain at least one point per object.")
        payload_id_set = set(payload_object_ids)
        missing = [object_id for object_id in object_ids if object_id not in payload_id_set]
        if missing:
            raise ValueError(f"Hand-surface point-cloud file {path} is missing object ids: {missing}")

        payload_dims_raw = payload.get("object_dims_m", None)
        payload_dims = None
        if payload_dims_raw is not None:
            payload_dims = torch.as_tensor(payload_dims_raw, dtype=torch.float32)
            if tuple(payload_dims.shape) != (len(payload_object_ids), 3):
                raise ValueError(
                    f"Hand-surface point-cloud object_dims_m in {path} has shape {tuple(payload_dims.shape)}, "
                    f"expected ({len(payload_object_ids)}, 3)."
                )

        id_to_index = {object_id: idx for idx, object_id in enumerate(payload_object_ids)}
        rows = []
        for local_idx, object_id in enumerate(object_ids):
            payload_idx = id_to_index[object_id]
            if payload_dims is not None:
                expected_dims = object_dims_m[local_idx].detach().cpu().to(dtype=torch.float32)
                got_dims = payload_dims[payload_idx]
                if not bool(torch.allclose(got_dims, expected_dims, atol=1.0e-6)):
                    raise ValueError(
                        f"Hand-surface point-cloud dims mismatch for {object_id!r} in {path}: "
                        f"got {tuple(float(v) for v in got_dims.tolist())}, "
                        f"expected {tuple(float(v) for v in expected_dims.tolist())}."
                    )
            rows.append(surface_points[payload_idx])
        return torch.stack(rows, dim=0).to(device=self.device, dtype=torch.float32)

    def _nearest_surface_delta_obj_chunked(self, anchors_obj: torch.Tensor, points_obj: torch.Tensor) -> torch.Tensor:
        chunk_size = max(1, int(getattr(self.cfg, "primitive_hand_surface_delta_chunk_size", 64)))
        env_count, anchor_count, _xyz = anchors_obj.shape
        best_dist2 = torch.full((env_count, anchor_count), float("inf"), device=self.device, dtype=anchors_obj.dtype)
        best_delta = torch.zeros((env_count, anchor_count, 3), device=self.device, dtype=anchors_obj.dtype)
        for start in range(0, points_obj.shape[0], chunk_size):
            points_chunk = points_obj[start : start + chunk_size].to(device=self.device, dtype=anchors_obj.dtype)
            diff = points_chunk.view(1, 1, -1, 3) - anchors_obj.unsqueeze(2)
            dist2 = torch.sum(diff * diff, dim=-1)
            chunk_dist2, chunk_idx = torch.min(dist2, dim=-1)
            gather_idx = chunk_idx.view(env_count, anchor_count, 1, 1).expand(-1, -1, 1, 3)
            chunk_delta = torch.gather(diff, dim=2, index=gather_idx).squeeze(2)
            update = chunk_dist2 < best_dist2
            best_dist2 = torch.where(update, chunk_dist2, best_dist2)
            best_delta = torch.where(update.unsqueeze(-1), chunk_delta, best_delta)
        return best_delta

    def _compute_hand_surface_delta_geometry_latent(self) -> torch.Tensor:
        if self.multiobject_surface_points_obj.numel() == 0:
            raise RuntimeError("Hand-surface delta geometry obs requires loaded multiobject surface points.")
        anchor_count = len(self.hand_surface_delta_body_ids)
        anchors_w = self.hand.data.body_pos_w[:, self.hand_surface_delta_body_ids].to(dtype=torch.float32)
        object_pos_w = self._active_cube_root_pos_w().to(dtype=torch.float32)
        object_quat_w = normalize_quat_wxyz(self._active_cube_root_quat_w().to(dtype=torch.float32))
        inv_object_quat_w = quat_conjugate_wxyz(object_quat_w).unsqueeze(1).expand(self.num_envs, anchor_count, 4)
        anchors_obj = quat_apply_wxyz(inv_object_quat_w, anchors_w - object_pos_w.unsqueeze(1))

        delta_palm = torch.zeros_like(anchors_obj)
        for size_id in range(len(self.multisize_object_ids)):
            mask = self.active_cube_size_id == size_id
            if not bool(mask.any()):
                continue
            env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            delta_obj = self._nearest_surface_delta_obj_chunked(
                anchors_obj[env_ids],
                self.multiobject_surface_points_obj[size_id],
            )
            obj_quat = object_quat_w[env_ids].unsqueeze(1).expand(env_ids.numel(), anchor_count, 4)
            delta_w = quat_apply_wxyz(obj_quat, delta_obj)
            delta_palm[env_ids] = self._world_vectors_to_hand(delta_w, env_ids)

        scale_m = max(float(getattr(self.cfg, "primitive_hand_surface_delta_scale_m", 0.04)), 1.0e-6)
        clip_value = float(getattr(self.cfg, "primitive_hand_surface_delta_clip", 2.0))
        if clip_value > 0.0:
            delta_palm = torch.clamp(delta_palm / scale_m, -clip_value, clip_value)
        else:
            delta_palm = delta_palm / scale_m
        return delta_palm.reshape(self.num_envs, -1)

    def _active_cube_data(self, field_name: str) -> torch.Tensor:
        if not bool(getattr(self, "multiobject_enabled", False)):
            return getattr(self.cube.data, field_name)
        if self.primitive_single_asset_per_env_enabled:
            return getattr(self.cube.data, field_name)
        reference = getattr(next(iter(self.multisize_cubes.values())).data, field_name)
        result = torch.zeros_like(reference)
        for size_id, object_id in enumerate(self.multisize_object_ids):
            mask = self.active_cube_size_id == size_id
            if bool(mask.any()):
                result[mask] = getattr(self.multisize_cubes[object_id].data, field_name)[mask]
        return result

    def _active_cube_root_pose_w(self) -> torch.Tensor:
        return self._active_cube_data("root_pose_w")

    def _active_cube_root_pos_w(self) -> torch.Tensor:
        return self._active_cube_data("root_pos_w")

    def _active_cube_root_quat_w(self) -> torch.Tensor:
        return self._active_cube_data("root_quat_w")

    def _active_cube_root_lin_vel_w(self) -> torch.Tensor:
        return self._active_cube_data("root_lin_vel_w")

    def _active_cube_root_ang_vel_w(self) -> torch.Tensor:
        return self._active_cube_data("root_ang_vel_w")


    def _multisize_env_mask(self, object_id: str, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        size_id = self.multisize_object_ids.index(object_id)
        if env_ids is None:
            return self.active_cube_size_id == size_id
        return self.active_cube_size_id[env_ids] == size_id

    def _cube_park_pose_w(self, object_id: str, env_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        size_id = self.multisize_object_ids.index(object_id)
        dims = self.multiobject_dims_m[size_id]
        height_m = float(dims[2].item())
        y_center = 0.5 * float(len(self.multisize_object_ids) - 1)
        if self.primitive_multi_shape_enabled:
            park_y_attr = "primitive_park_y_spacing_m"
            park_x_attr = "primitive_park_x_m"
        elif self.multicuboid_enabled:
            park_y_attr = "multicuboid_park_y_spacing_m"
            park_x_attr = "multicuboid_park_x_m"
        else:
            park_y_attr = "multisize_cube_park_y_spacing_m"
            park_x_attr = "multisize_cube_park_x_m"
        y_offset = (float(size_id) - y_center) * float(getattr(self.cfg, park_y_attr, 0.22))
        pos_offset = torch.tensor(
            (
                float(getattr(self.cfg, park_x_attr, -0.22)),
                y_offset,
                0.5 * height_m,
            ),
            device=self.device,
            dtype=dtype,
        )
        pos = self.scene.env_origins[env_ids].to(dtype=dtype) + pos_offset.unsqueeze(0)
        quat = torch.zeros((env_ids.numel(), 4), device=self.device, dtype=dtype)
        quat[:, 0] = 1.0
        return torch.cat((pos, quat), dim=-1)

    def _write_multisize_cube_reset(
        self,
        env_ids: torch.Tensor,
        active_object_pose_w: torch.Tensor,
    ) -> None:
        zeros_vel = torch.zeros(env_ids.numel(), 6, device=self.device)
        if self.primitive_single_asset_per_env_enabled:
            self.cube.write_root_pose_to_sim(active_object_pose_w, env_ids=env_ids)
            self.cube.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
            return
        for object_id in self.multisize_object_ids:
            cube = self.multisize_cubes[object_id]
            active_mask = self._multisize_env_mask(object_id, env_ids)
            pose_w = self._cube_park_pose_w(object_id, env_ids, active_object_pose_w.dtype)
            if bool(active_mask.any()):
                pose_w[active_mask] = active_object_pose_w[active_mask]
            cube.write_root_pose_to_sim(pose_w, env_ids=env_ids)
            cube.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)

    def _update_active_cube_size_buffers(self, env_ids: torch.Tensor, size_ids: torch.Tensor) -> None:
        self.active_cube_size_id[env_ids] = size_ids
        dims = self.multiobject_dims_m[size_ids].to(dtype=torch.float32)
        if getattr(self, "object_scale_randomization_enabled", False):
            env_scale = self.object_scale_values[env_ids].to(dtype=dims.dtype).unsqueeze(-1)
            dims = dims * env_scale
        side_lengths = dims[:, 0]
        self.active_cube_side_length_m[env_ids] = side_lengths
        self.active_cube_size_norm[env_ids] = side_lengths / CUBE_4CM_SIDE_LENGTH_M - 1.0
        self.active_object_dims_m[env_ids] = dims
        self.active_object_dim_norm[env_ids] = dims / CUBE_4CM_SIDE_LENGTH_M - 1.0
        if self.multiobject_is_cylinder.numel() > 0:
            self.active_object_is_cylinder[env_ids] = self.multiobject_is_cylinder[size_ids].to(dtype=torch.float32)
        if self.multiobject_shape_ids.numel() > 0:
            shape_ids = self.multiobject_shape_ids[size_ids]
            self.active_object_shape_id[env_ids] = shape_ids
            onehot = torch.nn.functional.one_hot(shape_ids, num_classes=len(self.primitive_shape_names)).to(
                dtype=torch.float32
            )
            self.active_object_shape_onehot[env_ids] = onehot
        if self.multiobject_geometry_latents.numel() > 0:
            self.active_object_geometry_latent[env_ids] = self.multiobject_geometry_latents[size_ids].to(
                dtype=torch.float32
            )

    def _explicit_local_random_workspace_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not bool(self.cfg.local_random_use_explicit_workspace_bounds):
            return None
        values = (
            self.cfg.local_random_workspace_min_x_m,
            self.cfg.local_random_workspace_min_y_m,
            self.cfg.local_random_workspace_min_z_m,
            self.cfg.local_random_workspace_max_x_m,
            self.cfg.local_random_workspace_max_y_m,
            self.cfg.local_random_workspace_max_z_m,
        )
        lo = torch.tensor(values[:3], device=self.device, dtype=torch.float32)
        hi = torch.tensor(values[3:], device=self.device, dtype=torch.float32)
        if bool((hi <= lo).any()):
            raise ValueError(f"Invalid explicit local-random workspace bounds: lo={lo.tolist()} hi={hi.tolist()}.")
        return lo, hi

    def _compute_local_random_workspace_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        explicit_bounds = self._explicit_local_random_workspace_bounds()
        if explicit_bounds is not None:
            return explicit_bounds
        return self._compute_quantile_local_random_workspace_bounds()

    def _compute_quantile_local_random_workspace_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.stable_cache["object_pose_palm"][:, :3].to(device=self.device, dtype=torch.float32)
        q_low = float(self.cfg.local_random_workspace_quantile_low)
        q_high = float(self.cfg.local_random_workspace_quantile_high)
        q_low = min(max(q_low, 0.0), 1.0)
        q_high = min(max(q_high, q_low), 1.0)
        lo = torch.quantile(positions, q_low, dim=0)
        hi = torch.quantile(positions, q_high, dim=0)
        margin = max(0.0, float(self.cfg.local_random_workspace_margin_m))
        lo = lo + margin
        hi = hi - margin
        cfg_lo = torch.tensor(
            [self.cfg.min_cube_palm_x, -self.cfg.max_cube_abs_y, -self.cfg.max_cube_abs_z],
            device=self.device,
            dtype=torch.float32,
        )
        cfg_hi = torch.tensor(
            [self.cfg.max_cube_palm_x, self.cfg.max_cube_abs_y, self.cfg.max_cube_abs_z],
            device=self.device,
            dtype=torch.float32,
        )
        lo = torch.maximum(lo, cfg_lo)
        hi = torch.minimum(hi, cfg_hi)
        bad = hi <= lo
        if bool(bad.any()):
            cache_lo = torch.min(positions, dim=0).values
            cache_hi = torch.max(positions, dim=0).values
            lo = torch.where(bad, torch.maximum(cache_lo, cfg_lo), lo)
            hi = torch.where(bad, torch.minimum(cache_hi, cfg_hi), hi)
        return lo, hi

    def _local_random_workspace_mask_for_pose(self, pose_palm: torch.Tensor) -> torch.Tensor:
        pos = pose_palm[:, :3]
        return ((pos >= self.local_random_workspace_lo) & (pos <= self.local_random_workspace_hi)).all(dim=-1)


    def _workspace_margin_for_pose(self, pose_palm: torch.Tensor) -> torch.Tensor:
        pos = pose_palm[:, :3]
        lo_margin = pos - self.reset_score_workspace_lo.unsqueeze(0)
        hi_margin = self.reset_score_workspace_hi.unsqueeze(0) - pos
        return torch.minimum(lo_margin, hi_margin).min(dim=-1).values

    def _compute_reset_bad_scores(self) -> torch.Tensor:
        pose = self.stable_cache["object_pose_palm"].to(device=self.device, dtype=torch.float32)
        contact_count = self.stable_cache["contact_bits"].to(device=self.device, dtype=torch.float32).sum(dim=-1)
        q = self.stable_cache["q"].to(device=self.device, dtype=torch.float32)
        span = torch.clamp(self.action_upper - self.action_lower, min=1.0e-6).unsqueeze(0)
        q_margin_norm = (torch.minimum(q - self.action_lower.unsqueeze(0), self.action_upper.unsqueeze(0) - q) / span).min(
            dim=-1
        ).values
        workspace_margin = self._workspace_margin_for_pose(pose)

        weighted_terms = (
            float(self.cfg.reset_score_low_z_weight) * (1.0 - rank01(pose[:, 2])),
            float(self.cfg.reset_score_low_q_margin_weight) * (1.0 - rank01(q_margin_norm)),
            float(self.cfg.reset_score_low_contact_weight) * (1.0 - rank01(contact_count)),
            float(self.cfg.reset_score_boundary_weight) * (1.0 - rank01(workspace_margin)),
        )
        weights = (
            float(self.cfg.reset_score_low_z_weight),
            float(self.cfg.reset_score_low_q_margin_weight),
            float(self.cfg.reset_score_low_contact_weight),
            float(self.cfg.reset_score_boundary_weight),
        )
        weight_sum = sum(max(0.0, weight) for weight in weights)
        if weight_sum <= 0.0:
            return torch.zeros(self.num_stable_grasps, device=self.device)
        score = torch.zeros(self.num_stable_grasps, device=self.device)
        for term, weight in zip(weighted_terms, weights, strict=True):
            if weight > 0.0:
                score += term
        return score / weight_sum

    def _build_reset_grasp_candidate_indices(self) -> torch.Tensor | None:
        selection = str(self.cfg.reset_grasp_selection)
        if selection not in {"random", "easy_score", "hard_score"}:
            raise ValueError(
                f"Unsupported reset_grasp_selection={self.cfg.reset_grasp_selection!r}. "
                "Expected 'random', 'easy_score', or 'hard_score'."
            )

        if selection == "random":
            indices = torch.arange(self.num_stable_grasps, device=self.device, dtype=torch.long)
        else:
            self.reset_bad_score[:] = self._compute_reset_bad_scores()
            pool_fraction = min(max(float(self.cfg.reset_grasp_score_pool_fraction), 0.0), 1.0)
            pool_size = max(1, int(round(pool_fraction * self.num_stable_grasps)))
            pool_size = min(pool_size, self.num_stable_grasps)
            order = torch.argsort(self.reset_bad_score)
            indices = order[:pool_size] if selection == "easy_score" else order[-pool_size:]

        filter_mask = torch.ones(indices.shape, device=self.device, dtype=torch.bool)
        if bool(self.cfg.reset_grasp_filter_by_local_random_workspace):
            pose = self.stable_cache["object_pose_palm"][indices].to(device=self.device, dtype=torch.float32)
            filter_mask &= self._local_random_workspace_mask_for_pose(pose)

        min_margin = float(self.cfg.reset_grasp_min_manipulability_q_margin)
        if min_margin >= 0.0:
            if "manipulability_q_margin_min" in self.stable_cache:
                q_margin = self.stable_cache["manipulability_q_margin_min"][indices].to(device=self.device, dtype=torch.float32)
            else:
                q = self.stable_cache["q"][indices].to(device=self.device, dtype=torch.float32)
                span = torch.clamp(self.action_upper - self.action_lower, min=1.0e-6).unsqueeze(0)
                q_margin = (
                    torch.minimum(q - self.action_lower.unsqueeze(0), self.action_upper.unsqueeze(0) - q) / span
                ).min(dim=-1).values
            filter_mask &= q_margin >= min_margin

        indices = indices[filter_mask].contiguous()
        if indices.numel() == 0:
            raise ValueError(
                "Reset grasp filters produced an empty candidate pool. "
                "Relax reset_grasp_selection, reset_grasp_score_pool_fraction, "
                "reset_grasp_filter_by_local_random_workspace, or reset_grasp_min_manipulability_q_margin."
            )
        if selection == "random" and not bool(self.cfg.reset_grasp_filter_by_local_random_workspace) and min_margin < 0.0:
            return None
        return indices

    def _sample_reset_grasp_indices(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioned_cache_kind: str | None = None
        conditioned_ids: torch.Tensor | None = None
        conditioned_offsets: torch.Tensor | None = None
        if self.scale_conditioned_stable_cache_enabled:
            conditioned_cache_kind = "scale-cache bucket"
            conditioned_ids = self.scale_conditioned_cache_bucket_ids_by_env
            conditioned_offsets = self.scale_conditioned_cache_offsets
        elif self.geometry_conditioned_stable_cache_enabled:
            conditioned_cache_kind = "geometry-cache anchor"
            conditioned_ids = self.geometry_conditioned_cache_anchor_ids_by_env
            conditioned_offsets = self.geometry_conditioned_cache_offsets
        if conditioned_ids is not None and conditioned_offsets is not None:
            if env_ids is None:
                raise ValueError(f"{conditioned_cache_kind} reset sampling requires explicit env_ids.")
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if env_ids.shape != (count,):
                raise ValueError(
                    f"Conditioned-cache reset env_ids must have shape {(count,)}, got {tuple(env_ids.shape)}."
                )
            group_ids = conditioned_ids[env_ids]
            sampled = torch.empty((count,), device=self.device, dtype=torch.long)
            for group_id in torch.unique(group_ids).tolist():
                group_mask = group_ids == int(group_id)
                group_count = int(group_mask.sum().item())
                start = int(conditioned_offsets[int(group_id)].item())
                stop = int(conditioned_offsets[int(group_id) + 1].item())
                if stop <= start:
                    raise ValueError(f"{conditioned_cache_kind} {group_id} has an empty row interval.")
                if self.reset_grasp_candidate_indices is None:
                    local = torch.randint(
                        stop - start,
                        (group_count,),
                        device=self.device,
                        generator=self.target_rng,
                        dtype=torch.long,
                    )
                    sampled[group_mask] = local + start
                else:
                    candidates = self.reset_grasp_candidate_indices
                    group_candidates = candidates[(candidates >= start) & (candidates < stop)]
                    if group_candidates.numel() == 0:
                        raise ValueError(
                            f"Reset-grasp filters produced no candidates for {conditioned_cache_kind} {group_id}."
                        )
                    slots = torch.randint(
                        group_candidates.numel(),
                        (group_count,),
                        device=self.device,
                        generator=self.target_rng,
                        dtype=torch.long,
                    )
                    sampled[group_mask] = group_candidates[slots]
            return sampled
        if self.reset_grasp_candidate_indices is None:
            return torch.randint(
                self.num_stable_grasps,
                (count,),
                device=self.device,
                generator=self.target_rng,
                dtype=torch.long,
            )
        pool = self.reset_grasp_candidate_indices
        pool_slots = torch.randint(pool.numel(), (count,), device=self.device, generator=self.target_rng, dtype=torch.long)
        return pool[pool_slots]

    def _sample_multisize_reset_entries(
        self,
        count: int,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if str(self.cfg.reset_grasp_selection) != "random":
            raise ValueError("Multi-object reset currently supports reset_grasp_selection='random' only.")
        num_sizes = len(self.multisize_object_ids)
        if self.primitive_single_asset_per_env_enabled:
            if env_ids is None:
                raise ValueError("single-asset-per-env reset requires env_ids.")
            size_ids = self.fixed_object_size_ids[env_ids]
        elif (
            self.primitive_multi_shape_enabled
            and bool(getattr(self.cfg, "primitive_shape_balanced_sampling", True))
            and len(self.primitive_shape_size_ids) > 0
        ):
            nonempty_shape_pools = [
                candidate_size_ids
                for candidate_size_ids in self.primitive_shape_size_ids.values()
                if candidate_size_ids.numel() > 0
            ]
            if len(nonempty_shape_pools) > 1:
                sampled_shape_slots = torch.randint(
                    len(nonempty_shape_pools),
                    (count,),
                    device=self.device,
                    generator=self.target_rng,
                    dtype=torch.long,
                )
                size_ids = torch.empty(count, device=self.device, dtype=torch.long)
                for shape_slot, candidate_size_ids in enumerate(nonempty_shape_pools):
                    mask = sampled_shape_slots == shape_slot
                    if not bool(mask.any()):
                        continue
                    slots = torch.randint(
                        candidate_size_ids.numel(),
                        (int(mask.sum().item()),),
                        device=self.device,
                        generator=self.target_rng,
                        dtype=torch.long,
                    )
                    size_ids[mask] = candidate_size_ids[slots]
            else:
                size_ids = torch.randint(num_sizes, (count,), device=self.device, generator=self.target_rng, dtype=torch.long)
        else:
            size_ids = torch.randint(num_sizes, (count,), device=self.device, generator=self.target_rng, dtype=torch.long)
        fixed_reset_grasps = getattr(self, "fixed_reset_grasp_indices_by_env", None)
        fixed_indices = None
        if fixed_reset_grasps is not None:
            if env_ids is None:
                raise ValueError("Multi-object fixed reset grasp indices require explicit env_ids.")
            fixed_indices = fixed_reset_grasps[env_ids].to(device=self.device, dtype=torch.long)
            if tuple(fixed_indices.shape) != (count,):
                raise ValueError(
                    f"Multi-object fixed reset grasp indices must have shape {(count,)}, "
                    f"got {tuple(fixed_indices.shape)}."
                )

        q = torch.empty((count, self.action_dim), device=self.device, dtype=torch.float32)
        qdot = torch.empty_like(q)
        q_cmd = torch.empty_like(q)
        object_pose_palm = torch.empty((count, 7), device=self.device, dtype=torch.float32)
        reset_grasp_indices = torch.empty(count, device=self.device, dtype=torch.long)
        anchor_local_pos_m = torch.zeros((count, 3), device=self.device, dtype=torch.float32)
        reset_from_post_success_cache = torch.zeros(count, device=self.device, dtype=torch.bool)
        for size_id, object_id in enumerate(self.multisize_object_ids):
            mask = size_ids == size_id
            if not bool(mask.any()):
                continue
            post_prob = 0.0
            if self.multisize_post_success_reset_sample_probs.numel() > size_id:
                post_prob = float(self.multisize_post_success_reset_sample_probs[size_id].item())
            post_cache = self.multisize_post_success_stable_caches.get(object_id)
            post_mask = torch.zeros_like(mask)
            # An explicit per-environment reset-row contract must never be
            # replaced by the stochastic post-success reset cache.
            if fixed_indices is None and post_cache is not None and post_prob > 0.0:
                object_slots = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                post_choice = (
                    torch.rand(
                        int(object_slots.numel()),
                        device=self.device,
                        generator=self.target_rng,
                    )
                    < post_prob
                )
                if bool(post_choice.any()):
                    post_mask[object_slots[post_choice]] = True
                    reset_from_post_success_cache[post_mask] = True

            normal_mask = mask & ~post_mask
            if bool(normal_mask.any()):
                n = int(normal_mask.sum().item())
                cache = self.multisize_stable_caches[object_id]
                if fixed_indices is not None:
                    indices = fixed_indices[normal_mask]
                    num_grasps = int(self.multisize_num_stable_grasps[object_id])
                    invalid = (indices < 0) | (indices >= num_grasps)
                    if bool(invalid.any()):
                        first_local = int(torch.nonzero(invalid, as_tuple=False)[0].item())
                        bad_index = int(indices[first_local].item())
                        object_slots = torch.nonzero(normal_mask, as_tuple=False).squeeze(-1)
                        bad_env = int(env_ids[object_slots[first_local]].item())
                        raise IndexError(
                            f"Fixed reset grasp index {bad_index} for env {bad_env} and "
                            f"object_id={object_id!r} must be in [0, {num_grasps})."
                        )
                else:
                    head_fraction = self._reset_head_grasp_fraction()
                    if head_fraction is not None:
                        handle_pool, head_pool = self._multisize_grasp_pools(size_id, object_id)
                        indices = self._sample_reset_indices_by_head_fraction(handle_pool, head_pool, n, head_fraction)
                    else:
                        indices = torch.randint(
                            self.multisize_num_stable_grasps[object_id],
                            (n,),
                            device=self.device,
                            generator=self.target_rng,
                            dtype=torch.long,
                        )
                q[normal_mask] = cache["q"][indices].to(dtype=torch.float32)
                qdot[normal_mask] = cache["qdot"][indices].to(dtype=torch.float32)
                q_cmd[normal_mask] = cache["q_cmd"][indices].to(dtype=torch.float32)
                object_pose_palm[normal_mask] = cache["object_pose_palm"][indices].to(dtype=torch.float32)
                anchor_local_pos_m[normal_mask] = self._cache_anchor_local_pos(cache, indices)
                reset_grasp_indices[normal_mask] = indices

            if bool(post_mask.any()):
                n = int(post_mask.sum().item())
                post_indices = torch.randint(
                    self.multisize_num_post_success_stable_grasps[object_id],
                    (n,),
                    device=self.device,
                    generator=self.target_rng,
                    dtype=torch.long,
                )
                q[post_mask] = post_cache["q"][post_indices].to(dtype=torch.float32)
                qdot[post_mask] = post_cache["qdot"][post_indices].to(dtype=torch.float32)
                q_cmd[post_mask] = post_cache["q_cmd"][post_indices].to(dtype=torch.float32)
                object_pose_palm[post_mask] = post_cache["object_pose_palm"][post_indices].to(dtype=torch.float32)
                anchor_local_pos_m[post_mask] = self._cache_anchor_local_pos(post_cache, post_indices)
                reset_grasp_indices[post_mask] = post_indices + self.multisize_num_stable_grasps[object_id]
        return (
            size_ids,
            reset_grasp_indices,
            q,
            qdot,
            q_cmd,
            object_pose_palm,
            anchor_local_pos_m,
            reset_from_post_success_cache,
        )

    def _cache_anchor_local_pos(self, cache: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
        count = int(indices.numel())
        if count <= 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)
        if "anchor_local_pos_m" in cache:
            anchor = cache["anchor_local_pos_m"][indices].to(device=self.device, dtype=torch.float32)
            if tuple(anchor.shape[-1:]) == (3,):
                return anchor
        anchor = torch.zeros((count, 3), device=self.device, dtype=torch.float32)
        if "anchor_y_local_m" in cache:
            anchor[:, 1] = cache["anchor_y_local_m"][indices].to(device=self.device, dtype=torch.float32)
        elif "axial_grasp_s_m" in cache:
            anchor[:, 1] = cache["axial_grasp_s_m"][indices].to(device=self.device, dtype=torch.float32)
        return anchor

    def _anchor_pos_from_pose(self, pose_palm: torch.Tensor, anchor_local_pos_m: torch.Tensor) -> torch.Tensor:
        return pose_palm[:, :3] + quat_apply_wxyz(pose_palm[:, 3:7], anchor_local_pos_m.to(dtype=pose_palm.dtype))

    def _bounds_tensor(self, values: Sequence[float] | torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values.to(device=self.device, dtype=dtype)
        else:
            tensor = torch.tensor(tuple(float(v) for v in values), device=self.device, dtype=dtype)
        if tuple(tensor.shape) != (6,):
            raise ValueError(f"{name} must contain 6 values: min_x, max_x, min_y, max_y, min_z, max_z.")
        return tensor

    def _points_inside_bounds(self, points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        lo = torch.stack((bounds[0], bounds[2], bounds[4]))
        hi = torch.stack((bounds[1], bounds[3], bounds[5]))
        return ((points >= lo) & (points <= hi)).all(dim=-1)

    def _pose6d_feasible_target_workspace_bounds(self, dtype: torch.dtype) -> torch.Tensor:
        values = _tuple_cfg(getattr(self.cfg, "pose6d_feasible_target_workspace_bounds_m", ()))
        if len(values) == 0:
            values = _tuple_cfg(
                getattr(self.cfg, "pose6d_anchor_workspace_bounds_m", (0.020, 0.070, -0.040, 0.040, 0.070, 0.130))
            )
        return self._bounds_tensor(values, name="pose6d_feasible_target_workspace_bounds_m", dtype=dtype)

    def _feasible_graspable_long_axis_range(self) -> tuple[float, float] | None:
        """Optional object-local [min, max] graspable span along the long axis.

        When configured, feasibility proxy points are restricted to this span so a
        bulky non-graspable end (e.g. the hammer head) is excluded from both target
        sampling and head-grasp reset filtering. ``None`` keeps the full-bbox range.
        """

        lo = getattr(self.cfg, "pose6d_feasible_target_graspable_long_axis_min_m", None)
        hi = getattr(self.cfg, "pose6d_feasible_target_graspable_long_axis_max_m", None)
        if lo is None or hi is None:
            return None
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        return (lo_f, hi_f)

    def _pose6d_feasible_proxy_points_from_dims(
        self, dims_m: torch.Tensor, dtype: torch.dtype, *, force_full_range: bool = False
    ) -> torch.Tensor:
        count = int(dims_m.shape[0])
        samples = max(1, int(getattr(self.cfg, "pose6d_feasible_target_proxy_samples", 9)))
        points = torch.zeros((count, samples, 3), device=self.device, dtype=dtype)
        mode = str(getattr(self.cfg, "pose6d_feasible_target_proxy_mode", "dominant_axis")).lower()
        if mode == "center" or samples <= 1:
            return points
        if mode not in {"dominant_axis", "local_y"}:
            raise ValueError(
                "pose6d_feasible_target_proxy_mode must be 'dominant_axis', 'local_y', or 'center', "
                f"got {getattr(self.cfg, 'pose6d_feasible_target_proxy_mode', None)!r}."
            )

        dims_m = dims_m.to(device=self.device, dtype=dtype)
        if mode == "local_y":
            axis_ids = torch.full((count,), 1, device=self.device, dtype=torch.long)
        else:
            axis_ids = torch.argmax(dims_m, dim=-1)
        half = 0.5 * dims_m.gather(1, axis_ids.unsqueeze(-1)).squeeze(-1)
        graspable = None if force_full_range else self._feasible_graspable_long_axis_range()
        if graspable is not None:
            lo_val, hi_val = graspable
            lo = torch.minimum(torch.maximum(torch.full_like(half, lo_val), -half), half)
            hi = torch.minimum(torch.maximum(torch.full_like(half, hi_val), -half), half)
            frac = torch.linspace(0.0, 1.0, samples, device=self.device, dtype=dtype).unsqueeze(0)
            axis_values = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * frac
        else:
            margin = max(float(getattr(self.cfg, "pose6d_feasible_target_proxy_endpoint_margin_m", 0.0)), 0.0)
            half_extent = torch.clamp(half - margin, min=0.0)
            unit = torch.linspace(-1.0, 1.0, samples, device=self.device, dtype=dtype).unsqueeze(0)
            axis_values = half_extent.unsqueeze(-1) * unit
        for axis in range(3):
            mask = axis_ids == axis
            if bool(mask.any()):
                points[mask, :, axis] = axis_values[mask]
        return points

    def _pose6d_feasible_proxy_points_local(self, env_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self._pose6d_feasible_proxy_points_from_dims(self.active_object_dims_m[env_ids], dtype)

    def _reset_head_grasp_fraction(self) -> float | None:
        """Target fraction of resets that start from a head grasp, or None (natural).

        Returns None when the graspable long-axis range is unset (feature off) or
        when neither the fraction nor the exclude flag is configured.
        """
        if self._feasible_graspable_long_axis_range() is None:
            return None
        frac = getattr(self.cfg, "pose6d_feasible_target_reset_head_grasp_fraction", None)
        if frac is not None:
            return float(min(max(float(frac), 0.0), 1.0))
        if bool(getattr(self.cfg, "pose6d_feasible_target_reset_exclude_head_enabled", False)):
            return 0.0
        return None

    def _grip_is_head_grasp_mask(self, object_pose_palm: torch.Tensor, dims_row: torch.Tensor) -> torch.Tensor:
        """True where a grasp is a HEAD grasp.

        A grasp is a head grasp when the object's full-centerline segment that falls
        inside the thumb workspace has its centroid beyond the handle's graspable max
        along the long axis (i.e. the bulky head is what the hand is holding). Grasps
        with no centerline inside the thumb workspace are not head grasps.
        """
        graspable = self._feasible_graspable_long_axis_range()
        if graspable is None:
            return torch.zeros(int(object_pose_palm.shape[0]), device=self.device, dtype=torch.bool)
        handle_max = float(graspable[1])
        dtype = torch.float32
        poses = object_pose_palm.to(device=self.device, dtype=dtype)
        m = int(poses.shape[0])
        dims_m = dims_row.to(device=self.device, dtype=dtype).view(1, 3).expand(m, 3)
        proxy_local = self._pose6d_feasible_proxy_points_from_dims(dims_m, dtype, force_full_range=True)
        samples = int(proxy_local.shape[1])
        long_axis = self._pose6d_feasible_target_object_long_axis_index()
        proxy_palm = poses[:, None, :3] + quat_apply_wxyz(poses[:, None, 3:7].expand(m, samples, 4), proxy_local)
        bounds = self._pose6d_feasible_target_workspace_bounds(dtype)
        inside = self._points_inside_bounds(proxy_palm.reshape(-1, 3), bounds).view(m, samples)
        hit_count = inside.sum(dim=-1)
        local_long = proxy_local[:, :, long_axis]
        hit_sum = (local_long * inside.to(dtype=dtype)).sum(dim=-1)
        hit_y = torch.where(hit_count > 0, hit_sum / torch.clamp(hit_count.to(dtype=dtype), min=1.0), torch.zeros_like(hit_sum))
        return (hit_count > 0) & (hit_y > handle_max)

    def _cache_object_dims_row(self, cache: dict, fallback_size_id: int | None) -> torch.Tensor:
        dims_raw = cache.get("object_dims_m")
        if dims_raw is None and isinstance(cache.get("metadata"), dict):
            dims_raw = cache["metadata"].get("object_dims_m")
        if dims_raw is not None:
            return torch.as_tensor(dims_raw, device=self.device, dtype=torch.float32).view(3)
        if fallback_size_id is not None and hasattr(self, "multiobject_dims_m"):
            return self.multiobject_dims_m[fallback_size_id].to(device=self.device, dtype=torch.float32).view(3)
        return self.active_object_dims_m[0].to(device=self.device, dtype=torch.float32).view(3)

    def _multisize_grasp_pools(self, size_id: int, object_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (handle_pool, head_pool) index tensors into the object's stable cache."""
        pools = getattr(self, "_multisize_grasp_pools_cache", None)
        if pools is None:
            pools = {}
            self._multisize_grasp_pools_cache = pools
        if object_id not in pools:
            cache = self.multisize_stable_caches[object_id]
            poses = cache["object_pose_palm"].to(device=self.device, dtype=torch.float32)
            dims_row = self._cache_object_dims_row(cache, size_id)
            head_mask = self._grip_is_head_grasp_mask(poses, dims_row)
            total = int(poses.shape[0])
            head_idx = torch.nonzero(head_mask, as_tuple=False).squeeze(-1).to(dtype=torch.long)
            handle_idx = torch.nonzero(~head_mask, as_tuple=False).squeeze(-1).to(dtype=torch.long)
            print(
                f"[INFO] reset grasp pools for {object_id!r}: handle={int(handle_idx.numel())}, "
                f"head={int(head_idx.numel())} ({100.0 * head_idx.numel() / max(total, 1):.1f}% head) of {total}.",
                flush=True,
            )
            pools[object_id] = (handle_idx, head_idx)
        return pools[object_id]

    def _sample_reset_indices_by_head_fraction(
        self, handle_pool: torch.Tensor, head_pool: torch.Tensor, n: int, head_fraction: float
    ) -> torch.Tensor:
        n_head = int(round(head_fraction * n))
        if head_pool.numel() == 0:
            n_head = 0
        if handle_pool.numel() == 0:
            n_head = n
        n_handle = n - n_head
        parts: list[torch.Tensor] = []
        if n_handle > 0:
            slots = torch.randint(handle_pool.numel(), (n_handle,), device=self.device, generator=self.target_rng, dtype=torch.long)
            parts.append(handle_pool[slots])
        if n_head > 0:
            slots = torch.randint(head_pool.numel(), (n_head,), device=self.device, generator=self.target_rng, dtype=torch.long)
            parts.append(head_pool[slots])
        indices = torch.cat(parts, dim=0)
        perm = torch.randperm(indices.numel(), device=self.device, generator=self.target_rng)
        return indices[perm]

    def _pose6d_forbidden_sdf_field(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the configured XYZ SDF once and retain a GPU grid_sample view."""

        cached = getattr(self, "_pose6d_forbidden_sdf_cache", None)
        if cached is not None:
            return cached
        configured = str(
            getattr(self.cfg, "pose6d_feasible_target_forbidden_sdf_path", "")
        ).strip()
        if not configured:
            raise ValueError(
                "sdf_grid feasible-target forbidden mode requires "
                "pose6d_feasible_target_forbidden_sdf_path."
            )
        path = resolve_workspace_path(configured)
        if not path.is_file():
            raise FileNotFoundError(f"Configured forbidden SDF does not exist: {path}")
        expected_sha256 = str(
            getattr(self.cfg, "pose6d_feasible_target_forbidden_sdf_sha256", "")
        ).strip()
        if expected_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise ValueError(
                    "Configured forbidden SDF SHA256 mismatch: "
                    f"expected {expected_sha256}, got {digest}."
                )
        with np.load(path, allow_pickle=False) as payload:
            sdf_xyz = np.asarray(payload["sdf_m"], dtype=np.float32)
            origin = np.asarray(payload["origin_m"], dtype=np.float32)
            voxel = float(np.asarray(payload["voxel_size_m"]).item())
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        if sdf_xyz.ndim != 3 or min(sdf_xyz.shape) < 2:
            raise ValueError(f"Forbidden SDF must be a 3D field, got {sdf_xyz.shape}.")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Forbidden SDF origin must be finite XYZ.")
        if not np.isfinite(voxel) or voxel <= 0.0 or not np.all(np.isfinite(sdf_xyz)):
            raise ValueError("Forbidden SDF values/voxel size are invalid.")
        if metadata.get("sign") != "positive_inside_negative_outside":
            raise ValueError("Forbidden SDF sign convention is unsafe.")
        # grid_sample expects [N,C,D(Z),H(Y),W(X)].
        field = torch.from_numpy(sdf_xyz).to(device=self.device).permute(2, 1, 0)
        field = field.contiguous().view(1, 1, *field.shape)
        origin_t = torch.as_tensor(origin, device=self.device, dtype=torch.float32)
        maximum_t = origin_t + voxel * torch.as_tensor(
            np.asarray(sdf_xyz.shape, dtype=np.float32) - 1.0,
            device=self.device,
        )
        cached = (field, origin_t, maximum_t)
        self._pose6d_forbidden_sdf_cache = cached
        print(
            "[INFO] loaded feasible-target forbidden SDF: "
            f"{path} shape={tuple(sdf_xyz.shape)} voxel={voxel * 1000.0:.1f}mm",
            flush=True,
        )
        return cached

    def _pose6d_forbidden_sdf_hits(
        self,
        proxy_palm: torch.Tensor,
        clearance_m: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized trilinear SDF predicate; outside-grid points are allowed."""

        field, origin, maximum = self._pose6d_forbidden_sdf_field()
        points = proxy_palm.to(dtype=torch.float32)
        valid = ((points >= origin) & (points <= maximum)).all(dim=-1)
        normalized = 2.0 * (points - origin) / (maximum - origin) - 1.0
        # One input batch with D=count, H=samples, W=1 query locations.
        query = normalized.unsqueeze(0).unsqueeze(3)
        sampled = F.grid_sample(
            field,
            query,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, :, 0]
        threshold = clearance_m.to(dtype=torch.float32)
        if threshold.ndim == 1:
            threshold = threshold.unsqueeze(-1)
        if threshold.ndim != 2 or threshold.shape != sampled.shape:
            raise ValueError(
                "Forbidden SDF clearance must have shape [env] or [env, proxy]."
            )
        return (valid & (sampled >= -threshold)).any(dim=-1)

    def _pose6d_compound_sdf_proxy_for_envs(
        self,
        env_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned compound-sphere proxy points/radii for each env.

        Multi-object tasks may pin one fixed-size sphere cover per object. The
        configured object order must exactly match the environment object order
        so active_cube_size_id is a safe gather index. Empty per-object fields
        retain the historical single-object proxy behavior.
        """

        count = int(env_ids.numel())
        per_object_ids = tuple(
            str(value)
            for value in getattr(
                self.cfg,
                "pose6d_feasible_target_forbidden_sdf_proxy_per_object_ids",
                (),
            )
        )
        per_object_points = tuple(
            getattr(
                self.cfg,
                "pose6d_feasible_target_forbidden_sdf_proxy_per_object_points_local_m",
                (),
            )
        )
        per_object_radii = tuple(
            getattr(
                self.cfg,
                "pose6d_feasible_target_forbidden_sdf_proxy_per_object_radii_m",
                (),
            )
        )
        any_per_object = bool(per_object_ids or per_object_points or per_object_radii)
        if any_per_object:
            expected_ids = tuple(str(value) for value in self.multisize_object_ids)
            if per_object_ids != expected_ids:
                raise ValueError(
                    "Per-object compound SDF proxy IDs must exactly match "
                    f"multisize_object_ids: expected {expected_ids}, got {per_object_ids}."
                )
            cache = getattr(self, "_pose6d_compound_sdf_proxy_per_object_cache", None)
            if cache is None:
                points = torch.as_tensor(
                    per_object_points, device=self.device, dtype=torch.float32
                )
                radii = torch.as_tensor(
                    per_object_radii, device=self.device, dtype=torch.float32
                )
                object_count = len(expected_ids)
                if (
                    points.ndim != 3
                    or points.shape[0] != object_count
                    or points.shape[1] == 0
                    or points.shape[2] != 3
                    or radii.shape != points.shape[:2]
                    or not bool(torch.isfinite(points).all())
                    or not bool(torch.isfinite(radii).all())
                    or bool((radii < 0.0).any())
                ):
                    raise ValueError(
                        "Per-object compound SDF proxies require finite "
                        "[object, proxy, 3] points and matching non-negative "
                        "[object, proxy] radii with at least one proxy point."
                    )
                cache = (points, radii)
                self._pose6d_compound_sdf_proxy_per_object_cache = cache
            points, radii = cache
            active = self.active_cube_size_id[env_ids].to(dtype=torch.long)
            gathered_points = points[active].to(dtype=dtype)
            gathered_radii = radii[active].to(dtype=dtype)
            if bool(getattr(self, "object_scale_randomization_enabled", False)):
                scale = self.object_scale_values[env_ids].to(dtype=dtype)
                gathered_points = gathered_points * scale[:, None, None]
                gathered_radii = gathered_radii * scale[:, None]
            return gathered_points, gathered_radii

        local_raw = tuple(
            getattr(
                self.cfg,
                "pose6d_feasible_target_forbidden_sdf_proxy_points_local_m",
                (),
            )
        )
        radii_raw = tuple(
            getattr(
                self.cfg,
                "pose6d_feasible_target_forbidden_sdf_proxy_radii_m",
                (),
            )
        )
        collision_local = torch.as_tensor(local_raw, device=self.device, dtype=dtype)
        collision_radii = torch.as_tensor(radii_raw, device=self.device, dtype=dtype)
        if (
            collision_local.ndim != 2
            or collision_local.shape[1] != 3
            or collision_local.shape[0] == 0
            or collision_radii.shape != (collision_local.shape[0],)
            or not bool(torch.isfinite(collision_local).all())
            or not bool(torch.isfinite(collision_radii).all())
            or bool((collision_radii < 0.0).any())
        ):
            raise ValueError(
                "Compound SDF proxy requires matching non-empty finite local "
                "XYZ points and non-negative finite radii."
            )
        return (
            collision_local.unsqueeze(0).expand(count, -1, -1),
            collision_radii.unsqueeze(0).expand(count, -1),
        )

    def _pose6d_target_proxy_workspace_info(
        self,
        target_pose: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(env_ids.numel())
        dtype = target_pose.dtype
        proxy_local = self._pose6d_feasible_proxy_points_local(env_ids, dtype)
        samples = int(proxy_local.shape[1])
        proxy_palm = target_pose[:, None, :3] + quat_apply_wxyz(
            target_pose[:, None, 3:7].expand(count, samples, 4),
            proxy_local,
        )
        bounds = self._pose6d_feasible_target_workspace_bounds(dtype)
        inside = self._points_inside_bounds(proxy_palm.reshape(-1, 3), bounds).view(count, samples)
        hit_count = inside.sum(dim=-1).to(dtype=dtype)
        hit_y_sum = (proxy_local[:, :, 1] * inside.to(dtype=dtype)).sum(dim=-1)
        hit_y = torch.where(hit_count > 0.0, hit_y_sum / torch.clamp(hit_count, min=1.0), torch.zeros_like(hit_y_sum))
        forbidden_hit = torch.zeros(count, device=self.device, dtype=torch.bool)
        if bool(getattr(self.cfg, "pose6d_feasible_target_forbidden_block_enabled", True)):
            dims_m = self.active_object_dims_m[env_ids].to(dtype=dtype)
            radius = 0.5 * torch.minimum(dims_m[:, 0], dims_m[:, 2])
            forbidden_mode = str(
                getattr(
                    self.cfg,
                    "pose6d_feasible_target_forbidden_mode",
                    "legacy_palm_wrist_boxes",
                )
            ).lower()
            if forbidden_mode == "sdf_grid":
                allowed_penetration_m = max(
                    float(
                        getattr(
                            self.cfg,
                            "pose6d_feasible_target_forbidden_sdf_allowed_penetration_m",
                            0.0,
                        )
                    ),
                    0.0,
                )
                clearance_base = (
                    max(
                        float(
                            getattr(
                                self.cfg,
                                "pose6d_feasible_target_forbidden_sdf_margin_m",
                                0.0,
                            )
                        ),
                        0.0,
                    )
                    + max(
                        float(
                            getattr(
                                self.cfg,
                                "pose6d_feasible_target_forbidden_sdf_interpolation_guard_m",
                                0.0,
                            )
                        ),
                        0.0,
                    )
                    - allowed_penetration_m
                )
                sdf_proxy_mode = str(
                    getattr(
                        self.cfg,
                        "pose6d_feasible_target_forbidden_sdf_proxy_mode",
                        "uniform_object_radius",
                    )
                ).strip().lower()
                if sdf_proxy_mode in {
                    "compound_sphere_cover_v1",
                    "hexhammer_compound_sphere_cover_v1",
                }:
                    collision_local, collision_radii = (
                        self._pose6d_compound_sdf_proxy_for_envs(env_ids, dtype)
                    )
                    collision_samples = int(collision_local.shape[1])
                    collision_palm = target_pose[:, None, :3] + quat_apply_wxyz(
                        target_pose[:, None, 3:7].expand(
                            count, collision_samples, 4
                        ),
                        collision_local,
                    )
                    clearance = torch.clamp(
                        clearance_base + collision_radii,
                        min=0.0,
                    )
                    if bool(
                        getattr(
                            self.cfg,
                            "pose6d_feasible_target_forbidden_sdf_inflate_by_object_radius",
                            False,
                        )
                    ):
                        raise ValueError(
                            "Compound SDF proxy already carries per-sphere radii; "
                            "uniform object-radius inflation must be disabled."
                        )
                elif sdf_proxy_mode == "uniform_object_radius":
                    collision_palm = proxy_palm
                    clearance = torch.full(
                        (count,), clearance_base, device=self.device, dtype=dtype
                    )
                    if bool(
                        getattr(
                            self.cfg,
                            "pose6d_feasible_target_forbidden_sdf_inflate_by_object_radius",
                            False,
                        )
                    ):
                        clearance = clearance + radius
                    clearance = torch.clamp(clearance, min=0.0)
                else:
                    raise ValueError(
                        "Unsupported feasible-target forbidden SDF proxy mode: "
                        f"{sdf_proxy_mode!r}."
                    )
                forbidden_hit = self._pose6d_forbidden_sdf_hits(
                    collision_palm, clearance
                )
            elif forbidden_mode == "finite_boxes":
                blocks_raw = tuple(
                    getattr(
                        self.cfg,
                        "pose6d_feasible_target_forbidden_block_bounds_m",
                        (),
                    )
                )
                if not blocks_raw:
                    raise ValueError(
                        "finite_boxes feasible-target forbidden mode requires "
                        "pose6d_feasible_target_forbidden_block_bounds_m."
                    )
                blocks = torch.tensor(blocks_raw, device=self.device, dtype=dtype)
                if blocks.ndim != 2 or tuple(blocks.shape[1:]) != (6,):
                    raise ValueError(
                        "pose6d_feasible_target_forbidden_block_bounds_m must "
                        "be a sequence of 6D bounds."
                    )
                names = tuple(
                    str(value)
                    for value in getattr(
                        self.cfg,
                        "pose6d_feasible_target_forbidden_block_names",
                        (),
                    )
                )
                if names and len(names) != int(blocks.shape[0]):
                    raise ValueError(
                        "pose6d_feasible_target_forbidden_block_names must be "
                        "empty or match the finite block count."
                    )
                if bool(
                    (
                        (blocks[:, 0] > blocks[:, 1])
                        | (blocks[:, 2] > blocks[:, 3])
                        | (blocks[:, 4] > blocks[:, 5])
                    ).any()
                ):
                    raise ValueError("Each feasible-target forbidden block must have ordered bounds.")
                inflate = torch.full(
                    (count,),
                    max(
                        float(
                            getattr(
                                self.cfg,
                                "pose6d_feasible_target_forbidden_block_margin_m",
                                0.0,
                            )
                        ),
                        0.0,
                    ),
                    device=self.device,
                    dtype=dtype,
                )
                if bool(
                    getattr(
                        self.cfg,
                        "pose6d_feasible_target_forbidden_block_inflate_by_object_radius",
                        False,
                    )
                ):
                    inflate = inflate + radius
                mins = torch.stack((blocks[:, 0], blocks[:, 2], blocks[:, 4]), dim=-1)
                maxs = torch.stack((blocks[:, 1], blocks[:, 3], blocks[:, 5]), dim=-1)
                forbidden_hit = (
                    (
                        proxy_palm[:, None, :, :]
                        >= mins[None, :, None, :] - inflate[:, None, None, None]
                    )
                    & (
                        proxy_palm[:, None, :, :]
                        <= maxs[None, :, None, :] + inflate[:, None, None, None]
                    )
                ).all(dim=-1).any(dim=(1, 2))
            elif forbidden_mode == "legacy_palm_wrist_boxes":
                palm_min_x = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_min_x_m", -0.020)) - radius
                palm_max_x = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_x_m", 0.080)) + radius
                palm_max_abs_y = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_abs_y_m", 0.070)) + radius
                palm_max_z = float(getattr(self.cfg, "pose6d_anchor_palm_forbidden_max_z_m", 0.030)) + radius
                in_palm_block = (
                    (proxy_palm[..., 0] >= palm_min_x.unsqueeze(-1))
                    & (proxy_palm[..., 0] <= palm_max_x.unsqueeze(-1))
                    & (torch.abs(proxy_palm[..., 1]) <= palm_max_abs_y.unsqueeze(-1))
                    & (proxy_palm[..., 2] <= palm_max_z.unsqueeze(-1))
                ).any(dim=-1)

                wrist_min_x = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_min_x_m", -0.080)) - radius
                wrist_max_x = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_x_m", 0.005)) + radius
                wrist_max_abs_y = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_abs_y_m", 0.090)) + radius
                wrist_max_z = float(getattr(self.cfg, "pose6d_anchor_wrist_forbidden_max_z_m", 0.090)) + radius
                in_wrist_block = (
                    (proxy_palm[..., 0] >= wrist_min_x.unsqueeze(-1))
                    & (proxy_palm[..., 0] <= wrist_max_x.unsqueeze(-1))
                    & (torch.abs(proxy_palm[..., 1]) <= wrist_max_abs_y.unsqueeze(-1))
                    & (proxy_palm[..., 2] <= wrist_max_z.unsqueeze(-1))
                ).any(dim=-1)
                forbidden_hit = in_palm_block | in_wrist_block
            else:
                raise ValueError(
                    "pose6d_feasible_target_forbidden_mode must be "
                    "'legacy_palm_wrist_boxes', 'finite_boxes', or 'sdf_grid'; got "
                    f"{forbidden_mode!r}."
                )
        return hit_count > 0, hit_count, hit_y, forbidden_hit

    def _perpendicular_basis_from_axis(self, axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(axis.shape[0])
        x_ref = torch.zeros((count, 3), device=self.device, dtype=axis.dtype)
        x_ref[:, 0] = 1.0
        z_ref = torch.zeros_like(x_ref)
        z_ref[:, 2] = 1.0
        ref = torch.where(torch.abs(axis[:, :1]) < 0.8, x_ref, z_ref)
        perp1 = torch.linalg.cross(axis, ref, dim=-1)
        perp1 = perp1 / torch.clamp(torch.linalg.norm(perp1, dim=-1, keepdim=True), min=1.0e-6)
        perp2 = torch.linalg.cross(axis, perp1, dim=-1)
        perp2 = perp2 / torch.clamp(torch.linalg.norm(perp2, dim=-1, keepdim=True), min=1.0e-6)
        return perp1, perp2

    def _current_pose6d_anchor_perp_delta_max_m(self) -> float:
        start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.010))
        end = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", start))
        perp_start = float(getattr(self.cfg, "pose6d_anchor_perp_translation_curriculum_start_m", 0.0))
        perp_end = float(getattr(self.cfg, "pose6d_anchor_perp_translation_curriculum_end_m", perp_start))
        if abs(end - start) <= 1.0e-8:
            return perp_end
        alpha = (float(self.adaptive_translation_delta_max.item()) - start) / (end - start)
        alpha = min(max(alpha, 0.0), 1.0)
        return perp_start + alpha * (perp_end - perp_start)

    def _pose6d_anchor_perp_delta_max_m_for_envs(self, env_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if not self._multiobject_per_object_curriculum_enabled():
            return torch.full(
                (int(env_ids.numel()),),
                float(self._current_pose6d_anchor_perp_delta_max_m()),
                device=self.device,
                dtype=dtype,
            )
        start = float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.010))
        end = float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", start))
        perp_start = float(getattr(self.cfg, "pose6d_anchor_perp_translation_curriculum_start_m", 0.0))
        perp_end = float(getattr(self.cfg, "pose6d_anchor_perp_translation_curriculum_end_m", perp_start))
        if abs(end - start) <= 1.0e-8:
            return torch.full((int(env_ids.numel()),), perp_end, device=self.device, dtype=dtype)
        object_ids = self.active_cube_size_id[env_ids].to(dtype=torch.long)
        current_max = self.adaptive_translation_delta_max_by_object[object_ids].to(dtype=dtype)
        alpha = torch.clamp((current_max - start) / (end - start), min=0.0, max=1.0)
        return perp_start + alpha * (perp_end - perp_start)

    def _random_unit_vectors(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        vec = torch.randn((count, 3), device=self.device, dtype=dtype, generator=self.target_rng)
        return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=1.0e-6)





    def _pose6d_feasible_target_object_long_axis_index(self) -> int:
        axis_name = str(getattr(self.cfg, "pose6d_feasible_target_object_long_axis", "y")).lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis_name not in axis_map:
            raise ValueError(
                "pose6d_feasible_target_object_long_axis must be one of 'x', 'y', or 'z'; "
                f"got {getattr(self.cfg, 'pose6d_feasible_target_object_long_axis', None)!r}."
            )
        return axis_map[axis_name]

    def _sample_pose6d_feasible_target_translation_offset(
        self,
        source_pose_palm: torch.Tensor,
        env_ids: torch.Tensor,
        translation_delta: torch.Tensor,
        translation_delta_max: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(source_pose_palm.shape[0])
        dtype = source_pose_palm.dtype
        mode = str(getattr(self.cfg, "pose6d_feasible_target_translation_mode", "random_axis")).lower()
        if mode in {"random_axis", "random_direction", "isotropic_direction"}:
            translation_dir = self._sample_pose6d_translation_axes_palm(count, dtype)
            return translation_dir * translation_delta.unsqueeze(-1), translation_dir
        if mode not in {
            "object_local_box",
            "local_box",
            "object_local_long_frontier",
            "local_long_frontier",
            "long_frontier",
            "object_local_short_frontier",
            "local_short_frontier",
            "short_frontier",
        }:
            raise ValueError(
                "pose6d_feasible_target_translation_mode must be one of "
                "'random_axis', 'object_local_box', 'object_local_long_frontier', or "
                f"'object_local_short_frontier'; got {getattr(self.cfg, 'pose6d_feasible_target_translation_mode', None)!r}."
            )

        long_axis = self._pose6d_feasible_target_object_long_axis_index()
        short_axes = tuple(axis for axis in range(3) if axis != long_axis)
        long_max = torch.clamp(translation_delta.to(dtype=dtype), min=0.0)
        if translation_delta_max is None:
            short_source = long_max
        else:
            short_source = torch.clamp(translation_delta_max.to(dtype=dtype), min=0.0)
        short_fraction = max(float(getattr(self.cfg, "pose6d_feasible_target_short_axis_fraction", 1.0)), 0.0)
        short_max = short_source * short_fraction

        local_delta = torch.zeros((count, 3), device=self.device, dtype=dtype)
        if mode in {"object_local_box", "local_box"}:
            local_delta[:, long_axis] = (
                torch.rand((count,), device=self.device, dtype=dtype, generator=self.target_rng) * 2.0 - 1.0
            ) * long_max
        elif mode in {"object_local_long_frontier", "local_long_frontier", "long_frontier"}:
            sign = torch.where(
                torch.rand((count,), device=self.device, dtype=dtype, generator=self.target_rng) < 0.5,
                torch.full((count,), -1.0, device=self.device, dtype=dtype),
                torch.full((count,), 1.0, device=self.device, dtype=dtype),
            )
            local_delta[:, long_axis] = sign * long_max
        else:
            local_delta[:, long_axis] = (
                torch.rand((count,), device=self.device, dtype=dtype, generator=self.target_rng) * 2.0 - 1.0
            ) * long_max

        for axis in short_axes:
            if mode in {"object_local_short_frontier", "local_short_frontier", "short_frontier"}:
                sign = torch.where(
                    torch.rand((count,), device=self.device, dtype=dtype, generator=self.target_rng) < 0.5,
                    torch.full((count,), -1.0, device=self.device, dtype=dtype),
                    torch.full((count,), 1.0, device=self.device, dtype=dtype),
                )
                local_delta[:, axis] = sign * short_max
            else:
                local_delta[:, axis] = (
                    torch.rand((count,), device=self.device, dtype=dtype, generator=self.target_rng) * 2.0 - 1.0
                ) * short_max

        fallback_local_axis = torch.zeros_like(local_delta)
        fallback_local_axis[:, long_axis] = 1.0
        source_quat = source_pose_palm[:, 3:7]
        translation_offset = quat_apply_wxyz(source_quat, local_delta)
        fallback_axis = quat_apply_wxyz(source_quat, fallback_local_axis)
        fallback_axis = fallback_axis / torch.clamp(
            torch.linalg.norm(fallback_axis, dim=-1, keepdim=True),
            min=1.0e-6,
        )
        return translation_offset, fallback_axis

    def _sample_pose6d_feasible_object_targets(
        self,
        env_ids: torch.Tensor,
        source_pose_palm: torch.Tensor,
        target_quat_palm: torch.Tensor,
        translation_delta: torch.Tensor,
        translation_delta_max: torch.Tensor | None = None,
        target_twist_axis: torch.Tensor | None = None,
        target_twist_angle: torch.Tensor | None = None,
        target_twist_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        count = int(env_ids.numel())
        dtype = source_pose_palm.dtype
        target_pose = source_pose_palm.clone()
        target_pose[:, 3:7] = target_quat_palm
        translation_axis = self.target_translation_axis_palm[env_ids].to(dtype=dtype)
        selected_twist_axis = (
            target_twist_axis.to(dtype=dtype).clone()
            if target_twist_axis is not None
            else torch.zeros((count, 3), device=self.device, dtype=dtype)
        )
        selected_twist_angle = (
            target_twist_angle.to(dtype=dtype).clone()
            if target_twist_angle is not None
            else torch.zeros(count, device=self.device, dtype=dtype)
        )
        actual_delta = torch.zeros(count, device=self.device, dtype=dtype)
        accepted = torch.zeros(count, device=self.device, dtype=torch.bool)
        fail = torch.ones(count, device=self.device, dtype=dtype)
        reject_count = torch.zeros(count, device=self.device, dtype=dtype)
        forbidden_reject_count = torch.zeros(count, device=self.device, dtype=dtype)
        attempt_count = torch.zeros(count, device=self.device, dtype=dtype)
        proxy_hit_count = torch.zeros(count, device=self.device, dtype=dtype)
        hit_proxy_y = torch.zeros(count, device=self.device, dtype=dtype)
        max_attempts = max(1, int(getattr(self.cfg, "pose6d_feasible_target_max_attempts", 16)))
        resample_rotation = bool(getattr(self.cfg, "pose6d_feasible_target_resample_rotation_each_attempt", False))
        resample_rotation = resample_rotation and target_twist_delta is not None

        for _ in range(max_attempts):
            unresolved = torch.nonzero(~accepted, as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            translation_delta_max_subset = None
            if translation_delta_max is not None:
                translation_delta_max_subset = translation_delta_max[unresolved]
            translation_offset, fallback_translation_axis = self._sample_pose6d_feasible_target_translation_offset(
                source_pose_palm[unresolved],
                env_ids[unresolved],
                translation_delta[unresolved],
                translation_delta_max_subset,
            )
            cand_pose = source_pose_palm[unresolved].clone()
            attempt_twist_axis = selected_twist_axis[unresolved]
            attempt_twist_angle = selected_twist_angle[unresolved]
            if resample_rotation:
                attempt_twist_axis = self._sample_twist_axes_palm(int(unresolved.numel()), dtype)
                if bool(getattr(self.cfg, "randomize_target_twist_direction", True)):
                    direction = torch.randint(
                        0,
                        2,
                        (int(unresolved.numel()),),
                        device=self.device,
                        dtype=torch.long,
                        generator=self.target_rng,
                    ).to(dtype=dtype)
                    direction = direction * 2.0 - 1.0
                else:
                    direction = torch.full(
                        (int(unresolved.numel()),),
                        float(getattr(self.cfg, "target_direction", 1.0)),
                        device=self.device,
                        dtype=dtype,
                    )
                    direction = torch.where(direction >= 0.0, torch.ones_like(direction), -torch.ones_like(direction))
                attempt_twist_angle = direction * target_twist_delta[unresolved].to(dtype=dtype)
                delta_quat = quat_from_angle_axis_wxyz(attempt_twist_angle, attempt_twist_axis)
                cand_pose[:, 3:7] = normalize_quat_wxyz(
                    quat_mul_wxyz(delta_quat, source_pose_palm[unresolved, 3:7])
                )
            else:
                cand_pose[:, 3:7] = target_quat_palm[unresolved]
            cand_pos = source_pose_palm[unresolved, :3] + translation_offset
            cand_pose[:, :3] = cand_pos
            workspace_ok, hit_count, hit_y, forbidden_hit = self._pose6d_target_proxy_workspace_info(
                cand_pose,
                env_ids[unresolved],
            )
            ok = workspace_ok & (~forbidden_hit)
            attempt_count[unresolved] += 1.0
            reject_count[unresolved] += (~workspace_ok).to(dtype=dtype)
            forbidden_reject_count[unresolved] += forbidden_hit.to(dtype=dtype)

            if bool(ok.any()):
                slots = unresolved[ok]
                target_pose[slots] = cand_pose[ok]
                target_offset = target_pose[slots, :3] - source_pose_palm[slots, :3]
                delta_norm = torch.linalg.norm(target_offset, dim=-1)
                safe_norm = torch.clamp(delta_norm, min=1.0e-6)
                translation_axis[slots] = target_offset / safe_norm.unsqueeze(-1)
                translation_axis[slots] = torch.where(
                    (delta_norm > 1.0e-6).unsqueeze(-1),
                    translation_axis[slots],
                    fallback_translation_axis[ok],
                )
                actual_delta[slots] = delta_norm
                proxy_hit_count[slots] = hit_count[ok]
                hit_proxy_y[slots] = hit_y[ok]
                selected_twist_axis[slots] = attempt_twist_axis[ok]
                selected_twist_angle[slots] = attempt_twist_angle[ok]
                accepted[slots] = True
                fail[slots] = 0.0

        failed = ~accepted
        if bool(failed.any()):
            # Keep the sampled rotation but remove the infeasible translation.
            # This preserves the rotation curriculum while avoiding an
            # arbitrary unreachable translation target.
            target_pose[failed, :3] = source_pose_palm[failed, :3]
            target_pose[failed, 3:7] = target_quat_palm[failed]
            translation_axis[failed] = self.target_translation_axis_palm[env_ids[failed]].to(dtype=dtype)
            _, hit_count, hit_y, _ = self._pose6d_target_proxy_workspace_info(target_pose[failed], env_ids[failed])
            proxy_hit_count[failed] = hit_count
            hit_proxy_y[failed] = hit_y

        workspace_reject = reject_count / torch.clamp(attempt_count, min=1.0)
        forbidden_reject = forbidden_reject_count / torch.clamp(attempt_count, min=1.0)
        return {
            "target_pose": target_pose,
            "translation_axis": translation_axis,
            "actual_delta": actual_delta,
            "fail": fail,
            "workspace_reject": workspace_reject,
            "forbidden_reject": forbidden_reject,
            "proxy_hit_count": proxy_hit_count,
            "hit_proxy_y": hit_proxy_y,
            "target_twist_axis": selected_twist_axis,
            "target_twist_angle": selected_twist_angle,
        }

    def _set_hand_frame_twist_targets(self, env_ids: torch.Tensor, source_pose_palm: torch.Tensor) -> None:
        count = int(env_ids.numel())
        if self._chain_post_success_target_curriculum_enabled():
            chain_target_mask = (self.achieved_goals[env_ids] > 0) | self.reset_from_post_success_cache[env_ids]
        else:
            chain_target_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        target_delta = torch.empty(count, device=self.device)
        sample_mode = torch.zeros(count, device=self.device, dtype=torch.long)
        curriculum_max_at_reset = torch.empty(count, device=self.device)
        if bool(chain_target_mask.any()):
            chain_slots = torch.nonzero(chain_target_mask, as_tuple=False).squeeze(-1)
            chain_delta, chain_mode, chain_max = self._sample_chain_twist_target_delta_rad(
                int(chain_slots.numel()), env_ids[chain_slots]
            )
            target_delta[chain_slots] = chain_delta
            sample_mode[chain_slots] = chain_mode
            curriculum_max_at_reset[chain_slots] = chain_max
        normal_target_mask = ~chain_target_mask
        if bool(normal_target_mask.any()):
            normal_slots = torch.nonzero(normal_target_mask, as_tuple=False).squeeze(-1)
            normal_delta, normal_mode, normal_max = self._sample_twist_target_delta_rad(
                int(normal_slots.numel()), env_ids[normal_slots]
            )
            target_delta[normal_slots] = normal_delta
            sample_mode[normal_slots] = normal_mode
            curriculum_max_at_reset[normal_slots] = normal_max
        axis = self._sample_twist_axes_palm(count, source_pose_palm.dtype)
        if bool(getattr(self.cfg, "randomize_target_twist_direction", True)):
            direction = torch.randint(
                0,
                2,
                (count,),
                device=self.device,
                dtype=torch.long,
                generator=self.target_rng,
            ).to(dtype=source_pose_palm.dtype)
            direction = direction * 2.0 - 1.0
        else:
            direction = torch.full(
                (count,),
                float(getattr(self.cfg, "target_direction", 1.0)),
                device=self.device,
                dtype=source_pose_palm.dtype,
            )
            direction = torch.where(direction >= 0.0, torch.ones_like(direction), -torch.ones_like(direction))

        target_twist_angle = direction * target_delta.to(dtype=source_pose_palm.dtype)
        delta_quat = quat_from_angle_axis_wxyz(target_twist_angle, axis)
        target_pose = source_pose_palm.clone()
        target_pose[:, 3:7] = normalize_quat_wxyz(quat_mul_wxyz(delta_quat, source_pose_palm[:, 3:7]))
        feasible_target_fail = torch.zeros(count, device=self.device, dtype=source_pose_palm.dtype)
        feasible_target_reject = torch.zeros_like(feasible_target_fail)
        feasible_target_forbidden_reject = torch.zeros_like(feasible_target_fail)
        feasible_target_hit_count = torch.zeros_like(feasible_target_fail)
        feasible_target_hit_y = torch.zeros_like(feasible_target_fail)
        translation_delta = torch.empty(count, device=self.device)
        translation_sample_mode = torch.zeros(count, device=self.device, dtype=torch.long)
        translation_max_at_reset = torch.empty(count, device=self.device)
        if self._chain_pose6d_translation_curriculum_enabled():
            chain_translation_mask = chain_target_mask
        else:
            chain_translation_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        if bool(chain_translation_mask.any()):
            chain_slots = torch.nonzero(chain_translation_mask, as_tuple=False).squeeze(-1)
            chain_delta, chain_mode, chain_max = self._sample_chain_pose6d_translation_delta_m(
                int(chain_slots.numel()), env_ids[chain_slots]
            )
            translation_delta[chain_slots] = chain_delta
            translation_sample_mode[chain_slots] = chain_mode
            translation_max_at_reset[chain_slots] = chain_max
        normal_translation_mask = ~chain_translation_mask
        if bool(normal_translation_mask.any()):
            normal_slots = torch.nonzero(normal_translation_mask, as_tuple=False).squeeze(-1)
            normal_delta, normal_mode, normal_max = self._sample_pose6d_translation_delta_m(
                int(normal_slots.numel()), env_ids[normal_slots]
            )
            translation_delta[normal_slots] = normal_delta
            translation_sample_mode[normal_slots] = normal_mode
            translation_max_at_reset[normal_slots] = normal_max
        if self._uses_pose6d_feasible_target_filter():
            feasible_sample = self._sample_pose6d_feasible_object_targets(
                env_ids,
                source_pose_palm,
                target_pose[:, 3:7],
                translation_delta.to(dtype=source_pose_palm.dtype),
                translation_max_at_reset.to(dtype=source_pose_palm.dtype),
                axis.to(dtype=source_pose_palm.dtype),
                target_twist_angle.to(dtype=source_pose_palm.dtype),
                target_delta.to(dtype=source_pose_palm.dtype),
            )
            target_pose = feasible_sample["target_pose"]
            translation_axis = feasible_sample["translation_axis"]
            actual_translation_delta = feasible_sample["actual_delta"]
            axis = feasible_sample["target_twist_axis"]
            target_twist_angle = feasible_sample["target_twist_angle"]
            feasible_target_fail = feasible_sample["fail"]
            feasible_target_reject = feasible_sample["workspace_reject"]
            feasible_target_forbidden_reject = feasible_sample["forbidden_reject"]
            feasible_target_hit_count = feasible_sample["proxy_hit_count"]
            feasible_target_hit_y = feasible_sample["hit_proxy_y"]
            source_anchor_pos = self._anchor_pos_from_pose(
                source_pose_palm,
                self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype),
            )
            target_anchor_pos = self._anchor_pos_from_pose(
                target_pose,
                self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype),
            )
            target_anchor_local = self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype)
            anchor_resample_fail = torch.zeros(count, device=self.device, dtype=source_pose_palm.dtype)
            anchor_workspace_reject = torch.zeros_like(anchor_resample_fail)
            anchor_forbidden_reject = torch.zeros_like(anchor_resample_fail)
            anchor_overlap_count = torch.zeros_like(anchor_resample_fail)
            anchor_min_centerline_z = torch.zeros_like(anchor_resample_fail)
        elif self._uses_pose6d_anchor_relaxed_targets():
            anchor_sample = self._sample_pose6d_anchor_relaxed_targets(
                env_ids,
                source_pose_palm,
                target_pose[:, 3:7],
                translation_delta.to(dtype=source_pose_palm.dtype),
            )
            target_pose = anchor_sample["target_pose"]
            translation_axis = anchor_sample["translation_axis"]
            actual_translation_delta = anchor_sample["actual_delta"]
            source_anchor_pos = anchor_sample["source_anchor_pos"]
            target_anchor_pos = anchor_sample["target_anchor_pos"]
            target_anchor_local = anchor_sample["target_anchor_local"]
            anchor_resample_fail = anchor_sample["fail"]
            anchor_workspace_reject = anchor_sample["workspace_reject"]
            anchor_forbidden_reject = anchor_sample["forbidden_reject"]
            anchor_overlap_count = anchor_sample["overlap_count"]
            anchor_min_centerline_z = anchor_sample["min_centerline_z"]
        else:
            translation_dir = self._sample_pose6d_translation_axes_palm(count, source_pose_palm.dtype)
            target_pos = source_pose_palm[:, :3] + translation_dir * translation_delta.unsqueeze(-1).to(
                dtype=source_pose_palm.dtype
            )
            if bool(getattr(self.cfg, "pose6d_translation_workspace_clamp_enabled", True)):
                workspace_lo = self.local_random_workspace_lo.to(dtype=source_pose_palm.dtype).unsqueeze(0)
                workspace_hi = self.local_random_workspace_hi.to(dtype=source_pose_palm.dtype).unsqueeze(0)
                target_pos = torch.minimum(torch.maximum(target_pos, workspace_lo), workspace_hi)
            target_offset = target_pos - source_pose_palm[:, :3]
            actual_translation_delta = torch.linalg.norm(target_offset, dim=-1)
            safe_translation_delta = torch.clamp(actual_translation_delta, min=1.0e-6)
            translation_axis = target_offset / safe_translation_delta.unsqueeze(-1)
            translation_axis = torch.where(
                (actual_translation_delta > 1.0e-6).unsqueeze(-1),
                translation_axis,
                translation_dir,
            )
            target_pose[:, :3] = target_pos
            source_anchor_pos = self._anchor_pos_from_pose(
                source_pose_palm,
                self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype),
            )
            target_anchor_pos = self._anchor_pos_from_pose(
                target_pose,
                self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype),
            )
            target_anchor_local = self.source_anchor_local_pos_m[env_ids].to(dtype=source_pose_palm.dtype)
            anchor_resample_fail = torch.zeros(count, device=self.device, dtype=source_pose_palm.dtype)
            anchor_workspace_reject = torch.zeros_like(anchor_resample_fail)
            anchor_forbidden_reject = torch.zeros_like(anchor_resample_fail)
            anchor_overlap_count = torch.zeros_like(anchor_resample_fail)
            anchor_min_centerline_z = torch.zeros_like(anchor_resample_fail)

        self.target_node_ids[env_ids] = -1
        self.graph_target_grasp_indices[env_ids] = -1
        self.edge_bin[env_ids] = 0
        self.edge_translation[env_ids] = 0.0
        self.edge_score[env_ids] = 0.0
        self.edge_q_dist[env_ids] = 0.0
        self.edge_contact_mismatch[env_ids] = 0.0
        self.edge_q_limit_risk[env_ids] = 0.0
        self.target_pose_palm[env_ids] = target_pose.to(dtype=torch.float32)
        self.target_keypoints_palm[env_ids] = self._object_keypoints_from_pose(target_pose, env_ids).to(dtype=torch.float32)
        self.target_source_pose_palm[env_ids] = source_pose_palm.to(device=self.device, dtype=torch.float32)
        self.target_twist_axis_palm[env_ids] = axis.to(dtype=torch.float32)
        self.target_translation_axis_palm[env_ids] = translation_axis.to(dtype=torch.float32)
        self.source_anchor_pos_palm[env_ids] = source_anchor_pos.to(dtype=torch.float32)
        self.target_anchor_pos_palm[env_ids] = target_anchor_pos.to(dtype=torch.float32)
        self.target_anchor_local_pos_m[env_ids] = target_anchor_local.to(dtype=torch.float32)
        self.pose6d_anchor_resample_fail[env_ids] = anchor_resample_fail.to(dtype=torch.float32)
        self.pose6d_anchor_workspace_reject[env_ids] = anchor_workspace_reject.to(dtype=torch.float32)
        self.pose6d_anchor_forbidden_reject[env_ids] = anchor_forbidden_reject.to(dtype=torch.float32)
        self.pose6d_anchor_segment_overlap_count[env_ids] = anchor_overlap_count.to(dtype=torch.float32)
        self.pose6d_anchor_min_centerline_z_m[env_ids] = anchor_min_centerline_z.to(dtype=torch.float32)
        self.pose6d_feasible_target_resample_fail[env_ids] = feasible_target_fail.to(dtype=torch.float32)
        self.pose6d_feasible_target_workspace_reject[env_ids] = feasible_target_reject.to(dtype=torch.float32)
        self.pose6d_feasible_target_forbidden_reject[env_ids] = feasible_target_forbidden_reject.to(dtype=torch.float32)
        self.pose6d_feasible_target_proxy_hit_count[env_ids] = feasible_target_hit_count.to(dtype=torch.float32)
        self.pose6d_feasible_target_hit_proxy_y_m[env_ids] = feasible_target_hit_y.to(dtype=torch.float32)
        self.edge_dist[env_ids] = self._object_pose_distance(source_pose_palm, target_pose, env_ids).detach()
        self.local_random_target_dist[env_ids] = self.edge_dist[env_ids]
        self.edge_translation[env_ids] = actual_translation_delta.to(dtype=torch.float32)
        self.local_random_target_translation[env_ids] = actual_translation_delta.to(dtype=torch.float32)
        self.local_random_target_rotation[env_ids] = target_delta.to(dtype=torch.float32)
        self.local_random_resample_fail[env_ids] = 0.0
        self.local_random_workspace_reject[env_ids] = 0.0
        self.target_delta[env_ids] = target_delta.to(dtype=torch.float32)
        self.target_delta_sample_mode[env_ids] = sample_mode
        self.target_delta_curriculum_max_at_reset[env_ids] = curriculum_max_at_reset.to(dtype=torch.float32)
        self.target_chain_phase[env_ids] = chain_target_mask.to(dtype=torch.long)
        self.translation_delta_sample_mode[env_ids] = translation_sample_mode
        self.translation_delta_curriculum_max_at_reset[env_ids] = translation_max_at_reset.to(dtype=torch.float32)
        self.target_twist_angle[env_ids] = target_twist_angle.to(dtype=torch.float32)
        self.current_twist_angle[env_ids] = 0.0
        self.twist_progress_raw[env_ids] = 0.0
        self.twist_progress[env_ids] = 0.0
        self.best_twist_progress[env_ids] = 0.0
        self.twist_remaining[env_ids] = torch.abs(target_twist_angle).to(dtype=torch.float32)
        self.twist_swing_drift[env_ids] = 0.0
        self.twist_position_drift[env_ids] = actual_translation_delta.to(dtype=torch.float32)
        self.target_translation_delta[env_ids] = actual_translation_delta.to(dtype=torch.float32)
        self.translation_progress_raw[env_ids] = 0.0
        self.translation_progress[env_ids] = 0.0
        self.best_translation_progress[env_ids] = 0.0
        self.translation_remaining[env_ids] = actual_translation_delta.to(dtype=torch.float32)
        self.translation_orthogonal_drift[env_ids] = 0.0

    def _refresh_hand_frame_twist_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() <= 0:
            return
        source_pose_palm = self.current_pose_palm[env_ids].detach()
        self._set_hand_frame_twist_targets(env_ids, source_pose_palm)
        self.near_success_steps[env_ids] = 0
        self.target_reached[env_ids] = False
        self._compute_intermediate_values()

    def _hand_to_world_pose(self, object_pose_palm: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        hand_pos = self._hand_root_pos_w(object_pose_palm.dtype, env_ids)
        hand_quat = self._hand_root_quat_w(object_pose_palm.dtype, env_ids)
        world_pos = hand_pos + quat_apply_wxyz(hand_quat, object_pose_palm[:, :3])
        world_quat = normalize_quat_wxyz(quat_mul_wxyz(hand_quat, object_pose_palm[:, 3:7]))
        return torch.cat((world_pos, world_quat), dim=-1)

    def _world_to_hand_pose(self, world_pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(world_pose.shape[0], device=self.device)
        hand_pos = self._hand_root_pos_w(world_pose.dtype, env_ids)
        hand_quat = self._hand_root_quat_w(world_pose.dtype, env_ids)
        inv_hand_quat = quat_conjugate_wxyz(hand_quat)
        local_pos = quat_apply_wxyz(inv_hand_quat, world_pose[:, :3] - hand_pos)
        local_quat = normalize_quat_wxyz(quat_mul_wxyz(inv_hand_quat, world_pose[:, 3:7]))
        return torch.cat((local_pos, local_quat), dim=-1)

    def _world_points_to_hand(self, points_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(points_w.shape[0], device=self.device)
        hand_pos = self._hand_root_pos_w(points_w.dtype, env_ids).unsqueeze(1)
        hand_quat = self._hand_root_quat_w(points_w.dtype, env_ids).unsqueeze(1)
        inv_hand_quat = quat_conjugate_wxyz(hand_quat).expand(points_w.shape[0], points_w.shape[1], 4)
        return quat_apply_wxyz(inv_hand_quat, points_w - hand_pos)

    def _world_vectors_to_hand(self, vectors_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(vectors_w.shape[0], device=self.device)
        hand_quat = self._hand_root_quat_w(vectors_w.dtype, env_ids).unsqueeze(1)
        inv_hand_quat = quat_conjugate_wxyz(hand_quat).expand(vectors_w.shape[0], vectors_w.shape[1], 4)
        return quat_apply_wxyz(inv_hand_quat, vectors_w)


    def _get_cube_contact_forces_w(self) -> torch.Tensor:
        if self.cube_contact_sensor is None:
            return torch.zeros((self.num_envs, len(self.contact_sensor_body_names), 3), device=self.device)
        if bool(getattr(self, "multiobject_enabled", False)):
            if self.primitive_single_asset_per_env_enabled:
                force_matrix = self.cube_contact_sensor.data.force_matrix_w
                forces_w = (
                    force_matrix[:, 0, :, :]
                    if force_matrix is not None
                    else torch.zeros((self.num_envs, len(self.contact_sensor_body_names), 3), device=self.device)
                )
            else:
                forces_w = torch.zeros((self.num_envs, len(self.contact_sensor_body_names), 3), device=self.device)
                for size_id, object_id in enumerate(self.multisize_object_ids):
                    sensor = self.multisize_cube_contact_sensors[object_id]
                    force_matrix = sensor.data.force_matrix_w
                    size_forces_w = (
                        force_matrix[:, 0, :, :]
                        if force_matrix is not None
                        else torch.zeros_like(forces_w)
                    )
                    mask = self.active_cube_size_id == size_id
                    if bool(mask.any()):
                        forces_w[mask] = size_forces_w[mask]
        else:
            force_matrix = self.cube_contact_sensor.data.force_matrix_w
            if force_matrix is not None:
                forces_w = force_matrix[:, 0, :, :]
            else:
                forces_w = torch.zeros((self.num_envs, len(self.contact_sensor_body_names), 3), device=self.device)
        expected_filters = len(self.contact_sensor_body_names)
        if forces_w.shape[1] != expected_filters:
            raise RuntimeError(
                f"Expected {expected_filters} cube contact filters, got {forces_w.shape[1]}. "
                "Check the active contact sensor body list and ContactSensor filter paths."
            )
        return forces_w

    def _get_cube_net_contact_force_w(self, forces_w: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, "multiobject_enabled", False)):
            if self.primitive_single_asset_per_env_enabled:
                net_forces_w = getattr(self.cube_contact_sensor.data, "net_forces_w", None)
                if net_forces_w is not None:
                    return net_forces_w[:, 0, :].to(dtype=forces_w.dtype)
                return -torch.sum(forces_w, dim=1)
            net_force = torch.zeros((self.num_envs, 3), device=self.device, dtype=forces_w.dtype)
            for size_id, object_id in enumerate(self.multisize_object_ids):
                sensor = self.multisize_cube_contact_sensors.get(object_id)
                net_forces_w = getattr(sensor.data, "net_forces_w", None) if sensor is not None else None
                if net_forces_w is None:
                    continue
                mask = self.active_cube_size_id == size_id
                if bool(mask.any()):
                    net_force[mask] = net_forces_w[:, 0, :][mask].to(dtype=forces_w.dtype)
            return net_force
        if self.cube_contact_sensor is not None:
            net_forces_w = getattr(self.cube_contact_sensor.data, "net_forces_w", None)
            if net_forces_w is not None:
                return net_forces_w[:, 0, :]
        return -torch.sum(forces_w, dim=1)

    def _select_contact_forces_w(self, forces_w: torch.Tensor, body_names: tuple[str, ...]) -> torch.Tensor:
        missing = tuple(name for name in body_names if name not in self.contact_sensor_body_name_to_idx)
        if missing:
            raise RuntimeError(f"Contact sensor was not configured with required body names: {missing}")
        indices = [self.contact_sensor_body_name_to_idx[name] for name in body_names]
        return forces_w[:, indices, :]

    def _update_contact_force_buffers(self, forces_w: torch.Tensor) -> None:
        contact_bit_forces_w = self._select_contact_forces_w(forces_w, self.hand_spec.contact_bit_bodies)
        self.contact_bit_force_norm[:] = torch.linalg.norm(contact_bit_forces_w, dim=-1)
        fingertip_forces_w = self._select_contact_forces_w(forces_w, self.hand_spec.force_sensor_bodies)
        self.contact_force_norm[:] = torch.linalg.norm(self._world_vectors_to_hand(fingertip_forces_w), dim=-1)
        if self.all_body_contact_force_obs_enabled:
            all_body_forces_w = self._select_contact_forces_w(forces_w, self.hand_spec.all_contact_bodies)
            object_force_w = self._get_cube_net_contact_force_w(forces_w).unsqueeze(1)
            all_body_with_object_forces_w = torch.cat((all_body_forces_w, object_force_w), dim=1)
            self.all_body_contact_force_norm[:] = torch.linalg.norm(all_body_with_object_forces_w, dim=-1)

    def _get_contact_gated_excess_load_terms(
        self,
        q_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the Sharpa contact-gated elastic-load reward terms."""

        forces_w = self._get_cube_contact_forces_w()
        paired_forces_w = forces_w[:, self.dexndm_excess_load_contact_body_indices, :]
        finger_contact_force_norm = torch.linalg.norm(paired_forces_w, dim=-1).amax(dim=-1)
        return contact_gated_excess_load_terms(
            q_action,
            self.action_targets,
            self.dexndm_controller_pgain_action,
            self.dexndm_controller_pgain_nominal_action,
            finger_contact_force_norm,
            self.dexndm_excess_load_joint_finger_ids,
            reference_lead_rad=self.dexndm_contact_gated_excess_load_reference_lead_rad,
            contact_force_threshold_n=self.dexndm_contact_gated_excess_load_contact_threshold_n,
            normalized_excess_clip=self.dexndm_contact_gated_excess_load_normalized_clip,
        )

    def _get_grasp_wrench_coverage_terms(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute CSGWCD diagnostics and the unscaled normalized positive reward."""

        if not self.dexndm_grasp_wrench_maintenance_reward_enabled:
            zero = self.grasp_wrench_quality
            return zero, zero, zero, zero, zero, zero, zero
        forces_w = self._get_cube_contact_forces_w()
        contact_forces_w = self._select_contact_forces_w(
            forces_w, self.dexndm_grasp_wrench_contact_bodies
        )
        force_norm_n = torch.linalg.norm(contact_forces_w, dim=-1)
        body_pos_w = self.hand.data.body_pos_w[:, self.dexndm_grasp_wrench_contact_body_ids, :]
        object_pose_w = self._active_cube_root_pose_w()
        object_quat_w = normalize_quat_wxyz(object_pose_w[:, 3:7])
        inverse_object_quat = quat_conjugate_wxyz(object_quat_w).unsqueeze(1).expand(
            -1, body_pos_w.shape[1], -1
        )
        body_pos_object = quat_apply_wxyz(
            inverse_object_quat,
            body_pos_w - object_pose_w[:, :3].unsqueeze(1),
        )
        projected_contact_points = None
        outward_contact_normals = None
        if bool(getattr(self.cfg, "grasp_wrench_mesh_projection_enabled", False)):
            projected_contact_points, outward_contact_normals = lookup_mesh_surface_field(
                body_pos_object,
                self.active_cube_size_id,
                self.multiobject_csgwcd_surface_delta_grid_m,
                self.multiobject_csgwcd_outward_normal_grid,
                self.csgwcd_mesh_grid_half_extent_m,
                object_scale_xyz=self.object_scale_xyz_values,
                validate_values=False,
            )
        (
            quality,
            force_margin_min,
            torque_margin_min,
            directional_margins,
            confidence,
            _,
        ) = conservative_saturated_grasp_wrench_coverage(
            body_pos_object,
            force_norm_n,
            self.active_object_dims_m.to(dtype=body_pos_object.dtype),
            self.dexndm_grasp_wrench_contact_friction.to(dtype=body_pos_object.dtype),
            projected_contact_points_object_m=projected_contact_points,
            outward_contact_normals_object=outward_contact_normals,
            contact_confidence_cap=(
                None
                if self.dexndm_grasp_wrench_contact_confidence_cap is None
                else self.dexndm_grasp_wrench_contact_confidence_cap.to(dtype=body_pos_object.dtype)
            ),
            contact_primary_mask=self.dexndm_grasp_wrench_contact_primary_mask,
            contact_group_ids=self.dexndm_grasp_wrench_contact_group_ids,
            secondary_group_confidence_cap=(
                self.dexndm_grasp_wrench_secondary_group_confidence_cap
                if self.dexndm_grasp_wrench_body_mode == "all_collision_capped_secondary"
                else None
            ),
            spatial_dedup_distance_m=(
                self.dexndm_grasp_wrench_spatial_dedup_distance_m
                if self.dexndm_grasp_wrench_body_mode == "all_collision_capped_secondary"
                else 0.0
            ),
            contact_on_n=self.dexndm_grasp_wrench_contact_on_n,
            contact_sat_n=self.dexndm_grasp_wrench_contact_sat_n,
            force_margin_erosion=self.dexndm_grasp_wrench_force_margin_erosion,
            torque_margin_erosion=self.dexndm_grasp_wrench_torque_margin_erosion,
            softmin_power=self.dexndm_grasp_wrench_softmin_power,
            validate_values=False,
        )

        warmup_steps = self.dexndm_grasp_wrench_reset_warmup_steps
        if warmup_steps > 0:
            warmup_active = self.grasp_wrench_warmup_count < warmup_steps
            self.grasp_wrench_warmup_quality_sum += torch.where(
                warmup_active, quality.detach(), torch.zeros_like(quality)
            )
            self.grasp_wrench_warmup_count += warmup_active.to(dtype=torch.long)
            warmup_complete = warmup_active & (self.grasp_wrench_warmup_count >= warmup_steps)
            reset_quality = self.grasp_wrench_warmup_quality_sum / float(warmup_steps)
            target_candidate = torch.clamp(
                torch.maximum(
                    reset_quality,
                    torch.full_like(reset_quality, self.dexndm_grasp_wrench_quality_floor),
                ),
                max=self.dexndm_grasp_wrench_quality_cap,
            )
            self.grasp_wrench_target_quality[:] = torch.where(
                warmup_complete,
                target_candidate,
                self.grasp_wrench_target_quality,
            )
            reward_ready = ~warmup_active
        else:
            self.grasp_wrench_target_quality.fill_(self.dexndm_grasp_wrench_quality_floor)
            reward_ready = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        deficit, maintenance_reward = normalized_grasp_wrench_maintenance_reward(
            quality,
            self.grasp_wrench_target_quality,
            validate_values=False,
        )
        deficit = torch.where(reward_ready, deficit, torch.zeros_like(deficit))
        maintenance_reward = torch.where(
            reward_ready,
            maintenance_reward,
            torch.zeros_like(maintenance_reward),
        )
        self.grasp_wrench_quality[:] = quality.detach()
        self.grasp_wrench_force_margin_min[:] = force_margin_min.detach()
        self.grasp_wrench_torque_margin_min[:] = torque_margin_min.detach()
        self.grasp_wrench_directional_margins[:] = directional_margins.detach()
        self.grasp_wrench_contact_confidence[:] = confidence.detach()
        self.grasp_wrench_deficit[:] = deficit.detach()
        self.grasp_wrench_reward_quality[:] = quality.detach()
        self.grasp_wrench_reward_force_margin_min[:] = force_margin_min.detach()
        self.grasp_wrench_reward_torque_margin_min[:] = torque_margin_min.detach()
        self.grasp_wrench_reward_directional_margins[:] = directional_margins.detach()
        self.grasp_wrench_reward_contact_confidence[:] = confidence.detach()
        self.grasp_wrench_reward_target_quality[:] = self.grasp_wrench_target_quality.detach()
        self.grasp_wrench_reward_deficit[:] = deficit.detach()
        self.grasp_wrench_reward_maintenance_reward[:] = maintenance_reward.detach()
        self.grasp_wrench_reward_warmup_count[:] = self.grasp_wrench_warmup_count.detach()
        self.grasp_wrench_reward_contact_force_w[:] = contact_forces_w.detach()
        confidence_count = (confidence > 0.0).to(dtype=quality.dtype).sum(dim=-1)
        below_target = reward_ready & (deficit > 0.0)
        return (
            quality,
            force_margin_min,
            torque_margin_min,
            deficit,
            below_target.float(),
            confidence_count,
            maintenance_reward,
        )





    def _get_dexndm_history_obs(self, q_action: torch.Tensor) -> torch.Tensor:
        current_step = int(self.common_step_counter)
        needs_update = self.dexndm_obs_history_prev_step != current_step
        if bool(needs_update.any()):
            env_ids = needs_update.nonzero(as_tuple=False).squeeze(-1)
            # Legacy critic-only noise; zero in the paper recipe. Actor joint
            # sensing retains its separate sharpa_deployable_* noise setting.
            noise_scale = float(getattr(self.cfg, "dexndm_joint_noise_scale", 0.0))
            if noise_scale > 0.0:
                q_for_obs = q_action + (2.0 * torch.rand_like(q_action) - 1.0) * noise_scale
            else:
                q_for_obs = q_action
            q_scaled = scale_to_unit(q_for_obs, self.action_lower, self.action_upper)
            frame = torch.cat((q_scaled, self.action_targets), dim=-1)
            self.dexndm_obs_history[env_ids, :-1] = self.dexndm_obs_history[env_ids, 1:].clone()
            self.dexndm_obs_history[env_ids, -1] = frame[env_ids]
            self.dexndm_obs_history_prev_step[env_ids] = current_step
        return self.dexndm_obs_history.reshape(self.num_envs, -1)

    def _apply_sharpa_deployable_joint_sensing_noise(self, q_action: torch.Tensor) -> torch.Tensor:
        noise_scale = max(float(getattr(self.cfg, "sharpa_deployable_joint_sensing_noise_scale", 0.0)), 0.0)
        if noise_scale <= 0.0:
            return q_action
        noise = (2.0 * torch.rand(
            q_action.shape,
            device=q_action.device,
            dtype=q_action.dtype,
            generator=self.target_rng,
        ) - 1.0) * noise_scale
        return torch.minimum(torch.maximum(q_action + noise, self.action_lower), self.action_upper)


    def _get_sharpa_deployable_history_obs(self, q_action: torch.Tensor) -> torch.Tensor:
        current_step = int(self.common_step_counter)
        needs_update = self.sharpa_deployable_obs_history_prev_step != current_step
        if bool(needs_update.any()):
            env_ids = needs_update.nonzero(as_tuple=False).squeeze(-1)
            q_noisy = self._apply_sharpa_deployable_joint_sensing_noise(q_action)
            q_scaled = scale_to_unit(q_noisy, self.action_lower, self.action_upper)
            q_cmd_scaled = scale_to_unit(self.action_targets, self.action_lower, self.action_upper)
            tactile_obs = torch.empty((self.num_envs, 0), device=self.device, dtype=torch.float32)
            frame = torch.cat((q_scaled, q_cmd_scaled, tactile_obs), dim=-1).to(dtype=torch.float32)
            if frame.shape[1] != self.sharpa_deployable_frame_dim:
                raise RuntimeError(
                    f"Sharpa deployable history frame has width {frame.shape[1]}, "
                    f"expected {self.sharpa_deployable_frame_dim}."
                )
            self.sharpa_deployable_obs_history[env_ids, :-1] = self.sharpa_deployable_obs_history[
                env_ids, 1:
            ].clone()
            self.sharpa_deployable_obs_history[env_ids, -1] = frame[env_ids]
            self.sharpa_deployable_obs_history_prev_step[env_ids] = current_step
        return self.sharpa_deployable_obs_history.reshape(self.num_envs, -1)

    def _reset_sharpa_deployable_object_pose_obs_delay(self, env_ids: torch.Tensor) -> None:
        if (
            not self.sharpa_deployable_object_pose_obs_delay_noise_enabled
            or self.sharpa_deployable_object_pose_obs_delay_max_steps <= 0
        ):
            self.sharpa_deployable_object_pose_obs_delay_steps[env_ids] = 0
            return
        self.sharpa_deployable_object_pose_obs_delay_steps[env_ids] = torch.randint(
            0,
            self.sharpa_deployable_object_pose_obs_delay_max_steps + 1,
            (int(env_ids.numel()),),
            device=self.device,
            generator=self.target_rng,
            dtype=torch.long,
        )

    def _reset_sharpa_deployable_object_pose_observation_state(
        self,
        env_ids: torch.Tensor,
        object_pose_palm: torch.Tensor,
    ) -> None:
        pose = object_pose_palm.to(device=self.device, dtype=torch.float32).clone()
        pose[:, 3:7] = positive_quat_wxyz(pose[:, 3:7])
        self._reset_sharpa_deployable_object_pose_obs_delay(env_ids)
        self.sharpa_deployable_object_pose_obs_queue[env_ids] = pose.unsqueeze(1).repeat(
            1,
            self.sharpa_deployable_object_pose_obs_queue_len,
            1,
        )
        self.sharpa_deployable_object_pose_obs_palm[env_ids] = pose
        self.sharpa_deployable_object_pose_obs_last_valid_palm[env_ids] = pose
        self.sharpa_deployable_object_pose_obs_confidence[env_ids] = float(
            getattr(self.cfg, "sharpa_deployable_pose_tracker_confidence", 1.0)
        )
        self.sharpa_deployable_object_pose_obs_dropout[env_ids] = 0.0
        self.sharpa_deployable_object_pose_obs_pos_noise_norm_m[env_ids] = 0.0
        self.sharpa_deployable_object_pose_obs_rot_noise_abs_deg[env_ids] = 0.0
        self.sharpa_deployable_object_pose_obs_prev_step[env_ids] = int(self.common_step_counter)
        self.sharpa_deployable_object_velocity_est_palm[env_ids] = 0.0
        self.sharpa_deployable_prev_object_pos_palm[env_ids] = pose[:, :3]
        self.sharpa_deployable_prev_object_quat_palm[env_ids] = pose[:, 3:7]
        self.sharpa_deployable_velocity_prev_step[env_ids] = int(self.common_step_counter)

    def _get_sharpa_deployable_observed_object_pose_palm(self, true_pose_palm: torch.Tensor) -> torch.Tensor:
        if not self.sharpa_deployable_object_pose_obs_delay_noise_enabled:
            return true_pose_palm

        current_step = int(self.common_step_counter)
        needs_update = self.sharpa_deployable_object_pose_obs_prev_step != current_step
        if bool(needs_update.any()):
            env_ids = needs_update.nonzero(as_tuple=False).squeeze(-1)
            true_pose = true_pose_palm[env_ids].detach().to(dtype=torch.float32).clone()
            true_pose[:, 3:7] = positive_quat_wxyz(true_pose[:, 3:7])
            if self.sharpa_deployable_object_pose_obs_queue_len > 1:
                self.sharpa_deployable_object_pose_obs_queue[env_ids, :-1] = (
                    self.sharpa_deployable_object_pose_obs_queue[env_ids, 1:].clone()
                )
            self.sharpa_deployable_object_pose_obs_queue[env_ids, -1] = true_pose

            if self.sharpa_deployable_object_pose_obs_delay_mode == "per_step_random":
                self._reset_sharpa_deployable_object_pose_obs_delay(env_ids)
            delay_steps = torch.clamp(
                self.sharpa_deployable_object_pose_obs_delay_steps[env_ids],
                min=0,
                max=self.sharpa_deployable_object_pose_obs_queue_len - 1,
            )
            queue_index = self.sharpa_deployable_object_pose_obs_queue_len - 1 - delay_steps
            delayed_pose = self.sharpa_deployable_object_pose_obs_queue[env_ids, queue_index].clone()

            observed_pose = delayed_pose
            pos_noise_norm = torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.float32)
            rot_noise_abs_deg = torch.zeros_like(pos_noise_norm)
            pos_std = self.sharpa_deployable_object_pose_obs_pos_noise_std_m
            if pos_std > 0.0:
                pos_noise = torch.randn(
                    (env_ids.numel(), 3),
                    device=self.device,
                    dtype=torch.float32,
                    generator=self.target_rng,
                ) * pos_std
                observed_pose[:, :3] = observed_pose[:, :3] + pos_noise
                pos_noise_norm = torch.linalg.norm(pos_noise, dim=-1)

            rot_std = self.sharpa_deployable_object_pose_obs_rot_noise_rad
            if rot_std > 0.0:
                axis = self._random_unit_vectors(int(env_ids.numel()), torch.float32)
                if self.sharpa_deployable_object_pose_obs_rot_noise_distribution == "uniform_bounded":
                    angle = (
                        2.0
                        * torch.rand(
                            (env_ids.numel(),),
                            device=self.device,
                            dtype=torch.float32,
                            generator=self.target_rng,
                        )
                        - 1.0
                    ) * rot_std
                else:
                    angle = torch.randn(
                        (env_ids.numel(),),
                        device=self.device,
                        dtype=torch.float32,
                        generator=self.target_rng,
                    ) * rot_std
                    angle = torch.clamp(angle, -3.0 * rot_std, 3.0 * rot_std)
                delta_quat = quat_from_angle_axis_wxyz(angle, axis)
                observed_pose[:, 3:7] = positive_quat_wxyz(quat_mul_wxyz(delta_quat, observed_pose[:, 3:7]))
                rot_noise_abs_deg = torch.abs(angle) * (180.0 / torch.pi)

            base_confidence = float(getattr(self.cfg, "sharpa_deployable_pose_tracker_confidence", 1.0))
            confidence = torch.full(
                (env_ids.numel(), 1),
                base_confidence,
                device=self.device,
                dtype=torch.float32,
            )
            dropout_prob = self.sharpa_deployable_object_pose_obs_dropout_prob
            dropout = (
                torch.rand((env_ids.numel(),), device=self.device, generator=self.target_rng) < dropout_prob
                if dropout_prob > 0.0
                else torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.bool)
            )
            if bool(dropout.any()):
                observed_pose[dropout] = self.sharpa_deployable_object_pose_obs_last_valid_palm[env_ids[dropout]]
                confidence[dropout] = self.sharpa_deployable_object_pose_obs_dropout_confidence
                pos_noise_norm[dropout] = 0.0
                rot_noise_abs_deg[dropout] = 0.0
            valid = ~dropout
            if bool(valid.any()):
                self.sharpa_deployable_object_pose_obs_last_valid_palm[env_ids[valid]] = observed_pose[valid]

            self.sharpa_deployable_object_pose_obs_palm[env_ids] = observed_pose
            self.sharpa_deployable_object_pose_obs_confidence[env_ids] = confidence
            self.sharpa_deployable_object_pose_obs_dropout[env_ids] = dropout.to(dtype=torch.float32)
            self.sharpa_deployable_object_pose_obs_pos_noise_norm_m[env_ids] = pos_noise_norm
            self.sharpa_deployable_object_pose_obs_rot_noise_abs_deg[env_ids] = rot_noise_abs_deg
            self.sharpa_deployable_object_pose_obs_prev_step[env_ids] = current_step
        return self.sharpa_deployable_object_pose_obs_palm

    def _update_sharpa_deployable_object_velocity_est_palm(
        self,
        object_pose_palm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if object_pose_palm is None:
            object_pose_palm = self.current_pose_palm
        current_pos_palm = object_pose_palm[:, :3].detach()
        current_quat_palm = positive_quat_wxyz(object_pose_palm[:, 3:7].detach())
        current_step = int(self.common_step_counter)
        needs_update = self.sharpa_deployable_velocity_prev_step != current_step
        if bool(needs_update.any()):
            env_ids = needs_update.nonzero(as_tuple=False).squeeze(-1)
            dt = max(float(self.dt), 1.0e-8)
            lin_vel_palm = (
                current_pos_palm[env_ids] - self.sharpa_deployable_prev_object_pos_palm[env_ids]
            ) / dt
            delta_quat_palm = quat_mul_wxyz(
                current_quat_palm[env_ids],
                quat_conjugate_wxyz(self.sharpa_deployable_prev_object_quat_palm[env_ids]),
            )
            ang_vel_palm = quat_to_rotvec_wxyz(delta_quat_palm) / dt
            self.sharpa_deployable_object_velocity_est_palm[env_ids] = torch.cat(
                (lin_vel_palm, ang_vel_palm), dim=-1
            )
            self.sharpa_deployable_prev_object_pos_palm[env_ids] = current_pos_palm[env_ids]
            self.sharpa_deployable_prev_object_quat_palm[env_ids] = current_quat_palm[env_ids]
            self.sharpa_deployable_velocity_prev_step[env_ids] = current_step
        return self.sharpa_deployable_object_velocity_est_palm

    def _get_sharpa_deployable_object_descriptor_obs(self) -> torch.Tensor:
        """Return deployable object descriptor obs: normalized dimensions and cuboid/cylinder onehot."""

        if self.sharpa_deployable_object_descriptor_dim <= 0:
            return torch.empty((self.num_envs, 0), device=self.device, dtype=torch.float32)

        if self.sharpa_deployable_mesh_bps_obs_enabled:
            return self.active_object_geometry_latent.to(dtype=torch.float32)

        if self.multiobject_enabled:
            dim_obs = self.active_object_dim_norm.to(dtype=torch.float32)
            shape_obs_full = self.active_object_shape_onehot.to(dtype=torch.float32)
            shape_obs = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
            if shape_obs_full.numel() > 0:
                width = min(2, int(shape_obs_full.shape[1]))
                shape_obs[:, :width] = shape_obs_full[:, :width]
        else:
            if self._single_object_variable_dims_enabled():
                dim_obs = self.active_object_dim_norm.to(dtype=torch.float32)
            else:
                side = float(
                    getattr(self, "active_cube_side_length_m", torch.tensor(CUBE_4CM_SIDE_LENGTH_M)).mean().item()
                )
                dims = torch.full((self.num_envs, 3), side, device=self.device, dtype=torch.float32)
                dim_obs = dims / CUBE_4CM_SIDE_LENGTH_M - 1.0
            shape_obs = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
            shape_obs[:, 0] = 1.0

        return torch.cat((dim_obs, shape_obs), dim=-1)

    def _get_sharpa_deployable_pose6d_task_obs(self) -> torch.Tensor:
        self._update_twist_target_terms()
        true_object_pose_obs = self._get_object_pose_obs()
        object_pose_obs = self._get_sharpa_deployable_observed_object_pose_palm(true_object_pose_obs)
        target_pose_obs = self.target_pose_palm.clone()
        target_pose_obs[:, 3:7] = positive_quat_wxyz(target_pose_obs[:, 3:7])

        pose_delta_pos = target_pose_obs[:, :3] - object_pose_obs[:, :3]
        pose_delta_quat = positive_quat_wxyz(
            quat_mul_wxyz(target_pose_obs[:, 3:7], quat_conjugate_wxyz(object_pose_obs[:, 3:7]))
        )
        pose_error_obs = torch.cat((pose_delta_pos, pose_delta_quat), dim=-1)

        current_anchor_pos_obs_palm = self._anchor_pos_from_pose(
            object_pose_obs,
            self.source_anchor_local_pos_m.to(dtype=object_pose_obs.dtype),
        ).to(dtype=torch.float32)
        anchor_delta = self.target_anchor_pos_palm - current_anchor_pos_obs_palm
        anchor_obs = torch.cat(
            (current_anchor_pos_obs_palm, self.target_anchor_pos_palm, anchor_delta),
            dim=-1,
        )

        translation_scale = max(
            float(getattr(self.cfg, "sharpa_deployable_target_translation_scale_m", 0.03)),
            1.0e-6,
        )
        target_command = torch.cat(
            (
                self.target_twist_axis_palm.to(dtype=torch.float32),
                (self.target_twist_angle / torch.pi).unsqueeze(-1).to(dtype=torch.float32),
                self.target_translation_axis_palm.to(dtype=torch.float32),
                (self.target_translation_delta / translation_scale).unsqueeze(-1).to(dtype=torch.float32),
            ),
            dim=-1,
        )

        object_velocity_est_palm = self._update_sharpa_deployable_object_velocity_est_palm(object_pose_obs)
        lin_clip = float(getattr(self.cfg, "sharpa_deployable_object_lin_vel_clip_mps", 10.0))
        ang_clip = float(getattr(self.cfg, "sharpa_deployable_object_ang_vel_clip_radps", 50.0))
        lin_vel_obs = torch.clamp(object_velocity_est_palm[:, :3], -lin_clip, lin_clip)
        ang_vel_obs = torch.clamp(object_velocity_est_palm[:, 3:6], -ang_clip, ang_clip)
        object_descriptor_obs = self._get_sharpa_deployable_object_descriptor_obs()
        if self.sharpa_deployable_object_pose_obs_delay_noise_enabled:
            pose_confidence = self.sharpa_deployable_object_pose_obs_confidence
        else:
            pose_confidence = torch.full(
                (self.num_envs, 1),
                float(getattr(self.cfg, "sharpa_deployable_pose_tracker_confidence", 1.0)),
                device=self.device,
                dtype=torch.float32,
            )
        gravity_palm_obs = self._get_gravity_palm_obs()

        task_obs = torch.cat(
            (
                object_pose_obs,
                target_pose_obs,
                pose_error_obs,
                anchor_obs,
                target_command,
                lin_vel_obs,
                ang_vel_obs,
                object_descriptor_obs,
                pose_confidence,
                gravity_palm_obs,
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        if task_obs.shape[1] != self.sharpa_deployable_task_dim:
            raise RuntimeError(
                f"Sharpa deployable task obs has width {task_obs.shape[1]}, "
                f"expected {self.sharpa_deployable_task_dim}."
            )
        return task_obs

    def _get_sharpa_deployable_pose6d_observations(self, q_action: torch.Tensor) -> torch.Tensor:
        obs = torch.cat(
            (
                self._get_sharpa_deployable_history_obs(q_action),
                self._get_sharpa_deployable_pose6d_task_obs(),
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        if obs.shape[1] != self.sharpa_deployable_observation_dim:
            raise RuntimeError(
                f"Sharpa deployable policy observation has width {obs.shape[1]}, "
                f"expected {self.sharpa_deployable_observation_dim}."
            )
        return obs

    def _get_sharpa_deployable_asymmetric_critic_observations(self, q_action: torch.Tensor) -> torch.Tensor:
        obs = self._get_dexndm_zpos_observations(q_action, force_zpos_tail=True)
        expected_dim = int(
            getattr(
                self.cfg,
                "sharpa_deployable_critic_observation_space",
                getattr(self.cfg, "state_space", obs.shape[1]),
            )
        )
        if obs.shape[1] != expected_dim:
            raise RuntimeError(
                f"Sharpa asymmetric critic observation has width {obs.shape[1]}, "
                f"expected {expected_dim}."
            )
        return obs

    def _dexndm_state_w_to_xyzw_obs(self, state_w: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                state_w[..., :3],
                quat_wxyz_to_xyzw(normalize_quat_wxyz(state_w[..., 3:7])),
                state_w[..., 7:13],
            ),
            dim=-1,
        )

    def _get_dexndm_guide_obs(self) -> torch.Tensor:
        hand_quat = self._hand_root_quat_w(dtype=torch.float32)
        if bool(getattr(self.cfg, "dexndm_zpos_twist_target_obs_enabled", False)):
            guide_quat_palm = normalize_quat_wxyz(self.target_pose_palm[:, 3:7])
        else:
            guide_quat_palm = normalize_quat_wxyz(self.dexndm_guided_rotation_guide_pose_palm[:, 3:7])
        guide_quat_w = normalize_quat_wxyz(quat_mul_wxyz(hand_quat, guide_quat_palm))
        current_quat_w = normalize_quat_wxyz(self._active_cube_root_quat_w())
        delta_quat_w = normalize_quat_wxyz(quat_mul_wxyz(current_quat_w, quat_conjugate_wxyz(guide_quat_w)))
        return torch.cat((quat_wxyz_to_xyzw(guide_quat_w), quat_wxyz_to_xyzw(delta_quat_w)), dim=-1)

    def _get_dexndm_force_obs(self, forces_w: torch.Tensor) -> torch.Tensor:
        applied_torque = sim_to_action_order(self.hand.data.applied_torque, self.mapping.action_joint_indices)
        fingertip_forces_w = self._select_contact_forces_w(forces_w, self.hand_spec.force_sensor_bodies)
        fingertip_moments_w = torch.zeros_like(fingertip_forces_w)
        fingertip_wrench = torch.cat((fingertip_forces_w, fingertip_moments_w), dim=-1).reshape(self.num_envs, -1)
        return torch.cat((applied_torque, fingertip_wrench), dim=-1)

    def _get_dexndm_contact_force_with_binary_obs(self, forces_w: torch.Tensor) -> torch.Tensor:
        all_body_forces_w = self._select_contact_forces_w(forces_w, self.hand_spec.all_contact_bodies)
        object_force_w = self._get_cube_net_contact_force_w(forces_w).unsqueeze(1)
        all_body_with_object_forces_w = torch.cat((all_body_forces_w, object_force_w), dim=1)
        contacts = (
            torch.linalg.norm(all_body_with_object_forces_w, dim=-1, keepdim=True)
            >= float(self.cfg.contact_sensor_binary_threshold_n)
        ).to(dtype=all_body_with_object_forces_w.dtype)
        return torch.cat((all_body_with_object_forces_w, contacts), dim=-1).reshape(self.num_envs, -1)

    def _get_gravity_palm_obs(self) -> torch.Tensor:
        gravity_w = torch.zeros((self.num_envs, 1, 3), device=self.device, dtype=torch.float32)
        gravity_w[:, :, 2] = -1.0
        return self._world_vectors_to_hand(gravity_w).squeeze(1)

    def _get_dexndm_zpos_observations(
        self, q_action: torch.Tensor, *, force_zpos_tail: bool = False
    ) -> torch.Tensor:
        obs_dim = int(
            self.cfg.dexndm_zpos_observation_space
        )
        zpos_tail_enabled = force_zpos_tail
        obs = torch.zeros(
            (self.num_envs, obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        cursor = 0

        history_obs = self._get_dexndm_history_obs(q_action)
        history_keep_frames = max(
            0,
            min(
                int(getattr(self.cfg, "dexndm_zpos_history_keep_frames", self.dexndm_obs_history_len)),
                self.dexndm_obs_history_len,
            ),
        )
        if history_keep_frames < self.dexndm_obs_history_len:
            history_frame_dim = 2 * self.action_dim
            history_obs = history_obs.view(self.num_envs, self.dexndm_obs_history_len, history_frame_dim).clone()
            if history_keep_frames == 0:
                history_obs.zero_()
            else:
                history_obs[:, : self.dexndm_obs_history_len - history_keep_frames] = 0.0
            history_obs = history_obs.reshape(self.num_envs, -1)
        obs[:, cursor : cursor + history_obs.shape[1]] = history_obs
        cursor += history_obs.shape[1]

        noise_scale = float(getattr(self.cfg, "dexndm_joint_noise_scale", 0.0)) * 0.1
        fingertip_state_w = self.hand.data.body_link_state_w[:, self.fingertip_body_ids]
        fingertip_state_obs = self._dexndm_state_w_to_xyzw_obs(fingertip_state_w).reshape(self.num_envs, -1)
        if noise_scale > 0.0:
            fingertip_state_obs = fingertip_state_obs + (2.0 * torch.rand_like(fingertip_state_obs) - 1.0) * noise_scale
        obs[:, cursor : cursor + fingertip_state_obs.shape[1]] = fingertip_state_obs
        cursor += fingertip_state_obs.shape[1]

        # DexNDM zpos allocates 384 dims, but its object-state write overwrites the
        # intended dof-velocity slot. Preserve that observed layout and leave the tail padded.
        object_state_w = torch.cat(
            (
                self._active_cube_root_pos_w(),
                self._active_cube_root_quat_w(),
                self._active_cube_root_lin_vel_w(),
                self._active_cube_root_ang_vel_w(),
            ),
            dim=-1,
        ).unsqueeze(1)
        object_state_obs = self._dexndm_state_w_to_xyzw_obs(object_state_w).reshape(self.num_envs, -1)
        if noise_scale > 0.0:
            object_state_obs = object_state_obs + (2.0 * torch.rand_like(object_state_obs) - 1.0) * noise_scale
        obs[:, cursor : cursor + object_state_obs.shape[1]] = object_state_obs
        cursor += object_state_obs.shape[1]

        guide_obs = self._get_dexndm_guide_obs()
        obs[:, cursor : cursor + guide_obs.shape[1]] = guide_obs
        cursor += guide_obs.shape[1]

        forces_w = self._get_cube_contact_forces_w()
        self._update_contact_force_buffers(forces_w)
        force_obs = self._get_dexndm_force_obs(forces_w)
        obs[:, cursor : cursor + force_obs.shape[1]] = force_obs
        cursor += force_obs.shape[1]

        if bool(getattr(self.cfg, "dexndm_zpos_all_body_contact_force_obs_enabled", True)):
            contact_obs = self._get_dexndm_contact_force_with_binary_obs(forces_w)
        else:
            contact_obs_dim = (len(self.hand_spec.all_contact_bodies) + 1) * 4
            contact_obs = torch.zeros((self.num_envs, contact_obs_dim), device=self.device, dtype=torch.float32)
        contact_dim = min(contact_obs.shape[1], obs_dim - cursor)
        if contact_dim > 0:
            obs[:, cursor : cursor + contact_dim] = contact_obs[:, :contact_dim]
            cursor += contact_dim
        if zpos_tail_enabled:
            reset_threshold = max(float(self.cfg.dexndm_omni_wrist_reset_pos_dist_m), 1.0e-6)
            init_pos_delta = (self._active_cube_root_pos_w() - self.dexndm_init_cube_pos_w) / reset_threshold
            if cursor + init_pos_delta.shape[1] > obs_dim:
                raise RuntimeError(
                    f"DexNDM zpos observation layout has no room for init position delta: "
                    f"cursor={cursor}, obs_dim={obs_dim}."
                )
            obs[:, cursor : cursor + init_pos_delta.shape[1]] = init_pos_delta
            cursor += init_pos_delta.shape[1]
            if bool(getattr(self.cfg, "dexndm_zpos_rot_axis_obs_enabled", False)):
                rot_axis_obs = self.dexndm_rot_axis_buf.to(dtype=torch.float32)
                if cursor + rot_axis_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for rotation axis: "
                        f"cursor={cursor}, obs_dim={obs_dim}."
                    )
                obs[:, cursor : cursor + rot_axis_obs.shape[1]] = rot_axis_obs
                cursor += rot_axis_obs.shape[1]
            if bool(getattr(self.cfg, "dexndm_zpos_twist_target_obs_enabled", False)):
                twist_target_obs = self._get_dexndm_twist_target_obs()
                if cursor + twist_target_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for twist target obs: "
                        f"cursor={cursor}, obs_dim={obs_dim}, twist_dim={twist_target_obs.shape[1]}."
                    )
                obs[:, cursor : cursor + twist_target_obs.shape[1]] = twist_target_obs
                cursor += twist_target_obs.shape[1]
            if bool(getattr(self.cfg, "dexndm_zpos_gravity_palm_obs_enabled", False)):
                gravity_palm_obs = self._get_gravity_palm_obs()
                if cursor + gravity_palm_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for gravity-in-palm obs: "
                        f"cursor={cursor}, obs_dim={obs_dim}, gravity_dim={gravity_palm_obs.shape[1]}."
                    )
                obs[:, cursor : cursor + gravity_palm_obs.shape[1]] = gravity_palm_obs
                cursor += gravity_palm_obs.shape[1]
            if self.multisize_cube_enabled and self.multisize_cube_size_obs_enabled:
                if cursor + 1 > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for cube size scalar: "
                        f"cursor={cursor}, obs_dim={obs_dim}."
                )
                obs[:, cursor] = self.active_cube_size_norm.to(dtype=torch.float32)
                cursor += 1
            if self.multicuboid_enabled and self.multicuboid_dim_obs_enabled:
                dim_obs = self.active_object_dim_norm.to(dtype=torch.float32)
                if cursor + dim_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for cuboid dim scalars: "
                        f"cursor={cursor}, obs_dim={obs_dim}, dim_obs={dim_obs.shape[1]}."
                )
                obs[:, cursor : cursor + dim_obs.shape[1]] = dim_obs
                cursor += dim_obs.shape[1]
            if self.primitive_multi_shape_enabled and self.primitive_dim_obs_enabled:
                dim_obs = self.active_object_dim_norm.to(dtype=torch.float32)
                if cursor + dim_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for primitive dim scalars: "
                        f"cursor={cursor}, obs_dim={obs_dim}, dim_obs={dim_obs.shape[1]}."
                )
                obs[:, cursor : cursor + dim_obs.shape[1]] = dim_obs
                cursor += dim_obs.shape[1]
            if self.primitive_multi_shape_enabled and self.primitive_geometry_latent_obs_enabled:
                if self.primitive_geometry_latent_source == "hand_surface_delta":
                    geometry_obs = self._compute_hand_surface_delta_geometry_latent().to(dtype=torch.float32)
                    self.active_object_geometry_latent[:] = geometry_obs
                else:
                    geometry_obs = self.active_object_geometry_latent.to(dtype=torch.float32)
                if geometry_obs.shape[1] != self.primitive_geometry_latent_dim:
                    raise RuntimeError(
                        f"Primitive geometry latent obs has width {geometry_obs.shape[1]}, "
                        f"expected {self.primitive_geometry_latent_dim}."
                    )
                if cursor + geometry_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for primitive geometry latent obs: "
                        f"cursor={cursor}, obs_dim={obs_dim}, geometry_obs={geometry_obs.shape[1]}."
                    )
                obs[:, cursor : cursor + geometry_obs.shape[1]] = geometry_obs
                cursor += geometry_obs.shape[1]
            if self.primitive_multi_shape_enabled and self.primitive_shape_obs_enabled:
                shape_obs_mode = str(getattr(self.cfg, "primitive_shape_obs_mode", "scalar")).lower()
                if shape_obs_mode == "onehot":
                    shape_obs = self.active_object_shape_onehot.to(dtype=torch.float32)
                elif shape_obs_mode == "scalar":
                    shape_obs = self.active_object_is_cylinder.to(dtype=torch.float32).unsqueeze(-1)
                else:
                    raise ValueError(
                        "primitive_shape_obs_mode must be one of 'scalar' or 'onehot', "
                        f"got {getattr(self.cfg, 'primitive_shape_obs_mode', None)!r}."
                    )
                if cursor + shape_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for primitive shape obs: "
                        f"cursor={cursor}, obs_dim={obs_dim}, shape_obs={shape_obs.shape[1]}."
                )
                obs[:, cursor : cursor + shape_obs.shape[1]] = shape_obs
                cursor += shape_obs.shape[1]
            if bool(getattr(self.cfg, "pose6d_anchor_obs_enabled", False)):
                anchor_obs = self._get_pose6d_anchor_obs()
                if cursor + anchor_obs.shape[1] > obs_dim:
                    raise RuntimeError(
                        f"DexNDM zpos observation layout has no room for pose6d anchor obs: "
                        f"cursor={cursor}, obs_dim={obs_dim}, anchor_dim={anchor_obs.shape[1]}."
                    )
                obs[:, cursor : cursor + anchor_obs.shape[1]] = anchor_obs
                cursor += anchor_obs.shape[1]
        return obs



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        raw_targets = self.action_targets + self.cfg.action_scale * self.actions
        raw_targets = torch.minimum(torch.maximum(raw_targets, self.action_lower), self.action_upper)
        self.prev_action_targets[:] = self.action_targets
        self.action_targets[:] = (
            self.cfg.action_smoothing * raw_targets + (1.0 - self.cfg.action_smoothing) * self.action_targets
        )
        self.sim_joint_targets[:] = action_to_sim_order(self.action_targets, self.mapping.action_joint_indices, self.num_joints)
        self._update_simtoolreal_external_wrench()

    def _apply_action(self) -> None:
        self._update_random_external_force()
        if self.dexndm_torque_control:
            q_action = sim_to_action_order(self.hand.data.joint_pos, self.mapping.action_joint_indices)
            dof_vel_finite_diff = (q_action - self.dexndm_prev_control_q_action) / max(float(self.physics_dt), 1.0e-8)
            dof_vel_for_control = dof_vel_finite_diff
            if self.dexndm_torque_control_velocity_source == "joint_vel":
                dof_vel_for_control = sim_to_action_order(self.hand.data.joint_vel, self.mapping.action_joint_indices)
            if self.dexndm_torque_control_gain_source == "default_joint_pd":
                torque = (
                    self.dexndm_controller_pgain_action * (self.action_targets - q_action)
                    - self.dexndm_controller_dgain_action * dof_vel_for_control
                )
            else:
                torque = (
                    self.dexndm_controller_pgain * (self.action_targets - q_action)
                    - self.dexndm_controller_dgain * dof_vel_for_control
                )
            self.dexndm_raw_commanded_torque_action[:] = torque.detach()
            if self.dexndm_torque_limit > 0.0:
                torque = torch.clamp(torque, -self.dexndm_torque_limit, self.dexndm_torque_limit)
            self.dexndm_commanded_torque_action[:] = torque.detach()
            self.dexndm_dof_vel_finite_diff[:] = dof_vel_for_control.detach()
            self.dexndm_prev_control_q_action[:] = q_action.detach()
            self.dexndm_sim_joint_efforts[:] = action_to_sim_order(
                torque,
                self.mapping.action_joint_indices,
                self.num_joints,
            )
            self.hand.set_joint_effort_target(self.dexndm_sim_joint_efforts)
            return
        self.hand.set_joint_position_target(self.sim_joint_targets)


    def _get_object_pose_obs(self) -> torch.Tensor:
        pose_obs = self.current_pose_palm.clone()
        quat = normalize_quat_wxyz(pose_obs[:, 3:7])
        pose_obs[:, 3:7] = torch.where(quat[:, :1] < 0.0, -quat, quat)
        return pose_obs



    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._compute_intermediate_values()
        q_action = sim_to_action_order(self.hand.data.joint_pos, self.mapping.action_joint_indices)
        qdot_action = sim_to_action_order(self.hand.data.joint_vel, self.mapping.action_joint_indices)
        obs = {"policy": self._get_sharpa_deployable_pose6d_observations(q_action)}
        obs["critic"] = self._get_sharpa_deployable_asymmetric_critic_observations(q_action)
        if self.tactile_enabled:
            tactile = self._get_tactile_history()
            obs = {key: torch.cat((value, tactile), dim=-1) for key, value in obs.items()}
        return obs

    def _get_tactile_history(self):
        step = int(self.common_step_counter)
        if step == self.tactile.previous_step:
            return self.tactile.frames.flatten(1)
        indices = [self.contact_sensor_body_name_to_idx[n] for n in self.hand_spec.force_sensor_bodies]
        if self.multiobject_enabled and not self.primitive_single_asset_per_env_enabled:
            forces = torch.zeros(self.num_envs, 2, 5, 3, device=self.device)
            points = torch.full((self.num_envs, 5, 3), float("nan"), device=self.device)
            for size_id, object_id in enumerate(self.multisize_object_ids):
                data = self.multisize_cube_contact_sensors[object_id].data
                mask = self.active_cube_size_id == size_id
                forces[mask] = data.force_matrix_w_history[:, :2, 0, indices][mask]
                points[mask] = data.contact_pos_w[:, 0, indices][mask]
        else:
            data = self.cube_contact_sensor.data
            forces = data.force_matrix_w_history[:, :2, 0, indices]
            points = data.contact_pos_w[:, 0, indices]
        frame = contact_frame(forces, points, self.hand.data.body_link_state_w[:, self.tactile_body_ids, :7])
        return self.tactile.update(frame, step)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        cube_pos_w = self._active_cube_root_pos_w()
        pos = self.current_pose_palm[:, :3]
        out_of_workspace = (
            (pos[:, 0] < self.cfg.min_cube_palm_x)
            | (pos[:, 0] > self.cfg.max_cube_palm_x)
            | (torch.abs(pos[:, 1]) > self.cfg.max_cube_abs_y)
            | (torch.abs(pos[:, 2]) > self.cfg.max_cube_abs_z)
            | (cube_pos_w[:, 2] < self.cfg.min_cube_world_z)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_workspace, time_out

    def _training_metric_mean(self, value: torch.Tensor) -> torch.Tensor:
        """Mean across training environments."""

        if value.ndim == 0:
            return value
        if value.shape[0] != self.num_envs:
            raise ValueError(
                "Training metric tensors must use environments as their first dimension; "
                f"got shape {tuple(value.shape)} for num_envs={self.num_envs}."
            )
        selected = value[self.training_metric_env_mask]
        if not (selected.is_floating_point() or selected.is_complex()):
            selected = selected.float()
        return selected.mean()

    def _training_metric_min(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value
        return value[self.training_metric_env_mask].min()

    def _training_metric_max(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value
        return value[self.training_metric_env_mask].max()

    def _get_dexndm_twist_rewards(self, q_action: torch.Tensor) -> torch.Tensor:
        self._update_twist_target_terms()
        target_sign = torch.where(
            self.target_twist_angle >= 0.0,
            torch.ones_like(self.target_twist_angle),
            -torch.ones_like(self.target_twist_angle),
        )
        target_abs = torch.abs(self.target_twist_angle)
        self.twist_progress_raw[:] = (target_sign * self.current_twist_angle).detach()
        self.twist_progress[:] = torch.minimum(
            torch.clamp(self.twist_progress_raw, min=0.0),
            target_abs,
        ).detach()

        progress = torch.clamp(self.twist_progress - self.best_twist_progress, min=0.0)
        progress_clip = max(float(getattr(self.cfg, "twist_progress_clip_rad", 0.0)), 0.0)
        if progress_clip > 0.0:
            progress = torch.clamp(progress, max=progress_clip)
        self.best_twist_progress[:] = torch.maximum(self.best_twist_progress, self.twist_progress.detach())

        uses_pose6d_translation = True
        translation_progress = torch.zeros_like(progress)
        translation_overshoot = torch.zeros_like(progress)
        if uses_pose6d_translation:
            translation_progress = torch.clamp(self.translation_progress - self.best_translation_progress, min=0.0)
            translation_progress_clip = max(
                float(getattr(self.cfg, "pose6d_translation_progress_clip_m", 0.0015)),
                0.0,
            )
            if translation_progress_clip > 0.0:
                translation_progress = torch.clamp(translation_progress, max=translation_progress_clip)
            self.best_translation_progress[:] = torch.maximum(
                self.best_translation_progress,
                self.translation_progress.detach(),
            )
            translation_overshoot = torch.clamp(-self.translation_remaining, min=0.0)

        drop = self.reset_terminated
        swing_success_threshold_per_env = self._active_twist_swing_success_threshold().to(dtype=self.twist_swing_drift.dtype)
        twist_success = (
            (self.twist_remaining < float(getattr(self.cfg, "twist_success_threshold_rad", 0.0872665)))
            & (self.twist_swing_drift < swing_success_threshold_per_env)
        )
        anchor_success = torch.ones_like(drop, dtype=torch.bool)
        if self._uses_pose6d_anchor_relaxed_targets():
            anchor_success = self.pose6d_anchor_error_m < self._active_pose6d_anchor_success_threshold().to(
                dtype=self.pose6d_anchor_error_m.dtype
            )
        if uses_pose6d_translation:
            if self._uses_pose6d_anchor_relaxed_targets():
                position_success = anchor_success
            else:
                position_success = (
                    (
                        torch.abs(self.translation_remaining)
                        < float(getattr(self.cfg, "pose6d_translation_success_threshold_m", 0.005))
                    )
                    & (
                        self.translation_orthogonal_drift
                        < float(getattr(self.cfg, "pose6d_translation_orthogonal_success_threshold_m", 0.010))
                    )
                )
        else:
            position_success = self.twist_position_drift < float(
                getattr(self.cfg, "twist_position_success_threshold_m", 0.008)
            )
        if self.pose6d_success_criterion == "full_pose":
            position_success, twist_success, _, _ = full_pose_success_terms(
                self.current_pose_palm, self.target_pose_palm,
                self.cfg.pose6d_success_position_threshold_m,
                self.cfg.pose6d_success_orientation_threshold_rad,
            )
        # This one predicate drives the dwell counter, reward, goal refresh,
        # episode success and every frontier curriculum success event below.
        near_success = twist_success & position_success & ~drop
        self.near_success_steps[:] = (self.near_success_steps + near_success.long()) * near_success.long()
        success = self.near_success_steps >= int(self.cfg.success_steps)
        new_success = success & ~self.target_reached
        self.target_reached[:] = self.target_reached | new_success
        target_chain_phase_for_metrics = self.target_chain_phase.detach().clone()

        done_envs = self.reset_terminated | self.reset_time_outs
        metric_env_mask = self.training_metric_env_mask
        metric_done_envs = done_envs & metric_env_mask
        done_count = metric_done_envs.float().sum()
        episode_has_success = self.target_reached | (self.achieved_goals > 0)
        episode_success = (episode_has_success.float() * metric_done_envs.float()).sum() / torch.clamp(
            done_count, min=1.0
        )
        self.episode_success_rate.copy_(torch.where(done_count > 0.0, episode_success, self.episode_success_rate))

        refresh_on_success = bool(getattr(self.cfg, "twist_target_refresh_on_success", False))
        refresh_envs = torch.zeros_like(new_success, dtype=torch.bool)
        if refresh_on_success:
            refresh_envs = new_success & ~done_envs
        curriculum_event_envs = (done_envs | refresh_envs) & metric_env_mask
        if bool(getattr(self.cfg, "adaptive_frontier_soft_promotion_enabled", False)):
            # Ordering defines soft-promotion priority for the rescue recipe.
            self._update_adaptive_chain_twist_target_curriculum(curriculum_event_envs, self.target_reached)
            if uses_pose6d_translation:
                self._update_adaptive_pose6d_translation_curriculum(curriculum_event_envs, self.target_reached)
            self._update_adaptive_twist_target_curriculum(curriculum_event_envs, self.target_reached)
        else:
            self._update_adaptive_twist_target_curriculum(curriculum_event_envs, self.target_reached)
            self._update_adaptive_chain_twist_target_curriculum(curriculum_event_envs, self.target_reached)
            if uses_pose6d_translation:
                self._update_adaptive_pose6d_translation_curriculum(curriculum_event_envs, self.target_reached)
        if uses_pose6d_translation:
            self._update_adaptive_chain_pose6d_translation_curriculum(curriculum_event_envs, self.target_reached)

        use_torque_work_penalty = bool(getattr(self.cfg, "dexndm_twist_torque_work_penalty_enabled", False))
        zero_penalty = torch.zeros_like(self.twist_remaining)
        action_penalty = zero_penalty
        joint_vel_penalty = zero_penalty
        torque_penalty = zero_penalty
        work_penalty = zero_penalty
        excess_load_penalty = zero_penalty
        excess_load_contact_joint_rate = zero_penalty
        excess_load_exceed_joint_rate = zero_penalty
        excess_load_contact_ratio_mean = zero_penalty
        excess_load_contact_ratio_max = zero_penalty
        if float(getattr(self.cfg, "action_penalty_scale", 0.0)) != 0.0:
            action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        if float(getattr(self.cfg, "joint_vel_penalty_scale", 0.0)) != 0.0:
            qdot_action = sim_to_action_order(self.hand.data.joint_vel, self.mapping.action_joint_indices)
            joint_vel_penalty = torch.sum(torch.square(qdot_action), dim=-1)
        if use_torque_work_penalty:
            qdot_action = sim_to_action_order(self.hand.data.joint_vel, self.mapping.action_joint_indices)
            applied_torque = sim_to_action_order(self.hand.data.applied_torque, self.mapping.action_joint_indices)
            torque_penalty = torch.sum(torch.square(applied_torque), dim=-1)
            work_penalty = torch.square(torch.sum(applied_torque * qdot_action, dim=-1))
        if self.dexndm_contact_gated_excess_load_penalty_enabled:
            (
                excess_load_penalty,
                normalized_excess_load,
                excess_load_joint_contact,
                excess_load_ratio,
            ) = self._get_contact_gated_excess_load_terms(q_action)
            joint_contact_float = excess_load_joint_contact.to(dtype=q_action.dtype)
            excess_load_contact_joint_rate = joint_contact_float.mean(dim=-1)
            excess_load_exceed_joint_rate = (
                excess_load_joint_contact & (normalized_excess_load > 0.0)
            ).to(dtype=q_action.dtype).mean(dim=-1)
            contact_joint_count = torch.clamp(joint_contact_float.sum(dim=-1), min=1.0)
            excess_load_contact_ratio_mean = (
                excess_load_ratio * joint_contact_float
            ).sum(dim=-1) / contact_joint_count
            excess_load_contact_ratio_max = torch.where(
                excess_load_joint_contact,
                excess_load_ratio,
                torch.zeros_like(excess_load_ratio),
            ).amax(dim=-1)
        (
            grasp_wrench_quality,
            grasp_wrench_force_margin_min,
            grasp_wrench_torque_margin_min,
            grasp_wrench_deficit,
            grasp_wrench_below_target,
            grasp_wrench_contact_confidence_count,
            grasp_wrench_maintenance_reward,
        ) = self._get_grasp_wrench_coverage_terms()
        remaining_reward = torch.exp(
            -self.twist_remaining / max(float(getattr(self.cfg, "twist_remaining_reward_temperature_rad", 0.0872665)), 1.0e-6)
        )
        twist_overshoot = torch.clamp(self.twist_progress_raw - target_abs, min=0.0)
        swing_success_threshold = torch.clamp(swing_success_threshold_per_env, min=1.0e-6)
        swing_soft_barrier = float(getattr(self.cfg, "twist_swing_soft_barrier_rad", 0.0))
        swing_barrier_fraction = torch.zeros_like(self.twist_swing_drift)
        swing_reward_margin = torch.ones_like(self.twist_swing_drift)
        if swing_soft_barrier > 0.0:
            swing_barrier_width = torch.clamp(swing_success_threshold - swing_soft_barrier, min=1.0e-6)
            swing_barrier_fraction = torch.clamp(
                (self.twist_swing_drift - swing_soft_barrier) / swing_barrier_width,
                min=0.0,
                max=1.0,
            )
            swing_barrier_fraction = torch.where(
                swing_soft_barrier < swing_success_threshold,
                swing_barrier_fraction,
                torch.zeros_like(swing_barrier_fraction),
            )
            if bool(getattr(self.cfg, "twist_near_success_swing_margin_enabled", False)):
                swing_reward_margin = 1.0 - swing_barrier_fraction
        translation_remaining_reward = torch.exp(
            -torch.abs(self.translation_remaining)
            / max(float(getattr(self.cfg, "pose6d_translation_remaining_reward_temperature_m", 0.006)), 1.0e-6)
        )
        if self._uses_pose6d_anchor_relaxed_targets():
            basic_pose_position_error = self.pose6d_anchor_error_m
        else:
            basic_pose_position_error = torch.linalg.norm(
                self.current_pose_palm[:, :3] - self.target_pose_palm[:, :3],
                dim=-1,
            )
        basic_pose_orientation_error = quat_angle_error_wxyz(
            self.target_pose_palm[:, 3:7],
            self.current_pose_palm[:, 3:7],
        )
        basic_pose_position_gate_weight = torch.ones_like(basic_pose_position_error)
        basic_pose_position_gate_temperature = torch.full_like(
            basic_pose_position_error,
            max(float(getattr(self.cfg, "basic_pose_position_reward_temperature_m", 0.006)), 1.0e-6),
        )
        basic_pose_position_gate_loose = torch.zeros_like(basic_pose_position_error, dtype=torch.bool)
        basic_pose_position_gate_mid = torch.zeros_like(basic_pose_position_error, dtype=torch.bool)
        basic_pose_position_gate_tight = torch.ones_like(basic_pose_position_error, dtype=torch.bool)
        if bool(getattr(self.cfg, "basic_pose_position_orientation_gate_enabled", False)):
            tight_threshold = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_tight_threshold_rad", 0.436332)),
                0.0,
            )
            loose_threshold = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_loose_threshold_rad", 0.785398)),
                tight_threshold,
            )
            basic_pose_position_gate_loose = basic_pose_orientation_error > loose_threshold
            basic_pose_position_gate_mid = (
                (basic_pose_orientation_error > tight_threshold) & (~basic_pose_position_gate_loose)
            )
            basic_pose_position_gate_tight = ~(
                basic_pose_position_gate_loose | basic_pose_position_gate_mid
            )
            loose_weight = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_loose_weight", 0.15)),
                0.0,
            )
            mid_weight = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_mid_weight", 0.40)),
                0.0,
            )
            tight_weight = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_tight_weight", 1.0)),
                0.0,
            )
            loose_temperature = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_loose_temperature_m", 0.030)),
                1.0e-6,
            )
            mid_temperature = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_mid_temperature_m", 0.020)),
                1.0e-6,
            )
            tight_temperature = max(
                float(getattr(self.cfg, "basic_pose_position_orientation_gate_tight_temperature_m", 0.015)),
                1.0e-6,
            )
            basic_pose_position_gate_weight = torch.where(
                basic_pose_position_gate_loose,
                torch.full_like(basic_pose_position_gate_weight, loose_weight),
                torch.where(
                    basic_pose_position_gate_mid,
                    torch.full_like(basic_pose_position_gate_weight, mid_weight),
                    torch.full_like(basic_pose_position_gate_weight, tight_weight),
                ),
            )
            basic_pose_position_gate_temperature = torch.where(
                basic_pose_position_gate_loose,
                torch.full_like(basic_pose_position_gate_temperature, loose_temperature),
                torch.where(
                    basic_pose_position_gate_mid,
                    torch.full_like(basic_pose_position_gate_temperature, mid_temperature),
                    torch.full_like(basic_pose_position_gate_temperature, tight_temperature),
                ),
            )
        basic_pose_position_reward = torch.exp(
            -basic_pose_position_error / basic_pose_position_gate_temperature
        )
        basic_pose_orientation_reward = torch.exp(
            -basic_pose_orientation_error
            / max(float(getattr(self.cfg, "basic_pose_orientation_reward_temperature_rad", 0.174533)), 1.0e-6)
        )

        reward_twist_progress_scaled = float(getattr(self.cfg, "twist_progress_reward_scale", 50.0)) * progress
        reward_twist_remaining_scaled = float(getattr(self.cfg, "twist_remaining_reward_scale", 0.5)) * remaining_reward
        reward_twist_position_drift_penalty_scaled = (
            -float(getattr(self.cfg, "twist_position_drift_penalty_scale", 20.0)) * self.twist_position_drift
        )
        reward_twist_swing_drift_penalty_scaled = (
            -float(getattr(self.cfg, "twist_swing_drift_penalty_scale", 1.0)) * self.twist_swing_drift
        )
        reward_twist_swing_barrier_penalty_scaled = (
            -float(getattr(self.cfg, "twist_swing_barrier_penalty_scale", 0.0))
            * torch.square(swing_barrier_fraction)
        )
        reward_twist_overshoot_penalty_scaled = (
            -float(getattr(self.cfg, "twist_overshoot_penalty_scale", 0.0)) * twist_overshoot
        )
        reward_translation_progress_scaled = (
            float(getattr(self.cfg, "pose6d_translation_progress_reward_scale", 0.0)) * translation_progress
        )
        reward_translation_remaining_scaled = (
            float(getattr(self.cfg, "pose6d_translation_remaining_reward_scale", 0.0)) * translation_remaining_reward
        )
        reward_translation_orthogonal_drift_penalty_scaled = (
            -float(getattr(self.cfg, "pose6d_translation_orthogonal_drift_penalty_scale", 0.0))
            * self.translation_orthogonal_drift
        )
        reward_translation_overshoot_penalty_scaled = (
            -float(getattr(self.cfg, "pose6d_translation_overshoot_penalty_scale", 0.0)) * translation_overshoot
        )
        reward_basic_pose_position_scaled = (
            float(getattr(self.cfg, "basic_pose_position_reward_scale", 0.0))
            * basic_pose_position_gate_weight
            * basic_pose_position_reward
        )
        reward_basic_pose_orientation_scaled = (
            float(getattr(self.cfg, "basic_pose_orientation_reward_scale", 0.0)) * basic_pose_orientation_reward
        )
        reward_basic_pose_position_error_penalty_scaled = (
            -float(getattr(self.cfg, "basic_pose_position_error_penalty_scale", 0.0)) * basic_pose_position_error
        )
        reward_basic_pose_orientation_error_penalty_scaled = (
            -float(getattr(self.cfg, "basic_pose_orientation_error_penalty_scale", 0.0))
            * basic_pose_orientation_error
        )
        if not uses_pose6d_translation:
            reward_translation_progress_scaled.zero_()
            reward_translation_remaining_scaled.zero_()
            reward_translation_orthogonal_drift_penalty_scaled.zero_()
            reward_translation_overshoot_penalty_scaled.zero_()
        reward_near_success_scaled = (
            float(getattr(self.cfg, "near_success_bonus", 0.0)) * near_success.float() * swing_reward_margin
        )
        near_success_steps_reward_scale = float(getattr(self.cfg, "near_success_steps_reward_scale", 0.0))
        if near_success_steps_reward_scale != 0.0:
            near_success_steps_fraction = torch.clamp(
                self.near_success_steps.to(dtype=reward_near_success_scaled.dtype)
                / max(float(getattr(self.cfg, "success_steps", 1)), 1.0),
                min=0.0,
                max=1.0,
            )
            near_success_steps_reward_power = max(
                float(getattr(self.cfg, "near_success_steps_reward_power", 2.0)),
                0.0,
            )
            reward_near_success_steps_scaled = near_success_steps_reward_scale * torch.pow(
                near_success_steps_fraction,
                near_success_steps_reward_power,
            ) * swing_reward_margin
        else:
            reward_near_success_steps_scaled = torch.zeros_like(reward_near_success_scaled)
        reward_success_scaled = float(self.cfg.success_bonus) * new_success.float()
        reward_success_hold_scaled = float(getattr(self.cfg, "success_hold_bonus", 0.0)) * success.float()
        reward_drop_scaled = float(self.cfg.drop_penalty) * drop.float()
        reward_action_penalty_scaled = -float(self.cfg.action_penalty_scale) * action_penalty
        reward_joint_vel_penalty_scaled = -float(self.cfg.joint_vel_penalty_scale) * joint_vel_penalty
        reward_torque_penalty_scaled = (
            -float(getattr(self.cfg, "dexndm_torque_penalty_scale", 0.0)) * torque_penalty
        )
        reward_work_penalty_scaled = -float(getattr(self.cfg, "dexndm_work_penalty_scale", 0.0)) * work_penalty
        reward_excess_load_penalty_scaled = (
            -self.dexndm_contact_gated_excess_load_penalty_scale * excess_load_penalty
        )
        reward_grasp_wrench_maintenance_scaled = (
            self.dexndm_grasp_wrench_maintenance_reward_scale * grasp_wrench_maintenance_reward
        )
        if not use_torque_work_penalty:
            reward_torque_penalty_scaled = torch.zeros_like(reward_torque_penalty_scaled)
            reward_work_penalty_scaled = torch.zeros_like(reward_work_penalty_scaled)

        reward_variant = str(getattr(self.cfg, "dexndm_twist_reward_variant", "progress")).lower()
        if reward_variant in {"progress", "directional_progress"}:
            reward_basic_pose_position_scaled = torch.zeros_like(reward_basic_pose_position_scaled)
            reward_basic_pose_orientation_scaled = torch.zeros_like(reward_basic_pose_orientation_scaled)
            reward_basic_pose_position_error_penalty_scaled = torch.zeros_like(
                reward_basic_pose_position_error_penalty_scaled
            )
            reward_basic_pose_orientation_error_penalty_scaled = torch.zeros_like(
                reward_basic_pose_orientation_error_penalty_scaled
            )
            reward = (
                reward_twist_progress_scaled
                + reward_twist_remaining_scaled
                + reward_twist_position_drift_penalty_scaled
                + reward_twist_swing_drift_penalty_scaled
                + reward_twist_swing_barrier_penalty_scaled
                + reward_twist_overshoot_penalty_scaled
                + reward_translation_progress_scaled
                + reward_translation_remaining_scaled
                + reward_translation_orthogonal_drift_penalty_scaled
                + reward_translation_overshoot_penalty_scaled
                + reward_near_success_scaled
                + reward_near_success_steps_scaled
                + reward_success_scaled
                + reward_success_hold_scaled
                + reward_drop_scaled
                + reward_action_penalty_scaled
                + reward_joint_vel_penalty_scaled
                + reward_torque_penalty_scaled
                + reward_work_penalty_scaled
                + reward_excess_load_penalty_scaled
                + reward_grasp_wrench_maintenance_scaled
            )
        elif reward_variant in {"basic", "basic_pose", "basic_pose_error", "pose_error"}:
            reward_twist_progress_scaled = torch.zeros_like(reward_twist_progress_scaled)
            reward_twist_remaining_scaled = torch.zeros_like(reward_twist_remaining_scaled)
            reward_twist_position_drift_penalty_scaled = torch.zeros_like(
                reward_twist_position_drift_penalty_scaled
            )
            reward_twist_swing_drift_penalty_scaled = torch.zeros_like(reward_twist_swing_drift_penalty_scaled)
            reward_twist_swing_barrier_penalty_scaled = torch.zeros_like(
                reward_twist_swing_barrier_penalty_scaled
            )
            reward_twist_overshoot_penalty_scaled = torch.zeros_like(reward_twist_overshoot_penalty_scaled)
            reward_translation_progress_scaled = torch.zeros_like(reward_translation_progress_scaled)
            reward_translation_remaining_scaled = torch.zeros_like(reward_translation_remaining_scaled)
            reward_translation_orthogonal_drift_penalty_scaled = torch.zeros_like(
                reward_translation_orthogonal_drift_penalty_scaled
            )
            reward_translation_overshoot_penalty_scaled = torch.zeros_like(
                reward_translation_overshoot_penalty_scaled
            )
            reward = (
                reward_basic_pose_position_scaled
                + reward_basic_pose_orientation_scaled
                + reward_basic_pose_position_error_penalty_scaled
                + reward_basic_pose_orientation_error_penalty_scaled
                + reward_near_success_scaled
                + reward_near_success_steps_scaled
                + reward_success_scaled
                + reward_success_hold_scaled
                + reward_drop_scaled
                + reward_action_penalty_scaled
                + reward_joint_vel_penalty_scaled
                + reward_torque_penalty_scaled
                + reward_work_penalty_scaled
                + reward_excess_load_penalty_scaled
                + reward_grasp_wrench_maintenance_scaled
            )
        else:
            raise ValueError(
                "dexndm_twist_reward_variant must be 'progress' or 'basic_pose_error', "
                f"got {self.cfg.dexndm_twist_reward_variant!r}."
            )

        dt = max(float(self.dt), 1.0e-8)
        cube_pos_w = self._active_cube_root_pos_w()
        cube_quat_w = normalize_quat_wxyz(self._active_cube_root_quat_w())
        dof_vel_finite_diff = (q_action - self.dexndm_prev_q_action) / dt
        self.dexndm_prev_cube_pos_w[:] = cube_pos_w.detach()
        self.dexndm_prev_cube_quat_w[:] = cube_quat_w.detach()
        self.dexndm_prev_q_action[:] = q_action.detach()
        self.dexndm_dof_vel_finite_diff[:] = dof_vel_finite_diff.detach()
        self.dexndm_torque_penalty[:] = torque_penalty.detach()
        self.dexndm_work_penalty[:] = work_penalty.detach()
        self.last_success[:] = success.detach()
        self.last_new_goal[:] = new_success.detach()
        self.last_drop[:] = drop.detach()
        if self.dexndm_grasp_wrench_maintenance_reward_enabled:
            self.grasp_wrench_reward_pose_error[:] = self.pose_error.detach()
            self.grasp_wrench_reward_drop[:] = drop.detach()
            self.grasp_wrench_reward_success[:] = success.detach()
            self.grasp_wrench_reward_reset_grasp_index[:] = self.reset_grasp_indices.detach()
            self.grasp_wrench_reward_object_external_force_w[:] = self.object_external_force_w.detach()
            self.grasp_wrench_reward_object_external_torque_w[:] = self.object_external_torque_w.detach()
            self.grasp_wrench_reward_current_pose_palm[:] = self.current_pose_palm.detach()
            self.grasp_wrench_reward_target_pose_palm[:] = self.target_pose_palm.detach()
        self.last_actions[:] = self.actions

        refresh_count = refresh_envs.float().sum()
        if bool(refresh_count.item() > 0.0):
            refresh_env_ids = torch.nonzero(refresh_envs, as_tuple=False).squeeze(-1)
            self.achieved_goals[refresh_env_ids] += 1
            self._refresh_hand_frame_twist_targets(refresh_env_ids)

        if "log" not in self.extras:
            self.extras["log"] = {}
        rad_to_deg = 180.0 / torch.pi
        episode_achieved_goals = torch.zeros((), device=self.device)
        if bool(done_count.item() > 0.0):
            episode_achieved_goals = (
                self.achieved_goals.to(dtype=torch.float32) * metric_done_envs.float()
            ).sum() / torch.clamp(done_count, min=1.0)
        first_target_metric_mask = target_chain_phase_for_metrics == 0
        chain_target_metric_mask = target_chain_phase_for_metrics > 0
        chain_target_mask = self.target_chain_phase > 0

        def _masked_rate(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_f = (mask & metric_env_mask).float()
            return (value.float() * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)

        metric_mean = self._training_metric_mean
        metric_min = self._training_metric_min
        metric_max = self._training_metric_max

        self.extras["log"].update(
            {
                "reward_total": metric_mean(reward),
                "target_delta_rad": metric_mean(self.target_delta),
                "target_twist_angle_abs_rad": metric_mean(target_abs),
                "target_twist_angle_abs_deg": metric_mean(target_abs * rad_to_deg),
                "target_twist_axis_abs_x": metric_mean(torch.abs(self.target_twist_axis_palm[:, 0])),
                "target_twist_axis_abs_y": metric_mean(torch.abs(self.target_twist_axis_palm[:, 1])),
                "target_twist_axis_abs_z": metric_mean(torch.abs(self.target_twist_axis_palm[:, 2])),
                "target_translation_delta_m": metric_mean(self.target_translation_delta),
                "target_translation_axis_abs_x": metric_mean(torch.abs(self.target_translation_axis_palm[:, 0])),
                "target_translation_axis_abs_y": metric_mean(torch.abs(self.target_translation_axis_palm[:, 1])),
                "target_translation_axis_abs_z": metric_mean(torch.abs(self.target_translation_axis_palm[:, 2])),
                "current_twist_angle_rad": metric_mean(self.current_twist_angle),
                "twist_progress_raw_rad": metric_mean(self.twist_progress_raw),
                "twist_progress_rad": metric_mean(self.twist_progress),
                "twist_remaining_rad": metric_mean(self.twist_remaining),
                "twist_remaining_deg": metric_mean(self.twist_remaining * rad_to_deg),
                "twist_swing_drift_rad": metric_mean(self.twist_swing_drift),
                "twist_swing_drift_deg": metric_mean(self.twist_swing_drift * rad_to_deg),
                "twist_swing_barrier_fraction": metric_mean(swing_barrier_fraction),
                "twist_swing_reward_margin": metric_mean(swing_reward_margin),
                "twist_position_drift": metric_mean(self.twist_position_drift),
                "twist_swing_success_threshold_rad": metric_mean(swing_success_threshold_per_env),
                "basic_pose_position_error_m": metric_mean(basic_pose_position_error),
                "basic_pose_orientation_error_rad": metric_mean(basic_pose_orientation_error),
                "basic_pose_orientation_error_deg": metric_mean(basic_pose_orientation_error * rad_to_deg),
                "basic_pose_position_gate_weight": metric_mean(basic_pose_position_gate_weight),
                "basic_pose_position_gate_temperature_m": metric_mean(basic_pose_position_gate_temperature),
                "basic_pose_position_gate_loose_rate": metric_mean(basic_pose_position_gate_loose.float()),
                "basic_pose_position_gate_mid_rate": metric_mean(basic_pose_position_gate_mid.float()),
                "basic_pose_position_gate_tight_rate": metric_mean(basic_pose_position_gate_tight.float()),
                "translation_progress_raw_m": metric_mean(self.translation_progress_raw),
                "translation_progress_m": metric_mean(self.translation_progress),
                "translation_remaining_m": metric_mean(self.translation_remaining),
                "translation_remaining_abs_m": metric_mean(torch.abs(self.translation_remaining)),
                "translation_orthogonal_drift_m": metric_mean(self.translation_orthogonal_drift),
                "pose6d_anchor_error_m": metric_mean(self.pose6d_anchor_error_m),
                "pose6d_anchor_success_threshold_m": metric_mean(self._active_pose6d_anchor_success_threshold()),
                "pose6d_anchor_success_rate": metric_mean(anchor_success.float()),
                "pose6d_anchor_target_x_m": metric_mean(self.target_anchor_pos_palm[:, 0]),
                "pose6d_anchor_target_y_m": metric_mean(self.target_anchor_pos_palm[:, 1]),
                "pose6d_anchor_target_z_m": metric_mean(self.target_anchor_pos_palm[:, 2]),
                "pose6d_anchor_resample_fail_rate": metric_mean(self.pose6d_anchor_resample_fail),
                "pose6d_anchor_workspace_reject_rate": metric_mean(self.pose6d_anchor_workspace_reject),
                "pose6d_anchor_forbidden_reject_rate": metric_mean(self.pose6d_anchor_forbidden_reject),
                "pose6d_anchor_segment_overlap_count": metric_mean(self.pose6d_anchor_segment_overlap_count),
                "pose6d_anchor_min_centerline_z_m": metric_mean(self.pose6d_anchor_min_centerline_z_m),
                "pose6d_anchor_dynamic_contact_count": metric_mean(self.pose6d_anchor_dynamic_contact_count),
                "pose6d_anchor_dynamic_y_local_m": metric_mean(self.pose6d_anchor_dynamic_y_local_m),
                "pose6d_feasible_target_resample_fail_rate": metric_mean(self.pose6d_feasible_target_resample_fail),
                "pose6d_feasible_target_workspace_reject_rate": metric_mean(self.pose6d_feasible_target_workspace_reject),
                "pose6d_feasible_target_forbidden_reject_rate": metric_mean(self.pose6d_feasible_target_forbidden_reject),
                "pose6d_feasible_target_proxy_hit_count": metric_mean(self.pose6d_feasible_target_proxy_hit_count),
                "pose6d_feasible_target_hit_proxy_y_m": metric_mean(self.pose6d_feasible_target_hit_proxy_y_m),
                "twist_near_success_rate": metric_mean(near_success.float()),
                "success_rate": metric_mean(success.float()),
                "new_success_rate": metric_mean(new_success.float()),
                "episode_success_rate": self.episode_success_rate,
                "target_refresh_rate": metric_mean(refresh_envs.float()),
                "target_refresh_count": (refresh_envs & metric_env_mask).float().sum(),
                "achieved_goals": metric_mean(self.achieved_goals.to(dtype=torch.float32)),
                "episode_achieved_goals": episode_achieved_goals,
                "post_success_reset_rate": metric_mean(self.reset_from_post_success_cache.float()),
                "target_chain_phase_rate": metric_mean(chain_target_mask.float()),
                "first_target_success_rate": _masked_rate(success, first_target_metric_mask),
                "first_target_new_success_rate": _masked_rate(new_success, first_target_metric_mask),
                "first_target_near_success_rate": _masked_rate(near_success, first_target_metric_mask),
                "first_target_drop_rate": _masked_rate(drop, first_target_metric_mask),
                "chain_target_success_rate": _masked_rate(success, chain_target_metric_mask),
                "chain_target_new_success_rate": _masked_rate(new_success, chain_target_metric_mask),
                "chain_target_near_success_rate": _masked_rate(near_success, chain_target_metric_mask),
                "chain_target_drop_rate": _masked_rate(drop, chain_target_metric_mask),
                "episode_done_count": done_count,
                "drop_rate": metric_mean(drop.float()),
                "timeout_rate": metric_mean(self.reset_time_outs.float()),
                "reset_rate": metric_mean((drop | self.reset_time_outs).float()),
                "reward_variant_basic_pose_error": torch.tensor(
                    float(reward_variant in {"basic", "basic_pose", "basic_pose_error", "pose_error"}),
                    device=self.device,
                ),
                "reward_twist_progress_scaled": metric_mean(reward_twist_progress_scaled),
                "reward_twist_remaining_scaled": metric_mean(reward_twist_remaining_scaled),
                "reward_twist_position_drift_penalty_scaled": metric_mean(reward_twist_position_drift_penalty_scaled),
                "reward_twist_swing_drift_penalty_scaled": metric_mean(reward_twist_swing_drift_penalty_scaled),
                "reward_twist_swing_barrier_penalty_scaled": metric_mean(reward_twist_swing_barrier_penalty_scaled),
                "reward_twist_overshoot_penalty_scaled": metric_mean(reward_twist_overshoot_penalty_scaled),
                "reward_translation_progress_scaled": metric_mean(reward_translation_progress_scaled),
                "reward_translation_remaining_scaled": metric_mean(reward_translation_remaining_scaled),
                "reward_translation_orthogonal_drift_penalty_scaled": (
                    metric_mean(reward_translation_orthogonal_drift_penalty_scaled)
                ),
                "reward_translation_overshoot_penalty_scaled": metric_mean(reward_translation_overshoot_penalty_scaled),
                "reward_basic_pose_position_scaled": metric_mean(reward_basic_pose_position_scaled),
                "reward_basic_pose_orientation_scaled": metric_mean(reward_basic_pose_orientation_scaled),
                "reward_basic_pose_position_error_penalty_scaled": (
                    metric_mean(reward_basic_pose_position_error_penalty_scaled)
                ),
                "reward_basic_pose_orientation_error_penalty_scaled": (
                    metric_mean(reward_basic_pose_orientation_error_penalty_scaled)
                ),
                "reward_near_success_scaled": metric_mean(reward_near_success_scaled),
                "reward_near_success_steps_scaled": metric_mean(reward_near_success_steps_scaled),
                "reward_success_scaled": metric_mean(reward_success_scaled),
                "reward_success_hold_scaled": metric_mean(reward_success_hold_scaled),
                "reward_drop_scaled": metric_mean(reward_drop_scaled),
                "reward_action_penalty_scaled": metric_mean(reward_action_penalty_scaled),
                "reward_joint_vel_penalty_scaled": metric_mean(reward_joint_vel_penalty_scaled),
                "reward_torque_penalty_scaled": metric_mean(reward_torque_penalty_scaled),
                "reward_work_penalty_scaled": metric_mean(reward_work_penalty_scaled),
                "reward_excess_load_penalty": metric_mean(excess_load_penalty),
                "reward_excess_load_penalty_scaled": metric_mean(reward_excess_load_penalty_scaled),
                "excess_load_contact_joint_rate": metric_mean(excess_load_contact_joint_rate),
                "excess_load_exceed_joint_rate": metric_mean(excess_load_exceed_joint_rate),
                "excess_load_contact_ratio_mean": metric_mean(excess_load_contact_ratio_mean),
                "excess_load_contact_ratio_max": metric_max(excess_load_contact_ratio_max),
                "grasp_wrench_quality": metric_mean(grasp_wrench_quality),
                "grasp_wrench_force_margin_min": metric_mean(grasp_wrench_force_margin_min),
                "grasp_wrench_torque_margin_min": metric_mean(grasp_wrench_torque_margin_min),
                "grasp_wrench_target_quality": metric_mean(self.grasp_wrench_target_quality),
                "grasp_wrench_deficit": metric_mean(grasp_wrench_deficit),
                "grasp_wrench_below_target_rate": metric_mean(grasp_wrench_below_target),
                "grasp_wrench_contact_confidence_count": metric_mean(
                    grasp_wrench_contact_confidence_count
                ),
                "reward_grasp_wrench_maintenance_scaled": metric_mean(
                    reward_grasp_wrench_maintenance_scaled
                ),
                "dexndm_torque_penalty": metric_mean(torque_penalty),
                "dexndm_work_penalty": metric_mean(work_penalty),
                "dexndm_twist_torque_work_penalty_enabled": torch.tensor(
                    float(use_torque_work_penalty), device=self.device
                ),
                "target_delta_sample_frontier_rate": metric_mean((self.target_delta_sample_mode == 1).float()),
                "target_delta_sample_replay_rate": metric_mean((self.target_delta_sample_mode == 2).float()),
                "target_delta_sample_anchor_rate": metric_mean((self.target_delta_sample_mode == 3).float()),
                "target_delta_sample_uniform_rate": metric_mean((self.target_delta_sample_mode == 4).float()),
                "translation_delta_sample_frontier_rate": metric_mean((self.translation_delta_sample_mode == 1).float()),
                "translation_delta_sample_replay_rate": metric_mean((self.translation_delta_sample_mode == 2).float()),
                "translation_delta_sample_anchor_rate": metric_mean((self.translation_delta_sample_mode == 3).float()),
                "translation_delta_sample_uniform_rate": metric_mean((self.translation_delta_sample_mode == 4).float()),
            }
        )
        if bool(
            getattr(self, "object_scale_randomization_enabled", False)
            or getattr(self, "object_nearcube_geometry_randomization_enabled", False)
            or getattr(self, "object_mass_randomization_enabled", False)
            or getattr(self, "object_com_randomization_enabled", False)
            or getattr(self, "object_inertia_from_compound_cuboids_enabled", False)
            or getattr(self, "object_friction_randomization_enabled", False)
            or getattr(self, "random_external_force_enabled", False)
        ):
            self.extras["log"].update(
                {
                    "object_dr_scale_mean": metric_mean(self.object_scale_values),
                    "object_dr_scale_min": metric_min(self.object_scale_values),
                    "object_dr_scale_max": metric_max(self.object_scale_values),
                    "object_dr_dim_x_m_mean": metric_mean(self.active_object_dims_m[:, 0]),
                    "object_dr_dim_y_m_mean": metric_mean(self.active_object_dims_m[:, 1]),
                    "object_dr_dim_z_m_mean": metric_mean(self.active_object_dims_m[:, 2]),
                    "object_dr_aspect_ratio_mean": metric_mean(
                        self.active_object_dims_m.amax(dim=-1) / self.active_object_dims_m.amin(dim=-1)
                    ),
                    "object_dr_mass_kg_mean": metric_mean(self.object_randomized_mass_kg),
                    "object_dr_inertia_xx_kg_m2_mean": metric_mean(
                        self.object_randomized_inertia_diag_kg_m2[:, 0]
                    ),
                    "object_dr_inertia_yy_kg_m2_mean": metric_mean(
                        self.object_randomized_inertia_diag_kg_m2[:, 1]
                    ),
                    "object_dr_inertia_zz_kg_m2_mean": metric_mean(
                        self.object_randomized_inertia_diag_kg_m2[:, 2]
                    ),
                    "object_dr_compound_inertia_multiplier_mean": metric_mean(
                        self.object_inertia_compound_multiplier
                    ),
                    "object_dr_compound_inertia_multiplier_min": metric_min(
                        self.object_inertia_compound_multiplier
                    ),
                    "object_dr_compound_inertia_multiplier_max": metric_max(
                        self.object_inertia_compound_multiplier
                    ),
                    "object_dr_com_offset_abs_m": torch.linalg.norm(
                        self.object_randomized_com_offset_b,
                        dim=-1,
                    )[metric_env_mask].mean(),
                    "object_dr_base_com_x_m_mean": metric_mean(self.object_randomized_base_com_b[:, 0]),
                    "object_dr_base_com_y_m_mean": metric_mean(self.object_randomized_base_com_b[:, 1]),
                    "object_dr_base_com_z_m_mean": metric_mean(self.object_randomized_base_com_b[:, 2]),
                    "object_dr_friction_scale_mean": metric_mean(self.object_friction_scale),
                    "object_dr_external_force_active_rate": metric_mean(self.object_external_force_active),
                    "object_dr_external_force_norm_n": metric_mean(
                        torch.linalg.norm(self.object_external_force_w, dim=-1)
                    ),
                    "object_dr_external_force_prob_mean": metric_mean(self.object_external_force_prob),
                    "object_dr_external_force_reference_prob_mean": metric_mean(
                        self.object_external_force_reference_prob
                    ),
                    "object_dr_external_torque_active_rate": metric_mean(self.object_external_torque_active),
                    "object_dr_external_torque_norm_nm": metric_mean(
                        torch.linalg.norm(self.object_external_torque_w, dim=-1)
                    ),
                    "object_dr_external_target_angular_acceleration_norm_rad_s2": metric_mean(
                        torch.linalg.norm(
                            self.object_external_target_angular_acceleration_w,
                            dim=-1,
                        )
                    ),
                    "object_dr_external_torque_prob_mean": metric_mean(self.object_external_torque_prob),
                    "object_dr_external_torque_reference_prob_mean": metric_mean(
                        self.object_external_torque_reference_prob
                    ),
                }
            )
            if hasattr(self, "object_randomized_density_kg_m3"):
                self.extras["log"]["object_dr_density_kg_m3_mean"] = metric_mean(
                    self.object_randomized_density_kg_m3
                )
            if self.object_nearcube_geometry_randomization_enabled:
                branch_histogram = torch.bincount(self.object_geometry_branch_id, minlength=3).to(dtype=torch.float32)
                branch_fractions = branch_histogram / float(self.num_envs)
                self.extras["log"].update(
                    {
                        f"object_dr_geometry_branch_{branch_id}_fraction": branch_fractions[branch_id]
                        for branch_id in range(3)
                    }
                )
            if self.geometry_conditioned_stable_cache_enabled:
                anchor_count = int(self.geometry_conditioned_cache_row_counts.numel())
                anchor_histogram = torch.bincount(
                    self.geometry_conditioned_cache_anchor_ids_by_env,
                    minlength=anchor_count,
                ).to(dtype=torch.float32)
                anchor_fractions = anchor_histogram / float(self.num_envs)
                self.extras["log"].update(
                    {
                        "object_dr_geometry_cache_anchor_mean": metric_mean(
                            self.geometry_conditioned_cache_anchor_ids_by_env.float()
                        ),
                        **{
                            f"object_dr_geometry_cache_anchor_{anchor_id}_fraction": anchor_fractions[anchor_id]
                            for anchor_id in range(anchor_count)
                        },
                    }
                )
            if self.scale_conditioned_stable_cache_enabled:
                bucket_count = int(self.scale_conditioned_cache_row_counts.numel())
                bucket_histogram = torch.bincount(
                    self.scale_conditioned_cache_bucket_ids_by_env,
                    minlength=bucket_count,
                ).to(dtype=torch.float32)
                bucket_fractions = bucket_histogram / float(self.num_envs)
                self.extras["log"].update(
                    {
                        "object_dr_scale_cache_bucket_mean": metric_mean(
                            self.scale_conditioned_cache_bucket_ids_by_env.float()
                        ),
                        **{
                            f"object_dr_scale_cache_bucket_{bucket_id}_fraction": bucket_fractions[bucket_id]
                            for bucket_id in range(bucket_count)
                        },
                    }
                )
        if self.object_mass_bucket_metrics_enabled:
            edges = self.object_mass_bucket_edges_kg
            bucket_ids = mass_bucket_ids(self.object_randomized_mass_kg, edges)
            applied_torque_action = sim_to_action_order(
                self.hand.data.applied_torque,
                self.mapping.action_joint_indices,
            )
            commanded_torque_action = self.dexndm_commanded_torque_action
            raw_commanded_torque_action = self.dexndm_raw_commanded_torque_action
            commanded_norm = torch.linalg.norm(commanded_torque_action, dim=-1)
            applied_norm = torch.linalg.norm(applied_torque_action, dim=-1)
            applied_command_gap = torch.linalg.norm(
                applied_torque_action - commanded_torque_action, dim=-1
            )
            clamp_rate = (
                torch.abs(raw_commanded_torque_action - commanded_torque_action) > 1.0e-7
            ).to(dtype=torch.float32).mean(dim=-1)
            self.extras["log"].update(
                {
                    "mass_sampling_nominal_branch_fraction": metric_mean(
                        (self.object_mass_sampling_branch_id == 0).float()
                    ),
                    "mass_sampling_wide_log_branch_fraction": metric_mean(
                        (self.object_mass_sampling_branch_id == 1).float()
                    ),
                    "actuator_torque_limit_enabled": torch.tensor(
                        float(self.dexndm_torque_limit > 0.0), device=self.device
                    ),
                    "actuator_command_clamp_joint_rate": metric_mean(clamp_rate),
                    "actuator_commanded_torque_norm_nm": metric_mean(commanded_norm),
                    "actuator_applied_torque_norm_nm": metric_mean(applied_norm),
                    "actuator_applied_command_gap_norm_nm": metric_mean(applied_command_gap),
                    "actuator_commanded_torque_abs_max_nm": metric_max(
                        torch.abs(commanded_torque_action).amax(dim=-1)
                    ),
                    "actuator_applied_torque_abs_max_nm": metric_max(
                        torch.abs(applied_torque_action).amax(dim=-1)
                    ),
                }
            )
            for bucket_id, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                bucket_mask = bucket_ids == bucket_id
                bucket_metric_mask = bucket_mask & metric_env_mask
                count = bucket_metric_mask.float().sum()
                denominator = torch.clamp(count, min=1.0)
                label = f"{int(round(lo * 1000.0))}to{int(round(hi * 1000.0))}g"

                def bucket_mean(value: torch.Tensor) -> torch.Tensor:
                    scalar = value if value.ndim == 1 else value.reshape(value.shape[0], -1).mean(dim=-1)
                    return torch.where(
                        count > 0.0,
                        (scalar * bucket_metric_mask.to(dtype=scalar.dtype)).sum() / denominator,
                        torch.zeros((), device=self.device, dtype=scalar.dtype),
                    )

                self.extras["log"].update(
                    {
                        f"mass_bucket_{label}_count": count,
                        f"mass_bucket_{label}_present": (count > 0.0).to(dtype=torch.float32),
                        f"mass_bucket_{label}_fraction": count / torch.clamp(
                            metric_env_mask.float().sum(), min=1.0
                        ),
                        # Empty buckets deliberately emit finite zero rates and
                        # expose *_present=0, avoiding NaNs in long training logs.
                        f"mass_bucket_{label}_drop_rate": bucket_mean(
                            self.reset_terminated.float()
                        ),
                        f"mass_bucket_{label}_near_success_rate": bucket_mean(near_success.float()),
                        f"mass_bucket_{label}_success_rate": bucket_mean(success.float()),
                        f"mass_bucket_{label}_commanded_torque_norm_nm": bucket_mean(commanded_norm),
                        f"mass_bucket_{label}_applied_torque_norm_nm": bucket_mean(applied_norm),
                        f"mass_bucket_{label}_applied_command_gap_norm_nm": bucket_mean(
                            applied_command_gap
                        ),
                        f"mass_bucket_{label}_external_force_norm_n": bucket_mean(
                            torch.linalg.norm(self.object_external_force_w, dim=-1)
                        ),
                        f"mass_bucket_{label}_external_torque_norm_nm": bucket_mean(
                            torch.linalg.norm(self.object_external_torque_w, dim=-1)
                        ),
                    }
                )

        if bool(getattr(self, "sharpa_deployable_object_pose_obs_delay_noise_enabled", False)):
            self.extras["log"].update(
                {
                    "object_pose_obs_dr_delay_steps_mean": (
                        metric_mean(self.sharpa_deployable_object_pose_obs_delay_steps.to(dtype=torch.float32))
                    ),
                    "object_pose_obs_dr_delay_steps_max": (
                        metric_max(self.sharpa_deployable_object_pose_obs_delay_steps.to(dtype=torch.float32))
                    ),
                    "object_pose_obs_dr_dropout_rate": metric_mean(self.sharpa_deployable_object_pose_obs_dropout),
                    "object_pose_obs_dr_confidence_mean": metric_mean(
                        self.sharpa_deployable_object_pose_obs_confidence
                    ),
                    "object_pose_obs_dr_pos_noise_norm_m": (
                        metric_mean(self.sharpa_deployable_object_pose_obs_pos_noise_norm_m)
                    ),
                    "object_pose_obs_dr_rot_noise_abs_deg": (
                        metric_mean(self.sharpa_deployable_object_pose_obs_rot_noise_abs_deg)
                    ),
                }
            )
        if self._adaptive_twist_target_curriculum_enabled():
            shared_rotation_curriculum = self._primary_chain_shared_rotation_curriculum_enabled()
            self.extras["log"].update(
                {
                    "rotation_curriculum_shared": torch.tensor(
                        float(shared_rotation_curriculum), device=self.device
                    ),
                    "target_delta_frontier_success_rate": self.adaptive_frontier_success_rate,
                    "target_delta_frontier_done_count": self.adaptive_frontier_done_count,
                    "target_delta_frontier_done_total": self.adaptive_frontier_done_total,
                    "target_delta_curriculum_advances": self.adaptive_curriculum_advances,
                    "target_twist_curriculum_max_rad": self.adaptive_target_delta_max,
                    "target_twist_curriculum_max_deg": self.adaptive_target_delta_max * rad_to_deg,
                    "target_twist_soft_promotions": self.adaptive_soft_promotion_counts[0],
                    "target_twist_level_dwell_iterations": self._adaptive_soft_promotion_dwell_iterations("twist"),
                }
            )
            if shared_rotation_curriculum:
                self.extras["log"].update(
                    {
                        "shared_rotation_frontier_success_rate": self.adaptive_frontier_success_rate,
                        "shared_rotation_frontier_done_count": self.adaptive_frontier_done_count,
                        "shared_rotation_frontier_done_total": self.adaptive_frontier_done_total,
                        "shared_rotation_curriculum_advances": self.adaptive_curriculum_advances,
                        "shared_rotation_curriculum_max_rad": self.adaptive_target_delta_max,
                        "shared_rotation_curriculum_max_deg": self.adaptive_target_delta_max * rad_to_deg,
                        # Backward-compatible aliases for dashboards that still
                        # display the historical Chain Rotation frontier.
                        "chain_target_twist_curriculum_max_rad": self.adaptive_chain_target_delta_max,
                        "chain_target_twist_curriculum_max_deg": (
                            self.adaptive_chain_target_delta_max * rad_to_deg
                        ),
                        "twist_chain_curriculum_gap_rad": (
                            self.adaptive_target_delta_max - self.adaptive_chain_target_delta_max
                        ),
                        "twist_chain_curriculum_gap_deg": (
                            (self.adaptive_target_delta_max - self.adaptive_chain_target_delta_max) * rad_to_deg
                        ),
                        "shared_rotation_soft_promotions": self.adaptive_soft_promotion_counts[0],
                        "shared_rotation_level_dwell_iterations": (
                            self._adaptive_soft_promotion_dwell_iterations("twist")
                        ),
                    }
                )
            elif self._chain_post_success_target_curriculum_enabled():
                self.extras["log"].update(
                    {
                        "chain_target_delta_frontier_success_rate": self.adaptive_chain_frontier_success_rate,
                        "chain_target_delta_frontier_done_count": self.adaptive_chain_frontier_done_count,
                        "chain_target_delta_frontier_done_total": self.adaptive_chain_frontier_done_total,
                        "chain_target_delta_curriculum_advances": self.adaptive_chain_curriculum_advances,
                        "chain_target_twist_curriculum_max_rad": self.adaptive_chain_target_delta_max,
                        "chain_target_twist_curriculum_max_deg": self.adaptive_chain_target_delta_max * rad_to_deg,
                        "chain_target_twist_soft_promotions": self.adaptive_soft_promotion_counts[1],
                        "chain_target_twist_level_dwell_iterations": (
                            self._adaptive_soft_promotion_dwell_iterations("chain_twist")
                        ),
                        "twist_chain_curriculum_gap_rad": (
                            self.adaptive_target_delta_max - self.adaptive_chain_target_delta_max
                        ),
                        "twist_chain_curriculum_gap_deg": (
                            (self.adaptive_target_delta_max - self.adaptive_chain_target_delta_max) * rad_to_deg
                        ),
                        "twist_chain_curriculum_sync_limit_count": (
                            self.adaptive_twist_chain_sync_limited_count
                        ),
                    }
                )
        if self._adaptive_pose6d_translation_curriculum_enabled():
            self.extras["log"].update(
                {
                    "translation_delta_frontier_success_rate": self.adaptive_translation_frontier_success_rate,
                    "translation_delta_frontier_done_count": self.adaptive_translation_frontier_done_count,
                    "translation_delta_frontier_done_total": self.adaptive_translation_frontier_done_total,
                    "translation_delta_curriculum_advances": self.adaptive_translation_curriculum_advances,
                    "translation_delta_curriculum_max_m": self.adaptive_translation_delta_max,
                    "translation_delta_soft_promotions": self.adaptive_soft_promotion_counts[2],
                    "translation_delta_level_dwell_iterations": (
                        self._adaptive_soft_promotion_dwell_iterations("translation")
                    ),
                }
            )
            if self._chain_pose6d_translation_curriculum_enabled():
                self.extras["log"].update(
                    {
                        "chain_translation_delta_frontier_success_rate": (
                            self.adaptive_chain_translation_frontier_success_rate
                        ),
                        "chain_translation_delta_frontier_done_count": (
                            self.adaptive_chain_translation_frontier_done_count
                        ),
                        "chain_translation_delta_frontier_done_total": (
                            self.adaptive_chain_translation_frontier_done_total
                        ),
                        "chain_translation_delta_curriculum_advances": (
                            self.adaptive_chain_translation_curriculum_advances
                        ),
                        "chain_translation_delta_curriculum_max_m": self.adaptive_chain_translation_delta_max,
                        "translation_chain_curriculum_progress_gap": (
                            self._first_chain_translation_progress_gap()
                        ),
                        "translation_chain_curriculum_sync_limit_count": (
                            self.adaptive_translation_chain_sync_limited_count
                        ),
                    }
                )
        if self._multiobject_per_object_curriculum_enabled():
            per_object_curriculum_log = {}
            for size_id, object_id in enumerate(self.multisize_object_ids):
                tag = object_id.replace("/", "_")
                if self._adaptive_twist_target_curriculum_enabled():
                    per_object_curriculum_log.update(
                        {
                            f"rotation_curriculum_shared_{tag}": torch.tensor(
                                float(self._primary_chain_shared_rotation_curriculum_enabled()),
                                device=self.device,
                            ),
                            f"target_delta_frontier_success_rate_{tag}": (
                                self.adaptive_frontier_success_rate_by_object[size_id]
                            ),
                            f"target_delta_frontier_done_count_{tag}": (
                                self.adaptive_frontier_done_count_by_object[size_id]
                            ),
                            f"target_delta_frontier_done_total_{tag}": (
                                self.adaptive_frontier_done_total_by_object[size_id]
                            ),
                            f"target_delta_curriculum_advances_{tag}": (
                                self.adaptive_curriculum_advances_by_object[size_id]
                            ),
                            f"target_twist_curriculum_max_rad_{tag}": (
                                self.adaptive_target_delta_max_by_object[size_id]
                            ),
                            f"target_twist_curriculum_max_deg_{tag}": (
                                self.adaptive_target_delta_max_by_object[size_id] * rad_to_deg
                            ),
                            f"target_twist_soft_promotions_{tag}": (
                                self.adaptive_soft_promotion_counts_by_object[size_id, 0]
                            ),
                            f"target_twist_level_dwell_iterations_{tag}": (
                                self._adaptive_soft_promotion_dwell_iterations("twist", size_id)
                            ),
                        }
                    )
                    if self._primary_chain_shared_rotation_curriculum_enabled():
                        per_object_curriculum_log.update(
                            {
                                f"shared_rotation_frontier_success_rate_{tag}": (
                                    self.adaptive_frontier_success_rate_by_object[size_id]
                                ),
                                f"shared_rotation_frontier_done_count_{tag}": (
                                    self.adaptive_frontier_done_count_by_object[size_id]
                                ),
                                f"shared_rotation_frontier_done_total_{tag}": (
                                    self.adaptive_frontier_done_total_by_object[size_id]
                                ),
                                f"shared_rotation_curriculum_advances_{tag}": (
                                    self.adaptive_curriculum_advances_by_object[size_id]
                                ),
                                f"shared_rotation_curriculum_max_rad_{tag}": (
                                    self.adaptive_target_delta_max_by_object[size_id]
                                ),
                                f"shared_rotation_curriculum_max_deg_{tag}": (
                                    self.adaptive_target_delta_max_by_object[size_id] * rad_to_deg
                                ),
                                f"chain_target_twist_curriculum_max_rad_{tag}": (
                                    self.adaptive_chain_target_delta_max_by_object[size_id]
                                ),
                                f"chain_target_twist_curriculum_max_deg_{tag}": (
                                    self.adaptive_chain_target_delta_max_by_object[size_id] * rad_to_deg
                                ),
                                f"twist_chain_curriculum_gap_rad_{tag}": (
                                    self.adaptive_target_delta_max_by_object[size_id]
                                    - self.adaptive_chain_target_delta_max_by_object[size_id]
                                ),
                                f"twist_chain_curriculum_gap_deg_{tag}": (
                                    (
                                        self.adaptive_target_delta_max_by_object[size_id]
                                        - self.adaptive_chain_target_delta_max_by_object[size_id]
                                    )
                                    * rad_to_deg
                                ),
                                f"shared_rotation_soft_promotions_{tag}": (
                                    self.adaptive_soft_promotion_counts_by_object[size_id, 0]
                                ),
                                f"shared_rotation_level_dwell_iterations_{tag}": (
                                    self._adaptive_soft_promotion_dwell_iterations("twist", size_id)
                                ),
                            }
                        )
                    elif self._chain_post_success_target_curriculum_enabled():
                        per_object_curriculum_log.update(
                            {
                                f"chain_target_delta_frontier_success_rate_{tag}": (
                                    self.adaptive_chain_frontier_success_rate_by_object[size_id]
                                ),
                                f"chain_target_delta_frontier_done_count_{tag}": (
                                    self.adaptive_chain_frontier_done_count_by_object[size_id]
                                ),
                                f"chain_target_delta_frontier_done_total_{tag}": (
                                    self.adaptive_chain_frontier_done_total_by_object[size_id]
                                ),
                                f"chain_target_delta_curriculum_advances_{tag}": (
                                    self.adaptive_chain_curriculum_advances_by_object[size_id]
                                ),
                                f"chain_target_twist_curriculum_max_rad_{tag}": (
                                    self.adaptive_chain_target_delta_max_by_object[size_id]
                                ),
                                f"chain_target_twist_curriculum_max_deg_{tag}": (
                                    self.adaptive_chain_target_delta_max_by_object[size_id] * rad_to_deg
                                ),
                                f"chain_target_twist_soft_promotions_{tag}": (
                                    self.adaptive_soft_promotion_counts_by_object[size_id, 1]
                                ),
                                f"chain_target_twist_level_dwell_iterations_{tag}": (
                                    self._adaptive_soft_promotion_dwell_iterations("chain_twist", size_id)
                                ),
                                f"twist_chain_curriculum_gap_rad_{tag}": (
                                    self.adaptive_target_delta_max_by_object[size_id]
                                    - self.adaptive_chain_target_delta_max_by_object[size_id]
                                ),
                                f"twist_chain_curriculum_gap_deg_{tag}": (
                                    (
                                        self.adaptive_target_delta_max_by_object[size_id]
                                        - self.adaptive_chain_target_delta_max_by_object[size_id]
                                    )
                                    * rad_to_deg
                                ),
                                f"twist_chain_curriculum_sync_limit_count_{tag}": (
                                    self.adaptive_twist_chain_sync_limited_count_by_object[size_id]
                                ),
                            }
                        )
                if self._adaptive_pose6d_translation_curriculum_enabled():
                    per_object_curriculum_log.update(
                        {
                            f"translation_delta_frontier_success_rate_{tag}": (
                                self.adaptive_translation_frontier_success_rate_by_object[size_id]
                            ),
                            f"translation_delta_frontier_done_count_{tag}": (
                                self.adaptive_translation_frontier_done_count_by_object[size_id]
                            ),
                            f"translation_delta_frontier_done_total_{tag}": (
                                self.adaptive_translation_frontier_done_total_by_object[size_id]
                            ),
                            f"translation_delta_curriculum_advances_{tag}": (
                                self.adaptive_translation_curriculum_advances_by_object[size_id]
                            ),
                            f"translation_delta_curriculum_max_m_{tag}": (
                                self.adaptive_translation_delta_max_by_object[size_id]
                            ),
                            f"translation_delta_soft_promotions_{tag}": (
                                self.adaptive_soft_promotion_counts_by_object[size_id, 2]
                            ),
                            f"translation_delta_level_dwell_iterations_{tag}": (
                                self._adaptive_soft_promotion_dwell_iterations("translation", size_id)
                            ),
                        }
                    )
                    if self._chain_pose6d_translation_curriculum_enabled():
                        per_object_curriculum_log.update(
                            {
                                f"chain_translation_delta_frontier_success_rate_{tag}": (
                                    self.adaptive_chain_translation_frontier_success_rate_by_object[size_id]
                                ),
                                f"chain_translation_delta_frontier_done_count_{tag}": (
                                    self.adaptive_chain_translation_frontier_done_count_by_object[size_id]
                                ),
                                f"chain_translation_delta_frontier_done_total_{tag}": (
                                    self.adaptive_chain_translation_frontier_done_total_by_object[size_id]
                                ),
                                f"chain_translation_delta_curriculum_advances_{tag}": (
                                    self.adaptive_chain_translation_curriculum_advances_by_object[size_id]
                                ),
                                f"chain_translation_delta_curriculum_max_m_{tag}": (
                                    self.adaptive_chain_translation_delta_max_by_object[size_id]
                                ),
                                f"translation_chain_curriculum_progress_gap_{tag}": (
                                    torch.tensor(
                                        self._curriculum_progress(
                                            float(self.adaptive_translation_delta_max_by_object[size_id].item()),
                                            float(getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002)),
                                            float(getattr(self.cfg, "pose6d_translation_curriculum_end_m", 0.002)),
                                        )
                                        - self._curriculum_progress(
                                            float(self.adaptive_chain_translation_delta_max_by_object[size_id].item()),
                                            float(
                                                getattr(
                                                    self.cfg,
                                                    "chain_pose6d_translation_curriculum_start_m",
                                                    getattr(self.cfg, "pose6d_translation_curriculum_start_m", 0.002),
                                                )
                                            ),
                                            float(
                                                getattr(
                                                    self.cfg,
                                                    "chain_pose6d_translation_curriculum_end_m",
                                                    getattr(self.cfg, "pose6d_translation_curriculum_end_m", 0.002),
                                                )
                                            ),
                                        ),
                                        device=self.device,
                                    )
                                ),
                                f"translation_chain_curriculum_sync_limit_count_{tag}": (
                                    self.adaptive_translation_chain_sync_limited_count_by_object[size_id]
                                ),
                            }
                        )
            self.extras["log"].update(per_object_curriculum_log)
        if self.multiobject_enabled:
            if self.primitive_multi_shape_enabled:
                log_prefix = "primitive"
            elif self.multisize_cube_enabled:
                log_prefix = "multisize"
            else:
                log_prefix = "multicuboid"
            multisize_log = {}
            if self.multisize_cube_enabled:
                multisize_log["multisize_size_norm_mean"] = metric_mean(self.active_cube_size_norm)
            elif self.multicuboid_enabled:
                multisize_log.update(
                    {
                        "multicuboid_dim_norm_x_mean": metric_mean(self.active_object_dim_norm[:, 0]),
                        "multicuboid_dim_norm_y_mean": metric_mean(self.active_object_dim_norm[:, 1]),
                        "multicuboid_dim_norm_z_mean": metric_mean(self.active_object_dim_norm[:, 2]),
                    }
                )
            else:
                multisize_log.update(
                    {
                        "primitive_dim_norm_x_mean": metric_mean(self.active_object_dim_norm[:, 0]),
                        "primitive_dim_norm_y_mean": metric_mean(self.active_object_dim_norm[:, 1]),
                        "primitive_dim_norm_z_mean": metric_mean(self.active_object_dim_norm[:, 2]),
                        "primitive_is_cylinder_rate": metric_mean(self.active_object_is_cylinder),
                    }
                )
                if self.primitive_geometry_latent_obs_enabled:
                    multisize_log["primitive_geometry_latent_abs_mean"] = (
                        metric_mean(self.active_object_geometry_latent.abs())
                    )
                for shape_id, shape_name in enumerate(self.primitive_shape_names):
                    multisize_log[f"primitive_shape_{shape_name}_rate"] = metric_mean(
                        (self.active_object_shape_id == shape_id).float()
                    )
            reset_threshold = max(float(self.cfg.dexndm_omni_wrist_reset_pos_dist_m), 1.0e-6)
            reset_dist_norm = torch.linalg.norm(cube_pos_w - self.dexndm_init_cube_pos_w, dim=-1) / reset_threshold
            if self.primitive_multi_shape_enabled:
                for shape_id, shape_name in enumerate(self.primitive_shape_names):
                    shape_mask = (self.active_object_shape_id == shape_id) & metric_env_mask
                    denom = torch.clamp(shape_mask.float().sum(), min=1.0)
                    prefix = f"{log_prefix}_{shape_name}"
                    multisize_log.update(
                        {
                            f"{prefix}_env_rate": metric_mean(shape_mask.float()),
                            f"{prefix}_success_rate": (success.float() * shape_mask.float()).sum() / denom,
                            f"{prefix}_new_success_rate": (new_success.float() * shape_mask.float()).sum() / denom,
                            f"{prefix}_near_success_rate": (near_success.float() * shape_mask.float()).sum() / denom,
                            f"{prefix}_drop_rate": (drop.float() * shape_mask.float()).sum() / denom,
                            f"{prefix}_anchor_error_m": (self.pose6d_anchor_error_m * shape_mask.float()).sum() / denom,
                            f"{prefix}_reset_dist_norm": (reset_dist_norm * shape_mask.float()).sum() / denom,
                        }
                    )
            for size_id, object_id in enumerate(self.multisize_object_ids):
                mask = (self.active_cube_size_id == size_id) & metric_env_mask
                denom = torch.clamp(mask.float().sum(), min=1.0)
                prefix = f"{log_prefix}_{object_id}"
                multisize_log.update(
                    {
                        f"{prefix}_env_rate": metric_mean(mask.float()),
                        f"{prefix}_success_rate": (success.float() * mask.float()).sum() / denom,
                        f"{prefix}_new_success_rate": (new_success.float() * mask.float()).sum() / denom,
                        f"{prefix}_near_success_rate": (near_success.float() * mask.float()).sum() / denom,
                        f"{prefix}_drop_rate": (drop.float() * mask.float()).sum() / denom,
                        f"{prefix}_anchor_error_m": (self.pose6d_anchor_error_m * mask.float()).sum() / denom,
                        f"{prefix}_reset_dist_norm": (reset_dist_norm * mask.float()).sum() / denom,
                        f"{prefix}_grasp_wrench_quality": (
                            grasp_wrench_quality * mask.float()
                        ).sum() / denom,
                        f"{prefix}_grasp_wrench_force_margin_min": (
                            grasp_wrench_force_margin_min * mask.float()
                        ).sum() / denom,
                        f"{prefix}_grasp_wrench_torque_margin_min": (
                            grasp_wrench_torque_margin_min * mask.float()
                        ).sum() / denom,
                        f"{prefix}_grasp_wrench_below_target_rate": (
                            grasp_wrench_below_target * mask.float()
                        ).sum() / denom,
                        f"{prefix}_grasp_wrench_contact_confidence_count": (
                            grasp_wrench_contact_confidence_count * mask.float()
                        ).sum() / denom,
                        f"{prefix}_reward_grasp_wrench_maintenance_scaled": (
                            reward_grasp_wrench_maintenance_scaled * mask.float()
                        ).sum() / denom,
                    }
                )
            self.extras["log"].update(multisize_log)
        return reward


    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        q_action = sim_to_action_order(self.hand.data.joint_pos, self.mapping.action_joint_indices)
        if self.dexndm_grasp_wrench_maintenance_reward_enabled:
            self.grasp_wrench_reward_q_cmd_action_rad[:] = (
                self.action_targets.detach()
            )
            self.grasp_wrench_reward_q_action_rad[:] = q_action.detach()
            self.grasp_wrench_reward_qdot_action_rad_s[:] = sim_to_action_order(
                self.hand.data.joint_vel,
                self.mapping.action_joint_indices,
            ).detach()
            self.grasp_wrench_reward_raw_commanded_torque_action_nm[:] = (
                self.dexndm_raw_commanded_torque_action.detach()
            )
            self.grasp_wrench_reward_commanded_torque_action_nm[:] = (
                self.dexndm_commanded_torque_action.detach()
            )
            self.grasp_wrench_reward_applied_torque_action_nm[:] = sim_to_action_order(
                self.hand.data.applied_torque,
                self.mapping.action_joint_indices,
            ).detach()
        return self._get_dexndm_twist_rewards(q_action)

    def _reset_sharpa_deployable_observation_state(
        self,
        env_ids: torch.Tensor,
        q: torch.Tensor,
        q_cmd: torch.Tensor,
        object_pose_palm: torch.Tensor,
    ) -> None:
        q_scaled = scale_to_unit(self._apply_sharpa_deployable_joint_sensing_noise(q), self.action_lower, self.action_upper)
        q_cmd_scaled = scale_to_unit(q_cmd, self.action_lower, self.action_upper)
        tactile_obs = torch.zeros(
            (env_ids.numel(), self.sharpa_deployable_tactile_dim),
            device=self.device,
            dtype=torch.float32,
        )
        history_frame = torch.cat((q_scaled, q_cmd_scaled, tactile_obs), dim=-1).to(dtype=torch.float32)
        self.sharpa_deployable_obs_history[env_ids] = history_frame.unsqueeze(1).repeat(
            1,
            self.sharpa_deployable_history_len,
            1,
        )
        self.sharpa_deployable_obs_history_prev_step[env_ids] = int(self.common_step_counter)
        self._reset_sharpa_deployable_object_pose_observation_state(env_ids, object_pose_palm)

    def _randomize_dexndm_pd_gains_on_reset(self, env_ids: torch.Tensor) -> None:
        """Sample independent per-environment, per-joint P/D multipliers."""

        if not self.dexndm_torque_pd_gain_randomization_enabled:
            self.dexndm_controller_pgain_scale[env_ids] = 1.0
            self.dexndm_controller_dgain_scale[env_ids] = 1.0
            self.dexndm_controller_pgain_action[env_ids] = self.dexndm_controller_pgain_nominal_action[env_ids]
            self.dexndm_controller_dgain_action[env_ids] = self.dexndm_controller_dgain_nominal_action[env_ids]
            return

        p_lo, p_hi = self._validate_range_pair(
            getattr(self.cfg, "dexndm_torque_pd_p_scale_range", (0.5, 2.0)),
            "dexndm_torque_pd_p_scale_range",
            allow_zero=False,
        )
        d_lo, d_hi = self._validate_range_pair(
            getattr(self.cfg, "dexndm_torque_pd_d_scale_range", (0.5, 2.0)),
            "dexndm_torque_pd_d_scale_range",
            allow_zero=False,
        )
        shape = (int(env_ids.numel()), self.action_dim)
        p_scale = sample_independent_two_sided_scales(
            shape,
            low=p_lo,
            high=p_hi,
            device=self.device,
            dtype=torch.float32,
            generator=self.target_rng,
        )
        d_scale = sample_independent_two_sided_scales(
            shape,
            low=d_lo,
            high=d_hi,
            device=self.device,
            dtype=torch.float32,
            generator=self.target_rng,
        )
        self.dexndm_controller_pgain_scale[env_ids] = p_scale
        self.dexndm_controller_dgain_scale[env_ids] = d_scale
        self.dexndm_controller_pgain_action[env_ids] = (
            self.dexndm_controller_pgain_nominal_action[env_ids] * p_scale
        )
        self.dexndm_controller_dgain_action[env_ids] = (
            self.dexndm_controller_dgain_nominal_action[env_ids] * d_scale
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if self.tactile_enabled and hasattr(self, "tactile"):
            self.tactile.reset(env_ids, int(self.common_step_counter))
        self._randomize_dexndm_pd_gains_on_reset(env_ids)
        if self.contact_sensor_required:
            if self.multiobject_enabled:
                if self.primitive_single_asset_per_env_enabled:
                    self.cube_contact_sensor.reset(env_ids)
                else:
                    for sensor in self.multisize_cube_contact_sensors.values():
                        sensor.reset(env_ids)
            else:
                self.cube_contact_sensor.reset(env_ids)

        reset_node_ids = None
        reset_from_post_success_cache = torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        if self.multiobject_enabled:
            pass
            sample = None
            (
                size_ids,
                reset_grasp_indices,
                q,
                qdot,
                q_cmd,
                object_pose_palm,
                anchor_local_pos_m,
                reset_from_post_success_cache,
            ) = self._sample_multisize_reset_entries(int(env_ids.numel()), env_ids=env_ids)
            self._update_active_cube_size_buffers(env_ids, size_ids)
        else:
            fixed_reset_grasps = getattr(self, "fixed_reset_grasp_indices_by_env", None)
            fixed_reset_nodes = getattr(self, "fixed_reset_node_ids_by_env", None)
            if fixed_reset_grasps is not None:
                sample = None
                reset_grasp_indices = fixed_reset_grasps[env_ids].to(device=self.device, dtype=torch.long)
                if self.scale_conditioned_stable_cache_enabled or self.geometry_conditioned_stable_cache_enabled:
                    if bool(
                        ((reset_grasp_indices < 0) | (reset_grasp_indices >= self.num_stable_grasps)).any()
                    ):
                        raise IndexError(
                            f"Fixed reset grasp indices must be in [0, {self.num_stable_grasps}) "
                            "for the concatenated conditioned-cache bank."
                        )
                    if self.scale_conditioned_stable_cache_enabled:
                        offsets = self.scale_conditioned_cache_offsets
                        expected_group_ids = self.scale_conditioned_cache_bucket_ids_by_env[env_ids]
                        group_name = "continuous-scale cache bucket"
                    else:
                        offsets = self.geometry_conditioned_cache_offsets
                        expected_group_ids = self.geometry_conditioned_cache_anchor_ids_by_env[env_ids]
                        group_name = "geometry-cache anchor"
                    fixed_group_ids = torch.bucketize(reset_grasp_indices, offsets[1:-1], right=True)
                    if not bool(torch.equal(fixed_group_ids, expected_group_ids)):
                        mismatch = torch.nonzero(fixed_group_ids != expected_group_ids, as_tuple=False).squeeze(-1)
                        first = int(mismatch[0].item())
                        raise ValueError(
                            f"Fixed reset grasp index belongs to the wrong {group_name}: "
                            f"env={int(env_ids[first])}, grasp_index={int(reset_grasp_indices[first])}, "
                            f"grasp_group={int(fixed_group_ids[first])}, "
                            f"expected_group={int(expected_group_ids[first])}."
                        )
                if fixed_reset_nodes is not None:
                    reset_node_ids = fixed_reset_nodes[env_ids].to(device=self.device, dtype=torch.long)
                pass
            sample = None
            reset_grasp_indices = self._sample_reset_grasp_indices(int(env_ids.numel()), env_ids=env_ids)

            if (
                self.post_success_stable_cache is not None and self.post_success_reset_sample_prob > 0.0 and (fixed_reset_grasps is None)
            ):
                reset_from_post_success_cache = (
                    torch.rand(
                        env_ids.numel(),
                        device=self.device,
                        generator=self.target_rng,
                    )
                    < self.post_success_reset_sample_prob
                )

            if bool(reset_from_post_success_cache.any()):
                q = torch.empty((env_ids.numel(), self.action_dim), device=self.device, dtype=torch.float32)
                qdot = torch.empty_like(q)
                q_cmd = torch.empty_like(q)
                object_pose_palm = torch.empty((env_ids.numel(), 7), device=self.device, dtype=torch.float32)
                anchor_local_pos_m = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=torch.float32)

                normal_mask = ~reset_from_post_success_cache
                if bool(normal_mask.any()):
                    normal_indices = reset_grasp_indices[normal_mask]
                    q[normal_mask] = self.stable_cache["q"][normal_indices].to(dtype=torch.float32)
                    qdot[normal_mask] = self.stable_cache["qdot"][normal_indices].to(dtype=torch.float32)
                    q_cmd[normal_mask] = self.stable_cache["q_cmd"][normal_indices].to(dtype=torch.float32)
                    object_pose_palm[normal_mask] = self.stable_cache["object_pose_palm"][normal_indices].to(
                        dtype=torch.float32
                    )
                    anchor_local_pos_m[normal_mask] = self._cache_anchor_local_pos(self.stable_cache, normal_indices)

                post_mask = reset_from_post_success_cache
                post_count = int(post_mask.sum().item())
                post_indices = torch.randint(
                    self.num_post_success_stable_grasps,
                    (post_count,),
                    device=self.device,
                    generator=self.target_rng,
                    dtype=torch.long,
                )
                post_cache = self.post_success_stable_cache
                q[post_mask] = post_cache["q"][post_indices].to(dtype=torch.float32)
                qdot[post_mask] = post_cache["qdot"][post_indices].to(dtype=torch.float32)
                q_cmd[post_mask] = post_cache["q_cmd"][post_indices].to(dtype=torch.float32)
                object_pose_palm[post_mask] = post_cache["object_pose_palm"][post_indices].to(dtype=torch.float32)
                anchor_local_pos_m[post_mask] = self._cache_anchor_local_pos(post_cache, post_indices)
                reset_grasp_indices = reset_grasp_indices.clone()
                reset_grasp_indices[post_mask] = post_indices + self.num_stable_grasps
            else:
                q = self.stable_cache["q"][reset_grasp_indices].to(dtype=torch.float32)
                qdot = self.stable_cache["qdot"][reset_grasp_indices].to(dtype=torch.float32)
                q_cmd = self.stable_cache["q_cmd"][reset_grasp_indices].to(dtype=torch.float32)
                object_pose_palm = self.stable_cache["object_pose_palm"][reset_grasp_indices].to(dtype=torch.float32)
                anchor_local_pos_m = self._cache_anchor_local_pos(self.stable_cache, reset_grasp_indices)

        q_sim = action_to_sim_order(q, self.mapping.action_joint_indices, self.num_joints)
        qdot_sim = action_to_sim_order(qdot, self.mapping.action_joint_indices, self.num_joints)
        q_cmd_sim = action_to_sim_order(q_cmd, self.mapping.action_joint_indices, self.num_joints)

        self._reset_dexndm_wrist_orientation(env_ids)

        root_state = self.hand.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self._hand_root_pos_w(root_state.dtype, env_ids)
        root_state[:, 3:7] = self._hand_root_quat_w(root_state.dtype, env_ids)
        self.hand.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)
        self.hand.write_joint_state_to_sim(q_sim, qdot_sim, env_ids=env_ids)
        self.hand.set_joint_position_target(q_cmd_sim, env_ids=env_ids)

        object_pose_w = self._hand_to_world_pose(object_pose_palm, env_ids)
        if (
            not self.multiobject_enabled
            or self._multiobject_pool_size() == 1
            or self.primitive_single_asset_per_env_enabled
        ):
            self._randomize_single_object_mass_com_on_reset(env_ids)
        else:
            for size_id, object_id in enumerate(self.multisize_object_ids):
                object_env_ids = env_ids[self.active_cube_size_id[env_ids] == size_id]
                if object_env_ids.numel() > 0:
                    self._randomize_single_object_mass_com_on_reset(
                        object_env_ids, self.multisize_cubes[object_id]
                    )
        self._reset_random_external_wrench_state(env_ids)
        if self.multiobject_enabled:
            self._write_multisize_cube_reset(env_ids, object_pose_w)
        else:
            self.cube.write_root_pose_to_sim(object_pose_w, env_ids=env_ids)
            self.cube.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)
        self._compute_intermediate_values()

        self.action_targets[env_ids] = q_cmd
        self.prev_action_targets[env_ids] = q_cmd
        self.dexndm_prev_control_q_action[env_ids] = q
        self.dexndm_sim_joint_efforts[env_ids] = 0.0
        self.dexndm_commanded_torque_action[env_ids] = 0.0
        self.dexndm_raw_commanded_torque_action[env_ids] = 0.0
        if self.dexndm_torque_control:
            self.hand.set_joint_effort_target(self.dexndm_sim_joint_efforts[env_ids], env_ids=env_ids)
        q_scaled = scale_to_unit(q, self.action_lower, self.action_upper)
        q_cmd_scaled = scale_to_unit(q_cmd, self.action_lower, self.action_upper)
        history_frame = torch.cat((q_scaled, q_cmd_scaled), dim=-1)
        self.qtarget_history[env_ids] = history_frame.unsqueeze(1).repeat(1, self.qtarget_history_len, 1)
        self.qtarget_history_prev_step[env_ids] = int(self.common_step_counter)
        dexndm_history_frame = torch.cat((q_scaled, q), dim=-1)
        self.dexndm_obs_history[env_ids] = dexndm_history_frame.unsqueeze(1).repeat(
            1, self.dexndm_obs_history_len, 1
        )
        self.dexndm_obs_history_prev_step[env_ids] = -1
        self.hora_rotation_init_q[env_ids] = q
        self.hora_prev_cube_quat_w[env_ids] = normalize_quat_wxyz(object_pose_w[:, 3:7])
        self.dexndm_init_cube_pos_w[env_ids] = object_pose_w[:, :3]
        self.dexndm_prev_cube_pos_w[env_ids] = object_pose_w[:, :3]
        self.dexndm_prev_cube_quat_w[env_ids] = normalize_quat_wxyz(object_pose_w[:, 3:7])
        self.dexndm_prev_q_action[env_ids] = q
        self.dexndm_dof_vel_finite_diff[env_ids] = 0.0
        self.dexndm_torque_penalty[env_ids] = 0.0
        self.dexndm_work_penalty[env_ids] = 0.0
        self.grasp_wrench_quality[env_ids] = 0.0
        self.grasp_wrench_force_margin_min[env_ids] = 0.0
        self.grasp_wrench_torque_margin_min[env_ids] = 0.0
        self.grasp_wrench_directional_margins[env_ids] = 0.0
        self.grasp_wrench_contact_confidence[env_ids] = 0.0
        self.grasp_wrench_target_quality[env_ids] = 0.0
        self.grasp_wrench_deficit[env_ids] = 0.0
        self.grasp_wrench_warmup_quality_sum[env_ids] = 0.0
        self.grasp_wrench_warmup_count[env_ids] = 0
        self.dexndm_rotp[env_ids] = 0.0
        self._update_dexndm_rotp_coef_on_reset(env_ids)
        self.object_velocity_obs[env_ids] = 0.0
        self.object_velocity_obs_prev_pos_w[env_ids] = object_pose_w[:, :3]
        self.object_velocity_obs_prev_quat_w[env_ids] = normalize_quat_wxyz(object_pose_w[:, 3:7])
        self.object_velocity_obs_prev_step[env_ids] = int(self.common_step_counter)
        self._reset_sharpa_deployable_observation_state(env_ids, q, q_cmd, object_pose_palm)
        self.hora_axis_angvel[env_ids] = 0.0
        self.hora_orthogonal_angvel[env_ids] = 0.0
        self.hora_object_linvel[env_ids] = 0.0
        self.hora_cumulative_axis_rotation[env_ids] = 0.0
        self.hora_joint_pose_deviation[env_ids] = 0.0
        self._reset_dexndm_rot_axis(env_ids)
        self._reset_dexndm_guided_rotation(env_ids, object_pose_palm)
        self.source_anchor_local_pos_m[env_ids] = anchor_local_pos_m.to(dtype=torch.float32)
        self.target_anchor_local_pos_m[env_ids] = anchor_local_pos_m.to(dtype=torch.float32)
        self.source_anchor_pos_palm[env_ids] = self._anchor_pos_from_pose(
            object_pose_palm,
            anchor_local_pos_m.to(dtype=object_pose_palm.dtype),
        ).to(dtype=torch.float32)
        self.current_anchor_pos_palm[env_ids] = self.source_anchor_pos_palm[env_ids]
        self.target_anchor_pos_palm[env_ids] = self.source_anchor_pos_palm[env_ids]
        self.pose6d_anchor_error_m[env_ids] = 0.0
        self.pose6d_anchor_resample_fail[env_ids] = 0.0
        self.pose6d_anchor_workspace_reject[env_ids] = 0.0
        self.pose6d_anchor_forbidden_reject[env_ids] = 0.0
        self.pose6d_anchor_segment_overlap_count[env_ids] = 0.0
        self.pose6d_anchor_min_centerline_z_m[env_ids] = 0.0
        self.pose6d_anchor_dynamic_contact_count[env_ids] = 0.0
        self.pose6d_anchor_dynamic_y_local_m[env_ids] = anchor_local_pos_m[:, 1].to(dtype=torch.float32)
        self.pose6d_feasible_target_resample_fail[env_ids] = 0.0
        self.pose6d_feasible_target_workspace_reject[env_ids] = 0.0
        self.pose6d_feasible_target_forbidden_reject[env_ids] = 0.0
        self.pose6d_feasible_target_proxy_hit_count[env_ids] = 0.0
        self.pose6d_feasible_target_hit_proxy_y_m[env_ids] = 0.0
        self.sim_joint_targets[env_ids] = q_cmd_sim
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.reset_grasp_indices[env_ids] = reset_grasp_indices
        self.reset_from_post_success_cache[env_ids] = reset_from_post_success_cache
        self.achieved_goals[env_ids] = 0
        self.target_chain_phase[env_ids] = 0
        self.near_success_steps[env_ids] = 0
        self.target_reached[env_ids] = False
        self.target_refresh_fallback_level[env_ids] = 0
        self.local_random_target_dist[env_ids] = 0.0
        self.local_random_target_translation[env_ids] = 0.0
        self.local_random_target_rotation[env_ids] = 0.0
        self.local_random_resample_fail[env_ids] = 0.0
        self.local_random_workspace_reject[env_ids] = 0.0
        self.mixed_target_is_scripted[env_ids] = False
        self.scripted_rotation_target_index[env_ids] = 0

        if reset_node_ids is None:
            self.reset_node_ids[env_ids] = -1
        else:
            self.reset_node_ids[env_ids] = reset_node_ids
        self.graph_source_grasp_indices[env_ids] = reset_grasp_indices
        self._set_hand_frame_twist_targets(env_ids, object_pose_palm)

        self._reset_directional_rotation_progress(env_ids, object_pose_palm)
        reset_pose_error = self._object_pose_distance(object_pose_palm, self.target_pose_palm[env_ids], env_ids).detach()
        self.prev_pose_error[env_ids] = reset_pose_error
        self.best_pose_error[env_ids] = reset_pose_error



    def _compute_intermediate_values(self) -> None:
        self.current_pose_palm = self._world_to_hand_pose(self._active_cube_root_pose_w())
        self.current_keypoints_palm = self._object_keypoints_from_pose(self.current_pose_palm)
        self.current_anchor_pos_palm[:] = self._anchor_pos_from_pose(
            self.current_pose_palm,
            self.source_anchor_local_pos_m.to(dtype=self.current_pose_palm.dtype),
        ).to(dtype=torch.float32)
        self.keypoint_error = self.target_keypoints_palm - self.current_keypoints_palm
        self.pose_error = self._object_pose_distance(self.current_pose_palm, self.target_pose_palm)
