#!/usr/bin/env python3
"""Filter Sharpa Wave cube4cm reset-cache rows with an Isaac Lab hold replay."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import traceback
import math
import sys
from datetime import datetime
from pathlib import Path

from poise.paths import workspace_root
POISE_ROOT = workspace_root()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--input", type=Path, required=True, help="Sharpa schema-v2 reset-cache path.")
parser.add_argument("--output-filtered-cache", type=Path, required=True)
parser.add_argument("--report-json", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=64, help="Parallel IsaacLab envs per replay chunk.")
parser.add_argument("--max-grasps", type=int, default=0, help="Limit rows checked; 0 checks all rows.")
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--env-spacing", type=float, default=0.35)
parser.add_argument(
    "--hand-usd-path",
    type=Path,
    default=None,
    help="Optional explicit Sharpa Wave USDA. The resolved path and SHA-256 are recorded in the report.",
)
parser.add_argument(
    "--expected-hand-usd-sha256",
    type=str,
    default="",
    help="Optional lowercase SHA-256 contract for --hand-usd-path; mismatch aborts before simulation.",
)
parser.add_argument(
    "--joint-limit-reference-usd",
    type=Path,
    default=None,
    help=(
        "Independent ASCII USDA from which action-ordered joint endpoints are parsed. "
        "Required for --joint-limit-profile usda09; this does not change the simulated hand asset."
    ),
)
parser.add_argument(
    "--expected-joint-limit-reference-usd-sha256",
    type=str,
    default="",
    help="Optional SHA-256 contract for --joint-limit-reference-usd.",
)
parser.add_argument("--replay-steps", type=int, default=240, help="Hold replay steps at 120 Hz.")
parser.add_argument("--hand-root-orientation", choices=("palm_down", "palm_up"), default="palm_down")
parser.add_argument(
    "--hand-root-rot-wxyz",
    type=float,
    nargs=4,
    default=None,
    help="Optional custom hand root quaternion override in IsaacLab wxyz order.",
)
parser.add_argument(
    "--omni-wrist-trials",
    type=int,
    default=1,
    help=(
        "Number of wrist-orientation replay trials per grasp. Values greater than 1 require a grasp "
        "to survive every sampled wrist orientation."
    ),
)
parser.add_argument("--wrist-seed", type=int, default=20260703, help="Seed for sampled omni-wrist orientations.")
parser.add_argument(
    "--no-identity-wrist-trial",
    action="store_true",
    help="Do not reserve trial 0 for the configured fixed hand-root orientation.",
)
parser.add_argument(
    "--gravity",
    type=float,
    nargs=3,
    default=None,
    help="Optional simulation gravity vector override, for example: --gravity 0 0 -0.05.",
)
parser.add_argument(
    "--object-dims-m",
    type=float,
    nargs=3,
    default=None,
    help=(
        "Optional object dimension override. Useful for testing whether a nominal cube4cm cache "
        "survives small scale perturbations, e.g. --object-dims-m 0.039 0.039 0.039."
    ),
)
parser.add_argument("--min-contact-fingers", type=int, default=2)
parser.add_argument("--contact-threshold-m", type=float, default=0.025)
parser.add_argument("--max-replay-drift-m", type=float, default=0.04)
parser.add_argument("--max-lin-vel-mps", type=float, default=0.25)
parser.add_argument("--max-ang-vel-radps", type=float, default=4.0)
parser.add_argument("--min-palm-x", type=float, default=0.025)
parser.add_argument("--max-palm-x", type=float, default=0.18)
parser.add_argument("--max-abs-palm-y", type=float, default=0.14)
parser.add_argument("--min-palm-z", type=float, default=0.02)
parser.add_argument("--max-palm-z", type=float, default=0.22)
parser.add_argument(
    "--joint-limit-profile",
    choices=("usd", "usda09", "deployment", "deployment_intersect_usd_scaled"),
    default="usd",
    help=(
        "Joint limits used by PhysX and cache validation. Non-USD profiles are strict: "
        "out-of-range q/q_cmd abort rather than being projected."
    ),
)
parser.add_argument(
    "--no-clamp-joints",
    action="store_true",
    help="For the USD profile only, validate/replay original values without raw-USD projection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim import SimulationContext
from isaacsim.core.utils import prims as prim_utils

from poise.sim.assets.cuboid import get_cuboid_cfg
from poise.sim.assets.cube_4cm import CUBE_4CM_SIDE_LENGTH_M, get_cube_4cm_cfg
from poise.sim.assets.cylinder import get_cylinder_cfg
from poise.sim.assets.irregular import (
    get_irregular_object_cfg,
    is_irregular_object_id,
    sample_irregular_surface_points,
)
from poise.sim.assets.sharpa_manual_hex_prisms import (
    SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS,
    get_manual_hex_prism_asset_dir,
    get_manual_hex_prism_spec,
    get_sharpa_manual_hex_prism_cfg,
)
from poise.sim.assets.sharpa_manual_convex_objects import (
    SHARPA_MANUAL_CONVEX_OBJECT_IDS,
    get_manual_convex_object_asset_dir,
    get_manual_convex_object_spec,
    get_sharpa_manual_convex_object_cfg,
)
from poise.sim.assets.sharpa_hand import (
    SHARPA_DEFAULT_ROOT_POS,
    SHARPA_DEPLOY_DOF_LIMITS_DEG,
    get_sharpa_wave_hand_cfg,
    resolve_sharpa_root_rot,
)
from poise.sim.hb1_sim2real import (
    composite_deploy_usd_limits,
    parse_usda_action_joint_limits,
    scaled_signed_usd_limits,
)
from poise.sim.joint_mapping import SHARPA_WAVE_RIGHT_HAND_SPEC, resolve_hand_mapping
from poise.sim.metrics import (
    cuboid_keypoints_from_pose,
    cuboid_pose_distance,
    cube_keypoints_from_pose,
    cube_pose_distance,
    normalize_quat_wxyz,
    quat_apply_wxyz,
)
from poise.grasp.cache import (
    load_stable_grasp_cache,
    save_stable_grasp_cache,
    summarize_stable_grasp_cache,
)

LOCAL_Y_CYLINDER_OBJECT_IDS = {"cylinder_2x20cm", "cylinder_2cm_20cm_y"}


def is_local_y_cylinder_object_id(object_id: str) -> bool:
    return object_id in LOCAL_Y_CYLINDER_OBJECT_IDS or (
        object_id.startswith("cylinder_") and object_id.endswith("_y")
    )


def is_cuboid_object_id(object_id: str) -> bool:
    return object_id.startswith("cube_") or object_id.startswith("cuboid_")


def is_manual_hex_prism_object_id(object_id: str) -> bool:
    return object_id in SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS


def is_manual_convex_object_id(object_id: str) -> bool:
    return object_id in SHARPA_MANUAL_CONVEX_OBJECT_IDS


def is_imported_convex_mesh_object_id(object_id: str) -> bool:
    return is_manual_hex_prism_object_id(object_id) or is_manual_convex_object_id(object_id)


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else POISE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def quat_from_angle_axis_wxyz(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
    half_angle = 0.5 * angle
    quat = torch.cat((torch.cos(half_angle).unsqueeze(-1), axis * torch.sin(half_angle).unsqueeze(-1)), dim=-1)
    return normalize_quat_wxyz(quat)


def randomize_rotation_rpy_wxyz(rpy_scale: torch.Tensor) -> torch.Tensor:
    axis_x = torch.zeros((rpy_scale.shape[0], 3), device=rpy_scale.device, dtype=rpy_scale.dtype)
    axis_y = torch.zeros_like(axis_x)
    axis_z = torch.zeros_like(axis_x)
    axis_x[:, 0] = 1.0
    axis_y[:, 1] = 1.0
    axis_z[:, 2] = 1.0
    qx = quat_from_angle_axis_wxyz(rpy_scale[:, 0] * math.pi, axis_x)
    qy = quat_from_angle_axis_wxyz(rpy_scale[:, 1] * math.pi, axis_y)
    qz = quat_from_angle_axis_wxyz(rpy_scale[:, 2] * math.pi, axis_z)
    return normalize_quat_wxyz(quat_mul_wxyz(qx, quat_mul_wxyz(qy, qz)))


def make_origins(num_envs: int, spacing: float, device: str) -> torch.Tensor:
    cols = max(1, math.ceil(math.sqrt(num_envs)))
    origins = torch.zeros(num_envs, 3, device=device)
    for env_id in range(num_envs):
        origins[env_id, 0] = (env_id % cols) * spacing
        origins[env_id, 1] = (env_id // cols) * spacing
    return origins


def action_to_sim_order(action_values: torch.Tensor, action_joint_indices: tuple[int, ...], num_joints: int) -> torch.Tensor:
    result = torch.zeros(action_values.shape[0], num_joints, device=action_values.device, dtype=action_values.dtype)
    result[:, list(action_joint_indices)] = action_values
    return result


def clamp_action_to_limits(action_values: torch.Tensor, hand: Articulation, action_joint_indices: tuple[int, ...]) -> torch.Tensor:
    lower = hand.data.soft_joint_pos_limits[:, list(action_joint_indices), 0]
    upper = hand.data.soft_joint_pos_limits[:, list(action_joint_indices), 1]
    if lower.shape[0] != action_values.shape[0]:
        # Every replicated environment uses the same articulation asset. Use
        # one raw-USD limit row when projecting the full cache at once.
        lower = lower[:1]
        upper = upper[:1]
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    return torch.where(finite, torch.minimum(torch.maximum(action_values, lower), upper), action_values)


def pad_to_batch(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if values.shape[0] == batch_size:
        return values
    pad = values[:1].repeat(batch_size - values.shape[0], *([1] * (values.ndim - 1)))
    return torch.cat((values, pad), dim=0)


def hand_to_world_pose(object_pose_palm: torch.Tensor, origins: torch.Tensor, hand_quat_wxyz: torch.Tensor) -> torch.Tensor:
    num = object_pose_palm.shape[0]
    device = object_pose_palm.device
    hand_pos = torch.tensor(SHARPA_DEFAULT_ROOT_POS, device=device, dtype=object_pose_palm.dtype).repeat(num, 1) + origins
    hand_quat = hand_quat_wxyz.to(device=device, dtype=object_pose_palm.dtype)
    if hand_quat.ndim == 1:
        hand_quat = hand_quat.view(1, 4)
    if hand_quat.shape[0] == 1:
        hand_quat = hand_quat.expand(num, 4)
    world_pos = hand_pos + quat_apply_wxyz(hand_quat, object_pose_palm[:, :3])
    world_quat = normalize_quat_wxyz(quat_mul_wxyz(hand_quat, object_pose_palm[:, 3:7]))
    return torch.cat((world_pos, world_quat), dim=-1)


def world_to_hand_pose(world_pose: torch.Tensor, origins: torch.Tensor, hand_quat_wxyz: torch.Tensor) -> torch.Tensor:
    num = world_pose.shape[0]
    device = world_pose.device
    hand_pos = torch.tensor(SHARPA_DEFAULT_ROOT_POS, device=device, dtype=world_pose.dtype).repeat(num, 1) + origins
    hand_quat = hand_quat_wxyz.to(device=device, dtype=world_pose.dtype)
    if hand_quat.ndim == 1:
        hand_quat = hand_quat.view(1, 4)
    if hand_quat.shape[0] == 1:
        hand_quat = hand_quat.expand(num, 4)
    inv_hand_quat = quat_conjugate_wxyz(hand_quat)
    palm_pos = quat_apply_wxyz(inv_hand_quat, world_pose[:, :3] - hand_pos)
    palm_quat = normalize_quat_wxyz(quat_mul_wxyz(inv_hand_quat, world_pose[:, 3:7]))
    return torch.cat((palm_pos, palm_quat), dim=-1)


def cuboid_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    dims_m: tuple[float, float, float],
) -> torch.Tensor:
    object_pos = object_pose_w[:, None, :3]
    object_quat_inv = quat_conjugate_wxyz(normalize_quat_wxyz(object_pose_w[:, 3:7]))
    rel_w = body_pos_w - object_pos
    rel_obj = quat_apply_wxyz(object_quat_inv[:, None, :].expand(-1, rel_w.shape[1], -1), rel_w)
    half_dims = torch.tensor(dims_m, device=body_pos_w.device, dtype=body_pos_w.dtype) * 0.5
    outside = torch.clamp(torch.abs(rel_obj) - half_dims.view(1, 1, 3), min=0.0)
    return torch.linalg.norm(outside, dim=-1)


def cube_surface_distances(body_pos_w: torch.Tensor, cube_pose_w: torch.Tensor) -> torch.Tensor:
    return cuboid_surface_distances(
        body_pos_w,
        cube_pose_w,
        (CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M),
    )


def cylinder_local_y_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    dims_m: tuple[float, float, float],
) -> torch.Tensor:
    object_pos = object_pose_w[:, None, :3]
    object_quat_inv = quat_conjugate_wxyz(normalize_quat_wxyz(object_pose_w[:, 3:7]))
    rel_w = body_pos_w - object_pos
    rel_obj = quat_apply_wxyz(object_quat_inv[:, None, :].expand(-1, rel_w.shape[1], -1), rel_w)
    radius = 0.5 * float(dims_m[0])
    half_length = 0.5 * float(dims_m[1])
    radial = torch.linalg.norm(rel_obj[..., (0, 2)], dim=-1)
    radial_out = torch.clamp(radial - radius, min=0.0)
    axial_out = torch.clamp(torch.abs(rel_obj[..., 1]) - half_length, min=0.0)
    return torch.sqrt(radial_out.square() + axial_out.square())


def cylinder_local_y_bbox_keypoints_from_pose(
    pose: torch.Tensor,
    dims_m: tuple[float, float, float],
) -> torch.Tensor:
    radius = 0.5 * float(dims_m[0])
    half_length = 0.5 * float(dims_m[1])
    local_points = torch.tensor(
        [[x, y, z] for x in (-radius, radius) for y in (-half_length, half_length) for z in (-radius, radius)],
        device=pose.device,
        dtype=pose.dtype,
    )
    points = local_points.expand(pose.shape[:-1] + local_points.shape)
    quat = pose[..., 3:7].unsqueeze(-2).expand(points.shape[:-1] + (4,))
    return quat_apply_wxyz(quat, points) + pose[..., :3].unsqueeze(-2)


def cylinder_local_y_bbox_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    dims_m: tuple[float, float, float],
) -> torch.Tensor:
    current = cylinder_local_y_bbox_keypoints_from_pose(current_pose, dims_m)
    goal = cylinder_local_y_bbox_keypoints_from_pose(goal_pose, dims_m)
    return torch.linalg.norm(current - goal, dim=-1).mean(dim=-1)


def resolve_object_dims_m(cache: dict) -> tuple[float, float, float]:
    object_id = str(cache.get("object_id", "cube_4cm"))
    dims_raw = cache.get("object_dims_m", None)
    if dims_raw is None and isinstance(cache.get("metadata"), dict):
        dims_raw = cache["metadata"].get("object_dims_m", None)
    if object_id == "cube_4cm" and dims_raw is None:
        dims_raw = (CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M)
    if dims_raw is None:
        raise ValueError(f"Cache for object_id={object_id!r} is missing object_dims_m metadata.")
    dims_m = tuple(float(v) for v in dims_raw)
    if len(dims_m) != 3 or any(v <= 0.0 for v in dims_m):
        raise ValueError(f"Invalid object_dims_m for {object_id!r}: {dims_m!r}.")
    if is_cuboid_object_id(object_id):
        return dims_m
    if is_local_y_cylinder_object_id(object_id):
        if abs(float(dims_m[0]) - float(dims_m[2])) > 1.0e-6 or float(dims_m[1]) <= float(dims_m[0]):
            raise ValueError(f"Local-Y cylinder dimensions must be diameter, length, diameter; got {dims_m!r}.")
        return dims_m
    if is_irregular_object_id(object_id):
        return dims_m
    if is_manual_hex_prism_object_id(object_id):
        expected = get_manual_hex_prism_spec(object_id).extents_xyz_m
        if any(abs(float(a) - float(b)) > 1.0e-6 for a, b in zip(dims_m, expected, strict=True)):
            raise ValueError(
                f"Manual hex-prism dimensions must match its fixed USD asset: got {dims_m!r}, expected {expected!r}."
            )
        return dims_m
    if is_manual_convex_object_id(object_id):
        expected = get_manual_convex_object_spec(object_id).extents_xyz_m
        if any(abs(float(a) - float(b)) > 1.0e-6 for a, b in zip(dims_m, expected, strict=True)):
            raise ValueError(
                f"Manual convex-object dimensions must match its fixed USD asset: got {dims_m!r}, expected {expected!r}."
            )
        return dims_m
    raise ValueError(
        f"Unsupported object_id={object_id!r}; expected cube/cuboid, local-Y cylinder, irregular object, "
        "or registered manual hex prism."
    )


def nominal_cube4cm_dims() -> tuple[float, float, float]:
    return (CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M)


def is_nominal_cube4cm_dims(object_dims_m: tuple[float, float, float]) -> bool:
    return all(abs(float(a) - float(b)) <= 1.0e-7 for a, b in zip(object_dims_m, nominal_cube4cm_dims()))


def get_object_cfg(object_id: str, object_dims_m: tuple[float, float, float]):
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return get_cube_4cm_cfg(prim_path="/World/Env_.*/Cube", init_pos=(0.0, 0.0, 0.0), disable_gravity=False)
    if is_cuboid_object_id(object_id):
        return get_cuboid_cfg(
            prim_path="/World/Env_.*/Cube",
            dims_m=object_dims_m,
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
    if is_local_y_cylinder_object_id(object_id):
        return get_cylinder_cfg(
            prim_path="/World/Env_.*/Cube",
            radius_m=0.5 * float(object_dims_m[0]),
            height_m=float(object_dims_m[1]),
            axis="Y",
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
    if is_irregular_object_id(object_id):
        return get_irregular_object_cfg(
            object_id=object_id,
            dims_m=object_dims_m,
            prim_path="/World/Env_.*/Cube",
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
    if is_manual_hex_prism_object_id(object_id):
        expected = get_manual_hex_prism_spec(object_id).extents_xyz_m
        if any(abs(float(a) - float(b)) > 1.0e-6 for a, b in zip(object_dims_m, expected, strict=True)):
            raise ValueError(
                f"Manual hex-prism dimensions must match its fixed USD asset: got {object_dims_m!r}, expected {expected!r}."
            )
        return get_sharpa_manual_hex_prism_cfg(
            object_id=object_id,
            prim_path="/World/Env_.*/Cube",
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
    if is_manual_convex_object_id(object_id):
        expected = get_manual_convex_object_spec(object_id).extents_xyz_m
        if any(abs(float(a) - float(b)) > 1.0e-6 for a, b in zip(object_dims_m, expected, strict=True)):
            raise ValueError(
                f"Manual convex-object dimensions must match its fixed USD asset: got {object_dims_m!r}, expected {expected!r}."
            )
        return get_sharpa_manual_convex_object_cfg(
            object_id=object_id,
            prim_path="/World/Env_.*/Cube",
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def transform_local_points_wxyz(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    points = points.to(device=pose.device, dtype=pose.dtype)
    expand_shape = pose.shape[:-1] + points.shape
    points = points.expand(expand_shape)
    quat = pose[..., 3:7].unsqueeze(-2).expand(expand_shape[:-1] + (4,))
    return quat_apply_wxyz(quat, points) + pose[..., :3].unsqueeze(-2)


def irregular_surface_points_w(
    object_pose_w: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    local_points = sample_irregular_surface_points(object_id, object_dims_m, samples_per_edge=7)
    return transform_local_points_wxyz(local_points, object_pose_w)


def irregular_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    surface_w = irregular_surface_points_w(object_pose_w, object_id, object_dims_m)
    distances = torch.cdist(body_pos_w, surface_w)
    return distances.amin(dim=-1)


def irregular_surface_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    local_points = sample_irregular_surface_points(object_id, object_dims_m, samples_per_edge=5)
    current = transform_local_points_wxyz(local_points, current_pose)
    goal = transform_local_points_wxyz(local_points, goal_pose)
    return torch.linalg.norm(current - goal, dim=-1).mean(dim=-1)


@functools.lru_cache(maxsize=len(SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS) + len(SHARPA_MANUAL_CONVEX_OBJECT_IDS))
def load_imported_convex_mesh_triangles(object_id: str) -> torch.Tensor:
    """Load the exact triangulated visual/collision mesh in object-local meters."""

    asset_dir = (get_manual_hex_prism_asset_dir(object_id) if is_manual_hex_prism_object_id(object_id) else get_manual_convex_object_asset_dir(object_id))
    obj_path = asset_dir / f"{object_id}.obj"
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw_line in obj_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(v) for v in fields[1:4]))
        elif fields[0] == "f" and len(fields) == 4:
            faces.append(tuple(int(v.split("/", 1)[0]) - 1 for v in fields[1:4]))
    if not vertices or not faces:
        raise ValueError(f"Imported convex OBJ has no triangulated mesh: {obj_path}")
    vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
    face_tensor = torch.tensor(faces, dtype=torch.long)
    triangles = vertex_tensor[face_tensor]
    if not torch.isfinite(triangles).all():
        raise ValueError(f"Imported convex OBJ contains non-finite coordinates: {obj_path}")
    return triangles


def point_to_triangle_surface_distances(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Exact unsigned distance from batched points to a closed triangle surface."""

    triangles = triangles.to(device=points.device, dtype=points.dtype)
    point = points.unsqueeze(-2)
    a = triangles[:, 0].view(*([1] * (points.ndim - 1)), -1, 3)
    b = triangles[:, 1].view(*([1] * (points.ndim - 1)), -1, 3)
    c = triangles[:, 2].view(*([1] * (points.ndim - 1)), -1, 3)

    def segment_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        edge = end - start
        denom = torch.clamp((edge * edge).sum(dim=-1), min=1.0e-16)
        t = torch.clamp(((point - start) * edge).sum(dim=-1) / denom, 0.0, 1.0)
        closest = start + t.unsqueeze(-1) * edge
        return torch.linalg.norm(point - closest, dim=-1)

    ab = b - a
    ac = c - a
    normal = torch.linalg.cross(ab, ac, dim=-1)
    normal_sq = torch.clamp((normal * normal).sum(dim=-1), min=1.0e-20)
    signed_numerator = ((point - a) * normal).sum(dim=-1)
    projected = point - (signed_numerator / normal_sq).unsqueeze(-1) * normal
    relative = projected - a
    d00 = (ab * ab).sum(dim=-1)
    d01 = (ab * ac).sum(dim=-1)
    d11 = (ac * ac).sum(dim=-1)
    d20 = (relative * ab).sum(dim=-1)
    d21 = (relative * ac).sum(dim=-1)
    bary_denom = torch.clamp(d00 * d11 - d01 * d01, min=1.0e-20)
    bary_v = (d11 * d20 - d01 * d21) / bary_denom
    bary_w = (d00 * d21 - d01 * d20) / bary_denom
    bary_u = 1.0 - bary_v - bary_w
    inside = (bary_u >= -1.0e-6) & (bary_v >= -1.0e-6) & (bary_w >= -1.0e-6)
    plane_distance = signed_numerator.abs() / torch.sqrt(normal_sq)
    plane_distance = torch.where(inside, plane_distance, torch.full_like(plane_distance, torch.inf))

    triangle_distance = torch.minimum(
        plane_distance,
        torch.minimum(segment_distance(a, b), torch.minimum(segment_distance(b, c), segment_distance(c, a))),
    )
    return triangle_distance.amin(dim=-1)


def manual_hex_prism_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    object_id: str,
) -> torch.Tensor:
    """Mesh-exact body-point distance using the real imported convex-mesh triangles."""

    object_quat_inv = quat_conjugate_wxyz(normalize_quat_wxyz(object_pose_w[:, 3:7]))
    relative_w = body_pos_w - object_pose_w[:, None, :3]
    object_quat_inv = object_quat_inv[:, None, :].expand(relative_w.shape[:-1] + (4,))
    body_pos_object = quat_apply_wxyz(object_quat_inv, relative_w)
    return point_to_triangle_surface_distances(body_pos_object, load_imported_convex_mesh_triangles(object_id))


def manual_hex_prism_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    object_id: str,
) -> torch.Tensor:
    local_vertices = torch.unique(load_imported_convex_mesh_triangles(object_id).reshape(-1, 3), dim=0)
    current = transform_local_points_wxyz(local_vertices, current_pose)
    goal = transform_local_points_wxyz(local_vertices, goal_pose)
    return torch.linalg.norm(current - goal, dim=-1).mean(dim=-1)


def object_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return cube_surface_distances(body_pos_w, object_pose_w)
    if is_cuboid_object_id(object_id):
        return cuboid_surface_distances(body_pos_w, object_pose_w, object_dims_m)
    if is_local_y_cylinder_object_id(object_id):
        return cylinder_local_y_surface_distances(body_pos_w, object_pose_w, object_dims_m)
    if is_irregular_object_id(object_id):
        return irregular_surface_distances(body_pos_w, object_pose_w, object_id, object_dims_m)
    if is_imported_convex_mesh_object_id(object_id):
        return manual_hex_prism_surface_distances(body_pos_w, object_pose_w, object_id)
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def object_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return cube_pose_distance(current_pose, goal_pose)
    if is_cuboid_object_id(object_id):
        return cuboid_pose_distance(current_pose, goal_pose, object_dims_m)
    if is_local_y_cylinder_object_id(object_id):
        return cylinder_local_y_bbox_pose_distance(current_pose, goal_pose, object_dims_m)
    if is_irregular_object_id(object_id):
        return irregular_surface_pose_distance(current_pose, goal_pose, object_id, object_dims_m)
    if is_imported_convex_mesh_object_id(object_id):
        return manual_hex_prism_pose_distance(current_pose, goal_pose, object_id)
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def object_keypoints_from_pose(
    pose: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return cube_keypoints_from_pose(pose)
    if is_cuboid_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    if is_local_y_cylinder_object_id(object_id):
        return cylinder_local_y_bbox_keypoints_from_pose(pose, object_dims_m)
    if is_irregular_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    if is_imported_convex_mesh_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def step_sim(sim: SimulationContext, hand: Articulation, cube: RigidObject, target: torch.Tensor, steps: int) -> None:
    dt = sim.get_physics_dt()
    for _ in range(max(0, int(steps))):
        hand.set_joint_position_target(target)
        hand.write_data_to_sim()
        cube.write_data_to_sim()
        sim.step()
        hand.update(dt)
        cube.update(dt)


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    values = values.detach().cpu().float()
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def main() -> int:
    if args_cli.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args_cli.min_contact_fingers < 0 or args_cli.min_contact_fingers > 5:
        raise ValueError("--min-contact-fingers must be in [0, 5].")
    if args_cli.omni_wrist_trials < 1:
        raise ValueError("--omni-wrist-trials must be positive.")

    input_path = resolve_workspace_path(args_cli.input).resolve()
    hand_usd_path = (
        resolve_workspace_path(args_cli.hand_usd_path).resolve()
        if args_cli.hand_usd_path is not None
        else None
    )
    if hand_usd_path is not None and not hand_usd_path.is_file():
        raise FileNotFoundError(f"Sharpa Wave USDA not found: {hand_usd_path}")
    hand_usd_sha256 = sha256_file(hand_usd_path) if hand_usd_path is not None else ""
    expected_hand_usd_sha256 = str(args_cli.expected_hand_usd_sha256).strip().lower()
    if expected_hand_usd_sha256 and hand_usd_path is None:
        raise ValueError("--expected-hand-usd-sha256 requires --hand-usd-path.")
    if expected_hand_usd_sha256 and hand_usd_sha256 != expected_hand_usd_sha256:
        raise ValueError(
            "Sharpa Wave USDA SHA-256 mismatch: "
            f"path={hand_usd_path}, expected={expected_hand_usd_sha256}, actual={hand_usd_sha256}."
        )
    joint_limit_reference_usd = (
        resolve_workspace_path(args_cli.joint_limit_reference_usd).expanduser().resolve()
        if args_cli.joint_limit_reference_usd is not None
        else None
    )
    if joint_limit_reference_usd is not None and not joint_limit_reference_usd.is_file():
        raise FileNotFoundError(f"Joint-limit reference USDA not found: {joint_limit_reference_usd}")
    joint_limit_reference_usd_sha256 = (
        sha256_file(joint_limit_reference_usd) if joint_limit_reference_usd is not None else ""
    )
    expected_joint_limit_reference_usd_sha256 = str(
        args_cli.expected_joint_limit_reference_usd_sha256
    ).strip().lower()
    if expected_joint_limit_reference_usd_sha256 and joint_limit_reference_usd is None:
        raise ValueError(
            "--expected-joint-limit-reference-usd-sha256 requires --joint-limit-reference-usd."
        )
    if (
        expected_joint_limit_reference_usd_sha256
        and joint_limit_reference_usd_sha256 != expected_joint_limit_reference_usd_sha256
    ):
        raise ValueError(
            "Joint-limit reference USDA SHA-256 mismatch: "
            f"path={joint_limit_reference_usd}, "
            f"expected={expected_joint_limit_reference_usd_sha256}, "
            f"actual={joint_limit_reference_usd_sha256}."
        )
    if args_cli.joint_limit_profile == "usda09" and joint_limit_reference_usd is None:
        raise ValueError("--joint-limit-profile usda09 requires --joint-limit-reference-usd.")
    cache = load_stable_grasp_cache(input_path, map_location="cpu")
    if str(cache.get("hand")) != "sharpa_wave_right":
        raise ValueError(f"Expected hand='sharpa_wave_right', got {cache.get('hand')!r}.")
    object_id = str(cache.get("object_id", "cube_4cm"))
    object_dims_m = resolve_object_dims_m(cache)
    cache_object_dims_m = object_dims_m
    if args_cli.object_dims_m is not None:
        object_dims_m = tuple(float(v) for v in args_cli.object_dims_m)
        if len(object_dims_m) != 3 or any(v <= 0.0 for v in object_dims_m):
            raise ValueError(f"--object-dims-m must contain three positive dimensions, got {object_dims_m!r}.")

    total_cache_rows = int(cache["q"].shape[0])
    start = max(0, int(args_cli.start_index))
    stop = total_cache_rows if args_cli.max_grasps <= 0 else min(total_cache_rows, start + int(args_cli.max_grasps))
    if start >= stop:
        raise ValueError(f"Empty selection: start={start}, stop={stop}, total_cache_rows={total_cache_rows}.")
    selected = torch.arange(start, stop, dtype=torch.long)
    total = int(selected.numel())
    batch_size = min(int(args_cli.batch_size), total)
    if args_cli.hand_root_rot_wxyz is None:
        hand_root_rot = tuple(float(v) for v in resolve_sharpa_root_rot(args_cli.hand_root_orientation))
        hand_root_label = args_cli.hand_root_orientation
    else:
        hand_root_rot = tuple(float(v) for v in args_cli.hand_root_rot_wxyz)
        hand_root_label = "custom"
    hand_root_quat = torch.tensor(hand_root_rot, dtype=torch.float32).view(1, 4)
    hand_root_quat_all = hand_root_quat.expand(total, 4)
    hand_quat_trials: list[torch.Tensor] = []
    hand_quat_trial_labels: list[str] = []
    if not args_cli.no_identity_wrist_trial:
        hand_quat_trials.append(hand_root_quat_all.clone())
        hand_quat_trial_labels.append("identity")
    while len(hand_quat_trials) < int(args_cli.omni_wrist_trials):
        trial_index = len(hand_quat_trials)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args_cli.wrist_seed) + trial_index)
        rpy_scale = 2.0 * torch.rand((total, 3), generator=generator, dtype=torch.float32) - 1.0
        wrist_rot = randomize_rotation_rpy_wxyz(rpy_scale)
        hand_quat_trials.append(normalize_quat_wxyz(quat_mul_wxyz(wrist_rot, hand_root_quat_all)))
        hand_quat_trial_labels.append(f"random_{trial_index}")

    sim_cfg_kwargs = {"dt": 1.0 / 120.0, "device": args_cli.device}
    if args_cli.gravity is not None:
        sim_cfg_kwargs["gravity"] = tuple(float(v) for v in args_cli.gravity)
    sim = SimulationContext(sim_utils.SimulationCfg(**sim_cfg_kwargs))
    sim.set_camera_view([0.8, -0.8, 0.9], [0.0, 0.0, 0.5])

    ground_cfg = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg, translation=(0.0, 0.0, -0.05))
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)

    origins = make_origins(batch_size, args_cli.env_spacing, sim.device)
    for env_id, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Env_{env_id}", "Xform", translation=origin.tolist())

    hand = Articulation(
        get_sharpa_wave_hand_cfg(
            prim_path="/World/Env_.*/Robot",
            init_pos=SHARPA_DEFAULT_ROOT_POS,
            init_rot=hand_root_rot,
            usd_path=hand_usd_path,
        )
    )
    cube_cfg = get_object_cfg(object_id, object_dims_m)
    cube = RigidObject(cube_cfg)
    if not hasattr(cube_cfg.spawn, "physics_material"):
        # IsaacLab 2.3's UsdFileCfg cannot create a physics material inline.
        # Bind the same coefficient used by the Cube/GridDiverse replay to all
        # imported USD collision assets, including compound non-convex objects.
        material_path = "/World/ImportedObjectPhysicsMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        material_cfg.func(material_path, material_cfg)
        for env_id in range(batch_size):
            sim_utils.bind_physics_material(f"/World/Env_{env_id}/Cube", material_path)
    sim.reset()
    mapping = resolve_hand_mapping(SHARPA_WAVE_RIGHT_HAND_SPEC, hand.joint_names, hand.body_names)
    body_name_to_idx = {name: idx for idx, name in enumerate(hand.body_names)}
    contact_body_indices = tuple(body_name_to_idx[name] for name in SHARPA_WAVE_RIGHT_HAND_SPEC.contact_bit_bodies)

    raw_limits = hand.data.soft_joint_pos_limits[0, list(mapping.action_joint_indices)].to(sim.device).clone()
    joint_limit_profile = str(args_cli.joint_limit_profile)
    if joint_limit_profile == "usd":
        active_limits = raw_limits
        joint_limit_rule = "usd_soft_limits"
    elif joint_limit_profile == "usda09":
        reference_limits = parse_usda_action_joint_limits(
            joint_limit_reference_usd, SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order
        ).to(device=sim.device, dtype=raw_limits.dtype)
        active_limits = scaled_signed_usd_limits(reference_limits, endpoint_scale=0.9)
        joint_limit_rule = "sharpa_rl_lab_usda_signed_endpoints_times_0.9"
    elif joint_limit_profile == "deployment":
        active_limits = torch.deg2rad(
            torch.tensor(
                [SHARPA_DEPLOY_DOF_LIMITS_DEG[name] for name in SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order],
                device=sim.device,
                dtype=raw_limits.dtype,
            )
        )
        joint_limit_rule = "docs_deployment_profile_exact"
    else:
        active_limits = composite_deploy_usd_limits(
            raw_limits, SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order, SHARPA_DEPLOY_DOF_LIMITS_DEG,
            usd_endpoint_scale=0.9,
        )
        joint_limit_rule = "deployment_profile_intersect_usd_endpoints_times_0.9"
    if joint_limit_profile != "usd":
        hand.write_joint_position_limit_to_sim(
            active_limits.unsqueeze(0).expand(batch_size, -1, -1),
            joint_ids=list(mapping.action_joint_indices),
            warn_limit_violation=False,
        )

    q_all = cache["q"][selected].to(device=sim.device, dtype=torch.float32)
    qdot_all = cache["qdot"][selected].to(device=sim.device, dtype=torch.float32)
    q_cmd_all = cache["q_cmd"][selected].to(device=sim.device, dtype=torch.float32)
    object_pose_all = cache["object_pose_palm"][selected].to(device=sim.device, dtype=torch.float32)
    q_source_all = q_all.clone()
    q_cmd_source_all = q_cmd_all.clone()
    projection_enabled = joint_limit_profile == "usd" and not args_cli.no_clamp_joints
    if projection_enabled:
        q_all = clamp_action_to_limits(q_all, hand, mapping.action_joint_indices)
        q_cmd_all = clamp_action_to_limits(q_cmd_all, hand, mapping.action_joint_indices)
    else:
        lower, upper = active_limits[:, 0], active_limits[:, 1]
        for label, values in (("q", q_all), ("q_cmd", q_cmd_all)):
            outside = (values < lower.unsqueeze(0) - 1.0e-7) | (values > upper.unsqueeze(0) + 1.0e-7)
            if bool(outside.any()):
                row, joint = torch.nonzero(outside, as_tuple=False)[0].tolist()
                raise ValueError(
                    f"Input {label} violates joint-limit profile={joint_limit_profile}: "
                    f"selected_row={row}, source_row={int(selected[row])}, "
                    f"joint={SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order[joint]}, "
                    f"value={float(values[row, joint]):.9f}, "
                    f"interval=[{float(lower[joint]):.9f}, {float(upper[joint]):.9f}]."
                )
    q_projection_delta = torch.abs(q_all - q_source_all)
    q_cmd_projection_delta = torch.abs(q_cmd_all - q_cmd_source_all)
    q_projected_rows = q_projection_delta.gt(1.0e-8).any(dim=-1)
    q_cmd_projected_rows = q_cmd_projection_delta.gt(1.0e-8).any(dim=-1)
    print(
        "[INFO] cache joint-limit contract "
        f"profile={joint_limit_profile} projection_enabled={projection_enabled} "
        f"q_rows={int(q_projected_rows.sum().item())}/{total} "
        f"q_cmd_rows={int(q_cmd_projected_rows.sum().item())}/{total} "
        f"max_delta_rad={float(q_projection_delta.max().item()):.8f}/"
        f"{float(q_cmd_projection_delta.max().item()):.8f}",
        flush=True,
    )

    trial_summaries: list[dict[str, float | int | str]] = []
    trial_chunk_summaries: list[dict[str, float | int | str]] = []
    stable_trials: list[torch.Tensor] = []
    stability_score_trials: list[torch.Tensor] = []
    drift_trials: list[torch.Tensor] = []
    linear_speed_trials: list[torch.Tensor] = []
    angular_speed_trials: list[torch.Tensor] = []
    contact_count_trials: list[torch.Tensor] = []
    workspace_trials: list[torch.Tensor] = []
    contact_bits_reference: torch.Tensor | None = None

    for trial_idx, (trial_label, hand_quat_all_cpu) in enumerate(zip(hand_quat_trial_labels, hand_quat_trials)):
        hand_quat_all = hand_quat_all_cpu.to(device=sim.device, dtype=torch.float32)
        drift_items: list[torch.Tensor] = []
        linear_speed_items: list[torch.Tensor] = []
        angular_speed_items: list[torch.Tensor] = []
        contact_bits_items: list[torch.Tensor] = []
        contact_count_items: list[torch.Tensor] = []
        stability_score_items: list[torch.Tensor] = []
        stable_items: list[torch.Tensor] = []
        workspace_items: list[torch.Tensor] = []

        for chunk_start in range(0, total, batch_size):
            chunk_stop = min(total, chunk_start + batch_size)
            n = chunk_stop - chunk_start
            q = pad_to_batch(q_all[chunk_start:chunk_stop], batch_size)
            qdot = pad_to_batch(qdot_all[chunk_start:chunk_stop], batch_size)
            q_cmd = pad_to_batch(q_cmd_all[chunk_start:chunk_stop], batch_size)
            object_pose_palm = pad_to_batch(object_pose_all[chunk_start:chunk_stop], batch_size)
            hand_quat = pad_to_batch(hand_quat_all[chunk_start:chunk_stop], batch_size)
            object_pose_w = hand_to_world_pose(object_pose_palm, origins, hand_quat)
            root_state = hand.data.default_root_state.clone()
            root_state[:, :3] = (
                torch.tensor(SHARPA_DEFAULT_ROOT_POS, device=sim.device, dtype=root_state.dtype) + origins
            )
            root_state[:, 3:7] = hand_quat.to(dtype=root_state.dtype)
            q_sim = action_to_sim_order(q, mapping.action_joint_indices, len(hand.joint_names))
            qdot_sim = action_to_sim_order(qdot, mapping.action_joint_indices, len(hand.joint_names))
            q_cmd_sim = action_to_sim_order(q_cmd, mapping.action_joint_indices, len(hand.joint_names))

            hand.write_root_pose_to_sim(root_state[:, :7])
            hand.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
            hand.write_joint_state_to_sim(q_sim, qdot_sim)
            hand.set_joint_position_target(q_cmd_sim)
            cube.write_root_pose_to_sim(object_pose_w)
            cube.write_root_velocity_to_sim(torch.zeros(batch_size, 6, device=sim.device))
            hand.write_data_to_sim()
            cube.write_data_to_sim()
            sim.step()
            hand.update(sim.get_physics_dt())
            cube.update(sim.get_physics_dt())
            hand.reset()
            cube.reset()

            step_sim(sim, hand, cube, q_cmd_sim, args_cli.replay_steps)

            replay_pose_palm = world_to_hand_pose(cube.data.root_pose_w.detach(), origins, hand_quat)
            replay_drift = object_pose_distance(object_pose_palm, replay_pose_palm, object_id, object_dims_m)[:n]
            contact_dist = object_surface_distances(
                hand.data.body_pos_w[:, list(contact_body_indices)].detach(),
                cube.data.root_pose_w.detach(),
                object_id,
                object_dims_m,
            )
            contact_bits = contact_dist[:n] < float(args_cli.contact_threshold_m)
            contact_count = contact_bits.sum(dim=-1)
            linear_speed = torch.linalg.norm(cube.data.root_vel_w[:n, :3], dim=-1)
            angular_speed = torch.linalg.norm(cube.data.root_vel_w[:n, 3:6], dim=-1)
            replay_pos = replay_pose_palm[:n, :3]
            in_workspace = (
                (replay_pos[:, 0] > float(args_cli.min_palm_x))
                & (replay_pos[:, 0] < float(args_cli.max_palm_x))
                & (torch.abs(replay_pos[:, 1]) < float(args_cli.max_abs_palm_y))
                & (replay_pos[:, 2] > float(args_cli.min_palm_z))
                & (replay_pos[:, 2] < float(args_cli.max_palm_z))
            )
            stability_score = contact_count.float() - 4.0 * linear_speed - 0.1 * angular_speed - 10.0 * replay_drift
            stable = (
                in_workspace
                & (contact_count >= int(args_cli.min_contact_fingers))
                & (linear_speed < float(args_cli.max_lin_vel_mps))
                & (angular_speed < float(args_cli.max_ang_vel_radps))
                & (replay_drift < float(args_cli.max_replay_drift_m))
            )

            drift_items.append(replay_drift.detach().cpu())
            linear_speed_items.append(linear_speed.detach().cpu())
            angular_speed_items.append(angular_speed.detach().cpu())
            contact_bits_items.append(contact_bits.detach().cpu())
            contact_count_items.append(contact_count.detach().cpu())
            stability_score_items.append(stability_score.detach().cpu())
            stable_items.append(stable.detach().cpu())
            workspace_items.append(in_workspace.detach().cpu())
            trial_chunk_summaries.append(
                {
                    "trial": int(trial_idx),
                    "trial_label": trial_label,
                    "cache_start": int(start + chunk_start),
                    "cache_stop": int(start + chunk_stop),
                    "stable": int(stable.sum().item()),
                    "count": int(n),
                    "mean_drift_m": float(replay_drift.mean().item()),
                    "mean_lin_vel_mps": float(linear_speed.mean().item()),
                    "mean_ang_vel_radps": float(angular_speed.mean().item()),
                    "mean_contact_fingers": float(contact_count.float().mean().item()),
                }
            )
            print(
                f"[INFO] wrist_trial={trial_idx}/{len(hand_quat_trials)} {trial_label} "
                f"chunk {start + chunk_start}:{start + chunk_stop} "
                f"stable={int(stable.sum().item())}/{n} "
                f"mean_drift={float(replay_drift.mean().item()):.4f} "
                f"mean_lin_vel={float(linear_speed.mean().item()):.4f} "
                f"mean_contact={float(contact_count.float().mean().item()):.2f}",
                flush=True,
            )

        drift_trial = torch.cat(drift_items)
        linear_speed_trial = torch.cat(linear_speed_items)
        angular_speed_trial = torch.cat(angular_speed_items)
        contact_bits_trial = torch.cat(contact_bits_items)
        contact_count_trial = torch.cat(contact_count_items)
        stability_score_trial = torch.cat(stability_score_items)
        stable_trial = torch.cat(stable_items)
        workspace_trial = torch.cat(workspace_items)
        if contact_bits_reference is None:
            contact_bits_reference = contact_bits_trial
        drift_trials.append(drift_trial)
        linear_speed_trials.append(linear_speed_trial)
        angular_speed_trials.append(angular_speed_trial)
        contact_count_trials.append(contact_count_trial)
        stability_score_trials.append(stability_score_trial)
        stable_trials.append(stable_trial)
        workspace_trials.append(workspace_trial)
        trial_summaries.append(
            {
                "trial": int(trial_idx),
                "trial_label": trial_label,
                "stable_count": int(stable_trial.sum().item()),
                "stable_fraction": float(stable_trial.float().mean().item()),
                "mean_drift_m": float(drift_trial.float().mean().item()),
                "mean_lin_vel_mps": float(linear_speed_trial.float().mean().item()),
                "mean_ang_vel_radps": float(angular_speed_trial.float().mean().item()),
                "mean_contact_fingers": float(contact_count_trial.float().mean().item()),
            }
        )

    stable_by_trial = torch.stack(stable_trials, dim=0)
    stable = stable_by_trial.all(dim=0)
    drift = torch.stack(drift_trials, dim=0).max(dim=0).values
    linear_speed = torch.stack(linear_speed_trials, dim=0).max(dim=0).values
    angular_speed = torch.stack(angular_speed_trials, dim=0).max(dim=0).values
    contact_count = torch.stack(contact_count_trials, dim=0).min(dim=0).values
    stability_score = torch.stack(stability_score_trials, dim=0).min(dim=0).values
    in_workspace = torch.stack(workspace_trials, dim=0).all(dim=0)
    contact_bits = contact_bits_reference if contact_bits_reference is not None else torch.zeros((total, 5), dtype=torch.bool)
    stable_source_indices = selected.cpu()[stable]
    per_grasp = []
    for item_idx, source_index in enumerate(selected.cpu().tolist()):
        per_grasp.append(
            {
                "source_index": int(source_index),
                "stable": bool(stable[item_idx].item()),
                "in_workspace": bool(in_workspace[item_idx].item()),
                "drift_m": float(drift[item_idx].item()),
                "linear_speed_mps": float(linear_speed[item_idx].item()),
                "angular_speed_radps": float(angular_speed[item_idx].item()),
                "contact_fingers": int(contact_count[item_idx].item()),
                "stability_score": float(stability_score[item_idx].item()),
            }
        )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "object_id": object_id,
        "object_dims_m": object_dims_m,
        "cache_object_dims_m": cache_object_dims_m,
        "object_dims_override_enabled": args_cli.object_dims_m is not None,
        "total_cache_rows": total_cache_rows,
        "selection": {"start_index": start, "num_checked": total},
        "batch_size": batch_size,
        "replay_steps": int(args_cli.replay_steps),
        "sim_dt": 1.0 / 120.0,
        "hold_seconds": float(args_cli.replay_steps / 120.0),
        "hand_root_orientation": hand_root_label,
        "hand_root_rot_wxyz": hand_root_rot,
        "hand_usd_path": str(hand_usd_path) if hand_usd_path is not None else None,
        "hand_usd_sha256": hand_usd_sha256 or None,
        "expected_hand_usd_sha256": expected_hand_usd_sha256 or None,
        "joint_limit_reference_usd_path": (
            str(joint_limit_reference_usd) if joint_limit_reference_usd is not None else None
        ),
        "joint_limit_reference_usd_sha256": joint_limit_reference_usd_sha256 or None,
        "expected_joint_limit_reference_usd_sha256": (
            expected_joint_limit_reference_usd_sha256 or None
        ),
        "omni_wrist": {
            "trials": int(args_cli.omni_wrist_trials),
            "wrist_seed": int(args_cli.wrist_seed),
            "identity_wrist_trial_enabled": not bool(args_cli.no_identity_wrist_trial),
            "trial_labels": hand_quat_trial_labels,
            "trial_summaries": trial_summaries,
        },
        "gravity": tuple(float(v) for v in args_cli.gravity) if args_cli.gravity is not None else None,
        "criteria": {
            "min_contact_fingers": int(args_cli.min_contact_fingers),
            "contact_threshold_m": float(args_cli.contact_threshold_m),
            "max_replay_drift_m": float(args_cli.max_replay_drift_m),
            "max_lin_vel_mps": float(args_cli.max_lin_vel_mps),
            "max_ang_vel_radps": float(args_cli.max_ang_vel_radps),
            "min_palm_x": float(args_cli.min_palm_x),
            "max_palm_x": float(args_cli.max_palm_x),
            "max_abs_palm_y": float(args_cli.max_abs_palm_y),
            "min_palm_z": float(args_cli.min_palm_z),
            "max_palm_z": float(args_cli.max_palm_z),
        },
        "joint_projection": {
            "enabled": projection_enabled,
            "profile": joint_limit_profile,
            "rule": joint_limit_rule,
            "limits_rad": [[float(v) for v in row] for row in active_limits.detach().cpu().tolist()],
            "margin_rad": 0.0,
            "q_projected_row_count": int(q_projected_rows.sum().item()),
            "q_cmd_projected_row_count": int(q_cmd_projected_rows.sum().item()),
            "q_max_delta_rad": float(q_projection_delta.max().item()),
            "q_cmd_max_delta_rad": float(q_cmd_projection_delta.max().item()),
        },
        "stable_count": int(stable.sum().item()),
        "stable_fraction": float(stable.float().mean().item()),
        "stable_source_indices_preview": stable_source_indices[:32].tolist(),
        "workspace_count": int(in_workspace.sum().item()),
        "contact_ge_min_count": int((contact_count >= int(args_cli.min_contact_fingers)).sum().item()),
        "drift_m": tensor_stats(drift),
        "linear_speed_mps": tensor_stats(linear_speed),
        "angular_speed_radps": tensor_stats(angular_speed),
        "contact_fingers": tensor_stats(contact_count.float()),
        "per_grasp": per_grasp,
        "chunk_summaries": trial_chunk_summaries,
    }

    report_path = resolve_workspace_path(args_cli.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_cache_path = resolve_workspace_path(args_cli.output_filtered_cache).resolve()
    if stable.any():
        filtered = dict(cache)
        for key, value in cache.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (total_cache_rows,):
                filtered[key] = value[stable_source_indices]
        filtered["q"] = q_all.detach().cpu()[stable]
        filtered["q_cmd"] = q_cmd_all.detach().cpu()[stable]
        filtered["contact_bits"] = contact_bits[stable]
        filtered["stability_score"] = stability_score[stable]
        filtered["cube_keypoints_palm"] = object_keypoints_from_pose(
            filtered["object_pose_palm"],
            object_id,
            object_dims_m,
        )
        filtered["object_id"] = object_id
        filtered["object_dims_m"] = object_dims_m
        filtered["source_cache_index"] = stable_source_indices.clone()
        metadata = dict(filtered.get("metadata", {}))
        metadata["object_id"] = object_id
        metadata["object_dims_m"] = object_dims_m
        metadata["hold_filter"] = {
            "created_at": report["created_at"],
            "input": str(input_path),
            "output": str(output_cache_path),
            "stable_count": int(stable.sum().item()),
            "num_checked": total,
            "source_start_index": start,
            "replay_steps": int(args_cli.replay_steps),
            "hold_seconds": float(args_cli.replay_steps / 120.0),
            "hand_root_orientation": args_cli.hand_root_orientation,
            "hand_root_label": hand_root_label,
            "hand_root_rot_wxyz": hand_root_rot,
            "hand_usd_path": report["hand_usd_path"],
            "hand_usd_sha256": report["hand_usd_sha256"],
            "joint_limit_reference_usd_path": report["joint_limit_reference_usd_path"],
            "joint_limit_reference_usd_sha256": report["joint_limit_reference_usd_sha256"],
            "omni_wrist": report["omni_wrist"],
            "gravity": report["gravity"],
            "criteria": report["criteria"],
            "joint_clamping_enabled": projection_enabled,
            "joint_limit_profile": joint_limit_profile,
            "joint_projection": report["joint_projection"],
        }
        filtered["metadata"] = metadata
        save_stable_grasp_cache(filtered, output_cache_path)
        report["output_filtered_cache"] = str(output_cache_path)
        report["filtered_cache_summary"] = summarize_stable_grasp_cache(filtered)
        print(f"[INFO] output_filtered_cache={output_cache_path}", flush=True)
        print(f"[INFO] filtered_cache_summary={report['filtered_cache_summary']}", flush=True)
    else:
        report["output_filtered_cache"] = None
        print("[WARN] No stable rows found; not writing a filtered cache.", flush=True)

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact_report = {
        key: value
        for key, value in report.items()
        if key not in {"chunk_summaries", "per_grasp"}
    }
    print(json.dumps(compact_report, indent=2, sort_keys=True), flush=True)
    print(f"[INFO] report={report_path}", flush=True)
    return 0 if stable.any() else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)
