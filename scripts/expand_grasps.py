#!/usr/bin/env python3
"""Generate physically validated Sharpa Wave reset variants from stable parents.

Default perturbations match the reference POISE reset-state recipe:

* q noise: 0.025 rad
* q_cmd noise: 0.012 rad
* object position noise: 0.004 m
* object orientation noise: 0.08 rad
* settle: 80 steps
* replay/hold: 240 steps at 120 Hz
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

from poise.paths import workspace_root

POISE_ROOT = workspace_root()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=__doc__,
    allow_abbrev=False,
)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--target-count", type=int, default=2000)
parser.add_argument(
    "--target-per-seed",
    type=int,
    default=0,
    help=(
        "If positive, keep this many validated grasps for each seed row in --seed-cache. "
        "Requires --sample-seed-pool and overrides --target-count to seed_count * target_per_seed."
    ),
)
parser.add_argument(
    "--min-per-seed",
    type=int,
    default=0,
    help=(
        "If positive, require at least this many final grasps from each seed row while still honoring "
        "--target-count. Requires --sample-seed-pool."
    ),
)
parser.add_argument("--max-batches", type=int, default=120)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report-json", type=Path, required=True)
parser.add_argument("--seed-cache", type=Path, required=True)
parser.add_argument("--seed-index", type=int, default=-1)
parser.add_argument(
    "--sample-seed-pool",
    action="store_true",
    help="Sample one seed row per env from the full seed cache instead of perturbing only --seed-index.",
)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--env-spacing", type=float, default=0.36)
parser.add_argument(
    "--hand-root-orientation", choices=("palm_down", "palm_up"), default="palm_up"
)
parser.add_argument("--hand-root-rot-wxyz", type=float, nargs=4, default=None)
parser.add_argument(
    "--hand-usd-path",
    type=Path,
    default=POISE_ROOT / "data/robots/sharpa_wave/right_sharpa_wave.usda",
    help="Public Sharpa hand with the POISE collision overlay.",
)
parser.add_argument(
    "--expected-hand-usd-sha256",
    type=str,
    default="",
    help="Optional lowercase SHA-256 contract for --hand-usd-path.",
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
parser.add_argument(
    "--joint-limit-profile",
    choices=("usd", "usda09", "deployment", "deployment_intersect_usd_scaled"),
    default="usd",
    help=(
        "Joint limits used by PhysX, seed validation, perturbation clamping, and q-margin checks. "
        "'usda09' is the loaded Sharpa USDA signed endpoints multiplied by 0.9; "
        "'deployment' is the exact docs/deploy_dof_limit_joint_ranges.md profile; "
        "'deployment_intersect_usd_scaled' preserves the historical HB1 0.9-USD composite."
    ),
)
parser.add_argument(
    "--use-hb1-deploy-limits",
    action="store_true",
    help=(
        "Deprecated compatibility alias for "
        "--joint-limit-profile deployment_intersect_usd_scaled."
    ),
)
parser.add_argument(
    "--joint-margin-deg",
    type=float,
    default=0.0,
    help="Keep generated q and q_cmd this far inside the active joint limits.",
)
parser.add_argument(
    "--no-clamp-joints",
    action="store_true",
    help=(
        "Do not project noisy q/q_cmd or settled q to active limits. Instead, "
        "sample bounded perturbations by rejection and reject any settled state outside the limits."
    ),
)
parser.add_argument("--q-noise-rad", type=float, default=0.025)
parser.add_argument("--q-cmd-noise-rad", type=float, default=0.012)
parser.add_argument("--object-pos-noise-m", type=float, default=0.004)
parser.add_argument("--object-rot-noise-rad", type=float, default=0.08)
parser.add_argument(
    "--object-mass-kg",
    type=float,
    default=0.0,
    help="Positive value overrides the object mass; zero keeps the asset/config default.",
)
parser.add_argument(
    "--object-friction",
    type=float,
    default=-1.0,
    help="Non-negative value overrides static and dynamic object friction.",
)
parser.add_argument("--settle-steps", type=int, default=80)
parser.add_argument(
    "--replay-steps", type=int, default=240, help="240 steps is 2s at dt=1/120."
)
parser.add_argument("--force-threshold-n", type=float, default=0.05)
parser.add_argument("--min-elastomer-force-contacts", type=int, default=0)
parser.add_argument(
    "--require-thumb-force-contact",
    action="store_true",
    help="Require the thumb elastomer to satisfy --min-force-active-frac.",
)
parser.add_argument("--min-force-active-frac", type=float, default=0.05)
parser.add_argument("--min-surface-contacts", type=int, default=0)
parser.add_argument("--contact-proxy-threshold-m", type=float, default=0.012)
parser.add_argument("--max-any-force-n", type=float, default=80.0)
parser.add_argument("--max-object-drift-m", type=float, default=0.060)
parser.add_argument("--max-lin-vel", type=float, default=0.20)
parser.add_argument("--max-ang-vel", type=float, default=3.0)
parser.add_argument("--min-q-margin-rad", type=float, default=0.003)
parser.add_argument("--min-palm-x", type=float, default=0.025)
parser.add_argument("--max-palm-x", type=float, default=0.18)
parser.add_argument("--max-abs-palm-y", type=float, default=0.14)
parser.add_argument("--min-palm-z", type=float, default=0.02)
parser.add_argument("--max-palm-z", type=float, default=0.22)
parser.add_argument("--write-every", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaacsim.core.utils import prims as prim_utils

from poise.sim.assets.cuboid import get_cuboid_cfg
from poise.sim.assets.cube_4cm import CUBE_4CM_SIDE_LENGTH_M, get_cube_4cm_cfg
from poise.sim.assets.cylinder import get_cylinder_cfg
from poise.sim.assets.irregular import (
    get_irregular_object_cfg,
    is_irregular_object_id,
    sample_irregular_surface_points,
    validate_irregular_dims_m,
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
from poise.sim.joint_mapping import (
    SHARPA_ACTION_DIM,
    SHARPA_DP_BODIES,
    SHARPA_ELASTOMER_BODIES,
    SHARPA_WAVE_RIGHT_HAND_SPEC,
    resolve_hand_mapping,
)
from poise.sim.metrics import (
    cuboid_keypoints_from_pose,
    cuboid_pose_distance,
    cube_keypoints_from_pose,
    cube_pose_distance,
    normalize_quat_wxyz,
    quat_apply_wxyz,
)
from poise.grasp.cache import (
    SCHEMA_VERSION as SHARPA_STABLE_GRASP_SCHEMA_VERSION,
    save_stable_grasp_cache,
    summarize_stable_grasp_cache,
)


CONTACT_BODY_NAMES = tuple(SHARPA_WAVE_RIGHT_HAND_SPEC.contact_sensor_bodies)
ELASTOMER_FORCE_INDICES = tuple(
    CONTACT_BODY_NAMES.index(name) for name in SHARPA_ELASTOMER_BODIES
)
DP_FORCE_INDICES = tuple(CONTACT_BODY_NAMES.index(name) for name in SHARPA_DP_BODIES)
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
    return is_manual_hex_prism_object_id(object_id) or is_manual_convex_object_id(
        object_id
    )


def nominal_cube4cm_dims() -> tuple[float, float, float]:
    return (CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M)


def is_nominal_cube4cm_dims(object_dims_m: tuple[float, float, float]) -> bool:
    return all(
        abs(float(a) - float(b)) <= 1.0e-7
        for a, b in zip(object_dims_m, nominal_cube4cm_dims())
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


def action_to_sim_order(
    action_values: torch.Tensor, action_joint_indices: tuple[int, ...], num_joints: int
) -> torch.Tensor:
    result = torch.zeros(
        action_values.shape[0],
        num_joints,
        device=action_values.device,
        dtype=action_values.dtype,
    )
    result[:, list(action_joint_indices)] = action_values
    return result


def sim_to_action_order(
    sim_values: torch.Tensor, action_joint_indices: tuple[int, ...]
) -> torch.Tensor:
    return sim_values[:, list(action_joint_indices)]


def clamp_joint_targets(
    target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    finite = torch.isfinite(lower) & torch.isfinite(upper)
    return torch.where(
        finite, torch.minimum(torch.maximum(target, lower), upper), target
    )


def sample_bounded_joint_noise(
    center: torch.Tensor,
    std_rad: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample independent Gaussian perturbations without endpoint projection."""

    if float(std_rad) <= 0.0:
        return center.clone()
    result = center + torch.randn(
        center.shape, device=center.device, dtype=center.dtype, generator=generator
    ) * float(std_rad)
    invalid = (result < lower.unsqueeze(0)) | (result > upper.unsqueeze(0))
    attempts = 0
    while bool(invalid.any()):
        attempts += 1
        if attempts > 1000:
            row, joint = torch.nonzero(invalid, as_tuple=False)[0].tolist()
            raise RuntimeError(
                "Unable to sample an in-limit joint perturbation without clamping: "
                f"row={row}, joint={joint}, center={float(center[row, joint]):.9f}, std={float(std_rad):.9f}."
            )
        redraw = center + torch.randn(
            center.shape, device=center.device, dtype=center.dtype, generator=generator
        ) * float(std_rad)
        result = torch.where(invalid, redraw, result)
        invalid = (result < lower.unsqueeze(0)) | (result > upper.unsqueeze(0))
    return result


def resolve_hand_root_rot() -> tuple[float, float, float, float]:
    if args_cli.hand_root_rot_wxyz is not None:
        quat = torch.tensor(args_cli.hand_root_rot_wxyz, dtype=torch.float32).unsqueeze(
            0
        )
        return tuple(float(x) for x in normalize_quat_wxyz(quat)[0].tolist())
    return resolve_sharpa_root_rot(args_cli.hand_root_orientation)


def make_origins(num_envs: int, spacing: float, device: str) -> torch.Tensor:
    cols = max(1, math.ceil(math.sqrt(num_envs)))
    origins = torch.zeros(num_envs, 3, device=device)
    for env_id in range(num_envs):
        origins[env_id, 0] = (env_id % cols) * spacing
        origins[env_id, 1] = (env_id // cols) * spacing
    return origins


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else POISE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_small_quat_wxyz(
    num: int,
    std_rad: float,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if std_rad <= 0.0:
        result = torch.zeros(num, 4, device=device)
        result[:, 0] = 1.0
        return result
    axis = torch.randn(num, 3, device=device, generator=generator)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
    angle = torch.randn(num, device=device, generator=generator) * float(std_rad)
    half = 0.5 * angle
    quat = torch.cat(
        (torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), dim=-1
    )
    return normalize_quat_wxyz(quat)


def hand_to_world_pose(
    object_pose_palm: torch.Tensor, origins: torch.Tensor, hand_quat_wxyz: torch.Tensor
) -> torch.Tensor:
    num = object_pose_palm.shape[0]
    device = object_pose_palm.device
    hand_pos = (
        torch.tensor(
            SHARPA_DEFAULT_ROOT_POS, device=device, dtype=object_pose_palm.dtype
        ).repeat(num, 1)
        + origins
    )
    hand_quat = hand_quat_wxyz.to(device=device, dtype=object_pose_palm.dtype).repeat(
        num, 1
    )
    world_pos = hand_pos + quat_apply_wxyz(hand_quat, object_pose_palm[:, :3])
    world_quat = normalize_quat_wxyz(quat_mul_wxyz(hand_quat, object_pose_palm[:, 3:7]))
    return torch.cat((world_pos, world_quat), dim=-1)


def world_to_hand_pose(
    world_pose: torch.Tensor, origins: torch.Tensor, hand_quat_wxyz: torch.Tensor
) -> torch.Tensor:
    num = world_pose.shape[0]
    device = world_pose.device
    hand_pos = (
        torch.tensor(
            SHARPA_DEFAULT_ROOT_POS, device=device, dtype=world_pose.dtype
        ).repeat(num, 1)
        + origins
    )
    hand_quat = hand_quat_wxyz.to(device=device, dtype=world_pose.dtype).repeat(num, 1)
    inv_hand_quat = quat_conjugate_wxyz(hand_quat)
    local_pos = quat_apply_wxyz(inv_hand_quat, world_pose[:, :3] - hand_pos)
    local_quat = normalize_quat_wxyz(quat_mul_wxyz(inv_hand_quat, world_pose[:, 3:7]))
    return torch.cat((local_pos, local_quat), dim=-1)


def ensure_tensor_rows(payload: dict, key: str, width: int, path: Path) -> torch.Tensor:
    if key not in payload:
        raise KeyError(f"{path} is missing required key {key!r}.")
    tensor = payload[key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{path} key {key!r} must be a torch.Tensor, got {type(tensor)}."
        )
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim != 2 or int(tensor.shape[1]) != width:
        raise ValueError(
            f"{path} key {key!r} has shape {tuple(tensor.shape)}, expected (*, {width})."
        )
    return tensor


def resolve_object_metadata(payload: dict) -> tuple[str, tuple[float, float, float]]:
    object_id = str(payload.get("object_id", "cube_4cm"))
    dims_raw = payload.get("object_dims_m", None)
    if dims_raw is None and isinstance(payload.get("metadata"), dict):
        dims_raw = payload["metadata"].get("object_dims_m", None)
    if object_id == "cube_4cm" and dims_raw is None:
        dims_raw = (
            CUBE_4CM_SIDE_LENGTH_M,
            CUBE_4CM_SIDE_LENGTH_M,
            CUBE_4CM_SIDE_LENGTH_M,
        )
    if dims_raw is None:
        raise ValueError(
            f"Seed cache for object_id={object_id!r} is missing object_dims_m metadata."
        )
    dims_m = tuple(float(v) for v in dims_raw)
    if len(dims_m) != 3 or any(v <= 0.0 for v in dims_m):
        raise ValueError(f"Invalid object_dims_m for {object_id!r}: {dims_m!r}.")
    if is_cuboid_object_id(object_id):
        return object_id, dims_m
    if is_local_y_cylinder_object_id(object_id):
        if abs(float(dims_m[0]) - float(dims_m[2])) > 1.0e-6 or float(
            dims_m[1]
        ) <= float(dims_m[0]):
            raise ValueError(
                f"Local-Y cylinder dimensions must be diameter, length, diameter; got {dims_m!r}."
            )
        return object_id, dims_m
    if is_irregular_object_id(object_id):
        return object_id, validate_irregular_dims_m(dims_m)
    if is_manual_hex_prism_object_id(object_id):
        expected = get_manual_hex_prism_spec(object_id).extents_xyz_m
        if any(
            abs(float(a) - float(b)) > 1.0e-6
            for a, b in zip(dims_m, expected, strict=True)
        ):
            raise ValueError(
                f"Manual hex-prism dimensions must match its fixed USD asset: got {dims_m!r}, expected {expected!r}."
            )
        return object_id, dims_m
    if is_manual_convex_object_id(object_id):
        expected = get_manual_convex_object_spec(object_id).extents_xyz_m
        if any(
            abs(float(a) - float(b)) > 1.0e-6
            for a, b in zip(dims_m, expected, strict=True)
        ):
            raise ValueError(
                f"Manual convex-object dimensions must match its fixed USD asset: got {dims_m!r}, expected {expected!r}."
            )
        return object_id, dims_m
    raise ValueError(
        f"Unsupported seed cache object_id={object_id!r}; expected cube/cuboid, local-Y cylinder, "
        "irregular object, registered manual hex prism, or registered manual convex object."
    )


def load_seed_entry(path: Path, index: int) -> dict[str, torch.Tensor | str]:
    seed_path = resolve_workspace_path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed cache does not exist: {seed_path}")
    payload = torch.load(seed_path, map_location="cpu", weights_only=False)
    object_id, object_dims_m = resolve_object_metadata(payload)
    q_pool = ensure_tensor_rows(payload, "q", SHARPA_ACTION_DIM, seed_path)
    q_cmd_pool = ensure_tensor_rows(payload, "q_cmd", SHARPA_ACTION_DIM, seed_path)
    object_pose_pool = ensure_tensor_rows(payload, "object_pose_palm", 7, seed_path)
    object_pose_pool[:, 3:7] = normalize_quat_wxyz(object_pose_pool[:, 3:7])
    contact_bits_pool = payload.get("contact_bits", None)
    if isinstance(contact_bits_pool, torch.Tensor):
        contact_bits_pool = contact_bits_pool.detach().cpu().to(dtype=torch.bool)
        if contact_bits_pool.ndim != 2 or int(contact_bits_pool.shape[1]) != 5:
            raise ValueError(
                f"{seed_path} key 'contact_bits' has shape {tuple(contact_bits_pool.shape)}, expected (*, 5)."
            )
    else:
        contact_bits_pool = torch.ones(q_pool.shape[0], 5, dtype=torch.bool)
    surface_contact_bits_pool = payload.get("surface_contact_bits", None)
    if isinstance(surface_contact_bits_pool, torch.Tensor):
        surface_contact_bits_pool = (
            surface_contact_bits_pool.detach().cpu().to(dtype=torch.bool)
        )
        if (
            surface_contact_bits_pool.ndim != 2
            or int(surface_contact_bits_pool.shape[1]) != 5
        ):
            raise ValueError(
                f"{seed_path} key 'surface_contact_bits' has shape {tuple(surface_contact_bits_pool.shape)}, expected (*, 5)."
            )
    else:
        surface_contact_bits_pool = contact_bits_pool.clone()
    stability_score_pool = payload.get("stability_score", None)
    if isinstance(stability_score_pool, torch.Tensor):
        stability_score_pool = (
            stability_score_pool.detach().cpu().to(dtype=torch.float32).reshape(-1)
        )
        if int(stability_score_pool.shape[0]) != int(q_pool.shape[0]):
            raise ValueError(
                f"{seed_path} key 'stability_score' has length {int(stability_score_pool.shape[0])}, "
                f"expected {int(q_pool.shape[0])}."
            )
    else:
        stability_score_pool = (
            contact_bits_pool.sum(dim=-1).to(dtype=torch.float32) * 100.0
        )
    count = int(q_pool.shape[0])
    if count <= 0:
        raise ValueError(f"{seed_path} contains no seed entries.")
    if index < 0:
        index = count + index
    if index < 0 or index >= count:
        raise IndexError(
            f"--seed-index {index} is out of range for {seed_path} with {count} entries."
        )
    object_pose_palm = object_pose_pool[index].clone()
    object_pose_palm[3:7] = normalize_quat_wxyz(object_pose_palm[3:7].reshape(1, 4))[
        0
    ].cpu()
    return {
        "path": str(seed_path),
        "index": torch.tensor(index, dtype=torch.long),
        "count": torch.tensor(count, dtype=torch.long),
        "q": q_pool[index],
        "q_cmd": q_cmd_pool[index],
        "object_pose_palm": object_pose_palm,
        "q_pool": q_pool,
        "q_cmd_pool": q_cmd_pool,
        "object_pose_palm_pool": object_pose_pool,
        "contact_bits_pool": contact_bits_pool,
        "surface_contact_bits_pool": surface_contact_bits_pool,
        "stability_score_pool": stability_score_pool,
        "object_id": object_id,
        "object_dims_m": object_dims_m,
    }


def cuboid_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    dims_m: tuple[float, float, float],
) -> torch.Tensor:
    object_pos = object_pose_w[:, None, :3]
    object_quat_inv = quat_conjugate_wxyz(normalize_quat_wxyz(object_pose_w[:, 3:7]))
    rel_w = body_pos_w - object_pos
    rel_obj = quat_apply_wxyz(
        object_quat_inv[:, None, :].expand(-1, rel_w.shape[1], -1), rel_w
    )
    half_dims = (
        torch.tensor(dims_m, device=body_pos_w.device, dtype=body_pos_w.dtype) * 0.5
    )
    outside = torch.clamp(torch.abs(rel_obj) - half_dims.view(1, 1, 3), min=0.0)
    return torch.linalg.norm(outside, dim=-1)


def cube_surface_distances(
    body_pos_w: torch.Tensor, cube_pose_w: torch.Tensor
) -> torch.Tensor:
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
    rel_obj = quat_apply_wxyz(
        object_quat_inv[:, None, :].expand(-1, rel_w.shape[1], -1), rel_w
    )
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
        [
            [x, y, z]
            for x in (-radius, radius)
            for y in (-half_length, half_length)
            for z in (-radius, radius)
        ],
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


def transform_local_points_wxyz(
    points: torch.Tensor, pose: torch.Tensor
) -> torch.Tensor:
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
    local_points = sample_irregular_surface_points(
        object_id, object_dims_m, samples_per_edge=7
    )
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
    local_points = sample_irregular_surface_points(
        object_id, object_dims_m, samples_per_edge=5
    )
    current = transform_local_points_wxyz(local_points, current_pose)
    goal = transform_local_points_wxyz(local_points, goal_pose)
    return torch.linalg.norm(current - goal, dim=-1).mean(dim=-1)


@functools.lru_cache(
    maxsize=len(SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS)
    + len(SHARPA_MANUAL_CONVEX_OBJECT_IDS)
)
def load_imported_convex_mesh_triangles(object_id: str) -> torch.Tensor:
    """Load the exact triangulated chamfered-prism mesh in object-local meters."""

    asset_dir = (
        get_manual_hex_prism_asset_dir(object_id)
        if is_manual_hex_prism_object_id(object_id)
        else get_manual_convex_object_asset_dir(object_id)
    )
    obj_path = asset_dir / f"{object_id}.obj"
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw_line in obj_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.strip().split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f" and len(fields) == 4:
            faces.append(
                tuple(int(value.split("/", 1)[0]) - 1 for value in fields[1:4])
            )
    if not vertices or not faces:
        raise ValueError(f"Imported convex OBJ has no triangulated mesh: {obj_path}")
    triangles = torch.tensor(vertices, dtype=torch.float32)[
        torch.tensor(faces, dtype=torch.long)
    ]
    if not torch.isfinite(triangles).all():
        raise ValueError(
            f"Imported convex OBJ contains non-finite coordinates: {obj_path}"
        )
    return triangles


def point_to_triangle_surface_distances(
    points: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    """Exact unsigned distance from batched points to a closed triangle surface."""

    triangles = triangles.to(device=points.device, dtype=points.dtype)
    point = points.unsqueeze(-2)
    view = (*([1] * (points.ndim - 1)), -1, 3)
    a, b, c = (triangles[:, index].view(view) for index in range(3))

    def segment_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        edge = end - start
        denom = torch.clamp((edge * edge).sum(dim=-1), min=1.0e-16)
        fraction = torch.clamp(((point - start) * edge).sum(dim=-1) / denom, 0.0, 1.0)
        closest = start + fraction.unsqueeze(-1) * edge
        return torch.linalg.norm(point - closest, dim=-1)

    ab, ac = b - a, c - a
    normal = torch.linalg.cross(ab, ac, dim=-1)
    normal_sq = torch.clamp((normal * normal).sum(dim=-1), min=1.0e-20)
    signed_numerator = ((point - a) * normal).sum(dim=-1)
    projected = point - (signed_numerator / normal_sq).unsqueeze(-1) * normal
    relative = projected - a
    d00, d01, d11 = (ab * ab).sum(dim=-1), (ab * ac).sum(dim=-1), (ac * ac).sum(dim=-1)
    d20, d21 = (relative * ab).sum(dim=-1), (relative * ac).sum(dim=-1)
    denominator = torch.clamp(d00 * d11 - d01 * d01, min=1.0e-20)
    bary_v = (d11 * d20 - d01 * d21) / denominator
    bary_w = (d00 * d21 - d01 * d20) / denominator
    bary_u = 1.0 - bary_v - bary_w
    inside = (bary_u >= -1.0e-6) & (bary_v >= -1.0e-6) & (bary_w >= -1.0e-6)
    plane_distance = signed_numerator.abs() / torch.sqrt(normal_sq)
    plane_distance = torch.where(
        inside, plane_distance, torch.full_like(plane_distance, torch.inf)
    )
    triangle_distance = torch.minimum(
        plane_distance,
        torch.minimum(
            segment_distance(a, b),
            torch.minimum(segment_distance(b, c), segment_distance(c, a)),
        ),
    )
    return triangle_distance.amin(dim=-1)


def manual_hex_prism_surface_distances(
    body_pos_w: torch.Tensor,
    object_pose_w: torch.Tensor,
    object_id: str,
) -> torch.Tensor:
    object_quat_inv = quat_conjugate_wxyz(normalize_quat_wxyz(object_pose_w[:, 3:7]))
    relative_w = body_pos_w - object_pose_w[:, None, :3]
    object_quat_inv = object_quat_inv[:, None, :].expand(relative_w.shape[:-1] + (4,))
    body_pos_object = quat_apply_wxyz(object_quat_inv, relative_w)
    return point_to_triangle_surface_distances(
        body_pos_object, load_imported_convex_mesh_triangles(object_id)
    )


def manual_hex_prism_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    object_id: str,
) -> torch.Tensor:
    local_vertices = torch.unique(
        load_imported_convex_mesh_triangles(object_id).reshape(-1, 3), dim=0
    )
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
        return cylinder_local_y_surface_distances(
            body_pos_w, object_pose_w, object_dims_m
        )
    if is_irregular_object_id(object_id):
        return irregular_surface_distances(
            body_pos_w, object_pose_w, object_id, object_dims_m
        )
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
        return cylinder_local_y_bbox_pose_distance(
            current_pose, goal_pose, object_dims_m
        )
    if is_irregular_object_id(object_id):
        return irregular_surface_pose_distance(
            current_pose, goal_pose, object_id, object_dims_m
        )
    if is_imported_convex_mesh_object_id(object_id):
        return manual_hex_prism_pose_distance(current_pose, goal_pose, object_id)
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def object_keypoints_from_pose(
    pose: torch.Tensor,
    object_id: str,
    object_dims_m: tuple[float, float, float],
) -> torch.Tensor:
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return cube_keypoints_from_pose(pose, side_length=CUBE_4CM_SIDE_LENGTH_M)
    if is_cuboid_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    if is_local_y_cylinder_object_id(object_id):
        return cylinder_local_y_bbox_keypoints_from_pose(pose, object_dims_m)
    if is_irregular_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    if is_imported_convex_mesh_object_id(object_id):
        return cuboid_keypoints_from_pose(pose, object_dims_m)
    raise ValueError(f"Unsupported object_id={object_id!r}.")


def get_object_cfg(object_id: str, object_dims_m: tuple[float, float, float]):
    if object_id == "cube_4cm" and is_nominal_cube4cm_dims(object_dims_m):
        return get_cube_4cm_cfg(
            prim_path="/World/Env_.*/Cube",
            init_pos=(0.0, 0.0, 0.0),
            disable_gravity=False,
        )
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
        if any(
            abs(float(a) - float(b)) > 1.0e-6
            for a, b in zip(object_dims_m, expected, strict=True)
        ):
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
        if any(
            abs(float(a) - float(b)) > 1.0e-6
            for a, b in zip(object_dims_m, expected, strict=True)
        ):
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


def step_sim(
    sim: SimulationContext,
    hand: Articulation,
    cube: RigidObject,
    target: torch.Tensor,
    steps: int,
) -> None:
    dt = sim.get_physics_dt()
    for _ in range(max(0, int(steps))):
        hand.set_joint_position_target(target)
        hand.write_data_to_sim()
        cube.write_data_to_sim()
        sim.step()
        hand.update(dt)
        cube.update(dt)


def reset_candidates(
    sim: SimulationContext,
    hand: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
    mapping,
    origins: torch.Tensor,
    hand_quat_wxyz: torch.Tensor,
    q_action: torch.Tensor,
    q_cmd_action: torch.Tensor,
    object_pose_palm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = q_action.shape[0]
    q_sim = action_to_sim_order(
        q_action, mapping.action_joint_indices, len(hand.joint_names)
    )
    q_cmd_sim = action_to_sim_order(
        q_cmd_action, mapping.action_joint_indices, len(hand.joint_names)
    )
    object_pose_w = hand_to_world_pose(object_pose_palm, origins, hand_quat_wxyz)

    root_state = hand.data.default_root_state.clone()
    root_state[:, :3] = (
        torch.tensor(SHARPA_DEFAULT_ROOT_POS, device=sim.device, dtype=root_state.dtype)
        + origins
    )
    root_state[:, 3:7] = hand_quat_wxyz.to(
        device=sim.device, dtype=root_state.dtype
    ).repeat(num_envs, 1)
    hand.write_root_pose_to_sim(root_state[:, :7])
    hand.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    hand.write_joint_state_to_sim(q_sim, torch.zeros_like(q_sim))
    hand.set_joint_position_target(q_cmd_sim)
    cube.write_root_pose_to_sim(object_pose_w)
    cube.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=sim.device))
    hand.write_data_to_sim()
    cube.write_data_to_sim()
    sim.step()
    hand.update(sim.get_physics_dt())
    cube.update(sim.get_physics_dt())
    sensor.reset()
    hand.reset()
    cube.reset()
    return q_cmd_sim, object_pose_w


def replay_and_measure(
    sim: SimulationContext,
    hand: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
    q_cmd_sim: torch.Tensor,
    steps: int,
    threshold_n: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = sim.get_physics_dt()
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError("ContactSensor force_matrix_w is None.")
    force_sum = torch.zeros(
        force_matrix.shape[0], force_matrix.shape[2], device=sim.device
    )
    force_max = torch.zeros_like(force_sum)
    active_steps = torch.zeros_like(force_sum)
    count = max(1, int(steps))
    for _ in range(count):
        hand.set_joint_position_target(q_cmd_sim)
        hand.write_data_to_sim()
        cube.write_data_to_sim()
        sim.step()
        hand.update(dt)
        cube.update(dt)
        sensor.update(dt, force_recompute=True)
        force_norm = torch.linalg.norm(sensor.data.force_matrix_w[:, 0, :, :], dim=-1)
        force_sum += force_norm
        force_max = torch.maximum(force_max, force_norm)
        active_steps += (force_norm > threshold_n).to(dtype=torch.float32)
    return force_sum / float(count), force_max, active_steps / float(count)


def q_margin(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    margin = torch.minimum(q - lower.unsqueeze(0), upper.unsqueeze(0) - q)
    margin = torch.where(torch.isfinite(margin), margin, torch.full_like(margin, 1.0))
    return margin.min(dim=-1).values


def build_cache(saved: list[dict[str, torch.Tensor]], metadata: dict) -> dict:
    object_id = str(metadata.get("object_id", "cube_4cm"))
    object_dims_m = tuple(
        float(v) for v in metadata.get("object_dims_m", (CUBE_4CM_SIDE_LENGTH_M,) * 3)
    )
    if not saved:
        empty_q = torch.empty(0, SHARPA_ACTION_DIM)
        empty_pose = torch.empty(0, 7)
        empty_keypoints = object_keypoints_from_pose(
            empty_pose, object_id, object_dims_m
        )
        return {
            "schema_version": SHARPA_STABLE_GRASP_SCHEMA_VERSION,
            "hand": "sharpa_wave_right",
            "action_dim": SHARPA_ACTION_DIM,
            "object_id": object_id,
            "object_dims_m": object_dims_m,
            "q": empty_q,
            "qdot": empty_q.clone(),
            "q_cmd": empty_q.clone(),
            "object_pose_palm": empty_pose,
            "cube_keypoints_palm": empty_keypoints,
            "contact_bits": torch.empty(0, 5, dtype=torch.bool),
            "stability_score": torch.empty(0),
            "source_seed_index": torch.empty(0, dtype=torch.long),
            "metadata": metadata,
        }
    q = torch.cat([item["q"] for item in saved], dim=0)
    qdot = torch.cat([item["qdot"] for item in saved], dim=0)
    q_cmd = torch.cat([item["q_cmd"] for item in saved], dim=0)
    pose = torch.cat([item["object_pose_palm"] for item in saved], dim=0)
    contact_bits = torch.cat([item["contact_bits"] for item in saved], dim=0)
    score = torch.cat([item["score"] for item in saved], dim=0)
    source_seed_index = (
        torch.cat([item["source_seed_index"] for item in saved], dim=0)
        if "source_seed_index" in saved[0]
        else None
    )
    min_per_seed = int(args_cli.min_per_seed)
    seed_count = int(metadata.get("seed_count", 0))
    if min_per_seed > 0 and source_seed_index is not None and seed_count > 0:
        selected = torch.zeros(score.shape[0], dtype=torch.bool)
        forced_indices: list[int] = []
        for seed_id in range(seed_count):
            seed_rows = torch.nonzero(
                source_seed_index == seed_id, as_tuple=False
            ).squeeze(-1)
            if seed_rows.numel() == 0:
                continue
            seed_order = seed_rows[torch.argsort(score[seed_rows], descending=True)]
            forced = seed_order[: min(min_per_seed, int(seed_order.numel()))]
            selected[forced] = True
            forced_indices.extend(int(idx) for idx in forced.tolist())
        remaining_slots = max(
            0, int(args_cli.target_count) - int(selected.sum().item())
        )
        fill_indices: list[int] = []
        if remaining_slots > 0:
            for idx in torch.argsort(score, descending=True).tolist():
                if not bool(selected[idx]):
                    fill_indices.append(int(idx))
                    selected[idx] = True
                    if len(fill_indices) >= remaining_slots:
                        break
        keep = torch.tensor(forced_indices + fill_indices, dtype=torch.long)
        if keep.numel() > int(args_cli.target_count):
            keep = keep[: int(args_cli.target_count)]
        keep = keep[torch.argsort(score[keep], descending=True)]
    else:
        order = torch.argsort(score, descending=True)
        keep = order[: args_cli.target_count]
    cache = {
        "schema_version": SHARPA_STABLE_GRASP_SCHEMA_VERSION,
        "hand": "sharpa_wave_right",
        "action_dim": SHARPA_ACTION_DIM,
        "object_id": object_id,
        "object_dims_m": object_dims_m,
        "q": q[keep].cpu(),
        "qdot": qdot[keep].cpu(),
        "q_cmd": q_cmd[keep].cpu(),
        "object_pose_palm": pose[keep].cpu(),
        "cube_keypoints_palm": object_keypoints_from_pose(
            pose[keep], object_id, object_dims_m
        ).cpu(),
        "contact_bits": contact_bits[keep].cpu(),
        "stability_score": score[keep].cpu(),
        "metadata": metadata,
        "elastomer_force_mean_n": torch.cat(
            [item["elastomer_force_mean_n"] for item in saved], dim=0
        )[keep].cpu(),
        "elastomer_force_max_n": torch.cat(
            [item["elastomer_force_max_n"] for item in saved], dim=0
        )[keep].cpu(),
        "elastomer_force_active_frac": torch.cat(
            [item["elastomer_force_active_frac"] for item in saved], dim=0
        )[keep].cpu(),
        "surface_contact_bits": torch.cat(
            [item["surface_contact_bits"] for item in saved], dim=0
        )[keep].cpu(),
        "surface_dist_m": torch.cat([item["surface_dist_m"] for item in saved], dim=0)[
            keep
        ].cpu(),
        "dp_force_max_n": torch.cat([item["dp_force_max_n"] for item in saved], dim=0)[
            keep
        ].cpu(),
        "q_margin_rad": torch.cat([item["q_margin_rad"] for item in saved], dim=0)[
            keep
        ].cpu(),
        "replay_drift_m": torch.cat([item["replay_drift_m"] for item in saved], dim=0)[
            keep
        ].cpu(),
        "replay_lin_vel_mps": torch.cat(
            [item["replay_lin_vel_mps"] for item in saved], dim=0
        )[keep].cpu(),
        "replay_ang_vel_radps": torch.cat(
            [item["replay_ang_vel_radps"] for item in saved], dim=0
        )[keep].cpu(),
    }
    if source_seed_index is not None:
        cache["source_seed_index"] = source_seed_index[keep].cpu()
    return cache


def write_report(cache: dict, batch_summaries: list[dict]) -> Path:
    score = cache["stability_score"]
    top = []
    if score.numel() > 0:
        order = torch.argsort(score, descending=True)[: min(20, score.numel())]
        for rank, idx in enumerate(order.tolist(), start=1):
            top.append(
                {
                    "rank": rank,
                    "score": round(float(score[idx].item()), 6),
                    "q": [round(float(x), 6) for x in cache["q"][idx].tolist()],
                    "q_cmd": [round(float(x), 6) for x in cache["q_cmd"][idx].tolist()],
                    "object_pose_palm": [
                        round(float(x), 6)
                        for x in cache["object_pose_palm"][idx].tolist()
                    ],
                    "contact_bits": [
                        bool(x) for x in cache["contact_bits"][idx].tolist()
                    ],
                    "surface_contact_bits": [
                        bool(x) for x in cache["surface_contact_bits"][idx].tolist()
                    ],
                    "elastomer_force_mean_n": [
                        round(float(x), 5)
                        for x in cache["elastomer_force_mean_n"][idx].tolist()
                    ],
                    "elastomer_force_max_n": [
                        round(float(x), 5)
                        for x in cache["elastomer_force_max_n"][idx].tolist()
                    ],
                    "elastomer_force_active_frac": [
                        round(float(x), 4)
                        for x in cache["elastomer_force_active_frac"][idx].tolist()
                    ],
                    "replay_drift_m": round(
                        float(cache["replay_drift_m"][idx].item()), 6
                    ),
                    "replay_lin_vel_mps": round(
                        float(cache["replay_lin_vel_mps"][idx].item()), 6
                    ),
                    "replay_ang_vel_radps": round(
                        float(cache["replay_ang_vel_radps"][idx].item()), 6
                    ),
                    "q_margin_rad": round(float(cache["q_margin_rad"][idx].item()), 6),
                }
            )
    report = {
        "metadata": cache["metadata"],
        "summary": summarize_stable_grasp_cache(cache),
        "batch_summaries": batch_summaries,
        "top": top,
    }
    if "source_seed_index" in cache:
        seed_ids = cache["source_seed_index"].to(dtype=torch.long)
        seed_count = int(cache["metadata"].get("seed_count", 0))
        if seed_count > 0:
            counts = torch.bincount(seed_ids, minlength=seed_count)
            report["per_seed_counts"] = {
                "min": int(counts.min().item()) if counts.numel() else 0,
                "max": int(counts.max().item()) if counts.numel() else 0,
                "mean": (
                    float(counts.to(dtype=torch.float32).mean().item())
                    if counts.numel()
                    else 0.0
                ),
                "counts": [int(x) for x in counts.tolist()],
            }
    report_path = resolve_workspace_path(args_cli.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    if args_cli.num_envs < 1:
        raise ValueError("--num-envs must be positive.")
    if args_cli.target_count < 1:
        raise ValueError("--target-count must be positive.")
    if args_cli.joint_margin_deg < 0.0:
        raise ValueError("--joint-margin-deg must be non-negative.")
    hand_usd_path = None
    hand_usd_sha256 = None
    if args_cli.hand_usd_path is not None:
        hand_usd_path = (
            resolve_workspace_path(args_cli.hand_usd_path).expanduser().resolve()
        )
        if not hand_usd_path.is_file():
            raise FileNotFoundError(f"Sharpa hand USD not found: {hand_usd_path}")
        hand_usd_sha256 = sha256_file(hand_usd_path)
    expected_hand_usd_sha256 = str(args_cli.expected_hand_usd_sha256).strip().lower()
    if expected_hand_usd_sha256 and hand_usd_path is None:
        raise ValueError("--expected-hand-usd-sha256 requires --hand-usd-path.")
    if expected_hand_usd_sha256 and hand_usd_sha256 != expected_hand_usd_sha256:
        raise RuntimeError(
            "Sharpa hand USDA SHA-256 mismatch: "
            f"expected={expected_hand_usd_sha256}, actual={hand_usd_sha256}, path={hand_usd_path}"
        )
    joint_limit_reference_usd = (
        resolve_workspace_path(args_cli.joint_limit_reference_usd)
        .expanduser()
        .resolve()
        if args_cli.joint_limit_reference_usd is not None
        else None
    )
    if (
        joint_limit_reference_usd is not None
        and not joint_limit_reference_usd.is_file()
    ):
        raise FileNotFoundError(
            f"Joint-limit reference USDA not found: {joint_limit_reference_usd}"
        )
    joint_limit_reference_usd_sha256 = (
        sha256_file(joint_limit_reference_usd)
        if joint_limit_reference_usd is not None
        else None
    )
    expected_joint_limit_reference_usd_sha256 = (
        str(args_cli.expected_joint_limit_reference_usd_sha256).strip().lower()
    )
    if expected_joint_limit_reference_usd_sha256 and joint_limit_reference_usd is None:
        raise ValueError(
            "--expected-joint-limit-reference-usd-sha256 requires --joint-limit-reference-usd."
        )
    if (
        expected_joint_limit_reference_usd_sha256
        and joint_limit_reference_usd_sha256
        != expected_joint_limit_reference_usd_sha256
    ):
        raise RuntimeError(
            "Joint-limit reference USDA SHA-256 mismatch: "
            f"expected={expected_joint_limit_reference_usd_sha256}, "
            f"actual={joint_limit_reference_usd_sha256}, path={joint_limit_reference_usd}"
        )
    joint_limit_profile = str(args_cli.joint_limit_profile)
    if joint_limit_profile == "usda09" and joint_limit_reference_usd is None:
        raise ValueError(
            "--joint-limit-profile usda09 requires --joint-limit-reference-usd."
        )
    if args_cli.use_hb1_deploy_limits:
        if joint_limit_profile != "usd":
            raise ValueError(
                "Do not combine --use-hb1-deploy-limits with --joint-limit-profile."
            )
        joint_limit_profile = "deployment_intersect_usd_scaled"
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0, device=args_cli.device, use_fabric=False
        )
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0)
    light_cfg.func("/World/Light", light_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    origins = make_origins(args_cli.num_envs, args_cli.env_spacing, sim.device)
    for env_id, origin in enumerate(origins):
        prim_utils.create_prim(
            f"/World/Env_{env_id}", "Xform", translation=origin.tolist()
        )

    hand_quat = torch.tensor(
        resolve_hand_root_rot(), device=sim.device, dtype=torch.float32
    )
    seed_entry = load_seed_entry(args_cli.seed_cache, args_cli.seed_index)
    object_id = str(seed_entry["object_id"])
    object_dims_m = tuple(float(v) for v in seed_entry["object_dims_m"])
    hand = Articulation(
        get_sharpa_wave_hand_cfg(
            prim_path="/World/Env_.*/Robot",
            init_pos=SHARPA_DEFAULT_ROOT_POS,
            init_rot=tuple(float(x) for x in hand_quat.cpu()),
            usd_path=hand_usd_path,
        )
    )
    cube_cfg = get_object_cfg(object_id, object_dims_m)
    if args_cli.object_mass_kg > 0.0:
        cube_cfg.spawn.mass_props.mass = float(args_cli.object_mass_kg)
    # Primitive spawn configs expose an inline physics_material field, whereas
    # IsaacLab 2.3 UsdFileCfg does not. Imported USD objects (including the
    # compound hex-prism hammer) therefore need an explicitly bound material.
    has_inline_physics_material = hasattr(cube_cfg.spawn, "physics_material")
    if args_cli.object_friction >= 0.0 and has_inline_physics_material:
        cube_cfg.spawn.physics_material.static_friction = float(
            args_cli.object_friction
        )
        cube_cfg.spawn.physics_material.dynamic_friction = float(
            args_cli.object_friction
        )
    cube_cfg.spawn.activate_contact_sensors = True
    cube = RigidObject(cube_cfg)
    if not has_inline_physics_material:
        friction = (
            float(args_cli.object_friction) if args_cli.object_friction >= 0.0 else 1.0
        )
        material_path = "/World/ImportedObjectPhysicsMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        )
        material_cfg.func(material_path, material_cfg)
        for env_id in range(args_cli.num_envs):
            sim_utils.bind_physics_material(f"/World/Env_{env_id}/Cube", material_path)
    sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Env_.*/Cube",
            update_period=0.0,
            debug_vis=False,
            filter_prim_paths_expr=[
                f"/World/Env_.*/Robot/{body_name}" for body_name in CONTACT_BODY_NAMES
            ],
        )
    )

    sim.reset()
    mapping = resolve_hand_mapping(
        SHARPA_WAVE_RIGHT_HAND_SPEC, hand.joint_names, hand.body_names
    )
    missing = sorted(set(CONTACT_BODY_NAMES) - set(hand.body_names))
    if missing:
        raise RuntimeError(f"Missing contact bodies: {missing}")
    if sensor.contact_physx_view.filter_count != len(CONTACT_BODY_NAMES):
        raise RuntimeError(
            f"Resolved contact filter_count={sensor.contact_physx_view.filter_count}, expected {len(CONTACT_BODY_NAMES)}."
        )
    elastomer_body_indices = mapping_body_indices(hand, SHARPA_ELASTOMER_BODIES)

    generator = torch.Generator(device=sim.device).manual_seed(int(args_cli.seed))
    raw_limits = (
        hand.data.soft_joint_pos_limits[0, list(mapping.action_joint_indices)]
        .to(sim.device)
        .clone()
    )
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
                [
                    SHARPA_DEPLOY_DOF_LIMITS_DEG[name]
                    for name in SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order
                ],
                device=sim.device,
                dtype=raw_limits.dtype,
            )
        )
        joint_limit_rule = "docs_deployment_profile_exact"
    else:
        active_limits = composite_deploy_usd_limits(
            raw_limits,
            SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order,
            SHARPA_DEPLOY_DOF_LIMITS_DEG,
            usd_endpoint_scale=0.9,
        )
        joint_limit_rule = "deployment_profile_intersect_usd_endpoints_times_0.9"
    if joint_limit_profile != "usd":
        hand.write_joint_position_limit_to_sim(
            active_limits.unsqueeze(0).expand(args_cli.num_envs, -1, -1),
            joint_ids=list(mapping.action_joint_indices),
            warn_limit_violation=False,
        )
    lower = active_limits[:, 0]
    upper = active_limits[:, 1]
    margin_rad = math.radians(float(args_cli.joint_margin_deg))
    clamp_lower = lower + margin_rad
    clamp_upper = upper - margin_rad
    if bool((clamp_lower >= clamp_upper).any()):
        raise ValueError(
            f"--joint-margin-deg={args_cli.joint_margin_deg} leaves an empty joint interval."
        )
    seed_pool_q = seed_entry["q_pool"].to(device=sim.device)
    seed_pool_q_cmd = seed_entry["q_cmd_pool"].to(device=sim.device)
    for label, values in (("q", seed_pool_q), ("q_cmd", seed_pool_q_cmd)):
        outside = (values < clamp_lower.unsqueeze(0) - 1.0e-7) | (
            values > clamp_upper.unsqueeze(0) + 1.0e-7
        )
        if bool(outside.any()):
            row, joint = torch.nonzero(outside, as_tuple=False)[0].tolist()
            raise ValueError(
                f"Seed {label} violates joint-limit profile={joint_limit_profile}: "
                f"row={row}, joint={SHARPA_WAVE_RIGHT_HAND_SPEC.joint_order[joint]}, "
                f"value={float(values[row, joint]):.9f}, "
                f"interval=[{float(clamp_lower[joint]):.9f}, {float(clamp_upper[joint]):.9f}]."
            )
    center_q = seed_entry["q"].to(device=sim.device).clone()
    center_q_cmd = seed_entry["q_cmd"].to(device=sim.device).clone()
    center_object_pose_palm = seed_entry["object_pose_palm"].to(device=sim.device)
    seed_pool_object_pose_palm = seed_entry["object_pose_palm_pool"].to(
        device=sim.device
    )
    target_per_seed = int(args_cli.target_per_seed)
    min_per_seed = int(args_cli.min_per_seed)
    kept_per_seed = torch.zeros(seed_pool_q.shape[0], dtype=torch.long)
    if min_per_seed > 0:
        if not args_cli.sample_seed_pool:
            raise ValueError("--min-per-seed requires --sample-seed-pool.")
        required_min = int(seed_pool_q.shape[0]) * min_per_seed
        if int(args_cli.target_count) < required_min:
            raise ValueError(
                f"--target-count={args_cli.target_count} is smaller than seed_count*min_per_seed={required_min}."
            )
    if target_per_seed > 0:
        if not args_cli.sample_seed_pool:
            raise ValueError("--target-per-seed requires --sample-seed-pool.")
        args_cli.target_count = int(seed_pool_q.shape[0]) * target_per_seed
    saved: list[dict[str, torch.Tensor]] = []
    batch_summaries: list[dict] = []
    total_kept = 0

    metadata = {
        "source": "scripts/expand_grasps.py",
        "object_id": object_id,
        "object_dims_m": object_dims_m,
        "seed_cache": seed_entry["path"],
        "seed_index": int(seed_entry["index"].item()),
        "seed_count": int(seed_entry["count"].item()),
        "sample_seed_pool": bool(args_cli.sample_seed_pool),
        "target_per_seed": target_per_seed,
        "min_per_seed": min_per_seed,
        "target_count": int(args_cli.target_count),
        "hand_root_orientation": args_cli.hand_root_orientation,
        "hand_root_quat_wxyz": tuple(float(x) for x in hand_quat.cpu().tolist()),
        "hand_usd_path": str(hand_usd_path) if hand_usd_path is not None else None,
        "hand_usd_sha256": hand_usd_sha256,
        "expected_hand_usd_sha256": expected_hand_usd_sha256 or None,
        "joint_limit_reference_usd_path": (
            str(joint_limit_reference_usd)
            if joint_limit_reference_usd is not None
            else None
        ),
        "joint_limit_reference_usd_sha256": joint_limit_reference_usd_sha256,
        "expected_joint_limit_reference_usd_sha256": (
            expected_joint_limit_reference_usd_sha256 or None
        ),
        "hb1_deploy_limits_enabled": joint_limit_profile != "usd",
        "joint_limit_profile": joint_limit_profile,
        "joint_limit_rule": joint_limit_rule,
        "joint_limits_rad": [
            [float(v) for v in row] for row in active_limits.detach().cpu().tolist()
        ],
        "joint_margin_deg": float(args_cli.joint_margin_deg),
        "joint_projection": {
            "enabled": not bool(args_cli.no_clamp_joints),
            "mode": (
                "endpoint_clamp"
                if not args_cli.no_clamp_joints
                else "bounded_rejection_sampling"
            ),
        },
        "center_q": tuple(float(x) for x in center_q.detach().cpu().tolist()),
        "center_q_cmd": tuple(float(x) for x in center_q_cmd.detach().cpu().tolist()),
        "center_object_pose_palm": tuple(
            float(x) for x in center_object_pose_palm.detach().cpu().tolist()
        ),
        "q_noise_rad": float(args_cli.q_noise_rad),
        "q_cmd_noise_rad": float(args_cli.q_cmd_noise_rad),
        "object_pos_noise_m": float(args_cli.object_pos_noise_m),
        "object_rot_noise_rad": float(args_cli.object_rot_noise_rad),
        "object_mass_kg": float(args_cli.object_mass_kg),
        "object_friction": float(args_cli.object_friction),
        "settle_steps": int(args_cli.settle_steps),
        "replay_steps": int(args_cli.replay_steps),
        "replay_seconds": float(args_cli.replay_steps / 120.0),
        "filters": {
            "force_threshold_n": float(args_cli.force_threshold_n),
            "min_elastomer_force_contacts": int(args_cli.min_elastomer_force_contacts),
            "require_thumb_force_contact": bool(args_cli.require_thumb_force_contact),
            "min_force_active_frac": float(args_cli.min_force_active_frac),
            "min_surface_contacts": int(args_cli.min_surface_contacts),
            "contact_proxy_threshold_m": float(args_cli.contact_proxy_threshold_m),
            "max_any_force_n": float(args_cli.max_any_force_n),
            "max_object_drift_m": float(args_cli.max_object_drift_m),
            "max_lin_vel": float(args_cli.max_lin_vel),
            "max_ang_vel": float(args_cli.max_ang_vel),
            "min_q_margin_rad": float(args_cli.min_q_margin_rad),
            "workspace_palm_xyz": {
                "min_x": float(args_cli.min_palm_x),
                "max_x": float(args_cli.max_palm_x),
                "max_abs_y": float(args_cli.max_abs_palm_y),
                "min_z": float(args_cli.min_palm_z),
                "max_z": float(args_cli.max_palm_z),
            },
        },
    }

    for batch_idx in range(args_cli.max_batches):
        num = args_cli.num_envs
        if args_cli.sample_seed_pool:
            if target_per_seed > 0:
                active_seed_ids = torch.nonzero(
                    kept_per_seed < target_per_seed, as_tuple=False
                ).squeeze(-1)
                if active_seed_ids.numel() == 0:
                    break
                active_seed_ids = active_seed_ids.to(device=sim.device)
                seed_ids = active_seed_ids[
                    torch.randint(
                        active_seed_ids.numel(),
                        (num,),
                        device=sim.device,
                        generator=generator,
                    )
                ]
            elif min_per_seed > 0 and bool((kept_per_seed < min_per_seed).any()):
                active_seed_ids = (
                    torch.nonzero(kept_per_seed < min_per_seed, as_tuple=False)
                    .squeeze(-1)
                    .to(device=sim.device)
                )
                seed_ids = active_seed_ids[
                    torch.randint(
                        active_seed_ids.numel(),
                        (num,),
                        device=sim.device,
                        generator=generator,
                    )
                ]
            else:
                seed_ids = torch.randint(
                    seed_pool_q.shape[0], (num,), device=sim.device, generator=generator
                )
            center_q_batch = seed_pool_q[seed_ids]
            center_q_cmd_batch = seed_pool_q_cmd[seed_ids]
            center_object_pose_batch = seed_pool_object_pose_palm[seed_ids]
        else:
            seed_ids = torch.full(
                (num,),
                int(seed_entry["index"].item()),
                device=sim.device,
                dtype=torch.long,
            )
            center_q_batch = center_q.unsqueeze(0).expand(num, -1)
            center_q_cmd_batch = center_q_cmd.unsqueeze(0).expand(num, -1)
            center_object_pose_batch = center_object_pose_palm.unsqueeze(0).expand(
                num, -1
            )

        if args_cli.no_clamp_joints:
            q_action = sample_bounded_joint_noise(
                center_q_batch,
                float(args_cli.q_noise_rad),
                clamp_lower,
                clamp_upper,
                generator,
            )
            q_cmd_action = sample_bounded_joint_noise(
                center_q_cmd_batch,
                float(args_cli.q_cmd_noise_rad),
                clamp_lower,
                clamp_upper,
                generator,
            )
        else:
            q_action = center_q_batch + torch.randn(
                num, SHARPA_ACTION_DIM, device=sim.device, generator=generator
            ) * float(args_cli.q_noise_rad)
            q_action = clamp_joint_targets(q_action, clamp_lower, clamp_upper)
            q_cmd_action = center_q_cmd_batch + torch.randn(
                num, SHARPA_ACTION_DIM, device=sim.device, generator=generator
            ) * float(args_cli.q_cmd_noise_rad)
            q_cmd_action = clamp_joint_targets(q_cmd_action, clamp_lower, clamp_upper)
        object_pos_palm = center_object_pose_batch[:, :3] + torch.randn(
            num, 3, device=sim.device, generator=generator
        ) * float(args_cli.object_pos_noise_m)
        object_quat_palm = normalize_quat_wxyz(
            quat_mul_wxyz(
                random_small_quat_wxyz(
                    num, float(args_cli.object_rot_noise_rad), sim.device, generator
                ),
                center_object_pose_batch[:, 3:7],
            )
        )
        object_pose_palm = torch.cat((object_pos_palm, object_quat_palm), dim=-1)

        q_cmd_sim, _initial_pose_w = reset_candidates(
            sim,
            hand,
            cube,
            sensor,
            mapping,
            origins,
            hand_quat,
            q_action,
            q_cmd_action,
            object_pose_palm,
        )
        step_sim(sim, hand, cube, q_cmd_sim, int(args_cli.settle_steps))
        q_settled = (
            sim_to_action_order(hand.data.joint_pos, mapping.action_joint_indices)
            .detach()
            .clone()
        )
        q_reset = (
            q_settled
            if args_cli.no_clamp_joints
            else clamp_joint_targets(q_settled, clamp_lower, clamp_upper)
        )
        qdot_settled = (
            sim_to_action_order(hand.data.joint_vel, mapping.action_joint_indices)
            .detach()
            .clone()
        )
        pose_candidate_w = cube.data.root_pose_w.detach().clone()
        pose_candidate_palm = (
            world_to_hand_pose(pose_candidate_w, origins, hand_quat).detach().clone()
        )

        hand.write_joint_state_to_sim(
            action_to_sim_order(
                q_settled, mapping.action_joint_indices, len(hand.joint_names)
            ),
            torch.zeros(num, len(hand.joint_names), device=sim.device),
        )
        hand.set_joint_position_target(q_cmd_sim)
        cube.write_root_pose_to_sim(pose_candidate_w)
        cube.write_root_velocity_to_sim(torch.zeros(num, 6, device=sim.device))
        hand.write_data_to_sim()
        cube.write_data_to_sim()
        sim.step()
        hand.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        sensor.reset()

        force_mean, force_max, active_frac = replay_and_measure(
            sim,
            hand,
            cube,
            sensor,
            q_cmd_sim,
            int(args_cli.replay_steps),
            float(args_cli.force_threshold_n),
        )
        replay_pose_w = cube.data.root_pose_w.detach().clone()
        replay_pose_palm = (
            world_to_hand_pose(replay_pose_w, origins, hand_quat).detach().clone()
        )
        elastomer_pos_w = (
            hand.data.body_pos_w[:, list(elastomer_body_indices)].detach().clone()
        )
        surface_dist = object_surface_distances(
            elastomer_pos_w, replay_pose_w, object_id, object_dims_m
        )

        elastomer_active_frac = active_frac[:, list(ELASTOMER_FORCE_INDICES)]
        elastomer_force_mean = force_mean[:, list(ELASTOMER_FORCE_INDICES)]
        elastomer_force_max = force_max[:, list(ELASTOMER_FORCE_INDICES)]
        dp_force_max = force_max[:, list(DP_FORCE_INDICES)]
        elastomer_force_bits = elastomer_active_frac >= float(
            args_cli.min_force_active_frac
        )
        surface_bits = surface_dist < float(args_cli.contact_proxy_threshold_m)
        contact_bits = elastomer_force_bits | surface_bits
        force_count = elastomer_force_bits.sum(dim=-1)
        surface_count = surface_bits.sum(dim=-1)
        contact_count = contact_bits.sum(dim=-1)
        thumb_force_contact = (
            elastomer_force_bits[:, 0]
            if args_cli.require_thumb_force_contact
            else torch.ones(num, device=sim.device, dtype=torch.bool)
        )
        drift = object_pose_distance(
            pose_candidate_palm, replay_pose_palm, object_id, object_dims_m
        )
        lin_vel = torch.linalg.norm(cube.data.root_lin_vel_w.detach(), dim=-1)
        ang_vel = torch.linalg.norm(cube.data.root_ang_vel_w.detach(), dim=-1)
        margins = q_margin(q_reset, lower, upper)
        any_force_max = force_max.max(dim=-1).values
        replay_pos = replay_pose_palm[:, :3]
        in_workspace = (
            (replay_pos[:, 0] >= float(args_cli.min_palm_x))
            & (replay_pos[:, 0] <= float(args_cli.max_palm_x))
            & (torch.abs(replay_pos[:, 1]) <= float(args_cli.max_abs_palm_y))
            & (replay_pos[:, 2] >= float(args_cli.min_palm_z))
            & (replay_pos[:, 2] <= float(args_cli.max_palm_z))
        )

        keep = (
            (force_count >= int(args_cli.min_elastomer_force_contacts))
            & thumb_force_contact
            & (surface_count >= int(args_cli.min_surface_contacts))
            & (any_force_max <= float(args_cli.max_any_force_n))
            & (drift <= float(args_cli.max_object_drift_m))
            & (lin_vel <= float(args_cli.max_lin_vel))
            & (ang_vel <= float(args_cli.max_ang_vel))
            & (margins >= float(args_cli.min_q_margin_rad))
            & in_workspace
        )

        score = (
            80.0 * contact_count.to(torch.float32)
            + 20.0 * elastomer_active_frac.sum(dim=-1)
            + 8.0 * surface_count.to(torch.float32)
            - 600.0 * drift
            - 10.0 * lin_vel
            - 0.8 * ang_vel
            - 0.06 * torch.clamp(any_force_max - 15.0, min=0.0)
            + 20.0 * torch.clamp(margins, max=0.02)
        )

        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if target_per_seed > 0 and keep_idx.numel() > 0:
            balanced_keep: list[int] = []
            for env_idx in keep_idx.detach().cpu().tolist():
                seed_id = int(seed_ids[env_idx].detach().cpu().item())
                if int(kept_per_seed[seed_id].item()) < target_per_seed:
                    balanced_keep.append(env_idx)
                    kept_per_seed[seed_id] += 1
            keep_idx = torch.tensor(balanced_keep, device=sim.device, dtype=torch.long)
        elif min_per_seed > 0 and keep_idx.numel() > 0:
            for env_idx in keep_idx.detach().cpu().tolist():
                seed_id = int(seed_ids[env_idx].detach().cpu().item())
                kept_per_seed[seed_id] += 1
        if keep_idx.numel() > 0:
            saved.append(
                {
                    "q": q_reset[keep_idx].detach().cpu(),
                    "qdot": qdot_settled[keep_idx].detach().cpu(),
                    "q_cmd": q_cmd_action[keep_idx].detach().cpu(),
                    "object_pose_palm": pose_candidate_palm[keep_idx].detach().cpu(),
                    "contact_bits": contact_bits[keep_idx].detach().cpu(),
                    "score": score[keep_idx].detach().cpu(),
                    "elastomer_force_mean_n": elastomer_force_mean[keep_idx]
                    .detach()
                    .cpu(),
                    "elastomer_force_max_n": elastomer_force_max[keep_idx]
                    .detach()
                    .cpu(),
                    "elastomer_force_active_frac": elastomer_active_frac[keep_idx]
                    .detach()
                    .cpu(),
                    "surface_contact_bits": surface_bits[keep_idx].detach().cpu(),
                    "surface_dist_m": surface_dist[keep_idx].detach().cpu(),
                    "dp_force_max_n": dp_force_max[keep_idx].detach().cpu(),
                    "q_margin_rad": margins[keep_idx].detach().cpu(),
                    "replay_drift_m": drift[keep_idx].detach().cpu(),
                    "replay_lin_vel_mps": lin_vel[keep_idx].detach().cpu(),
                    "replay_ang_vel_radps": ang_vel[keep_idx].detach().cpu(),
                    "source_seed_index": seed_ids[keep_idx]
                    .detach()
                    .cpu()
                    .to(dtype=torch.long),
                }
            )
            if target_per_seed > 0:
                total_kept = int(kept_per_seed.sum().item())
            elif min_per_seed > 0:
                total_kept = sum(int(item["q"].shape[0]) for item in saved)
            else:
                total_kept += int(keep_idx.numel())

        batch_summary = {
            "batch": batch_idx + 1,
            "kept": int(keep_idx.numel()),
            "total_kept": total_kept,
            "force_contact_mean": float(force_count.to(torch.float32).mean().item()),
            "surface_contact_mean": float(
                surface_count.to(torch.float32).mean().item()
            ),
            "contact_mean": float(contact_count.to(torch.float32).mean().item()),
            "contact_max": int(contact_count.max().item()),
            "any_force_max_mean": float(any_force_max.mean().item()),
            "drift_mean": float(drift.mean().item()),
            "lin_vel_mean": float(lin_vel.mean().item()),
            "ang_vel_mean": float(ang_vel.mean().item()),
            "workspace_rate": float(in_workspace.to(torch.float32).mean().item()),
            "q_margin_min": float(margins.min().item()),
        }
        if target_per_seed > 0:
            batch_summary["per_seed_min"] = int(kept_per_seed.min().item())
            batch_summary["per_seed_max"] = int(kept_per_seed.max().item())
            batch_summary["seeds_complete"] = int(
                (kept_per_seed >= target_per_seed).sum().item()
            )
        elif min_per_seed > 0:
            batch_summary["per_seed_min"] = int(kept_per_seed.min().item())
            batch_summary["per_seed_max"] = int(kept_per_seed.max().item())
            batch_summary["seeds_complete"] = int(
                (kept_per_seed >= min_per_seed).sum().item()
            )
        batch_summaries.append(batch_summary)
        print(
            "[INFO] "
            f"batch={batch_idx + 1}/{args_cli.max_batches} kept={int(keep_idx.numel())}/{num} "
            f"total={total_kept}/{args_cli.target_count} "
            f"contact_mean={batch_summary['contact_mean']:.2f} "
            f"surface_mean={batch_summary['surface_contact_mean']:.2f} "
            f"force_mean={batch_summary['force_contact_mean']:.2f} "
            f"drift_mean={batch_summary['drift_mean']:.4f} "
            f"lin_vel_mean={batch_summary['lin_vel_mean']:.4f} "
            f"workspace={batch_summary['workspace_rate']:.3f}",
            flush=True,
        )

        if args_cli.write_every > 0 and total_kept >= args_cli.write_every:
            partial = build_cache(saved, metadata)
            save_stable_grasp_cache(partial, resolve_workspace_path(args_cli.output))
        if target_per_seed > 0:
            if bool((kept_per_seed >= target_per_seed).all()):
                break
        elif min_per_seed > 0:
            if total_kept >= args_cli.target_count and bool(
                (kept_per_seed >= min_per_seed).all()
            ):
                break
        elif total_kept >= args_cli.target_count:
            break

    cache = build_cache(saved, metadata)
    output = save_stable_grasp_cache(cache, resolve_workspace_path(args_cli.output))
    report = write_report(cache, batch_summaries)
    print(f"[INFO] output={output}", flush=True)
    print(f"[INFO] report={report}", flush=True)
    print(f"[INFO] summary={summarize_stable_grasp_cache(cache)}", flush=True)
    if int(cache["q"].shape[0]) < int(args_cli.target_count):
        print(
            f"[ERROR] saved only {int(cache['q'].shape[0])}/{args_cli.target_count}; "
            "increase --max-batches or relax thresholds.",
            flush=True,
        )
        return 2
    if target_per_seed > 0:
        min_count = int(kept_per_seed.min().item())
        if min_count < target_per_seed:
            print(
                f"[ERROR] only reached per-seed min={min_count}/{target_per_seed}; "
                "increase --max-batches or relax thresholds.",
                flush=True,
            )
            return 2
    if min_per_seed > 0 and "source_seed_index" in cache:
        counts = torch.bincount(
            cache["source_seed_index"].to(dtype=torch.long),
            minlength=int(seed_pool_q.shape[0]),
        )
        min_count = int(counts.min().item()) if counts.numel() else 0
        if min_count < min_per_seed:
            print(
                f"[ERROR] final cache only reached per-seed min={min_count}/{min_per_seed}; "
                "increase --max-batches or relax thresholds.",
                flush=True,
            )
            return 2
    return 0


def mapping_body_indices(
    hand: Articulation, body_names: tuple[str, ...]
) -> tuple[int, ...]:
    body_name_to_idx = {name: idx for idx, name in enumerate(hand.body_names)}
    missing = [name for name in body_names if name not in body_name_to_idx]
    if missing:
        raise RuntimeError(f"Missing body names: {missing}")
    return tuple(body_name_to_idx[name] for name in body_names)


if __name__ == "__main__":
    import os

    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if exit_code:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
