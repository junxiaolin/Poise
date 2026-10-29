#!/usr/bin/env python3
"""Convert a Sharpa RL Lab grasp cache to POISE stable-cache schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


from poise.paths import workspace_root
POISE_ROOT = workspace_root()

from poise.sim.assets.cube_4cm import CUBE_4CM_SIDE_LENGTH_M
from poise.sim.joint_mapping import SHARPA_JOINT_ORDER
from poise.sim.metrics import (
    normalize_quat_wxyz,
    quat_apply_wxyz,
)
from poise.grasp.cache import (
    SCHEMA_VERSION,
    save_stable_grasp_cache,
    summarize_stable_grasp_cache,
)


SHARPA_RL_LAB_HAND_ROOT_POS = (0.0, 0.0, 0.5)
SHARPA_RL_LAB_PALMDOWN_ROOT_ROT = (0.5735764, 0.0, 0.819152, 0.0)
DEFAULT_OBJECT_ID = "cube_4cm"
DEFAULT_OBJECT_DIMS_M = (CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M, CUBE_4CM_SIDE_LENGTH_M)


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


def world_to_hand_pose(
    local_world_pose: torch.Tensor,
    *,
    hand_root_pos: tuple[float, float, float],
    hand_root_rot_wxyz: tuple[float, float, float, float],
) -> torch.Tensor:
    """Convert env-local world object pose into the hand root frame."""

    num = local_world_pose.shape[0]
    hand_pos = torch.tensor(hand_root_pos, dtype=local_world_pose.dtype).view(1, 3).repeat(num, 1)
    hand_quat = normalize_quat_wxyz(
        torch.tensor(hand_root_rot_wxyz, dtype=local_world_pose.dtype).view(1, 4).repeat(num, 1)
    )
    inv_hand_quat = quat_conjugate_wxyz(hand_quat)
    local_pos = quat_apply_wxyz(inv_hand_quat, local_world_pose[:, :3] - hand_pos)
    local_quat = normalize_quat_wxyz(quat_mul_wxyz(inv_hand_quat, local_world_pose[:, 3:7]))
    local_quat = torch.where(local_quat[:, :1] < 0.0, -local_quat, local_quat)
    return torch.cat((local_pos, local_quat), dim=-1)


def bbox_keypoints_from_pose(pose: torch.Tensor, dims_m: tuple[float, float, float]) -> torch.Tensor:
    """Return the 8 local bbox corners transformed by a WXYZ pose."""

    hx, hy, hz = (0.5 * float(v) for v in dims_m)
    local_points = torch.tensor(
        [[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)],
        device=pose.device,
        dtype=pose.dtype,
    )
    expand_shape = pose.shape[:-1] + local_points.shape
    points = local_points.expand(expand_shape)
    quat = pose[..., 3:7].unsqueeze(-2).expand(expand_shape[:-1] + (4,))
    return quat_apply_wxyz(quat, points) + pose[..., :3].unsqueeze(-2)


def resolve_object_dims_m(object_id: str, dims_raw: tuple[float, float, float] | None) -> tuple[float, float, float]:
    if dims_raw is None:
        if object_id == DEFAULT_OBJECT_ID:
            return DEFAULT_OBJECT_DIMS_M
        raise ValueError(f"--object-dims-m is required for object_id={object_id!r}.")
    dims = tuple(float(v) for v in dims_raw)
    if len(dims) != 3 or any(v <= 0.0 for v in dims):
        raise ValueError(f"Invalid --object-dims-m for object_id={object_id!r}: {dims_raw!r}.")
    return dims


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(x) for x in values.tolist())


def rotation_matrices_to_quat_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Convert batched SO(3) matrices to normalized positive-w quaternions."""

    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation matrices shaped [N,3,3], got {rotations.shape}.")
    quats = np.empty((rotations.shape[0], 4), dtype=np.float64)
    for index, rotation in enumerate(rotations):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=2.0e-5
        ):
            raise ValueError(f"object_pose_hand row {index} does not contain an SO(3) rotation.")
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quat = np.array([0.25 * scale, (rotation[2,1]-rotation[1,2])/scale, (rotation[0,2]-rotation[2,0])/scale, (rotation[1,0]-rotation[0,1])/scale])
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(max(0.0, 1.0 + rotation[0,0] - rotation[1,1] - rotation[2,2])) * 2.0
                quat = np.array([(rotation[2,1]-rotation[1,2])/scale, 0.25*scale, (rotation[0,1]+rotation[1,0])/scale, (rotation[0,2]+rotation[2,0])/scale])
            elif axis == 1:
                scale = np.sqrt(max(0.0, 1.0 + rotation[1,1] - rotation[0,0] - rotation[2,2])) * 2.0
                quat = np.array([(rotation[0,2]-rotation[2,0])/scale, (rotation[0,1]+rotation[1,0])/scale, 0.25*scale, (rotation[1,2]+rotation[2,1])/scale])
            else:
                scale = np.sqrt(max(0.0, 1.0 + rotation[2,2] - rotation[0,0] - rotation[1,1])) * 2.0
                quat = np.array([(rotation[1,0]-rotation[0,1])/scale, (rotation[0,2]+rotation[2,0])/scale, (rotation[1,2]+rotation[2,1])/scale, 0.25*scale])
        quat /= np.linalg.norm(quat)
        quats[index] = -quat if quat[0] < 0.0 else quat
    return quats.astype(np.float32)


def load_source_cache(
    source: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Load native Sharpa RL Lab cache fields in POISE action order."""

    metadata: dict[str, Any] = {
        "source": str(source),
        "source_columns": "qpos[0:22], object_pos[22:25], object_quat_wxyz[25:29]",
    }
    if source.suffix == ".npz":
        data = np.load(source, allow_pickle=False)
        missing = [key for key in ("qpos", "object_pos_w", "object_quat_wxyz", "joint_names") if key not in data]
        if missing:
            raise ValueError(f"Named Sharpa IsaacLab NPZ is missing required keys: {missing}")
        source_joint_order = _strings(data["joint_names"])
        raw_q = np.asarray(data["qpos"], dtype=np.float32)
        raw_q_cmd = np.asarray(data["q_cmd"], dtype=np.float32) if "q_cmd" in data else None
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float32)
        object_quat = np.asarray(data["object_quat_wxyz"], dtype=np.float32)
        object_pose_hand = np.asarray(data["object_pose_hand"], dtype=np.float32) if "object_pose_hand" in data else None
        if "hand_root_pos_w" in data:
            metadata["source_hand_root_pos_w"] = [float(x) for x in data["hand_root_pos_w"].tolist()]
        if "hand_root_quat_wxyz" in data:
            metadata["source_hand_root_quat_wxyz"] = [float(x) for x in data["hand_root_quat_wxyz"].tolist()]
        metadata["source_format"] = "sharpa_rl_lab_named_isaaclab_npz"
        metadata["source_has_q_cmd"] = raw_q_cmd is not None
    else:
        raw = np.load(source)
        if raw.ndim != 2 or raw.shape[1] != 29:
            raise ValueError(f"Expected Sharpa RL Lab native cache shaped Nx29, got {raw.shape}.")
        raw_q = np.asarray(raw[:, :22], dtype=np.float32)
        raw_q_cmd = None
        object_pos = np.asarray(raw[:, 22:25], dtype=np.float32)
        object_quat = np.asarray(raw[:, 25:29], dtype=np.float32)
        object_pose_hand = None
        source_joint_order = SHARPA_JOINT_ORDER
        metadata["source_format"] = "sharpa_rl_lab_native_numpy_22dof_plus_object_pose_wxyz"
        metadata["source_has_q_cmd"] = False

    if raw_q.ndim != 2 or raw_q.shape[1] != len(SHARPA_JOINT_ORDER):
        raise ValueError(f"Expected qpos shape [N, 22], got {raw_q.shape}.")
    if raw_q_cmd is not None and raw_q_cmd.shape != raw_q.shape:
        raise ValueError(f"Expected q_cmd shape {raw_q.shape}, got {raw_q_cmd.shape}.")
    if object_pos.shape != (raw_q.shape[0], 3):
        raise ValueError(f"Expected object_pos shape [N, 3], got {object_pos.shape}.")
    if object_quat.shape != (raw_q.shape[0], 4):
        raise ValueError(f"Expected object_quat_wxyz shape [N, 4], got {object_quat.shape}.")
    if object_pose_hand is not None:
        if object_pose_hand.shape != (raw_q.shape[0], 4, 4):
            raise ValueError(f"Expected object_pose_hand shape [N,4,4], got {object_pose_hand.shape}.")
        if not np.isfinite(object_pose_hand).all() or not np.allclose(
            object_pose_hand[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1.0e-6
        ):
            raise ValueError(f"Invalid object_pose_hand transforms: {source}")
    finite_q_cmd = True if raw_q_cmd is None else np.isfinite(raw_q_cmd).all()
    if not np.isfinite(raw_q).all() or not finite_q_cmd or not np.isfinite(object_pos).all() or not np.isfinite(object_quat).all():
        raise ValueError(f"Sharpa RL Lab cache contains non-finite values: {source}")

    source_name_to_index = {name: index for index, name in enumerate(source_joint_order)}
    missing_names = [name for name in SHARPA_JOINT_ORDER if name not in source_name_to_index]
    if missing_names:
        raise ValueError(
            "Source joint_names cannot be mapped into POISE Sharpa action order. "
            f"Missing: {missing_names}; source order: {source_joint_order}"
        )
    q_indices = [source_name_to_index[name] for name in SHARPA_JOINT_ORDER]
    q = torch.as_tensor(raw_q[:, q_indices].copy(), dtype=torch.float32)
    if raw_q_cmd is None:
        raise ValueError("POISE requires q_cmd; enrich the optimizer cache from contact wrenches first.")
    else:
        q_cmd = torch.as_tensor(raw_q_cmd[:, q_indices].copy(), dtype=torch.float32)
    metadata["source_joint_order"] = tuple(source_joint_order)
    metadata["output_joint_order"] = SHARPA_JOINT_ORDER
    return (
        q,
        q_cmd,
        torch.as_tensor(object_pos, dtype=torch.float32),
        torch.as_tensor(object_quat, dtype=torch.float32),
        torch.as_tensor(object_pose_hand, dtype=torch.float32) if object_pose_hand is not None else None,
        metadata,
    )


def convert_cache(
    source: Path,
    output: Path,
    *,
    max_rows: int,
    hand_root_pos: tuple[float, float, float] | None,
    hand_root_rot_wxyz: tuple[float, float, float, float] | None,
    contact_bits_mode: str,
    object_id: str,
    object_dims_m: tuple[float, float, float],
    profile_name: str,
) -> dict[str, Any]:
    q, q_cmd, object_pos, object_quat, source_object_pose_hand, source_metadata = load_source_cache(source)
    if hand_root_pos is None:
        raw_pos = source_metadata.get("source_hand_root_pos_w")
        hand_root_pos = tuple(float(x) for x in raw_pos) if raw_pos is not None else SHARPA_RL_LAB_HAND_ROOT_POS
    if hand_root_rot_wxyz is None:
        raw_rot = source_metadata.get("source_hand_root_quat_wxyz")
        hand_root_rot_wxyz = (
            tuple(float(x) for x in raw_rot) if raw_rot is not None else SHARPA_RL_LAB_PALMDOWN_ROOT_ROT
        )
    if max_rows > 0:
        q = q[:max_rows]
        q_cmd = q_cmd[:max_rows]
        object_pos = object_pos[:max_rows]
        object_quat = object_quat[:max_rows]
        if source_object_pose_hand is not None:
            source_object_pose_hand = source_object_pose_hand[:max_rows]
    if q.shape[0] <= 0:
        raise ValueError("No rows selected for conversion.")

    q = q.contiguous()
    q_cmd = q_cmd.contiguous()
    object_pos = object_pos.contiguous()
    object_quat = normalize_quat_wxyz(object_quat.contiguous())
    if source_object_pose_hand is not None:
        direct_pos = source_object_pose_hand[:, :3, 3].contiguous()
        direct_quat = torch.as_tensor(
            rotation_matrices_to_quat_wxyz(source_object_pose_hand[:, :3, :3].cpu().numpy()), dtype=torch.float32
        )
        object_pose_palm = torch.cat((direct_pos, direct_quat), dim=-1)
        source_metadata["source_object_pose_hand_used"] = True
        source_metadata["source_object_pose_frame"] = "hand_root_direct"
    else:
        object_pose_palm = world_to_hand_pose(
            torch.cat((object_pos, object_quat), dim=-1),
            hand_root_pos=hand_root_pos,
            hand_root_rot_wxyz=hand_root_rot_wxyz,
        )
        source_metadata["source_object_pose_hand_used"] = False
        source_metadata["source_object_pose_frame"] = "env_local_world"

    if contact_bits_mode == "all":
        contact_bits = torch.ones((q.shape[0], 5), dtype=torch.bool)
    elif contact_bits_mode == "unknown":
        contact_bits = torch.zeros((q.shape[0], 5), dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported contact_bits_mode={contact_bits_mode!r}.")
    stability_score = contact_bits.to(dtype=torch.float32).sum(dim=-1)

    cache: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hand": "sharpa_wave_right",
        "action_dim": len(SHARPA_JOINT_ORDER),
        "object_id": object_id,
        "object_dims_m": object_dims_m,
        "q": q,
        "qdot": torch.zeros_like(q),
        "q_cmd": q_cmd,
        "object_pose_palm": object_pose_palm,
        "cube_keypoints_palm": bbox_keypoints_from_pose(object_pose_palm, object_dims_m),
        "contact_bits": contact_bits,
        "stability_score": stability_score,
        "profile_names": [profile_name],
        "profile_id": torch.zeros(q.shape[0], dtype=torch.long),
        "source_row_index": torch.arange(q.shape[0], dtype=torch.long),
        "metadata": {
            **source_metadata,
            "object_id": object_id,
            "object_dims_m": object_dims_m,
            "conversion_hand_root_pos": tuple(float(x) for x in hand_root_pos),
            "conversion_hand_root_rot_wxyz": tuple(float(x) for x in hand_root_rot_wxyz),
            "contact_bits_mode": contact_bits_mode,
            "notes": (
                "Converted from Sharpa RL Lab saved grasping states. "
                "q_cmd is preserved when the source NPZ provides it; otherwise q_cmd=q. "
                "contact_bits and stability_score are placeholders until IsaacLab hold filtering."
            ),
        },
    }

    save_stable_grasp_cache(cache, output)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional limit; 0 converts all rows.")
    parser.add_argument(
        "--source-hand-root-pos",
        type=float,
        nargs=3,
        default=None,
        help="Override source hand root position; by default use hand_root_pos_w from .npz when available.",
    )
    parser.add_argument(
        "--source-hand-root-rot-wxyz",
        type=float,
        nargs=4,
        default=None,
        help="Override source hand root quaternion; by default use hand_root_quat_wxyz from .npz when available.",
    )
    parser.add_argument(
        "--contact-bits-mode",
        choices=("all", "unknown"),
        default="unknown",
        help="Placeholder contact-bit policy for raw caches without per-finger contact labels.",
    )
    parser.add_argument(
        "--object-id",
        type=str,
        default=DEFAULT_OBJECT_ID,
        help="POISE object_id to write into the stable-cache.",
    )
    parser.add_argument(
        "--object-dims-m",
        type=float,
        nargs=3,
        default=None,
        help="Object bbox dimensions in meters, in object local XYZ order.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default="sharpa_rl_lab_converted",
        help="Stable-cache profile name for all converted rows.",
    )
    args = parser.parse_args()
    object_dims_m = resolve_object_dims_m(args.object_id, tuple(args.object_dims_m) if args.object_dims_m else None)

    cache = convert_cache(
        args.input,
        args.output,
        max_rows=int(args.max_rows),
        hand_root_pos=tuple(float(x) for x in args.source_hand_root_pos) if args.source_hand_root_pos else None,
        hand_root_rot_wxyz=(
            tuple(float(x) for x in args.source_hand_root_rot_wxyz) if args.source_hand_root_rot_wxyz else None
        ),
        contact_bits_mode=args.contact_bits_mode,
        object_id=str(args.object_id),
        object_dims_m=object_dims_m,
        profile_name=str(args.profile_name),
    )
    pose = cache["object_pose_palm"][:, :3]
    summary = summarize_stable_grasp_cache(cache)
    summary["object_pose_palm_pos_min"] = [float(x) for x in pose.min(dim=0).values.tolist()]
    summary["object_pose_palm_pos_max"] = [float(x) for x in pose.max(dim=0).values.tolist()]
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
