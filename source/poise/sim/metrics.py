"""Canonical full-pose cube metrics for AG2OP-Wuji-Cube."""

from __future__ import annotations

import torch

from poise.sim.assets.cube_4cm import CUBE_4CM_SIDE_LENGTH_M


def normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize quaternions in Isaac Lab's `(w, x, y, z)` convention."""

    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in `(w, x, y, z)` convention.

    Args:
        quat: Quaternion tensor with shape `(..., 4)`.
        vec: Vector tensor broadcastable to shape `(..., 3)`.

    Returns:
        Rotated vectors with shape broadcast from `quat` and `vec`.
    """

    quat = normalize_quat_wxyz(quat)
    q_w = quat[..., :1]
    q_xyz = quat[..., 1:]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def cuboid_local_corners(
    dims_m: tuple[float, float, float] | torch.Tensor = (
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
    ),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the 8 canonical cuboid corners in object frame.

    The object is treated as having a canonical marked frame. Rotational
    symmetries are intentionally not considered equivalent.
    """

    dims = torch.as_tensor(dims_m, device=device, dtype=dtype)
    if tuple(dims.shape) != (3,):
        raise ValueError(f"Expected dims_m to have shape (3,), got {tuple(dims.shape)}.")
    signs = torch.tensor(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        device=device,
        dtype=dtype,
    )
    return signs * (0.5 * dims).unsqueeze(0)


def cube_local_corners(
    side_length: float = CUBE_4CM_SIDE_LENGTH_M,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the 8 canonical cube corners in object frame."""

    return cuboid_local_corners((side_length, side_length, side_length), device=device, dtype=dtype)


def cuboid_keypoints_from_pose(
    pose: torch.Tensor,
    dims_m: tuple[float, float, float] | torch.Tensor = (
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
    ),
    local_corners: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform canonical cuboid corners by pose.

    Args:
        pose: Tensor shaped `(..., 7)` containing position and quaternion
            `(x, y, z, qw, qx, qy, qz)`.
        dims_m: Cuboid dimensions `(x, y, z)` in meters.
        local_corners: Optional precomputed corners shaped `(8, 3)`.

    Returns:
        Tensor shaped `(..., 8, 3)`.
    """

    if pose.shape[-1] != 7:
        raise ValueError(f"Expected pose last dimension to be 7, got {pose.shape}.")

    pos = pose[..., :3]
    quat = pose[..., 3:7]
    if local_corners is None:
        local_corners = cuboid_local_corners(dims_m, device=pose.device, dtype=pose.dtype)
    else:
        local_corners = local_corners.to(device=pose.device, dtype=pose.dtype)

    expand_shape = pose.shape[:-1] + local_corners.shape
    corners = local_corners.expand(expand_shape)
    quat_expand = quat.unsqueeze(-2).expand(expand_shape[:-1] + (4,))
    return quat_apply_wxyz(quat_expand, corners) + pos.unsqueeze(-2)


def cube_keypoints_from_pose(
    pose: torch.Tensor,
    side_length: float = CUBE_4CM_SIDE_LENGTH_M,
    local_corners: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform canonical cube corners by pose."""

    return cuboid_keypoints_from_pose(
        pose,
        dims_m=(side_length, side_length, side_length),
        local_corners=local_corners,
    )


def cuboid_keypoint_l2_errors(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    dims_m: tuple[float, float, float] | torch.Tensor = (
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
    ),
) -> torch.Tensor:
    """Return per-corner L2 errors for canonical full cuboid poses."""

    local = cuboid_local_corners(dims_m, device=current_pose.device, dtype=current_pose.dtype)
    current = cuboid_keypoints_from_pose(current_pose, dims_m, local)
    goal = cuboid_keypoints_from_pose(goal_pose.to(device=current_pose.device, dtype=current_pose.dtype), dims_m, local)
    return torch.linalg.norm(current - goal, dim=-1)


def cube_keypoint_l2_errors(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    side_length: float = CUBE_4CM_SIDE_LENGTH_M,
) -> torch.Tensor:
    """Return per-corner L2 errors for canonical full cube poses."""

    return cuboid_keypoint_l2_errors(current_pose, goal_pose, (side_length, side_length, side_length))


def cuboid_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    dims_m: tuple[float, float, float] | torch.Tensor = (
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
    ),
) -> torch.Tensor:
    """Return max-corner L2 distance for canonical full cuboid poses."""

    return torch.max(cuboid_keypoint_l2_errors(current_pose, goal_pose, dims_m), dim=-1).values


def cube_pose_distance(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    side_length: float = CUBE_4CM_SIDE_LENGTH_M,
) -> torch.Tensor:
    """Return max-corner L2 distance for canonical full cube poses."""

    return cuboid_pose_distance(current_pose, goal_pose, (side_length, side_length, side_length))


def cuboid_pose_success(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    threshold: float,
    dims_m: tuple[float, float, float] | torch.Tensor = (
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
        CUBE_4CM_SIDE_LENGTH_M,
    ),
) -> torch.Tensor:
    """Return success mask using canonical full-pose cuboid distance."""

    return cuboid_pose_distance(current_pose, goal_pose, dims_m) < threshold


def cube_pose_success(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    threshold: float,
    side_length: float = CUBE_4CM_SIDE_LENGTH_M,
) -> torch.Tensor:
    """Return success mask using canonical full-pose cube distance."""

    return cuboid_pose_success(current_pose, goal_pose, threshold, (side_length, side_length, side_length))
