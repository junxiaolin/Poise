"""Full object-origin / SO(3) pose tolerances, independent of Isaac Sim."""

import torch


def full_pose_success_terms(current_pose, target_pose, position_threshold_m, orientation_threshold_rad):
    """Return component acceptance and errors for poses with wxyz quaternions.

    Quaternion sign does not affect the shortest rotation angle. Drop exclusion
    and consecutive-step verification are applied by the environment.
    """
    position_error = torch.linalg.vector_norm(current_pose[:, :3] - target_pose[:, :3], dim=-1)
    current = current_pose[:, 3:7]
    target = target_pose[:, 3:7]
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)
    # Vector part of target * conjugate(current), with sign-invariant angle.
    scalar = (target * current).sum(dim=-1).abs()
    vector = (
        current[:, :1] * target[:, 1:] - target[:, :1] * current[:, 1:]
        - torch.linalg.cross(target[:, 1:], current[:, 1:], dim=-1)
    )
    orientation_error = 2.0 * torch.atan2(torch.linalg.vector_norm(vector, dim=-1), scalar)
    return (
        position_error < position_threshold_m,
        orientation_error < orientation_threshold_rad,
        position_error,
        orientation_error,
    )
