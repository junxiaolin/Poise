"""Reject targets outside the trained workspace or intersecting the palm."""

import numpy as np
import torch

from poise.core.observation import quat_apply_wxyz
from poise.deploy.sdf import forbidden_sdf_hits, load_forbidden_sdf_grid


def validate_target(bundle, pose):
    pose = torch.as_tensor(pose, dtype=torch.float32)
    if pose.shape != (7,) or not torch.isfinite(pose).all() or not torch.isclose(pose[3:].norm(), torch.tensor(1.0), atol=1e-4):
        raise ValueError("Target must be a finite pose with a unit quaternion")
    config = bundle.manifest["target_feasibility"]
    bounds = np.asarray(config["workspace_bounds_m"]).reshape(3, 2)
    if np.any(pose[:3].numpy() < bounds[:, 0]) or np.any(pose[:3].numpy() > bounds[:, 1]):
        raise ValueError("Target is outside the trained palm workspace")
    grid = load_forbidden_sdf_grid(bundle.root / config["sdf_filename"])
    if grid.sha256 != config["sdf_sha256"]:
        raise ValueError("Target collision field hash mismatch")
    points = torch.as_tensor(config["proxy_points_local_m"], dtype=torch.float32)
    points = quat_apply_wxyz(pose[3:].expand(len(points), -1), points) + pose[:3]
    clearance = np.maximum(0, np.asarray(config["proxy_radii_m"]) + config["margin_m"] + config["interpolation_guard_m"] - config["allowed_penetration_m"])
    if forbidden_sdf_hits(points.numpy(), grid=grid, clearance_m=clearance).any():
        raise ValueError("Target intersects the trained palm collision field")
