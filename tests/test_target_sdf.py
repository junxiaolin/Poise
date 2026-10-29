"""Check signed distance convention and compound target rejection."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from poise.deploy.sdf import load_forbidden_sdf_grid, sample_forbidden_sdf_trilinear
from poise.deploy.targets import validate_target


def test_planar_field_interpolation_and_target_collision(tmp_path):
    axis = np.linspace(-0.1, 0.1, 3)
    field = np.broadcast_to(axis[:, None, None], (3, 3, 3)).astype(np.float32)
    metadata = {"schema": "sharpa_hand_collision_sdf_v1", "sign": "positive_inside_negative_outside", "shape": [3, 3, 3]}
    path = tmp_path / "sdf.npz"
    np.savez(path, sdf_m=field, origin_m=np.full(3, -0.1), voxel_size_m=0.1, metadata_json=json.dumps(metadata))
    grid = load_forbidden_sdf_grid(path)
    values, valid = sample_forbidden_sdf_trilinear(np.array([[-0.025, 0, 0], [0.2, 0, 0]]), grid)
    assert values[0] == pytest.approx(-0.025)
    assert valid.tolist() == [True, False]
    bundle = SimpleNamespace(root=tmp_path, manifest={"target_feasibility": {
        "sdf_filename": "sdf.npz", "sdf_sha256": grid.sha256,
        "workspace_bounds_m": [-0.1, 0.1] * 3,
        "proxy_points_local_m": [[0, 0, 0]], "proxy_radii_m": [0.01],
        "margin_m": 0, "interpolation_guard_m": 0.0005, "allowed_penetration_m": 0.002,
    }})
    validate_target(bundle, [-0.025, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="intersects"):
        validate_target(bundle, [-0.005, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="outside"):
        validate_target(bundle, [0.2, 0, 0, 1, 0, 0, 0])
