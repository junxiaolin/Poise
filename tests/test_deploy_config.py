"""Deployment preflight uses local artifacts only, including corrupted inputs."""

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import trimesh

from poise.deploy.config import DeploymentConfig, prepare_deployment
from poise.core.policy_bundle import PolicyBundle


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    fp = tmp_path / "foundationpose"
    fp.mkdir()
    (fp / "estimater.py").write_text('raise RuntimeError("must not import during preflight")')
    trimesh.creation.box(extents=[0.04, 0.05, 0.06]).export(tmp_path / "object.obj")
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[300:400, 600:700] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    cfg = {
        "foundationpose_root": "foundationpose", "mesh": "object.obj", "registration_mask": "mask.png",
        "sdk_container": "fake", "sdk_root": "/opt/vendor sdk", "sdk_python": "python3",
        "tag_id": 0, "tag_size_m": 0.06, "t_hand_tag": np.eye(4).tolist(),
        "t_mesh_object": np.eye(4).tolist(), "t_hand_target": np.eye(4).tolist(),
        "gravity_direction_palm": [-1, 0, 0], "hard_joint_lower_rad": [-1.1] * 22,
        "hard_joint_upper_rad": [1.1] * 22,
    }
    filename = tmp_path / "robot.json"
    filename.write_text(json.dumps(cfg))
    initial = {"q": torch.zeros(22), "q_cmd": torch.zeros(22),
               "object_pose_palm": torch.tensor([0, 0, 0, 1., 0, 0, 0])}
    torch.save(initial, tmp_path / "initial.pt")
    field = np.full((3, 3, 3), -1.0, dtype=np.float32)
    np.savez(tmp_path / "sdf.npz", sdf_m=field, origin_m=np.full(3, -0.1), voxel_size_m=0.1,
             metadata_json=json.dumps({"schema": "sharpa_hand_collision_sdf_v1",
                 "sign": "positive_inside_negative_outside", "shape": [3, 3, 3]}))
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    bundle = SimpleNamespace(root=tmp_path, observation_dim=244, manifest={
        "initial_grasp": {"tensor_filename": "initial.pt", "tensor_sha256": sha("initial.pt")},
        "sim_contract": {"joint_lower": [-1.] * 22, "joint_upper": [1.] * 22, "control_dt_s": 0.05,
                         "target_translation_scale_m": 0.03},
        "control": {"control_dt_s": 0.05, "frequency_hz": 20, "action_scale_rad": 1 / 24, "action_smoothing": 1.0},
        "physical_object": {"dimensions_m": [0.04, 0.05, 0.06], "stable_cache_object_id": "box"},
        "target_feasibility": {"workspace_bounds_m": [-0.1, 0.1] * 3,
            "sdf_filename": "sdf.npz", "sdf_sha256": sha("sdf.npz"),
            "proxy_points_local_m": [[0, 0, 0]], "proxy_radii_m": [0.01],
            "margin_m": 0, "interpolation_guard_m": 0.0005, "allowed_penetration_m": 0.002},
    })
    bundle.object_descriptor = torch.zeros(64)
    bundle.infer = lambda obs: torch.zeros(22)
    bundle.next_joint_target = lambda previous, action: PolicyBundle.next_joint_target(bundle, previous, action)
    monkeypatch.setattr("poise.deploy.config.PolicyBundle.load", lambda _: bundle)
    return filename, cfg, bundle


def test_tactile_bundle_requires_hardware_adapter(deployment):
    filename, _, bundle = deployment
    bundle.observation_dim = 304
    with pytest.raises(ValueError, match="tactile hardware adapter"):
        prepare_deployment(filename, bundle.root)


def test_preflight_resolves_paths_and_never_starts_hardware(deployment, monkeypatch):
    filename, _, bundle = deployment
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not spawn processes or access hardware")
    monkeypatch.setattr("subprocess.Popen", forbidden)
    result = prepare_deployment(filename, bundle.root)
    assert result.report["valid"] and not result.report["hardware_contacted"]
    assert result.config.mesh == filename.parent / "object.obj"
    assert result.report["mask_pixels"] == 10000


@pytest.mark.parametrize("field,value", [
    ("t_hand_tag", None), ("t_hand_tag", np.zeros((4, 4)).tolist()),
    ("gravity_direction_palm", [0, 0, 0]), ("hard_joint_lower_rad", [-1]),
    ("hard_joint_upper_rad", [float("nan")] * 22), ("hard_joint_upper_rad", [-2] * 22),
    ("tag_id", 587), ("tag_id", True), ("tag_size_m", float("nan")),
    ("speed_coeff", 0.31), ("current_coeff", 0), ("sdk_root", "relative/path"),
    ("sdk_container", "-unexpected-option"), ("hand_serial", 123), ("typo", 1),
])
def test_bad_installation_config_fails_early(deployment, field, value):
    filename, cfg, _ = deployment
    cfg[field] = value
    filename.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        DeploymentConfig.load(filename)


def test_mesh_frame_is_applied_before_dimension_check(deployment):
    filename, cfg, bundle = deployment
    transform = np.array([[0., -1, 0, .3], [1, 0, 0, -.2], [0, 0, 1, .1], [0, 0, 0, 1]])
    mesh = trimesh.load(filename.parent / "object.obj", process=False)
    mesh.apply_transform(transform)
    mesh.export(filename.parent / "object.obj")
    cfg["t_mesh_object"] = transform.tolist()
    filename.write_text(json.dumps(cfg))
    assert prepare_deployment(filename, bundle.root).report["valid"]


def test_preflight_exercises_single_frame_policy_command(deployment):
    filename, _, bundle = deployment
    bundle.infer = lambda obs: torch.zeros(1, 22)
    with pytest.raises(RuntimeError, match="Invalid policy command"):
        prepare_deployment(filename, bundle.root)


@pytest.mark.parametrize("fault,match", [
    ("mesh", "size/frame mismatch"), ("mask", "Registration mask"),
    ("limits", "hardware envelope"), ("period", "20 Hz"),
    ("hash", "hash mismatch"), ("target", "outside"),
])
def test_incompatible_artifacts_are_rejected(deployment, fault, match):
    filename, cfg, bundle = deployment
    if fault == "mesh":
        trimesh.creation.box(extents=[40, 50, 60]).export(filename.parent / "object.obj")
    elif fault == "mask":
        cv2.imwrite(str(filename.parent / "mask.png"), np.zeros((10, 10), dtype=np.uint8))
    elif fault == "limits":
        cfg["hard_joint_upper_rad"] = [0.5] * 22
    elif fault == "period":
        bundle.manifest["sim_contract"]["control_dt_s"] = 0.1
    elif fault == "hash":
        with (filename.parent / "initial.pt").open("ab") as handle:
            handle.write(b"corrupt")
    elif fault == "target":
        cfg["t_hand_target"][0][3] = 0.2
    filename.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=match):
        prepare_deployment(filename, bundle.root)
