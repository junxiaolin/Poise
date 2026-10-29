"""Offline configuration and bundle checks shared by the CLI and live runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from poise.core.policy_bundle import PolicyBundle
from .calibration import as_transform
from .control import PolicyController, transform_to_pose
from .sdk_runtime import sdk_command
from .targets import validate_target


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class DeploymentConfig:
    foundationpose_root: Path
    mesh: Path
    registration_mask: Path
    sdk_container: str
    sdk_root: str
    sdk_python: str
    tag_id: int
    tag_size_m: float
    t_hand_tag: np.ndarray
    t_mesh_object: np.ndarray
    t_hand_target: np.ndarray
    gravity_direction_palm: np.ndarray
    hard_joint_lower_rad: np.ndarray
    hard_joint_upper_rad: np.ndarray
    camera_serial: str | None = None
    hand_serial: str | None = None
    speed_coeff: float = 0.30
    current_coeff: float = 0.30

    @classmethod
    def load(cls, filename: Path) -> DeploymentConfig:
        filename = Path(filename).expanduser().resolve()
        values = json.loads(filename.read_text())
        if not isinstance(values, dict):
            raise ValueError("Deployment configuration must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown deployment fields: {sorted(unknown)}")
        required = set(cls.__dataclass_fields__) - {
            "camera_serial", "hand_serial", "speed_coeff", "current_coeff"}
        missing = sorted(key for key in required if values.get(key) is None)
        if missing:
            raise ValueError(f"Fill installation-specific deployment fields: {missing}")
        for key in ("foundationpose_root", "mesh", "registration_mask"):
            if not isinstance(values[key], str) or not values[key].strip():
                raise ValueError(f"{key} must be a nonempty path")
            path = Path(values[key]).expanduser()
            values[key] = (filename.parent / path).resolve()
        for key in ("t_hand_tag", "t_mesh_object", "t_hand_target"):
            try:
                values[key] = as_transform(values[key])
            except ValueError as error:
                raise ValueError(f"{key}: {error}") from error
        for key in ("hard_joint_lower_rad", "hard_joint_upper_rad"):
            values[key] = _vector(values[key], 22, key)
        if np.any(values["hard_joint_lower_rad"] >= values["hard_joint_upper_rad"]):
            raise ValueError("Each hardware lower joint limit must be below its upper limit")
        gravity = _vector(values["gravity_direction_palm"], 3, "gravity_direction_palm")
        if not np.isclose(np.linalg.norm(gravity), 1.0, atol=1e-4, rtol=0):
            raise ValueError("gravity_direction_palm must be a unit vector")
        values["gravity_direction_palm"] = gravity
        if type(values["tag_id"]) is not int or not 0 <= values["tag_id"] < 587:
            raise ValueError("tag_id must be an integer in the AprilTag 36h11 range [0, 586]")
        for key, maximum in (("tag_size_m", 1.0), ("speed_coeff", 0.30), ("current_coeff", 0.30)):
            value = values.get(key, 0.30)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{key} must be finite and in (0, {maximum}]")
        for key in ("camera_serial", "hand_serial"):
            value = values.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be a nonempty string or null")
        sdk_command(values["sdk_container"], values["sdk_root"], values["sdk_python"])
        return cls(**values)


@dataclass(frozen=True)
class PreparedDeployment:
    config: DeploymentConfig
    bundle: PolicyBundle
    initial: dict
    mask: np.ndarray
    target_pose: torch.Tensor
    report: dict


def prepare_deployment(config_file: Path, bundle_dir: Path) -> PreparedDeployment:
    """Validate local files and policy compatibility without importing device SDKs.

    This does not discover hardware, launch Docker, load FoundationPose weights,
    open a camera, or assert that the installed native/CUDA dependencies work.
    """
    cfg = DeploymentConfig.load(config_file)
    bundle = PolicyBundle.load(bundle_dir)
    if bundle.observation_dim != 244:
        raise ValueError("Hardware deployment requires the 244D profile; the tactile hardware adapter is not implemented.")
    record = bundle.manifest["initial_grasp"]
    initial_path = bundle.root / record["tensor_filename"]
    if hashlib.sha256(initial_path.read_bytes()).hexdigest() != record["tensor_sha256"]:
        raise ValueError("Initial grasp hash mismatch")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    lower = _vector(bundle.manifest["sim_contract"]["joint_lower"], 22, "policy joint_lower")
    upper = _vector(bundle.manifest["sim_contract"]["joint_upper"], 22, "policy joint_upper")
    if np.any(lower >= upper) or np.any(lower < cfg.hard_joint_lower_rad) or np.any(upper > cfg.hard_joint_upper_rad):
        raise ValueError("Policy joint limits must fit the configured hardware envelope")
    for key in ("q", "q_cmd"):
        vector = _vector(initial[key], 22, f"initial grasp {key}")
        if np.any(vector < cfg.hard_joint_lower_rad) or np.any(vector > cfg.hard_joint_upper_rad):
            raise ValueError(f"Initial grasp {key} is outside hardware joint limits")
        if key == "q_cmd" and (np.any(vector < lower) or np.any(vector > upper)):
            raise ValueError("Initial grasp q_cmd is outside policy joint limits")
    pose = _vector(initial["object_pose_palm"], 7, "initial object pose")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Initial object pose must have a unit quaternion")
    dt = float(bundle.manifest["sim_contract"]["control_dt_s"])
    control = bundle.manifest["control"]
    if not all(np.isclose(float(value), 0.05, atol=1e-9, rtol=0) for value in (dt, control["control_dt_s"])) or not np.isclose(float(control["frequency_hz"]), 20.0):
        raise ValueError("This deployment runtime requires the trained 20 Hz control contract")
    target_pose = transform_to_pose(cfg.t_hand_target)
    validate_target(bundle, target_pose)
    # Exercise the actual single-frame actor -> incremental-command path. This
    # caught a real (1,22) vs (22,) integration failure missed by batch replays.
    controller = PolicyController(bundle, initial["q"], initial["q_cmd"], initial["object_pose_palm"],
                                  target_pose, cfg.gravity_direction_palm, timestamp_ms=0)
    proposed = controller.propose(initial["q"], initial["object_pose_palm"],
                                  timestamp_ms=dt * 1000, pose_age_s=0, joint_age_s=0)
    controller.acknowledge(proposed)  # Offline only; this target is never sent.
    if not (cfg.foundationpose_root / "estimater.py").is_file():
        raise FileNotFoundError(f"FoundationPose estimater.py missing in {cfg.foundationpose_root}")
    for path in (cfg.mesh, cfg.registration_mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    import cv2
    import trimesh

    mask = cv2.imread(str(cfg.registration_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (720, 1280) or np.count_nonzero(mask) < 50:
        raise ValueError("Registration mask must be a 1280x720 image containing at least 50 object pixels")
    mesh = trimesh.load(cfg.mesh, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Tracking mesh must be a finite triangle mesh")
    # Convert mesh coordinates into the simulator object frame before comparing size.
    vertices = (np.asarray(mesh.vertices) - cfg.t_mesh_object[:3, 3]) @ cfg.t_mesh_object[:3, :3]
    extents = np.ptp(vertices, axis=0)
    expected = _vector(bundle.manifest["physical_object"]["dimensions_m"], 3, "physical object dimensions")
    if np.any(expected <= 0) or not np.allclose(extents, expected, atol=0.001, rtol=0.02):
        raise ValueError(f"Tracking mesh size/frame mismatch: {extents.tolist()} m; bundle expects {expected.tolist()} m")
    report = {
        "valid": True, "hardware_contacted": False,
        "object_id": bundle.manifest["physical_object"]["stable_cache_object_id"],
        "control_frequency_hz": 20, "mesh_extents_m": extents.tolist(),
        "mask_pixels": int(np.count_nonzero(mask)),
        "sdk_container": cfg.sdk_container, "sdk_root": cfg.sdk_root,
        "checks": ["configuration", "policy_hash", "policy_inference", "initial_grasp_hash", "joint_limits",
                   "control_contract", "target_workspace_and_sdf", "mesh_dimensions", "mask"],
    }
    return PreparedDeployment(cfg, bundle, initial, mask, target_pose, report)
