"""Export a self-contained policy bundle using the live simulator contract."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from poise.core.policy_bundle import PolicyBundle, export_fused_policy
from poise.core.schema import observation_contract_for_dimension, SCHEMA_VERSION, joint_interface_manifest
from poise.grasp.cache import load_stable_grasp_cache
from poise.paths import resolve_path


def export_bundle(checkpoint: Path, output: Path, env, *, object_index=0, grasp_index=0):
    cfg = env.cfg
    observation_contract = observation_contract_for_dimension(cfg.observation_space)
    objects = list(cfg.primitive_object_ids)
    object_id = objects[object_index]
    geometry = torch.load(resolve_path(cfg.object_mesh_data_path), map_location="cpu", weights_only=False)
    descriptor = geometry["geometry_latents"][list(geometry["object_ids"]).index(object_id)]
    cache = load_stable_grasp_cache(resolve_path(cfg.primitive_stable_cache_paths[object_index]))
    if cache["object_id"] != object_id:
        raise ValueError("Object and reset-cache identity differ")
    output.mkdir(parents=True, exist_ok=False)
    initial = {key: cache[key][grasp_index].cpu() for key in ("q", "q_cmd", "object_pose_palm")}
    initial.update(object_id=object_id, cache_index=grasp_index)
    torch.save(initial, output / "initial_grasp.pt")
    policy = export_fused_policy(checkpoint, output / "policy.ts", activation="elu",
                                observation_dim=observation_contract.dimension)
    policy.update(filename="policy.ts", sha256=hashlib.sha256((output / "policy.ts").read_bytes()).hexdigest(), clip_actions=1.0)
    observation = asdict(observation_contract)
    observation["contract"] = observation.pop("contract_id")
    observation["object_descriptor"] = {"kind": "mesh_bps64", "dimension": 64, "object_id": object_id, "values": descriptor.tolist()}
    contract = {
        "joint_lower": env.action_lower.cpu().tolist(),
        "joint_upper": env.action_upper.cpu().tolist(),
        "control_dt_s": env.dt,
        "target_translation_scale_m": cfg.sharpa_deployable_target_translation_scale_m,
        "linear_velocity_clip_mps": cfg.sharpa_deployable_object_lin_vel_clip_mps,
        "angular_velocity_clip_radps": cfg.sharpa_deployable_object_ang_vel_clip_radps,
        "observation_contract": observation_contract.contract_id,
        "include_gravity_palm": True,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "policy": policy,
        "observation": observation, "joint_interface": joint_interface_manifest(),
        "physical_object": {"stable_cache_object_id": object_id, "dimensions_m": cfg.primitive_dims_m[object_index]},
        "control": {"control_dt_s": env.dt, "frequency_hz": 1.0 / env.dt,
                    "action_scale_rad": cfg.action_scale, "action_smoothing": cfg.action_smoothing},
        "sim_contract": contract,
        "initial_grasp": {"tensor_filename": "initial_grasp.pt", "cache_index": grasp_index,
                          "tensor_sha256": hashlib.sha256((output / "initial_grasp.pt").read_bytes()).hexdigest()},
        "training": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "target_mode": cfg.target_mode, "recipe": cfg.to_dict()},
    }
    if env.tactile_enabled:
        manifest["tactile"] = {
            "actor_dimension": 304, "critic_dimension": 552,
            "history_shape": [3, 20], "order": "oldest_to_newest",
            "fingers": list(env.hand_spec.force_sensor_bodies),
            "frame": "five_normal_resultant_magnitudes_N_then_five_local_mean_contact_XYZ_m",
            "force_source": "object_filtered_force_matrix_w_history",
            "force_samples": 2, "sample_dt_s": cfg.sim.dt, "history_dt_s": env.step_dt,
            "hold_probability": env.tactile.hold_probability,
            "hold_scope": "force_and_position_per_finger",
            "hardware_adapter_supported": False,
        }
    if Path(cfg.robot_cfg.spawn.usd_path).resolve() == resolve_path("data/robots/sharpa_wave/right_sharpa_wave.usda").resolve():
        from poise.assets.hand import check_hand, hand_manifest
        identity = check_hand()
        identity.pop("directory")
        manifest["training"]["hand_asset"] = {**identity, "files": hand_manifest()["files"]}
    if cfg.pose6d_feasible_target_forbidden_mode == "sdf_grid":
        sdf_path = resolve_path(cfg.pose6d_feasible_target_forbidden_sdf_path)
        sdf_hash = hashlib.sha256(sdf_path.read_bytes()).hexdigest()
        if sdf_hash != cfg.pose6d_feasible_target_forbidden_sdf_sha256:
            raise ValueError("Hand collision SDF hash mismatch")
        shutil.copy2(sdf_path, output / "hand_sdf.npz")
        manifest["target_feasibility"] = {
            "sdf_filename": "hand_sdf.npz", "sdf_sha256": sdf_hash,
            "workspace_bounds_m": cfg.pose6d_feasible_target_workspace_bounds_m,
            "proxy_points_local_m": cfg.pose6d_feasible_target_forbidden_sdf_proxy_points_local_m,
            "proxy_radii_m": cfg.pose6d_feasible_target_forbidden_sdf_proxy_radii_m,
            "margin_m": cfg.pose6d_feasible_target_forbidden_sdf_margin_m,
            "interpolation_guard_m": cfg.pose6d_feasible_target_forbidden_sdf_interpolation_guard_m,
            "allowed_penetration_m": cfg.pose6d_feasible_target_forbidden_sdf_allowed_penetration_m,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    bundle = PolicyBundle.load(output)
    probe = torch.linspace(-1, 1, observation_contract.dimension).reshape(1, -1)
    if not torch.isfinite(bundle.infer(probe)).all():
        raise RuntimeError("Exported policy produced non-finite actions")
    return manifest
