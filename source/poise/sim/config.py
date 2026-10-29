"""Flat POISE training configuration. Import after starting Isaac Sim."""

import json
from importlib.resources import files
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from poise.paths import resolve_path
from poise.core.tactile import TACTILE_PROFILE
from poise.sim.assets.cube_4cm import get_cube_4cm_cfg
from poise.sim.assets.sharpa_hand import get_sharpa_wave_hand_cfg


@configclass
class PoiseEnvCfg(DirectRLEnvCfg):
    decimation = 6
    episode_length_s = 20.0
    action_space = 22
    observation_space = 244
    state_space = 492
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=0.55)


def _update(target, values, namespace=""):
    """Merge a saved configuration into explicitly constructed Isaac configs."""
    for key, value in values.items():
        if not hasattr(target, key) and not isinstance(target, dict):
            raise KeyError(f"Unsupported Isaac Lab configuration field: {namespace}{key}")
        current = target.get(key) if isinstance(target, dict) else getattr(target, key)
        if isinstance(value, dict):
            if current is None:
                raise ValueError(f"Missing configuration type for {namespace}{key}")
            _update(current, value, f"{namespace}{key}.")
            continue
        if callable(current):
            # Callable types come from our constructors, never from recipe text.
            continue
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)


def load_env_cfg(recipe: str | Path = "hexprism40", *, num_envs=128, device="cuda:0", seed=42):
    """Load a built-in recipe or a user-authored JSON recipe."""
    path = Path(recipe)
    if not path.is_file():
        path = files("poise.sim.recipes").joinpath(f"{recipe}.json")
    values = json.loads(path.read_text())
    if values["hand_type"] != "sharpa_wave_right":
        raise ValueError("Only the 22-DoF Sharpa Wave right hand is supported.")
    if values["observation_profile"] not in ("sharpa_pose6d_deployable_v1_notactile", TACTILE_PROFILE):
        raise ValueError("POISE requires the BPS64 deployable observation profile.")
    if not values["sharpa_deployable_gravity_palm_obs_enabled"] or values["primitive_geometry_latent_dim"] != 64:
        raise ValueError("POISE requires BPS64 and palm-frame gravity observations.")
    if not values["sharpa_deployable_mesh_bps_obs_enabled"] or not values["sharpa_deployable_asymmetric_critic_enabled"] or not values["dexndm_reward_mode"]:
        raise ValueError("POISE requires BPS64, the asymmetric critic and the reaching reward.")
    if values["target_mode"] != "hand_frame_twist_translate" or values["reset_source"] != "stable_cache":
        raise ValueError("POISE requires stable-grasp resets and SE(3) reaching targets.")
    cfg = PoiseEnvCfg()
    cfg.robot_cfg = get_sharpa_wave_hand_cfg(usd_path=resolve_path(values["robot_cfg"]["spawn"]["usd_path"]))
    values["robot_cfg"]["spawn"]["usd_path"] = cfg.robot_cfg.spawn.usd_path
    cfg.cube_cfg = get_cube_4cm_cfg(prim_path="/World/envs/env_.*/Cube", init_pos=(0.0, 0.0, 0.0))
    cfg.ground_cfg = sim_utils.GroundPlaneCfg()
    nested = {"viewer", "sim", "scene", "robot_cfg", "cube_cfg", "ground_cfg"}
    for key, value in values.items():
        if key in nested:
            _update(getattr(cfg, key), value, f"{key}.")
        elif key == "ui_window_class_type":
            continue
        else:
            setattr(cfg, key, value)
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.seed = int(seed)
    if values["observation_profile"] == TACTILE_PROFILE:
        cfg.observation_space = 304
        cfg.state_space = 552
    return cfg


def load_ppo_cfg(recipe="ppo"):
    return json.loads(files("poise.sim.recipes").joinpath(f"{recipe}.json").read_text())
