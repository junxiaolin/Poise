"""Render rollout pose frames and errors without adding physics bodies."""

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from poise.sim.env import quat_angle_error_wxyz, quat_apply_wxyz


def _material(color):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=tuple(c * 0.25 for c in color))


class _Frame:
    """Three positive-axis arrows built from local USD primitives (no downloads)."""

    def __init__(self, name, length):
        self.length = length
        colors = ((1.0, 0.08, 0.08), (0.08, 1.0, 0.08), (0.15, 0.35, 1.0))
        shafts, tips = {}, {}
        for axis, color in zip("XYZ", colors):
            shafts[axis] = sim_utils.CylinderCfg(
                radius=length * 0.018, height=length * 0.8, axis=axis,
                visual_material=_material(color),
            )
            tips[axis] = sim_utils.ConeCfg(
                radius=length * 0.045, height=length * 0.2, axis=axis,
                visual_material=_material(color),
            )
        self.shafts = VisualizationMarkers(VisualizationMarkersCfg(prim_path=f"/Visuals/Poise/{name}/Axes", markers=shafts))
        self.tips = VisualizationMarkers(VisualizationMarkersCfg(prim_path=f"/Visuals/Poise/{name}/Tips", markers=tips))

    def update(self, pose):
        count = pose.shape[0]
        quats = pose[:, None, 3:7].expand(-1, 3, -1).reshape(-1, 4)
        directions = torch.eye(3, device=pose.device, dtype=pose.dtype).repeat(count, 1)
        directions = quat_apply_wxyz(quats, directions)
        origins = pose[:, None, :3].expand(-1, 3, -1).reshape(-1, 3)
        indices = torch.arange(3, device=pose.device).repeat(count)
        self.shafts.visualize(origins + directions * self.length * 0.4, quats, marker_indices=indices)
        self.tips.visualize(origins + directions * self.length * 0.9, quats, marker_indices=indices)


class RolloutVisuals:
    """Pose markers for all environments; English metrics for the selected one.

    Current pose is simulator ground truth. Both it and the palm-relative goal
    are transformed into world coordinates before visualization.
    """

    def __init__(self, env, args):
        self.env = env
        self.env_ids = torch.arange(env.num_envs, device=env.device)
        self.metric_env = args.metrics_env
        self.frames = {}
        if args.visualize_pose_frames:
            self.frames["current"] = _Frame("CurrentObject", args.object_frame_scale_m)
            self.frames["target"] = _Frame("TargetObject", args.target_frame_scale_m)
        if args.show_palm_frame:
            self.frames["palm"] = _Frame("Palm", args.palm_frame_scale_m)
        self.origins = None
        if args.show_target_position:
            self.origins = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Poise/ObjectOrigins",
                markers={
                    "current": sim_utils.SphereCfg(radius=0.002, visual_material=_material((0.0, 0.9, 1.0))),
                    "target": sim_utils.SphereCfg(radius=0.003, visual_material=_material((1.0, 0.8, 0.0))),
                },
            ))
        self.line = None
        if args.show_position_error:
            self.line = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/Poise/PositionError",
                markers={"line": sim_utils.CylinderCfg(radius=0.0006, height=1.0, visual_material=_material((1.0, 0.8, 0.0)))},
            ))
        self.window = None
        self.labels = {}
        if args.show_metrics and env.sim.has_gui():
            import omni.ui as ui

            self.window = ui.Window("POISE Rollout", width=390, height=340, position_x=12, position_y=80)
            with self.window.frame:
                with ui.VStack(spacing=4):
                    ui.Label(f"Environment {args.metrics_env} | X: red, Y: green, Z: blue", height=22)
                    ui.Label(
                        f"Current axes: {args.object_frame_scale_m * 1000:g} mm | Target axes: {args.target_frame_scale_m * 1000:g} mm",
                        height=22,
                    )
                    if args.show_target_position:
                        ui.Label("Origins: current cyan | target yellow", height=22)
                    ui.Label("Current pose: simulator ground truth", height=22)
                    ui.Separator(height=4)
                    for key in ("time", "position", "rotation", "strict", "dwell", "goals", "curriculum"):
                        self.labels[key] = ui.Label("", height=22)
        self.latest = {}

    def update(self, step):
        env = self.env
        current = env._active_cube_root_pose_w()
        target = env._hand_to_world_pose(env.target_pose_palm, self.env_ids)
        if "current" in self.frames:
            self.frames["current"].update(current)
            self.frames["target"].update(target)
        if "palm" in self.frames:
            palm = torch.cat((env._hand_root_pos_w(current.dtype), env._hand_root_quat_w(current.dtype)), dim=-1)
            self.frames["palm"].update(palm)
        if self.origins:
            positions = torch.stack((current[:, :3], target[:, :3]), dim=1).reshape(-1, 3)
            indices = torch.tensor([0, 1], device=current.device).repeat(env.num_envs)
            self.origins.visualize(positions, marker_indices=indices)
        delta = target[:, :3] - current[:, :3]
        distance = torch.linalg.norm(delta, dim=-1)
        if self.line:
            # Rotate the cylinder's +Z axis onto the current-to-target vector.
            direction = delta / distance.clamp_min(1.0e-8).unsqueeze(-1)
            orientation = torch.stack((1.0 + direction[:, 2], -direction[:, 1], direction[:, 0], torch.zeros_like(distance)), dim=-1)
            norm = torch.linalg.norm(orientation, dim=-1, keepdim=True)
            opposite = orientation.new_tensor([0.0, 1.0, 0.0, 0.0]).expand_as(orientation)
            orientation = torch.where(norm > 1.0e-6, orientation / norm.clamp_min(1.0e-6), opposite)
            scales = torch.ones_like(delta)
            scales[:, 2] = distance.clamp_min(1.0e-8)
            self.line.visualize((current[:, :3] + target[:, :3]) * 0.5, orientation, scales)
        i = self.metric_env
        position_mm = distance[i].item() * 1000.0
        rotation_deg = math.degrees(quat_angle_error_wxyz(target[i:i+1, 3:7], current[i:i+1, 3:7]).item())
        self.latest = {
            "step": step, "env_index": i, "position_error_mm": position_mm,
            "orientation_error_deg": rotation_deg,
            "strict_pose_within_tolerance": position_mm < 10.0 and rotation_deg < 10.0,
            "training_dwell_steps": int(env.near_success_steps[i]),
            "episode_goals": int(env.achieved_goals[i]),
        }
        if self.labels:
            self.labels["time"].text = f"Step: {step} | Sim time: {step * env.step_dt:.2f} s"
            self.labels["position"].text = f"Position error: {position_mm:.2f} mm (Euclidean)"
            self.labels["rotation"].text = f"Rotation error: {rotation_deg:.2f} deg (SO3)"
            self.labels["strict"].text = f"Inside 10 mm / 10 deg now: {'YES' if self.latest['strict_pose_within_tolerance'] else 'NO'}"
            self.labels["dwell"].text = f"Training dwell: {self.latest['training_dwell_steps']} / {env.cfg.success_steps} steps"
            self.labels["goals"].text = f"Episode goals (training criterion): {self.latest['episode_goals']}"
            rotation_max = (
                env.cfg.twist_target_uniform_max_rad
                if env.cfg.twist_target_sampling_mode == "full_range_uniform"
                else env.adaptive_target_delta_max
            )
            translation_max = (
                env.cfg.pose6d_translation_uniform_max_m
                if env.cfg.pose6d_translation_sampling_mode == "full_range_uniform"
                else env.adaptive_translation_delta_max
            )
            self.labels["curriculum"].text = (
                f"Goal limits: {math.degrees(float(rotation_max)):.1f} deg / "
                f"{float(translation_max) * 1000:.1f} mm"
            )

    def close(self):
        if self.window:
            self.window.destroy()
            self.window = None
