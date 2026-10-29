"""Run a checkpoint in Isaac Lab, export its bundle, and verify deployment inputs."""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--recipe", help="Legacy/custom recipe; defaults to hexprism40")
    selection.add_argument("--setup", help="Versioned setup and its explicit rollout settings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16, help="Policy steps; 0 plays until the window closes or Ctrl+C.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-interface", action="store_true", help="Disable sensor noise and verify every observation/action offline.")
    parser.add_argument("--visualize-pose-frames", action="store_true", help="Show current and target object XYZ axes (red/green/blue).")
    parser.add_argument("--object-frame-scale-m", type=float, default=0.045, help="Current object axis length in meters.")
    parser.add_argument("--target-frame-scale-m", type=float, default=0.075, help="Target object axis length in meters.")
    parser.add_argument("--show-palm-frame", action="store_true")
    parser.add_argument("--palm-frame-scale-m", type=float, default=0.03)
    parser.add_argument("--show-target-position", action="store_true", help="Mark current and target object origins in cyan and yellow.")
    parser.add_argument("--show-position-error", action="store_true", help="Draw a yellow line from current to target object origin.")
    parser.add_argument("--show-metrics", action="store_true", help="Show English pose-error and training-goal metrics in a GUI panel.")
    parser.add_argument("--metrics-env", type=int, default=0)
    parser.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Camera position relative to the selected environment.")
    parser.add_argument("--camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Camera look-at point relative to the selected environment.")
    parser.add_argument("--real-time", action="store_true", help="Pace playback at the policy control rate; slower rendering stays slower.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or not 0 <= args.metrics_env < args.num_envs:
        parser.error("num-envs must be positive, steps nonnegative, and metrics-env a valid environment index")
    if min(args.object_frame_scale_m, args.target_frame_scale_m, args.palm_frame_scale_m) <= 0:
        parser.error("Frame lengths must be positive")
    if args.verify_interface and args.steps == 0:
        parser.error("Interface verification requires a finite, positive --steps")
    setup = checkpoint_identity = rollout = None
    if args.setup:
        from poise.setups import load_setup, require_setup, check_checkpoint, setup_contract, repository_root, read_pinned
        setup = load_setup(args.setup)
        require_setup(setup, deep=True)
        checkpoint_identity = check_checkpoint(setup, args.checkpoint)
        rollout = read_pinned(repository_root() / setup["rollout"]["config"], setup["rollout"]["sha256"])
        args.recipe = setup["recipe"]["name"]
        if args.num_envs < len(setup["objects"]):
            parser.error("Use at least one environment per setup object")
    else:
        args.recipe = args.recipe or "hexprism40"
    app = AppLauncher(args).app
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from poise.core.golden import GOLDEN_SCHEMA_VERSION, verify_golden_reference
        from poise.sim.config import load_env_cfg, load_ppo_cfg
        from poise.sim.env import PoiseEnv
        from poise.training.ppo import BoundedStdActorCritic
        from poise.training.export import export_bundle
        from poise.training.golden import capture_frame, stack_frames

        torch.set_num_threads(8)
        cfg = load_env_cfg(args.recipe, num_envs=args.num_envs, device=args.device, seed=args.seed)
        if args.verify_interface and cfg.observation_space != 244:
            raise ValueError("Tactile simulation rollout is supported; hardware interface verification requires the 244D profile.")
        if rollout:
            for key, value in rollout["overrides"].items():
                if not hasattr(cfg, key):
                    raise ValueError(f"Unknown rollout override: {key}")
                setattr(cfg, key, value)
        if args.camera_eye is not None or args.camera_target is not None:
            cfg.viewer.origin_type = "env"
            cfg.viewer.env_index = args.metrics_env
            if args.camera_eye is not None:
                cfg.viewer.eye = tuple(args.camera_eye)
            if args.camera_target is not None:
                cfg.viewer.lookat = tuple(args.camera_target)
        if args.verify_interface:
            cfg.sharpa_deployable_joint_sensing_noise_scale = 0.0
            cfg.sharpa_deployable_object_pose_obs_delay_noise_enabled = False
            cfg.sharpa_deployable_object_pose_obs_pos_noise_std_m = 0.0
            cfg.sharpa_deployable_object_pose_obs_rot_noise_deg = 0.0
            cfg.sharpa_deployable_object_pose_obs_dropout_prob = 0.0
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
        if setup:
            (args.output / "setup.json").write_text(json.dumps(setup_contract(setup), indent=2))
            (args.output / "rollout.json").write_text(json.dumps(rollout, indent=2))
        env = RslRlVecEnvWrapper(PoiseEnv(cfg), clip_actions=1.0)
        base = env.unwrapped
        obs, _ = env.reset()
        agent = load_ppo_cfg(setup["ppo"]["name"] if setup else "ppo")
        policy_cfg = agent["policy"]
        policy_cfg.pop("class_name")
        policy = BoundedStdActorCritic(obs, agent["obs_groups"], 22, **policy_cfg).to(args.device)
        policy.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=False)["model_state_dict"])
        policy.eval()
        manifest = export_bundle(args.checkpoint, args.output / "bundle", base)
        object_bundles = [(args.output / "bundle", manifest)]
        for index in range(1, len(cfg.primitive_object_ids)):
            bundle_dir = args.output / "bundles" / cfg.primitive_object_ids[index]
            object_bundles.append((bundle_dir, export_bundle(args.checkpoint, bundle_dir, base, object_index=index)))
        visuals = None
        if any((args.visualize_pose_frames, args.show_palm_frame, args.show_target_position, args.show_position_error, args.show_metrics)):
            from poise.sim.rollout_visuals import RolloutVisuals
            visuals = RolloutVisuals(base, args)
            visuals.update(0)
            base.sim.render()
            print("POISE_VISUALS_READY " + json.dumps(visuals.latest), flush=True)
        frames = [capture_frame(base, obs["policy"])] if args.verify_interface else []
        initial_object_indices = base.active_cube_size_id.detach().cpu().clone() if args.verify_interface else None
        actions, terminations, reward_sum, completed_steps = [], 0, 0.0, 0
        stop_requested = False

        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True

        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        try:
            while not stop_requested and app.is_running() and (args.steps == 0 or completed_steps < args.steps):
                started = time.perf_counter()
                with torch.inference_mode():
                    action = policy.act_inference(obs)
                if args.verify_interface:
                    actions.append(action.detach().cpu().clone())
                obs, reward, done, _ = env.step(action)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("Non-finite simulator output")
                terminations += int(done.sum())
                if args.verify_interface and done.any():
                    raise RuntimeError(f"Interface rollout reset at step {completed_steps}; use a shorter window or a stable seed")
                reward_sum += reward.mean().item()
                completed_steps += 1
                if args.verify_interface:
                    frames.append(capture_frame(base, obs["policy"]))
                if visuals:
                    visuals.update(completed_steps)
                    base.sim.render()
                if args.real_time:
                    time.sleep(max(0.0, base.step_dt - (time.perf_counter() - started)))
        except KeyboardInterrupt:
            stop_requested = True
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        if stop_requested:
            print("POISE_PLAY_STOPPED by user", flush=True)
        report = {"steps": completed_steps, "num_envs": args.num_envs, "terminations": terminations,
                  "mean_reward": reward_sum / completed_steps if completed_steps else None}
        if setup:
            report.update(setup=setup_contract(setup), checkpoint=checkpoint_identity,
                          purpose=rollout["purpose"], object_ids=list(cfg.primitive_object_ids))
        if visuals:
            report["visualization"] = visuals.latest
            visuals.close()
        if args.verify_interface:
            if completed_steps != args.steps:
                raise RuntimeError("Interface verification stopped before completing the requested steps")
            all_frames, all_actions = stack_frames(frames), torch.stack(actions)
            checks = {}
            for index, (bundle_dir, bundle_manifest) in enumerate(object_bundles):
                mask = initial_object_indices == index
                if not mask.any():
                    raise ValueError(f"No rollout environment covers object {cfg.primitive_object_ids[index]}")
                golden = {"schema_version": GOLDEN_SCHEMA_VERSION,
                          "frames": {key: value[:, mask] for key, value in all_frames.items()},
                          "sim_contract": bundle_manifest["sim_contract"],
                          "reference_action": all_actions[:, mask],
                          "metadata": {"policy_sha256": bundle_manifest["policy"]["sha256"]}}
                golden_path = args.output / ("golden.pt" if len(object_bundles) == 1 else f"golden_{index}.pt")
                torch.save(golden, golden_path)
                checks[cfg.primitive_object_ids[index]] = verify_golden_reference(golden_path, bundle_dir)
            report["interface"] = next(iter(checks.values())) if len(checks) == 1 else checks
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print("POISE_PLAY_COMPLETE " + json.dumps(report), flush=True)
        env.close()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
