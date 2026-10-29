"""Audit filtered tactile observations in real Isaac physics before training."""
import argparse
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="hexprism40_omniwrist_tactile_v1")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import torch
        from poise.sim.config import load_env_cfg
        from poise.sim.env import PoiseEnv
        from poise.core.observation import quat_apply_wxyz
        cfg = load_env_cfg(args.recipe, num_envs=args.num_envs, device=args.device, seed=42)
        env = PoiseEnv(cfg)
        obs, _ = env.reset()
        assert obs['policy'].shape == (args.num_envs, 304)
        assert obs['critic'].shape == (args.num_envs, 552)
        assert not obs['policy'][:, -60:].any()
        indices = [env.contact_sensor_body_name_to_idx[n] for n in env.hand_spec.force_sensor_bodies]
        samples = 0
        maximum_force = maximum_local = maximum_error = 0.
        # Disable readout holds just for the direct sensor/coordinate audit.
        env.tactile.hold_probability = 0.
        with torch.inference_mode():
            for step in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 22, device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()
                assert torch.isfinite(reward).all()
                torch.testing.assert_close(obs['policy'][:, -60:], obs['critic'][:, -60:])
                repeated = env._get_observations()
                torch.testing.assert_close(repeated['policy'][:, -60:], obs['policy'][:, -60:])
                frame = env.tactile.frames[:, -1]
                live = (frame[:, :5] > 0) & ~env.tactile.pending[:, None]
                samples += int(live.sum())
                data = env.cube_contact_sensor.data
                pair_forces = data.force_matrix_w_history[:, :2, 0, indices]
                expected = pair_forces.norm(dim=-1).mean(1)
                torch.testing.assert_close(frame[:, :5][live], expected[live])
                local = frame[:, 5:].reshape(args.num_envs, 5, 3)
                pose = env.hand.data.body_link_state_w[:, env.tactile_body_ids, :7]
                world = pose[..., :3] + quat_apply_wxyz(pose[..., 3:], local)
                if live.any():
                    error = (world - data.contact_pos_w[:, 0, indices])[live].abs().max().item()
                    maximum_error = max(maximum_error, error)
                    maximum_force = max(maximum_force, frame[:, :5].max().item())
                    maximum_local = max(maximum_local, local[live].norm(dim=-1).max().item())
                if step == args.steps // 2:
                    ids = torch.arange(0, args.num_envs, 2, device=env.device)
                    env._reset_idx(ids)
                    reset_obs = env._get_observations()
                    assert not reset_obs['policy'][ids, -60:].any()
                    assert not reset_obs['critic'][ids, -60:].any()
            assert samples > 0, 'No actual fingertip contacts observed'
            assert maximum_error < 1e-5
            assert maximum_local < .1, 'Contact points are not local to fingertip geometry'
        result = dict(status='passed',num_envs=args.num_envs,steps=args.steps,
                      actor_dim=304,critic_dim=552,contact_samples=samples,
                      max_force_n=maximum_force,max_local_contact_norm_m=maximum_local,
                      max_world_reconstruction_error_m=maximum_error,
                      same_actor_critic_history=True,reset_clears_history=True,
                      repeated_observation_does_not_advance_history=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print('POISE_TACTILE_VERIFIED '+json.dumps(result),flush=True)
        env.close()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    finally:
        app.close()


if __name__ == '__main__':
    main()
