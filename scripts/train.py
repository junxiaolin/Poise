"""Train POISE with Isaac Lab and RSL-RL, optionally through torchrun."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--recipe", help="Legacy/custom JSON recipe; defaults to hexprism40")
    selection.add_argument("--setup", help="Versioned setup with pinned assets and PPO settings")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--physical-gpus", type=int, help="Visible GPUs per node; logical ranks wrap onto these GPUs")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--global-frontier-min", type=int,
                        help="Frontier completed-event gate per level, per object when enabled (all ranks combined)")
    parser.add_argument("--resume-global-frontier-from", type=int,
                        help="Original gate in a resume checkpoint; allow only the explicit global-frontier-min change")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("iterations and num-envs must be positive")
    if args.global_frontier_min is not None and args.global_frontier_min < 1:
        parser.error("global-frontier-min must be positive")
    if args.resume_global_frontier_from is not None:
        if args.resume_global_frontier_from < 1 or args.resume is None or args.global_frontier_min is None:
            parser.error("resume-global-frontier-from requires a positive old gate, --resume and --global-frontier-min")
    setup = None
    if args.setup:
        from poise.setups import load_setup, require_setup, setup_contract
        setup = load_setup(args.setup)
        if args.num_envs < len(setup["objects"]):
            parser.error("Use at least one environment per setup object")
        require_setup(setup, deep=True)
        if args.global_frontier_min is not None and args.global_frontier_min != setup["training"]["global_frontier_min"]:
            parser.error("Changing the setup curriculum gate requires a new/custom recipe")
        args.recipe = setup["recipe"]["name"]
    else:
        args.recipe = args.recipe or "hexprism40"
    from poise.training.distributed import configure_launch
    rank, world_size, physical_rank = configure_launch(args.distributed, args.physical_gpus, args.backend)
    if args.distributed:
        args.device = f"cuda:{physical_rank}"
    seed = args.seed + rank
    launcher = AppLauncher(args)
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from poise.core.policy_bundle import export_fused_policy
        from poise.paths import resolve_path
        from poise.sim.config import load_env_cfg, load_ppo_cfg
        from poise.sim.env import PoiseEnv
        from poise.training.runner import PoiseRunner
        from poise.training.export import export_bundle
        from poise.training.distributed import initialize_process_group, curriculum_launch_settings

        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        initialize_process_group(args.device)
        shared_output = [str((args.output or resolve_path("outputs/training") / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve())]
        if args.distributed:
            torch.distributed.broadcast_object_list(shared_output, src=0)
        output = Path(shared_output[0])
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        if args.distributed:
            torch.distributed.barrier()
        cfg = load_env_cfg(args.recipe, num_envs=args.num_envs, device=args.device, seed=seed)
        curriculum_settings = curriculum_launch_settings(cfg, args.global_frontier_min)
        cfg.log_dir = str(output)
        agent = load_ppo_cfg(setup["ppo"]["name"] if setup else "ppo")
        agent.update(seed=seed, device=args.device, max_iterations=args.iterations)
        config_dir = output if rank == 0 else output / f"rank_{rank}"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
        (config_dir / "agent.json").write_text(json.dumps(agent, indent=2))
        if setup:
            (config_dir / "setup.json").write_text(json.dumps(setup_contract(setup), indent=2))
        topology = {
            "rank": rank, "world_size": world_size, "device": args.device,
            "seed": seed, "base_seed": args.seed, "envs_per_rank": args.num_envs,
            "total_envs": args.num_envs * world_size,
            "backend": args.backend if args.distributed else None,
            **curriculum_settings,
        }
        (config_dir / "topology.json").write_text(json.dumps(topology, indent=2))
        print("POISE_TOPOLOGY " + json.dumps(topology), flush=True)
        env = RslRlVecEnvWrapper(PoiseEnv(cfg), clip_actions=1.0)
        runner = PoiseRunner(env, agent, log_dir=str(output), device=args.device)
        if setup:
            runner.setup_contract = setup_contract(setup)
        if args.resume:
            runner.load(str(args.resume.resolve()), map_location=args.device,
                        resume_global_frontier_from=args.resume_global_frontier_from)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=args.resume is None)
        if args.distributed:
            torch.distributed.barrier()
        if rank != 0:
            torch.distributed.destroy_process_group()
            env.close()
            return
        checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        metadata = export_fused_policy(checkpoint, output / "policy.ts", activation=agent["policy"]["activation"],
                                      observation_dim=cfg.observation_space)
        (output / "export.json").write_text(json.dumps(metadata, indent=2, default=str))
        for index, object_id in enumerate(cfg.primitive_object_ids):
            export_bundle(checkpoint, output / "bundles" / object_id, env.unwrapped, object_index=index)
        if any(name == "ap2ap_tasks" or name.startswith("ap2ap_tasks.") for name in sys.modules):
            raise RuntimeError("Unexpected research-repository import")
        report = {
            "num_envs": args.num_envs * world_size, "envs_per_rank": args.num_envs,
            "world_size": world_size, "iterations_this_run": args.iterations,
            "transitions_this_run": args.num_envs * world_size * agent["num_steps_per_env"] * args.iterations,
            "completed_iterations": runner.current_learning_iteration + 1,
            "checkpoint": str(checkpoint), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(), "research_imports": False,
            "export": metadata,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str))
        print("POISE_TRAINING_COMPLETE " + json.dumps(report, default=str), flush=True)
        if args.distributed:
            torch.distributed.destroy_process_group()
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit's shutdown can terminate Python with code 0, masking a failed run.
        # Exit the failed simulator process explicitly after flushing diagnostics.
        os._exit(1)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
