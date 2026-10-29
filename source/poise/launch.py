"""Resolve setup commands without importing a simulator or optional optimizer."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from poise.paths import workspace_root
from poise.setups import data_path, load_setup, read_pinned, repository_root


def command(args):
    setup = load_setup(args.setup)
    root = repository_root()
    python = sys.executable
    if args.command == "train":
        profile = args.profile
        training = setup["training"]
        if profile != "custom" and (args.num_envs is not None or args.iterations is not None):
            raise ValueError("Use --profile custom when overriding num-envs/iterations")
        if profile == "reference":
            prefix = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                      f"--nproc_per_node={training['world_size']}"]
            options = ["--distributed", "--physical-gpus", str(training["physical_gpus"]), "--backend", training["backend"]]
            num_envs, iterations = training["envs_per_rank"], training["iterations"]
        else:
            prefix, options = [python], []
            num_envs = training["smoke_num_envs"] if profile == "smoke" else (args.num_envs if args.num_envs is not None else 128)
            iterations = training["smoke_iterations"] if profile == "smoke" else (args.iterations if args.iterations is not None else 100)
        if num_envs < len(setup["objects"]) or iterations < 1:
            raise ValueError("Training needs at least one environment per setup object and positive iterations")
        cmd = prefix + [str(root / "scripts/train.py"), "--setup", setup["id"], "--num-envs", str(num_envs),
                        "--iterations", str(iterations), "--seed", str(args.seed if args.seed is not None else training["seed"]),
                        "--headless", *options]
        if args.resume:
            cmd += ["--resume", str(args.resume.absolute())]
    else:
        rollout = read_pinned(root / setup["rollout"]["config"], setup["rollout"]["sha256"])
        num_envs = args.num_envs if args.num_envs is not None else rollout["num_envs"]
        steps = args.steps if args.steps is not None else rollout["steps"]
        if num_envs < len(setup["objects"]) or steps < 0:
            raise ValueError("Rollout needs at least one environment per setup object and nonnegative steps")
        if args.verify_interface and steps == 0:
            raise ValueError("Interface verification requires finite positive steps")
        checkpoint = args.checkpoint.absolute() if args.checkpoint else data_path(workspace_root(), setup["reference_checkpoint"]["path"])
        cmd = [python, str(root / "scripts/play.py"), "--setup", setup["id"], "--checkpoint", str(checkpoint),
               "--num-envs", str(num_envs), "--steps", str(steps),
               "--seed", str(args.seed if args.seed is not None else rollout["seed"])]
        if not args.headless:
            if not 0 <= args.view_env < num_envs:
                raise ValueError("view-env must index an existing rollout environment")
            cmd += ["--visualize-pose-frames", "--show-metrics", "--real-time",
                    "--metrics-env", str(args.view_env), "--camera-eye", "0.45", "0.45", "0.85",
                    "--camera-target", "-0.08", "0.0", "0.58"]
        else:
            cmd += ["--headless"]
        if args.verify_interface:
            cmd += ["--verify-interface"]
    cmd += ["--device", args.device, "--output", str(args.output.absolute())]
    if args.kit_args:
        cmd += ["--kit_args", args.kit_args]
    return cmd


def run(args):
    cmd = command(args)
    if args.dry_run:
        return {"command": cmd, "data_root": str(workspace_root()), "launched": False}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository_root() / "source")
    env["POISE_ROOT"] = str(workspace_root())
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    subprocess.run(cmd, cwd=repository_root(), env=env, check=True)
    return {"command": cmd, "output": str(args.output.absolute()), "completed": True}
