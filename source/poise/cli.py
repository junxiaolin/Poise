"""POISE setups, local asset preparation, simulation and offline deployment checks."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("interface", help="Print the Sharpa Wave observation/action interface")
    cache = commands.add_parser("check-cache", help="Validate a trusted Sharpa reset cache")
    cache.add_argument("cache", type=Path)
    golden = commands.add_parser("verify-golden", help="Check an existing Isaac Lab golden recording")
    golden.add_argument("--bundle", type=Path, required=True)
    golden.add_argument("--golden", type=Path, required=True)
    template = commands.add_parser("init-deploy", help="Write a Sharpa deployment configuration template")
    template.add_argument("output", type=Path)
    deploy = commands.add_parser("check-deploy", help="Check deployment files offline; no hardware access")
    deploy.add_argument("--bundle", type=Path, required=True)
    deploy.add_argument("--config", type=Path, required=True)
    fetch = commands.add_parser("fetch-hand", help="Install pinned public Sharpa assets and POISE collision settings")
    fetch.add_argument("--source", type=Path, help="Optional local public-repository checkout at the pinned revision")
    fetch.add_argument("--destination", type=Path)
    fetch.add_argument("--replace", action="store_true", help="Keep a backup of previous assets and switch to the public source")
    check = commands.add_parser("check-hand", help="Verify all hand files against pinned checksums")
    check.add_argument("--destination", type=Path)
    commands.add_parser("list-setups", help="List versioned training/rollout setups")
    show = commands.add_parser("show-setup", help="Show configuration and reference identity")
    show.add_argument("setup")
    env = commands.add_parser("check-env", help="Inspect one independent environment; no camera/robot connection")
    env.add_argument("stage", choices=("train", "lygra-poise", "foundationpose-poise"))
    env.add_argument("--native", action="store_true")
    env.add_argument("--foundationpose-root", type=Path)
    grasp_assets = commands.add_parser("fetch-grasp-assets", help="Fetch Hex40 search inputs and optional joint-limit reference")
    grasp_assets.add_argument("--source", type=Path)
    grasp_assets.add_argument("--root", type=Path)
    for name in ("check-assets", "fetch-assets", "package-assets"):
        item = commands.add_parser(name, help={
            "check-assets": "Check setup assets before simulation",
            "fetch-assets": "Fetch pinned shared release packages or import local assets",
            "package-assets": "Create a local private bundle; never uploads",
        }[name])
        item.add_argument("--setup", required=True)
        item.add_argument("--root", type=Path, help="Data workspace; defaults to POISE_ROOT")
        item.add_argument("--with-reference", action="store_true")
        if name == "check-assets":
            item.add_argument("--deep", action="store_true", help="Also validate trusted cache/geometry/checkpoint contents")
        elif name == "fetch-assets":
            item.add_argument("--source", type=Path, help="Local release directory, workspace directory or tar bundle")
        else:
            item.add_argument("--output", type=Path, required=True)
    for name in ("train", "rollout"):
        item = commands.add_parser(name, help=f"Run setup {name} using this Python environment")
        item.add_argument("--setup", required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--num-envs", type=int)
        item.add_argument("--seed", type=int)
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--kit-args")
        item.add_argument("--dry-run", action="store_true")
        if name == "train":
            item.add_argument("--profile", choices=("smoke", "custom", "reference"), default="smoke")
            item.add_argument("--iterations", type=int)
            item.add_argument("--resume", type=Path)
        else:
            item.add_argument("--checkpoint", type=Path, help="Defaults to the setup's pinned internal reference")
            item.add_argument("--steps", type=int, help="Defaults to setup rollout settings; 0 runs until stopped")
            item.add_argument("--headless", action="store_true")
            item.add_argument("--verify-interface", action="store_true")
            item.add_argument("--view-env", type=int, default=0, help="GUI camera/metrics environment index")
    args = parser.parse_args()
    if args.command == "list-setups":
        from poise.setups import list_setups
        result = [{"id": s["id"], "description": s["description"], "status": s["status"],
                   "objects": [o["id"] for o in s["objects"]]} for s in list_setups()]
    elif args.command == "show-setup":
        from poise.setups import load_setup
        result = load_setup(args.setup)
    elif args.command == "check-env":
        from poise.environments import check_environment
        result = check_environment(args.stage, native=args.native, foundationpose_root=args.foundationpose_root)
        if not result["valid"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
    elif args.command in ("train", "rollout"):
        from poise.launch import run
        if args.num_envs is not None and args.num_envs < 1:
            parser.error("num-envs must be positive")
        if args.command == "train" and args.iterations is not None and args.iterations < 1:
            parser.error("iterations must be positive")
        result = run(args)
    elif args.command == "fetch-grasp-assets":
        from poise.setups import load_setup, repository_root, sha256
        from poise.assets.setup_data import install_assets
        path = repository_root() / 'manifests/grasp-assets.json'
        grasp = dict(id='grasp_hexprism40_v1', assets_manifest='manifests/grasp-assets.json', assets_manifest_sha256=sha256(path))
        result = dict(object_assets=install_assets(load_setup('hexprism40_palmup_v1'), source=args.source, root=args.root),
                      generation_assets=install_assets(grasp, source=args.source, root=args.root))
    elif args.command in ("check-assets", "fetch-assets", "package-assets"):
        from poise.setups import load_setup, check_setup
        from poise.assets.setup_data import install_assets, package_assets
        setup = load_setup(args.setup)
        options = dict(root=args.root, with_reference=args.with_reference)
        if args.command == "check-assets":
            result = check_setup(setup, deep=args.deep, **options)
            if not result["valid"]:
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
        elif args.command == "fetch-assets":
            result = install_assets(setup, source=args.source, **options)
        else:
            result = package_assets(setup, args.output, **options)
    elif args.command == "interface":
        from poise.core.schema import ACTION_DIM, HAND_ID, OBSERVATION_DIM, joint_interface_manifest, observation_layout_dicts
        result = {"hand": HAND_ID, "action_dim": ACTION_DIM, "observation_dim": OBSERVATION_DIM,
                  "joint_interface": joint_interface_manifest(), "fields": observation_layout_dicts()}
    elif args.command == "check-cache":
        from poise.grasp import load_stable_grasp_cache
        data = load_stable_grasp_cache(args.cache)
        result = {"valid": True, "hand": data["hand"], "object_id": data["object_id"], "rows": len(data["q"])}
    elif args.command == "verify-golden":
        from poise.core.golden import verify_golden_reference
        result = verify_golden_reference(args.golden, args.bundle)
    elif args.command == "init-deploy":
        from importlib.resources import files
        content = files("poise.deploy.configs").joinpath("sharpa_wave.example.json").read_text()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(content)
        result = {"config": str(args.output), "next": "Fill null fields from your installation, then run poise check-deploy"}
    elif args.command == "fetch-hand":
        from poise.assets.hand import install_hand
        result = install_hand(args.destination, source=args.source, replace=args.replace)
    elif args.command == "check-hand":
        from poise.assets.hand import check_hand
        result = check_hand(args.destination)
    else:
        from poise.deploy.config import prepare_deployment
        result = prepare_deployment(args.config, args.bundle).report
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
