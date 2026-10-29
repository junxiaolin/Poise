"""Build a diverse Sharpa reset cache from fresh Lightning search through physics.

Run with Isaac Lab Python; --optimizer-python selects a separate Lightning
Grasp environment. Every stage has a log, input/output hashes and a completion
receipt. --resume reuses only matching completed stages. No training job is
launched and no existing training cache is overwritten.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import torch
from poise.assets.hand import check_hand, sha256
from poise.grasp.cache import load_stable_grasp_cache, save_stable_grasp_cache
from poise.grasp.finalize import select_rows, seed_counts
from poise.paths import workspace_root, resolve_path


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(paths):
    return {str(p.resolve()): sha256(p) for p in paths}


def stop_child_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def remaining_seconds(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Generation wall-time budget exhausted; no successful result published")
    return remaining


def parent_targets(cfg, available):
    """Formal builds require exact counts; fast builds cap parents explicitly."""
    minimum = cfg.get("minimum_parents", cfg["parents"])
    if not 1 <= minimum <= cfg["parents"]:
        raise ValueError("minimum_parents must lie between 1 and parents")
    if available < minimum:
        raise RuntimeError(
            f"Only {available} stable parents; need {minimum}. Add fresh search passes to a new build config."
        )
    count = min(available, cfg["parents"])
    rows = cfg["rows"] if count == cfg["parents"] else count * cfg["rows"] // cfg["parents"]
    if not count * cfg["min_variants"] <= rows <= count * cfg["max_variants"]:
        raise ValueError("Resolved row count is incompatible with per-parent quotas")
    return count, rows


def optimizer_provenance(python, deadline=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace_root() / "source")
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    timeout = remaining_seconds(deadline)
    child = subprocess.Popen(
        [str(python), "-m", "poise.grasp.lightning.runtime"], text=True, env=env,
        stdout=subprocess.PIPE, start_new_session=True,
    )
    try:
        output, _ = child.communicate(timeout=timeout)
    except BaseException:
        stop_child_group(child)
        raise
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, child.args)
    return json.loads(output)


class Stages:
    def __init__(self, output, env, resume, contract, deadline=None):
        self.deadline = deadline
        self.output, self.env, self.resume, self.contract = (
            output,
            env,
            resume,
            contract,
        )

    def run(self, name, cmd, inputs, outputs, *, empty_filter=None):
        remaining_seconds(self.deadline)
        receipt = self.output / f"{name}.stage.json"
        expected = {
            "command": [str(x) for x in cmd],
            "inputs": fingerprint(inputs),
            "contract_sha256": self.contract,
        }
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if (
                self.resume
                and old.get("status") == "complete"
                and all(old.get(k) == v for k, v in expected.items())
                and all(p.is_file() for p in outputs)
                and old["outputs"] == fingerprint(outputs)
            ):
                print(f"[REUSE] {name}", flush=True)
                return
            raise RuntimeError(
                f"Stage {name} differs from saved inputs; use a fresh output directory"
            )
        if any(p.exists() for p in outputs):
            raise FileExistsError(
                f"Unreceipted outputs for {name}; inspect and move them before resuming"
            )
        record = dict(expected, started_utc=now(), status="running")
        running = self.output / f"{name}.running.json"
        running.write_text(json.dumps(record, indent=2))
        print(f"[START] {name}", flush=True)
        with (self.output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(
                expected["command"],
                cwd=workspace_root(),
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = child.pid
            running.write_text(json.dumps(record, indent=2))
            try:
                code = child.wait(timeout=remaining_seconds(self.deadline))
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                stop_child_group(child)
                record.update(status="timed_out", finished_utc=now())
                running.write_text(json.dumps(record, indent=2))
                raise TimeoutError(f"Generation budget exhausted during {name}; see {running}") from error
            except BaseException:
                stop_child_group(child)
                record.update(status="interrupted", finished_utc=now())
                running.write_text(json.dumps(record, indent=2))
                raise
        if empty_filter is not None and not outputs[0].exists() and outputs[1].exists():
            report = json.loads(outputs[1].read_text())
            if report.get("stable_count") == 0 and code in (0, 1):
                cache = load_stable_grasp_cache(empty_filter)
                cache = select_rows(cache, torch.empty(0, dtype=torch.long))
                cache.setdefault("metadata", {})["empty_filter_report"] = str(
                    outputs[1]
                )
                save_stable_grasp_cache(cache, outputs[0])
                code = 0
        if code:
            record.update(status="failed", exit_code=code, finished_utc=now())
            running.write_text(json.dumps(record, indent=2))
            raise RuntimeError(
                f'{name} exited {code}; see {self.output / (name + ".log")}'
            )
        record.update(
            status="complete",
            exit_code=code,
            finished_utc=now(),
            outputs=fingerprint(outputs),
        )
        receipt.write_text(json.dumps(record, indent=2))
        running.unlink()
        print(f"[DONE] {name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/grasp_hexprism40.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--optimizer-python", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--experience", type=Path)
    p.add_argument(
        "--search-results",
        type=Path,
        help="Optional manifest of freshly generated search outputs and their hashes",
    )
    args = p.parse_args()
    started = time.monotonic()
    started_utc = now()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Build interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    torch.set_num_threads(4)
    cfg = json.loads(args.config.read_text())
    budget = cfg.get("max_wall_seconds")
    if budget is not None and (not isinstance(budget, (int, float)) or not 0 < budget < float("inf")):
        raise ValueError("max_wall_seconds must be finite and positive")
    deadline = started + budget if budget is not None else None
    # Validate the optional fast-mode policy before launching anything.
    parent_targets(cfg, cfg["parents"])
    root = workspace_root()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".build.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A workspace-wide device lock also covers child GPU stages. This prevents
    # separate output directories from accidentally multiplying GPU load.
    device_lock_path = (
        root
        / "outputs/grasp_generation"
        / ("device_" + args.device.replace(":", "_") + ".lock")
    )
    device_lock_path.parent.mkdir(parents=True, exist_ok=True)
    device_lock = device_lock_path.open("a")
    fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hand = check_hand()
    limits = resolve_path(cfg["joint_limit_reference"])
    mesh = resolve_path(cfg["object_mesh"])
    if (
        sha256(limits) != cfg["joint_limit_reference_sha256"]
        or sha256(mesh) != cfg["object_mesh_sha256"]
    ):
        raise ValueError("Object or joint-limit reference hash mismatch")
    if not (
        cfg["parents"] * cfg["min_variants"]
        <= cfg["rows"]
        <= cfg["parents"] * cfg["max_variants"]
    ):
        raise ValueError("Requested row count is incompatible with per-parent quotas")
    source_files = (
        sorted((root / "source/poise/grasp").rglob("*.py"))
        + sorted((root / "source/poise/sim").rglob("*.py"))
        + sorted((root / "source/poise/assets").rglob("*.py"))
        + [
            root / "source/poise/paths.py",
            root / "scripts/expand_grasps.py",
            root / "scripts/filter_grasps.py",
            Path(__file__),
        ]
    )
    contract = {
        "config": cfg,
        "hand": hand,
        "source": fingerprint(source_files),
        "python": sys.executable,
        "optimizer_python": str(args.optimizer_python.absolute()),
        "optimizer": optimizer_provenance(args.optimizer_python, deadline),
        "device": args.device,
        "experience": str(args.experience) if args.experience else None,
        "object_assets": fingerprint(
            sorted(p for p in mesh.parent.rglob("*") if p.is_file())
        ),
        "simulator_versions": {"python": sys.version, "torch": torch.__version__},
    }
    imports = (
        json.loads(args.search_results.read_text()) if args.search_results else None
    )
    if imports is not None:
        if imports["hand"] != hand:
            raise ValueError(
                "Imported search hand contract differs from installed public hand"
            )
        contract["search_results"] = imports
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    contract_path = out / "contract.json"
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "Existing build contract differs; use a new output or unchanged --resume"
            )
    else:
        contract_path.write_text(json.dumps(contract, indent=2))
    env = os.environ.copy()
    env.update(
        POISE_ROOT=str(root),
        PYTHONPATH=str(root / "source"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    env.pop("SHARPA_WAVE_URDF_PATH", None)
    stages = Stages(out, env, args.resume, contract_hash, deadline)
    py = sys.executable
    opt = str(args.optimizer_python)
    isaac = ["--headless", "--device", args.device]
    if args.experience:
        isaac += ["--experience", str(args.experience)]
    physical = [
        "--hand-usd-path",
        str(root / "data/robots/sharpa_wave/right_sharpa_wave.usda"),
        "--expected-hand-usd-sha256",
        hand["entrypoint_sha256"],
        "--joint-limit-reference-usd",
        str(limits),
        "--expected-joint-limit-reference-usd-sha256",
        sha256(limits),
        "--joint-limit-profile",
        "usda09",
        "--hand-root-orientation",
        "palm_up",
    ]

    def filter_cache(name, source):
        output = out / f"{name}.pt"
        report = out / f"{name}.json"
        stages.run(
            name,
            [
                py,
                "scripts/filter_grasps.py",
                *isaac,
                *physical,
                "--input",
                source,
                "--output-filtered-cache",
                output,
                "--report-json",
                report,
                "--batch-size",
                str(cfg["filter_batch_size"]),
                "--omni-wrist-trials",
                "8",
                "--wrist-seed",
                str(cfg["wrist_seed"]),
                "--replay-steps",
                "240",
                "--min-contact-fingers",
                "2",
                "--contact-threshold-m",
                "0.025",
                "--max-replay-drift-m",
                "0.04",
                "--max-lin-vel-mps",
                "0.25",
                "--max-ang-vel-radps",
                "4.0",
                "--min-palm-x",
                "0.025",
                "--max-palm-x",
                "0.18",
                "--max-abs-palm-y",
                "0.14",
                "--min-palm-z",
                "0.02",
                "--max-palm-z",
                "0.22",
                "--no-clamp-joints",
            ],
            [source],
            [output, report],
            empty_filter=source,
        )
        return output

    converted = []
    for search in cfg["searches"]:
        name = "search_" + search["name"]
        raw = out / f"{name}.npy"
        meta = out / f"{name}.metadata.npz"
        if imports is None or search["name"] not in imports["searches"]:
            stages.run(
                name,
                [
                    opt,
                    "-m",
                    "poise.grasp.lightning.generate",
                    "--joint_limit_profile",
                    "usda09",
                    "--object_mesh_path",
                    mesh,
                    "--output_path",
                    raw,
                    "--n_assign_retry",
                    "20",
                    "--pose_batch_size",
                    str(cfg.get("search_pose_batch_size", 32)),
                    "--batch_size_inner",
                    str(cfg.get("search_inner_batch_size", 32)),
                    *map(str, search["args"]),
                ],
                [mesh, limits],
                [raw, meta, meta.with_suffix(".json")],
            )
        else:
            record = imports["searches"][search["name"]]
            if record["config_search"] != search:
                raise ValueError(f'Imported search arguments differ: {search["name"]}')
            sources = [Path(x) for x in record["files"]]
            # Manifest JSON object ordering must not determine file roles.
            sources = [
                next(p for p in sources if p.name.endswith(suffix))
                for suffix in (".npy", ".metadata.npz", ".metadata.json")
            ]
            if len(record["files"]) != 3:
                raise ValueError(
                    "Each imported search must provide exactly three output files"
                )
            if fingerprint(sources) != record["files"]:
                raise ValueError(
                    f'Imported search output hashes differ: {search["name"]}'
                )
            destinations = [raw, meta, meta.with_suffix(".json")]
            copy_code = "import shutil,sys; [shutil.copyfile(s,d) for s,d in zip(sys.argv[1::2],sys.argv[2::2])]"
            stages.run(
                name + "_import",
                [
                    py,
                    "-c",
                    copy_code,
                    *[str(x) for pair in zip(sources, destinations) for x in pair],
                ],
                sources,
                destinations,
            )
        if json.loads(meta.with_suffix(".json").read_text())["num_cache_entries"] == 0:
            print(f"[EMPTY] {name}; retained report", flush=True)
            continue
        qcmd = out / f"{name}.qcmd.npz"
        stages.run(
            name + "_qcmd",
            [
                opt,
                "-m",
                "poise.grasp.lightning.qcmd",
                "--cache_path",
                raw,
                "--metadata_path",
                meta,
                "--output_npz_path",
                qcmd,
                "--joint_limit_profile",
                "usda09",
                "--usd_path",
                limits,
                "--batch_size",
                str(cfg.get("qcmd_batch_size", 128)),
            ],
            [raw, meta, limits],
            [qcmd],
        )
        cache = out / f"{name}.pt"
        stages.run(
            name + "_convert",
            [
                py,
                "-m",
                "poise.grasp.convert",
                "--input",
                qcmd,
                "--output",
                cache,
                "--object-id",
                cfg["object_id"],
                "--object-dims-m",
                *map(str, cfg["object_dims_m"]),
                "--profile-name",
                search["name"],
            ],
            [qcmd],
            [cache],
        )
        converted.append(cache)
    if not converted:
        raise RuntimeError("No search candidates found")
    merged = out / "candidates.pt"
    stages.run(
        "merge_candidates",
        [
            py,
            "-m",
            "poise.grasp.merge",
            *[x for f in converted for x in ["--input", f]],
            "--output",
            merged,
            "--report-json",
            out / "candidates.json",
        ],
        converted,
        [merged, out / "candidates.json"],
    )
    stable = filter_cache("stable_candidates", merged)
    count = len(load_stable_grasp_cache(stable)["q"])
    parent_count, target_rows = parent_targets(cfg, count)
    (out / "targets.json").write_text(json.dumps({
        "requested_parents": cfg["parents"], "requested_rows": cfg["rows"],
        "available_stable_parents": count, "parents": parent_count, "rows": target_rows,
        "minimum_parents": cfg.get("minimum_parents", cfg["parents"]),
    }, indent=2))
    print(f"[TARGET] parents={parent_count}, rows={target_rows}", flush=True)
    parents = out / "parents.pt"
    stages.run(
        "select_parents",
        [
            py,
            "-m",
            "poise.grasp.select_parents",
            "--input",
            stable,
            "--output",
            parents,
            "--report-json",
            out / "parents.json",
            "--target-count",
            str(parent_count),
        ],
        [stable],
        [parents, out / "parents.json"],
    )

    def expand(name, source, per_seed, seed):
        output = out / f"{name}.pt"
        report = out / f"{name}.json"
        n = len(load_stable_grasp_cache(source)["q"])
        stages.run(
            name,
            [
                py,
                "scripts/expand_grasps.py",
                *isaac,
                *physical,
                "--seed-cache",
                source,
                "--sample-seed-pool",
                "--target-per-seed",
                str(per_seed),
                "--num-envs",
                str(min(cfg["expansion_envs"], n * per_seed)),
                "--max-batches",
                str(cfg["max_expansion_batches"]),
                "--seed",
                str(seed),
                "--output",
                output,
                "--report-json",
                report,
                "--object-mass-kg",
                "0.05",
                "--object-friction",
                "1.0",
                "--q-noise-rad",
                "0.025",
                "--q-cmd-noise-rad",
                "0.012",
                "--object-pos-noise-m",
                "0.004",
                "--object-rot-noise-rad",
                "0.08",
                "--settle-steps",
                "80",
                "--replay-steps",
                "240",
                "--min-surface-contacts",
                "2",
                "--contact-proxy-threshold-m",
                "0.025",
                "--max-object-drift-m",
                "0.04",
                "--max-lin-vel",
                "0.25",
                "--max-ang-vel",
                "4.0",
                "--min-palm-x",
                "0.025",
                "--max-palm-x",
                "0.18",
                "--max-abs-palm-y",
                "0.14",
                "--min-palm-z",
                "0.02",
                "--max-palm-z",
                "0.22",
                "--min-q-margin-rad",
                "0",
            ],
            [source],
            [output, report],
        )
        return output

    initial = filter_cache(
        "initial_filtered",
        expand(
            "initial_variants", parents, cfg["initial_variants"], cfg["expansion_seed"]
        ),
    )
    supplement = out / "supplement_empty.pt"
    if not supplement.exists():
        save_stable_grasp_cache(
            select_rows(
                load_stable_grasp_cache(initial), torch.empty(0, dtype=torch.long)
            ),
            supplement,
        )
    quota_args = [
        "--min-per-seed",
        str(cfg["min_variants"]),
        "--max-per-seed",
        str(cfg["max_variants"]),
        "--target-total-rows",
        str(target_rows),
    ]
    for i in range(cfg["supplement_rounds"] + 1):
        counts = seed_counts(
            load_stable_grasp_cache(initial), parent_count
        ) + seed_counts(load_stable_grasp_cache(supplement), parent_count)
        if (
            int(counts.min()) >= cfg["min_variants"]
            and int(counts.clamp(max=cfg["max_variants"]).sum()) >= target_rows
        ):
            break
        if i == cfg["supplement_rounds"]:
            raise RuntimeError(
                "Supplement budget exhausted; see per-parent counts. No final cache published."
            )
        subset = out / f"supplement_{i}_parents.pt"
        stages.run(
            f"subset_{i}",
            [
                py,
                "-m",
                "poise.grasp.finalize",
                "deficient-subset",
                "--seed-cache",
                parents,
                "--initial-filtered",
                initial,
                "--additional-filtered",
                supplement,
                *quota_args,
                "--output",
                subset,
            ],
            [parents, initial, supplement],
            [subset],
        )
        local = expand(
            f"supplement_{i}_local",
            subset,
            cfg["supplement_variants"],
            cfg["expansion_seed"] + 10000 + i,
        )
        mapped = out / f"supplement_{i}_mapped.pt"
        stages.run(
            f"remap_{i}",
            [
                py,
                "-m",
                "poise.grasp.finalize",
                "remap",
                "--seed-subset",
                subset,
                "--candidates",
                local,
                "--output",
                mapped,
            ],
            [subset, local],
            [mapped],
        )
        filtered = filter_cache(f"supplement_{i}_filtered", mapped)
        combined = out / f"supplement_{i}_combined.pt"
        stages.run(
            f"combine_{i}",
            [
                py,
                "-m",
                "poise.grasp.finalize",
                "merge",
                "--inputs",
                supplement,
                filtered,
                "--output",
                combined,
            ],
            [supplement, filtered],
            [combined],
        )
        supplement = combined
    final = out / "training.pt"
    stages.run(
        "finalize",
        [
            py,
            "-m",
            "poise.grasp.finalize",
            "finalize",
            "--seed-cache",
            parents,
            "--initial-filtered",
            initial,
            "--supplement-filtered",
            supplement,
            *quota_args,
            "--output",
            final,
            "--report-json",
            out / "training.json",
        ],
        [parents, initial, supplement],
        [final, out / "training.json"],
    )
    cache = load_stable_grasp_cache(final)
    counts = seed_counts(cache, parent_count)
    assert (
        len(cache["q"]) == target_rows
        and counts.min() >= cfg["min_variants"]
        and counts.max() <= cfg["max_variants"]
    )
    sig = torch.cat([cache["q"], cache["q_cmd"], cache["object_pose_palm"]], dim=1)
    assert len(torch.unique(sig, dim=0)) == len(sig), "Duplicate reset states"
    remaining_seconds(deadline)
    # Keep training.pt immutable after its stage receipt; provenance is adjacent.
    (out / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "finished_utc": now(),
                "started_utc": started_utc,
                "elapsed_seconds": time.monotonic() - started,
                "max_wall_seconds": budget,
                "requested_parents": cfg["parents"],
                "requested_rows": cfg["rows"],
                "parents": parent_count,
                "rows": len(sig),
                "per_parent_min": int(counts.min()),
                "per_parent_max": int(counts.max()),
                "cache": str(final),
                "cache_sha256": sha256(final),
                "parents_sha256": sha256(parents),
                "hand": hand,
                "build_contract_sha256": contract_hash,
            },
            indent=2,
        )
    )
    print("[COMPLETE] " + str(final), flush=True)


if __name__ == "__main__":
    main()
