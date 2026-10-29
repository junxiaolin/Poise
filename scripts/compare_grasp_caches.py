"""Compare trusted reset caches for coverage, diversity, limits and replay stability."""

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch
from poise.assets.hand import sha256
from poise.grasp.cache import load_stable_grasp_cache
from poise.sim.hb1_sim2real import parse_usda_action_joint_limits
from poise.sim.joint_mapping import SHARPA_JOINT_ORDER


def summary(x):
    x = np.asarray(x, dtype=float)
    if not x.size:
        return None
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in comparison statistics")
    return {
        "min": float(x.min()),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def position_bins(p, step=0.01):
    return np.floor(np.asarray(p) / step + 1e-7).astype(np.int64)


def histogram_difference(a, b):
    ca = Counter(map(tuple, a))
    cb = Counter(map(tuple, b))
    keys = sorted(set(ca) | set(cb))
    pa = np.array([ca[k] for k in keys], dtype=float) / len(a)
    pb = np.array([cb[k] for k in keys], dtype=float) / len(b)
    mean = (pa + pb) / 2

    def kl(p):
        use = p > 0
        return np.sum(p[use] * np.log2(p[use] / mean[use]))

    return {
        "occupied_reference": len(ca),
        "occupied_new": len(cb),
        "intersection": len(set(ca) & set(cb)),
        "reference_bin_coverage": len(set(ca) & set(cb)) / len(ca),
        "jensen_shannon_bits": float((kl(pa) + kl(pb)) / 2),
    }


def nearest(query, bank, self_match=False):
    distances = []
    for start in range(0, len(query), 256):
        # Direct differences avoid cancellation for millimeter-scale distances
        # between positions sharing a much larger palm-frame offset.
        d = torch.cdist(
            query[start : start + 256],
            bank,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        if self_match:
            i = torch.arange(len(d))
            d[i, start + i] = float("inf")
        distances.append(d.min(dim=1).values)
    return torch.cat(distances).numpy()


def inspect(cache, limits):
    q = cache["q"]
    cmd = cache["q_cmd"]
    pose = cache["object_pose_palm"]
    parents = cache.get("source_seed_index")
    sig = torch.cat([q, cmd, pose], 1)
    report = {
        "rows": len(q),
        "unique_exact_states": len(torch.unique(sig, dim=0)),
        "position_min_m": pose[:, :3].min(0).values.tolist(),
        "position_max_m": pose[:, :3].max(0).values.tolist(),
        "position_std_m": pose[:, :3].std(0, unbiased=False).tolist(),
        "joint_std_rad": q.std(0, unbiased=False).tolist(),
        "qcmd_minus_q_rms_rad": summary((cmd - q).square().mean(1).sqrt()),
        "quaternion_max_norm_error": float((pose[:, 3:].norm(dim=1) - 1).abs().max()),
        "contact_fingers": summary(cache["contact_bits"].sum(1)),
        "position_nearest_neighbor_mm": (
            summary(nearest(pose[:, :3], pose[:, :3], True) * 1000)
            if len(q) > 1
            else None
        ),
    }
    for label, x in [("q", q), ("q_cmd", cmd)]:
        bad = (x < limits[:, 0] - 2e-7) | (x > limits[:, 1] + 2e-7)
        report[label + "_joint_limit_violating_rows"] = int(bad.any(1).sum())
    if parents is not None:
        _, counts = torch.unique(parents, return_counts=True)
        report["parents"] = len(counts)
        report["per_parent_counts"] = summary(counts)
        report["parent_count_histogram"] = {
            str(int(k)): int(v) for k, v in sorted(Counter(counts.tolist()).items())
        }
        spread = []
        for parent in torch.unique(parents):
            xyz = pose[parents == parent, :3]
            spread.append(float(xyz.std(0, unbiased=False).norm()) * 1000)
        report["within_parent_position_spread_mm"] = summary(spread)
    return report


def nearest_orientation(query, bank):
    query = torch.nn.functional.normalize(query, dim=1)
    bank = torch.nn.functional.normalize(bank, dim=1)
    angular = []
    for start in range(0, len(query), 256):
        dot = (query[start : start + 256] @ bank.T).abs().clamp(max=1)
        angular.append(2 * torch.acos(dot.max(1).values) * 180 / math.pi)
    return torch.cat(angular)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--reference-replay", type=Path)
    p.add_argument("--new-replay", type=Path)
    p.add_argument(
        "--limits",
        type=Path,
        default=Path("data/robots/sharpa_limits/right_sharpa_wave.usda"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    a = load_stable_grasp_cache(args.reference)
    b = load_stable_grasp_cache(args.new)
    if a["object_id"] != b["object_id"] or not np.allclose(
        a["object_dims_m"], b["object_dims_m"], atol=1e-7, rtol=0
    ):
        raise ValueError("Comparison requires the same object and dimensions")
    if min(len(a["q"]), len(b["q"])) == 0:
        raise ValueError("Comparison requires two nonempty caches")
    limits = parse_usda_action_joint_limits(args.limits, SHARPA_JOINT_ORDER) * 0.9
    report = {
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256(args.reference),
            **inspect(a, limits),
        },
        "new": {
            "path": str(args.new.resolve()),
            "sha256": sha256(args.new),
            **inspect(b, limits),
        },
        "joint_limit_tolerance_rad": 2e-7,
        "position_bins_10mm": histogram_difference(
            position_bins(a["object_pose_palm"][:, :3]),
            position_bins(b["object_pose_palm"][:, :3]),
        ),
        "reference_to_new_nearest_position_mm": summary(
            nearest(a["object_pose_palm"][:, :3], b["object_pose_palm"][:, :3]) * 1000
        ),
        "new_to_reference_nearest_position_mm": summary(
            nearest(b["object_pose_palm"][:, :3], a["object_pose_palm"][:, :3]) * 1000
        ),
    }
    # Compare rotation representations through sign-invariant SO(3) distances.
    for name, query, bank in [("reference_to_new", a, b), ("new_to_reference", b, a)]:
        report[name + "_nearest_orientation_deg"] = summary(
            nearest_orientation(
                query["object_pose_palm"][:, 3:], bank["object_pose_palm"][:, 3:]
            )
        )
    signatures = [
        torch.cat([c["q"], c["q_cmd"], c["object_pose_palm"]], 1) for c in (a, b)
    ]
    report["exact_state_overlap"] = sum(
        len(torch.unique(s, dim=0)) for s in signatures
    ) - len(torch.unique(torch.cat(signatures), dim=0))
    for key, path in [("reference", args.reference_replay), ("new", args.new_replay)]:
        if path:
            replay = json.loads(path.read_text())
            if (
                replay["selection"]["num_checked"] != report[key]["rows"]
                or replay["selection"]["start_index"] != 0
            ):
                raise ValueError("Replay report must cover every cache row")
            replay_input = Path(replay["input"])
            if (
                not replay_input.is_file()
                or sha256(replay_input) != report[key]["sha256"]
            ):
                raise ValueError("Replay input does not match the compared cache")
            report[key]["fresh_wrist_replay"] = {
                k: replay[k]
                for k in [
                    "stable_count",
                    "stable_fraction",
                    "hold_seconds",
                    "criteria",
                    "omni_wrist",
                    "drift_m",
                    "linear_speed_mps",
                    "angular_speed_radps",
                    "hand_usd_sha256",
                    "joint_limit_reference_usd_sha256",
                ]
            }
    if args.reference_replay and args.new_replay:
        ra = json.loads(args.reference_replay.read_text())
        rb = json.loads(args.new_replay.read_text())
        for k in (
            "criteria",
            "batch_size",
            "replay_steps",
            "gravity",
            "hand_root_rot_wxyz",
            "hand_usd_sha256",
            "joint_limit_reference_usd_sha256",
        ):
            if ra[k] != rb[k]:
                raise ValueError(f"Replay protocol mismatch: {k}")
        for k in ("trials", "wrist_seed", "identity_wrist_trial_enabled"):
            if ra["omni_wrist"][k] != rb["omni_wrist"][k]:
                raise ValueError(f"Replay wrist protocol mismatch: {k}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    lines = [
        "# HexPrism40 reset-cache comparison",
        "",
        "Both caches are measured in the hand-root frame. Position and orientation nearest-neighbor distances are separate marginal coverage measures, not joint SE(3) matches or policy success.",
        "",
        "| Metric | Reference | Regenerated |",
        "|---|---:|---:|",
    ]
    for label, fn in [
        ("Rows", lambda x: x["rows"]),
        ("Parents", lambda x: x.get("parents", "—")),
        ("Exact unique states", lambda x: x["unique_exact_states"]),
        (
            "Variants per parent (min–max)",
            lambda x: (
                f"{x['per_parent_counts']['min']:g}–{x['per_parent_counts']['max']:g}"
                if "per_parent_counts" in x
                else "—"
            ),
        ),
        ("q limit violations", lambda x: x["q_joint_limit_violating_rows"]),
        ("q_cmd limit violations", lambda x: x["q_cmd_joint_limit_violating_rows"]),
        (
            "Median position nearest neighbor (mm)",
            lambda x: (
                round(x["position_nearest_neighbor_mm"]["median"], 3)
                if x["position_nearest_neighbor_mm"]
                else "—"
            ),
        ),
    ]:
        lines.append(f"| {label} | {fn(report['reference'])} | {fn(report['new'])} |")
    if args.reference_replay and args.new_replay:
        lines.append(
            f"| Fresh OmniWrist8 hold-2s success | {report['reference']['fresh_wrist_replay']['stable_fraction']:.2%} | {report['new']['fresh_wrist_replay']['stable_fraction']:.2%} |"
        )
    bins = report["position_bins_10mm"]
    lines += [
        "",
        f"10-mm occupied position bins: {bins['occupied_reference']} reference, {bins['occupied_new']} regenerated; {bins['intersection']} overlap. Jensen–Shannon divergence: {bins['jensen_shannon_bits']:.4f} bits.",
        "",
        "Physical replay is a reset stability measurement. It does not establish equivalent reaching-policy performance.",
        "",
    ]
    (args.output / "comparison.md").write_text("\n".join(lines))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        both = (
            torch.cat(
                [a["object_pose_palm"][:, :3], b["object_pose_palm"][:, :3]]
            ).numpy()
            * 1000
        )
        lower = both.min(0) - 2
        upper = both.max(0) + 2
        for row, (label, c) in enumerate([("Reference", a), ("Regenerated", b)]):
            xyz = c["object_pose_palm"][:, :3].numpy() * 1000
            for col, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
                ax = axes[row, col]
                ax.scatter(xyz[:, i], xyz[:, j], s=2, alpha=0.22)
                ax.set_title(label)
                ax.set_xlabel("XYZ"[i] + " (mm)")
                ax.set_ylabel("XYZ"[j] + " (mm)")
                ax.set_aspect("equal", "box")
                ax.set_xlim(lower[i], upper[i])
                ax.set_ylim(lower[j], upper[j])
        fig.savefig(args.output / "position_coverage.png", dpi=170)
        plt.close(fig)
    except ImportError:
        pass
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": [len(a["q"]), len(b["q"])],
                "position_bins": bins,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
