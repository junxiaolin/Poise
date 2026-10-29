#!/usr/bin/env python3
"""Prepare deficient seed subsets and finalize native GridDiverse grasp caches."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROW_KEYS_REQUIRED = ("q", "q_cmd", "object_pose_palm")


def load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary cache")
    for key in ROW_KEYS_REQUIRED:
        if key not in payload or not isinstance(payload[key], torch.Tensor):
            raise KeyError(f"{path} is missing tensor key {key!r}")
    return payload


def row_count(cache: dict[str, Any]) -> int:
    return int(cache["q"].shape[0])


def save_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def select_rows(cache: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    total = row_count(cache)
    result: dict[str, Any] = {}
    for key, value in cache.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == total
        ):
            result[key] = value[indices]
        elif key == "metadata" and isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value
    return result


def seed_counts(cache: dict[str, Any], seed_count: int) -> torch.Tensor:
    seed_ids = cache["source_seed_index"].to(dtype=torch.long)
    if seed_ids.numel() and (
        int(seed_ids.min()) < 0 or int(seed_ids.max()) >= seed_count
    ):
        raise ValueError(f"source_seed_index must lie in [0, {seed_count})")
    return torch.bincount(seed_ids, minlength=seed_count)


def quaternion_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True).clamp_min(1.0e-12)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def normalized_pose_distance(
    candidate_pose: torch.Tensor,
    selected_poses: torch.Tensor,
    pos_scale_m: float,
    rot_scale_rad: float,
) -> torch.Tensor:
    pos = (
        torch.linalg.norm(
            candidate_pose[:3].unsqueeze(0) - selected_poses[:, :3], dim=-1
        )
        / pos_scale_m
    )
    rot = (
        quaternion_geodesic_angle(
            candidate_pose[3:7].unsqueeze(0).expand(selected_poses.shape[0], -1),
            selected_poses[:, 3:7],
        )
        / rot_scale_rad
    )
    return torch.sqrt(pos.square() + rot.square())


def greedy_maximin_indices(
    candidate_poses: torch.Tensor,
    selected_poses: torch.Tensor,
    count: int,
    pos_scale_m: float,
    rot_scale_rad: float,
) -> list[int]:
    available = list(range(int(candidate_poses.shape[0])))
    chosen: list[int] = []
    selected = selected_poses.clone()
    while available and len(chosen) < count:
        if selected.shape[0] == 0:
            next_local = available[0]
        else:
            scores = torch.stack(
                [
                    normalized_pose_distance(
                        candidate_poses[idx], selected, pos_scale_m, rot_scale_rad
                    ).min()
                    for idx in available
                ]
            )
            next_local = available[int(torch.argmax(scores).item())]
        chosen.append(next_local)
        selected = torch.cat(
            (selected, candidate_poses[next_local : next_local + 1]), dim=0
        )
        available.remove(next_local)
    return chosen


def command_subset(args: argparse.Namespace) -> int:
    seeds = load_cache(args.seed_cache)
    filtered = load_cache(args.initial_filtered)
    n_seeds = row_count(seeds)
    counts = seed_counts(filtered, n_seeds)
    additional_paths = list(args.additional_filtered or [])
    for path in additional_paths:
        counts += seed_counts(load_cache(path), n_seeds)
    if args.target_total_rows is None:
        required_counts = torch.full(
            (n_seeds,), int(args.min_per_seed), dtype=torch.long
        )
    else:
        required_counts = allocate_exact_per_seed_quotas(
            torch.full((n_seeds,), int(args.max_per_seed), dtype=torch.long),
            min_per_seed=int(args.min_per_seed),
            max_per_seed=int(args.max_per_seed),
            target_total_rows=int(args.target_total_rows),
        )
    deficient = torch.nonzero(counts < required_counts, as_tuple=False).squeeze(-1)
    subset = select_rows(seeds, deficient)
    subset["source_seed_index"] = torch.arange(deficient.numel(), dtype=torch.long)
    subset["original_source_seed_index"] = deficient.clone()
    metadata = dict(subset.get("metadata", {}))
    metadata["native_grid_diverse_deficient_subset"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_cache": str(args.seed_cache.resolve()),
        "initial_filtered_cache": str(args.initial_filtered.resolve()),
        "additional_filtered_caches": [
            str(path.resolve()) for path in additional_paths
        ],
        "minimum_variants_per_seed": int(args.min_per_seed),
        "maximum_variants_per_seed": int(args.max_per_seed),
        "target_total_rows": args.target_total_rows,
        "required_counts_for_deficient_seeds": required_counts[deficient].tolist(),
        "original_seed_count": n_seeds,
        "deficient_seed_count": int(deficient.numel()),
        "original_source_seed_indices": deficient.tolist(),
        "initial_counts_for_deficient_seeds": counts[deficient].tolist(),
    }
    subset["metadata"] = metadata
    save_cache(subset, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "deficient_seed_count": int(deficient.numel()),
                "indices": deficient.tolist(),
            }
        )
    )
    return 0


def command_merge(args: argparse.Namespace) -> int:
    caches = [load_cache(path) for path in args.inputs]
    for cache in caches[1:]:
        for key in (
            "schema_version",
            "hand",
            "action_dim",
            "object_id",
            "object_dims_m",
        ):
            if cache.get(key) != caches[0].get(key):
                raise ValueError(f"Incompatible supplement cache: {key}")
        assert_matching_filter_contracts(caches[0], cache)
    row_counts = [row_count(cache) for cache in caches]
    result: dict[str, Any] = {}
    for key, first in caches[0].items():
        values = [cache.get(key) for cache in caches]
        if (
            isinstance(first, torch.Tensor)
            and first.ndim >= 1
            and int(first.shape[0]) == row_counts[0]
            and all(
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and int(value.shape[0]) == count
                and value.shape[1:] == first.shape[1:]
                and value.dtype == first.dtype
                for value, count in zip(values, row_counts, strict=True)
            )
        ):
            result[key] = torch.cat(values, dim=0)
        elif key == "metadata" and isinstance(first, dict):
            result[key] = dict(first)
        else:
            result[key] = first
    metadata = dict(result.get("metadata", {}))
    metadata["merged_filtered_caches"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "input_rows": row_counts,
        "output_rows": sum(row_counts),
    }
    result["metadata"] = metadata
    save_cache(result, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": row_count(result),
                "inputs": len(caches),
            }
        )
    )
    return 0


def command_remap(args: argparse.Namespace) -> int:
    subset = load_cache(args.seed_subset)
    candidates = load_cache(args.candidates)
    mapping = subset["original_source_seed_index"].to(dtype=torch.long)
    local_ids = candidates["source_seed_index"].to(dtype=torch.long)
    if local_ids.numel() and (
        int(local_ids.min()) < 0 or int(local_ids.max()) >= int(mapping.numel())
    ):
        raise ValueError("candidate source_seed_index exceeds deficient subset size")
    candidates["source_seed_index"] = mapping[local_ids]
    metadata = dict(candidates.get("metadata", {}))
    metadata["source_seed_index_remap"] = {
        "seed_subset": str(args.seed_subset.resolve()),
        "local_to_original": mapping.tolist(),
    }
    candidates["metadata"] = metadata
    save_cache(candidates, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": row_count(candidates),
                "mapped_seed_count": int(mapping.numel()),
            }
        )
    )
    return 0


def concatenate_selected(
    initial: dict[str, Any],
    supplement: dict[str, Any],
    initial_indices: torch.Tensor,
    supplement_indices: torch.Tensor,
) -> dict[str, Any]:
    n_initial = row_count(initial)
    n_supplement = row_count(supplement)
    result: dict[str, Any] = {}
    common_keys = set(initial) & set(supplement)
    for key in sorted(common_keys):
        a, b = initial[key], supplement[key]
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.ndim >= 1
            and b.ndim >= 1
            and int(a.shape[0]) == n_initial
            and int(b.shape[0]) == n_supplement
            and a.shape[1:] == b.shape[1:]
            and a.dtype == b.dtype
        ):
            result[key] = torch.cat((a[initial_indices], b[supplement_indices]), dim=0)
    for key in ("schema_version", "hand", "action_dim", "object_id", "object_dims_m"):
        if key in initial:
            result[key] = initial[key]
    result["variant_source_file_index"] = torch.cat(
        (
            torch.zeros(initial_indices.numel(), dtype=torch.long),
            torch.ones(supplement_indices.numel(), dtype=torch.long),
        )
    )
    return result


def filter_contract_from_cache(cache: dict[str, Any]) -> dict[str, Any] | None:
    metadata = cache.get("metadata")
    if not isinstance(metadata, dict):
        return None
    hold_filter = metadata.get("hold_filter")
    if not isinstance(hold_filter, dict):
        return None
    projection = hold_filter.get("joint_projection")
    result = {
        "hand_root_orientation": hold_filter.get("hand_root_orientation"),
        "hand_root_rot_wxyz": hold_filter.get("hand_root_rot_wxyz"),
        "gravity": hold_filter.get("gravity"),
        "omni_wrist": {
            key: value
            for key, value in (hold_filter.get("omni_wrist") or {}).items()
            if key != "trial_summaries"
        },
        "replay_steps": hold_filter.get("replay_steps"),
        "hold_seconds": hold_filter.get("hold_seconds"),
        "criteria": hold_filter.get("criteria"),
        "hand_usd_path": hold_filter.get("hand_usd_path"),
        "hand_usd_sha256": hold_filter.get("hand_usd_sha256"),
        "joint_limit_reference_usd_path": hold_filter.get(
            "joint_limit_reference_usd_path"
        ),
        "joint_limit_reference_usd_sha256": hold_filter.get(
            "joint_limit_reference_usd_sha256"
        ),
        "joint_limit_profile": hold_filter.get("joint_limit_profile"),
        "joint_projection": projection,
    }
    return result


def assert_matching_filter_contracts(
    initial: dict[str, Any],
    supplement: dict[str, Any],
) -> dict[str, Any] | None:
    initial_contract = filter_contract_from_cache(initial)
    supplement_contract = filter_contract_from_cache(supplement)
    if initial_contract is None:
        return supplement_contract
    if supplement_contract is None or row_count(supplement) == 0:
        return initial_contract
    for key in (
        "hand_root_orientation",
        "hand_root_rot_wxyz",
        "gravity",
        "hand_usd_sha256",
        "joint_limit_reference_usd_sha256",
        "joint_limit_profile",
        "joint_projection",
        "omni_wrist",
        "replay_steps",
        "hold_seconds",
        "criteria",
    ):
        if initial_contract.get(key) != supplement_contract.get(key):
            raise ValueError(
                f"Initial and supplement filter contracts disagree for {key!r}."
            )
    return initial_contract


def allocate_exact_per_seed_quotas(
    capacities: torch.Tensor,
    *,
    min_per_seed: int,
    max_per_seed: int,
    target_total_rows: int | None,
) -> torch.Tensor:
    """Allocate deterministic, as-even-as-possible per-parent variant quotas.

    When ``target_total_rows`` is omitted this returns capped capacities.
    With an explicit
    target, allocation starts at the minimum and adds one row per parent in
    stable parent-index order, layer by layer.  Capacity-deficient parents are
    skipped, so any spare quota is deterministically redistributed.
    """

    capacities = capacities.to(dtype=torch.long, device="cpu")
    if capacities.ndim != 1:
        raise ValueError("capacities must be a rank-1 tensor")
    if min_per_seed < 0 or max_per_seed < min_per_seed:
        raise ValueError("expected 0 <= min_per_seed <= max_per_seed")
    capped = torch.minimum(capacities, torch.full_like(capacities, int(max_per_seed)))
    deficient = torch.nonzero(capped < int(min_per_seed), as_tuple=False).squeeze(-1)
    if deficient.numel():
        details = {int(i): int(capped[i]) for i in deficient.tolist()}
        raise RuntimeError(
            f"{len(details)} parent(s) cannot reach min_per_seed={min_per_seed}: {details}"
        )

    if target_total_rows is None:
        return capped

    target = int(target_total_rows)
    minimum_total = int(min_per_seed) * int(capacities.numel())
    maximum_total = int(capped.sum().item())
    if target < minimum_total or target > maximum_total:
        raise RuntimeError(
            f"target_total_rows={target} is outside feasible [{minimum_total}, {maximum_total}] "
            f"for {int(capacities.numel())} parents and the available stable rows"
        )

    quotas = torch.full_like(capacities, int(min_per_seed))
    remaining = target - minimum_total
    while remaining > 0:
        progressed = False
        for seed_id in range(int(quotas.numel())):
            if remaining == 0:
                break
            if int(quotas[seed_id]) < int(capped[seed_id]):
                quotas[seed_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError(
                "unable to allocate exact target_total_rows despite capacity precheck"
            )
    return quotas


def command_finalize(args: argparse.Namespace) -> int:
    seeds = load_cache(args.seed_cache)
    initial = load_cache(args.initial_filtered)
    supplement = load_cache(args.supplement_filtered)
    for other in (initial, supplement):
        for key in (
            "schema_version",
            "hand",
            "action_dim",
            "object_id",
            "object_dims_m",
        ):
            if other.get(key) != seeds.get(key):
                raise ValueError(f"Parent and variant cache identities disagree: {key}")
    validated_filter_contract = assert_matching_filter_contracts(initial, supplement)
    if validated_filter_contract is None:
        raise ValueError(
            "Finalization requires recorded physical hold-filter provenance"
        )
    n_seeds = row_count(seeds)
    initial_counts = seed_counts(initial, n_seeds)
    supplement_counts = seed_counts(supplement, n_seeds)

    if args.max_per_seed < args.min_per_seed:
        raise ValueError("max_per_seed must be greater than or equal to min_per_seed")

    capacities = initial_counts + supplement_counts
    if args.target_total_rows is None:
        # Preserve the original finalizer behavior exactly: keep up to
        # max_per_seed rows from the initial pass, and use supplement rows only
        # to bring parents below min_per_seed up to the minimum.
        quotas = torch.minimum(
            initial_counts,
            torch.full_like(initial_counts, int(args.max_per_seed)),
        )
        quotas = torch.maximum(
            quotas,
            torch.full_like(quotas, int(args.min_per_seed)),
        )
        deficient = torch.nonzero(capacities < quotas, as_tuple=False).squeeze(-1)
        if deficient.numel():
            details = {int(i): int(capacities[i]) for i in deficient.tolist()}
            raise RuntimeError(
                f"{len(details)} parent(s) cannot reach min_per_seed={args.min_per_seed}: {details}"
            )
        quota_method = "legacy_initial_up_to_max_supplement_to_min"
    else:
        quotas = allocate_exact_per_seed_quotas(
            capacities,
            min_per_seed=int(args.min_per_seed),
            max_per_seed=int(args.max_per_seed),
            target_total_rows=int(args.target_total_rows),
        )
        quota_method = "balanced_parent_index_round_robin_with_capacity_redistribution"

    # Prefer the primary noisy pass and fill any remaining exact quota from the
    # supplement pool.  Selection within each source uses deterministic
    # object-pose maximin coverage.
    selected_initial: list[int] = []
    selected_initial_per_seed: dict[int, list[int]] = {}
    selected_supplement: list[int] = []
    selected_supplement_per_seed: dict[int, list[int]] = {}
    initial_seed_ids = initial["source_seed_index"].to(dtype=torch.long)
    supplement_seed_ids = supplement["source_seed_index"].to(dtype=torch.long)
    for seed_id in range(n_seeds):
        quota = int(quotas[seed_id].item())
        initial_rows = torch.nonzero(
            initial_seed_ids == seed_id, as_tuple=False
        ).squeeze(-1)
        keep_initial = min(int(initial_rows.numel()), quota)
        if keep_initial > 0:
            local_initial = greedy_maximin_indices(
                initial["object_pose_palm"][initial_rows],
                initial["object_pose_palm"].new_empty((0, 7)),
                keep_initial,
                args.position_scale_m,
                args.rotation_scale_rad,
            )
            chosen_initial = initial_rows[
                torch.tensor(local_initial, dtype=torch.long)
            ].tolist()
            selected_initial.extend(chosen_initial)
            selected_initial_per_seed[seed_id] = chosen_initial
        else:
            chosen_initial = []

        need = quota - len(chosen_initial)
        if need == 0:
            continue
        candidate_rows = torch.nonzero(
            supplement_seed_ids == seed_id, as_tuple=False
        ).squeeze(-1)
        if int(candidate_rows.numel()) < need:
            raise RuntimeError(
                f"seed {seed_id} has {len(chosen_initial)} selected initial + "
                f"{int(candidate_rows.numel())} supplement stable rows; cannot reach exact quota {quota}"
            )
        selected_initial_poses = (
            initial["object_pose_palm"][torch.tensor(chosen_initial, dtype=torch.long)]
            if chosen_initial
            else initial["object_pose_palm"].new_empty((0, 7))
        )
        local_selected = greedy_maximin_indices(
            supplement["object_pose_palm"][candidate_rows],
            selected_initial_poses,
            need,
            args.position_scale_m,
            args.rotation_scale_rad,
        )
        chosen_supplement = candidate_rows[
            torch.tensor(local_selected, dtype=torch.long)
        ].tolist()
        selected_supplement.extend(chosen_supplement)
        selected_supplement_per_seed[seed_id] = chosen_supplement

    initial_indices = torch.tensor(selected_initial, dtype=torch.long)
    supplement_indices = torch.tensor(selected_supplement, dtype=torch.long)
    final = concatenate_selected(
        initial, supplement, initial_indices, supplement_indices
    )
    final_counts = seed_counts(final, n_seeds)
    if (
        int(final_counts.min().item()) < args.min_per_seed
        or int(final_counts.max().item()) > args.max_per_seed
    ):
        raise RuntimeError("final per-seed count contract failed")
    if not torch.equal(final_counts.cpu(), quotas.cpu()):
        raise RuntimeError("final per-seed counts do not match allocated quotas")
    if args.target_total_rows is not None and row_count(final) != int(
        args.target_total_rows
    ):
        raise RuntimeError(
            f"final row count {row_count(final)} does not equal target_total_rows={args.target_total_rows}"
        )

    metadata = {
        "schema_name": "sharpa_native_grid_diverse_variants_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_cache": str(args.seed_cache.resolve()),
        "initial_filtered_cache": str(args.initial_filtered.resolve()),
        "supplement_filtered_cache": str(args.supplement_filtered.resolve()),
        "seed_count": n_seeds,
        "initial_stable_rows": row_count(initial),
        "initial_selected_rows": int(initial_indices.numel()),
        "supplement_stable_rows": row_count(supplement),
        "supplement_selected_rows": int(supplement_indices.numel()),
        "final_rows": row_count(final),
        "target_total_rows": args.target_total_rows,
        "quota_allocation_method": quota_method,
        "allocated_per_seed_quotas": quotas.tolist(),
        "per_seed_min": int(final_counts.min().item()),
        "per_seed_max": int(final_counts.max().item()),
        "per_seed_counts": final_counts.tolist(),
        "initial_per_seed_counts": initial_counts.tolist(),
        "supplement_available_per_seed_counts": supplement_counts.tolist(),
        "variant_selection": {
            "method": "greedy_maximin_object_pose_palm",
            "position_scale_m": float(args.position_scale_m),
            "rotation_scale_rad": float(args.rotation_scale_rad),
            "selected_initial_source_rows_by_seed": {
                str(k): v for k, v in selected_initial_per_seed.items()
            },
            "selected_supplement_source_rows_by_seed": {
                str(k): v for k, v in selected_supplement_per_seed.items()
            },
        },
        "candidate_perturbation": {
            key: initial.get("metadata", {}).get(key)
            for key in (
                "q_noise_rad",
                "q_cmd_noise_rad",
                "object_pos_noise_m",
                "object_rot_noise_rad",
                "joint_projection",
            )
        },
        "filter_contract": validated_filter_contract,
    }
    final["metadata"] = metadata
    save_cache(final, args.output)

    report = {
        "output": str(args.output.resolve()),
        "object_id": final.get("object_id"),
        "seed_count": n_seeds,
        "initial_stable_rows": row_count(initial),
        "initial_selected_rows": int(initial_indices.numel()),
        "supplement_stable_rows": row_count(supplement),
        "supplement_selected_rows": int(supplement_indices.numel()),
        "final_rows": row_count(final),
        "target_total_rows": args.target_total_rows,
        "allocated_per_seed_quotas": quotas.tolist(),
        "per_seed_min": int(final_counts.min().item()),
        "per_seed_max": int(final_counts.max().item()),
        "per_seed_counts": final_counts.tolist(),
        "filter_contract": validated_filter_contract,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("deficient-subset")
    subset.add_argument("--seed-cache", type=Path, required=True)
    subset.add_argument("--initial-filtered", type=Path, required=True)
    subset.add_argument("--additional-filtered", type=Path, action="append", default=[])
    subset.add_argument("--min-per-seed", type=int, default=10)
    subset.add_argument("--max-per-seed", type=int, default=20)
    subset.add_argument(
        "--target-total-rows",
        type=int,
        default=None,
        help="Select parents below the deterministic exact-total quota rather than only below min-per-seed.",
    )
    subset.add_argument("--output", type=Path, required=True)
    subset.set_defaults(func=command_subset)

    remap = subparsers.add_parser("remap")
    remap.add_argument("--seed-subset", type=Path, required=True)
    remap.add_argument("--candidates", type=Path, required=True)
    remap.add_argument("--output", type=Path, required=True)
    remap.set_defaults(func=command_remap)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(func=command_merge)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--seed-cache", type=Path, required=True)
    finalize.add_argument("--initial-filtered", type=Path, required=True)
    finalize.add_argument("--supplement-filtered", type=Path, required=True)
    finalize.add_argument("--min-per-seed", type=int, default=10)
    finalize.add_argument("--max-per-seed", type=int, default=20)
    finalize.add_argument(
        "--target-total-rows",
        type=int,
        default=None,
        help="Require an exact final row count while retaining per-parent min/max bounds.",
    )
    finalize.add_argument("--position-scale-m", type=float, default=0.004)
    finalize.add_argument("--rotation-scale-rad", type=float, default=0.08)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--report-json", type=Path, required=True)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
