#!/usr/bin/env python3
"""Merge compatible Sharpa stable-grasp caches and remove duplicate states."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


from poise.paths import workspace_root

ROOT = workspace_root()

from poise.grasp.cache import (  # noqa: E402
    load_stable_grasp_cache,
    save_stable_grasp_cache,
    summarize_stable_grasp_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--fallback-profile-name",
        action="append",
        default=[],
        help=(
            "Optional profile name for the corresponding --input when that cache has no "
            "profile_names/profile_id lineage. May be repeated; omitted entries default to the input stem."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def row_tensor_keys(cache: dict[str, Any], row_count: int) -> set[str]:
    return {
        key
        for key, value in cache.items()
        if isinstance(value, torch.Tensor)
        and value.ndim >= 1
        and int(value.shape[0]) == row_count
    }


def first_unique_indices(signature: torch.Tensor) -> torch.Tensor:
    """Return first occurrences in input order using exact tensor values."""

    seen: dict[bytes, int] = {}
    keep: list[int] = []
    contiguous = signature.detach().cpu().contiguous()
    for index in range(int(contiguous.shape[0])):
        key = contiguous[index].numpy().tobytes()
        if key not in seen:
            seen[key] = index
            keep.append(index)
    return torch.tensor(keep, dtype=torch.long)


def main() -> int:
    args = parse_args()
    input_paths = [resolve(path) for path in args.input]
    caches = [load_stable_grasp_cache(path) for path in input_paths]
    reference = caches[0]
    identity = (
        reference["schema_version"],
        reference["hand"],
        reference["action_dim"],
        reference["object_id"],
    )
    dims = tuple(float(value) for value in reference["object_dims_m"])
    row_counts = [int(cache["q"].shape[0]) for cache in caches]

    for path, cache in zip(input_paths, caches, strict=True):
        candidate_identity = (
            cache["schema_version"],
            cache["hand"],
            cache["action_dim"],
            cache["object_id"],
        )
        candidate_dims = tuple(float(value) for value in cache["object_dims_m"])
        if candidate_identity != identity or candidate_dims != dims:
            raise ValueError(
                f"Incompatible cache {path}: identity={candidate_identity}, dims={candidate_dims}; "
                f"expected identity={identity}, dims={dims}."
            )

    common_keys = set.intersection(
        *(
            row_tensor_keys(cache, count)
            for cache, count in zip(caches, row_counts, strict=True)
        )
    )
    required = {
        "q",
        "qdot",
        "q_cmd",
        "object_pose_palm",
        "cube_keypoints_palm",
        "contact_bits",
        "stability_score",
    }
    if missing := sorted(required - common_keys):
        raise ValueError(
            f"Required row tensors are not common to all inputs: {missing}"
        )

    # Old lineage keys refer to each input cache independently.  Rebuild a
    # single unambiguous row index after concatenation instead of propagating
    # those incompatible index spaces.
    common_keys -= {
        "source_cache_index",
        "rebuild_source_cache_index",
        "legacy_upstream_source_cache_index",
        "immediate_input_cache_index",
        "source_seed_index",
        # profile_id values are local to each input's profile_names table and
        # must be remapped into one merged global table below.
        "profile_id",
    }
    concatenated = {
        key: torch.cat([cache[key].cpu() for cache in caches], dim=0)
        for key in sorted(common_keys)
    }
    signature = torch.cat(
        (
            concatenated["q"].to(torch.float64),
            concatenated["q_cmd"].to(torch.float64),
            concatenated["object_pose_palm"].to(torch.float64),
        ),
        dim=-1,
    )
    keep = first_unique_indices(signature)
    merged = {
        "schema_version": reference["schema_version"],
        "hand": reference["hand"],
        "action_dim": reference["action_dim"],
        "object_id": reference["object_id"],
        "object_dims_m": dims,
        **{key: value[keep] for key, value in concatenated.items()},
    }
    merged_count = int(keep.numel())

    profile_names: list[str] = []
    profile_name_to_id: dict[str, int] = {}
    remapped_profile_ids: list[torch.Tensor] = []
    fallback_names = list(args.fallback_profile_name)
    if len(fallback_names) > len(input_paths):
        raise ValueError(
            "More --fallback-profile-name values were supplied than --input caches."
        )
    for input_index, (path, cache, count) in enumerate(
        zip(input_paths, caches, row_counts, strict=True)
    ):
        local_names = cache.get("profile_names")
        local_ids = cache.get("profile_id")
        if not isinstance(local_names, (list, tuple)) or not isinstance(
            local_ids, torch.Tensor
        ):
            fallback = (
                fallback_names[input_index]
                if input_index < len(fallback_names)
                else path.stem
            )
            local_names = [str(fallback)]
            local_ids = torch.zeros(count, dtype=torch.long)
        if local_ids.shape != (count,):
            raise ValueError(
                f"Cache {path} has invalid profile_id shape {tuple(local_ids.shape)}; expected {(count,)}."
            )
        local_to_global: list[int] = []
        for raw_name in local_names:
            name = str(raw_name)
            if name not in profile_name_to_id:
                profile_name_to_id[name] = len(profile_names)
                profile_names.append(name)
            local_to_global.append(profile_name_to_id[name])
        if local_ids.numel() and (
            int(local_ids.min()) < 0 or int(local_ids.max()) >= len(local_names)
        ):
            raise ValueError(
                f"Cache {path} profile_id contains values outside profile_names."
            )
        lookup = torch.tensor(local_to_global, dtype=torch.long)
        remapped_profile_ids.append(lookup[local_ids.to(dtype=torch.long).cpu()])
    merged["profile_names"] = profile_names
    merged["profile_id"] = torch.cat(remapped_profile_ids, dim=0)[keep]

    merged["source_cache_index"] = torch.arange(merged_count, dtype=torch.long)
    merged["rebuild_source_cache_index"] = torch.arange(merged_count, dtype=torch.long)
    merged["legacy_upstream_source_cache_index"] = torch.arange(
        merged_count, dtype=torch.long
    )

    offsets = torch.tensor(
        [0, *torch.tensor(row_counts).cumsum(dim=0).tolist()], dtype=torch.long
    )
    source_file = torch.bucketize(keep, offsets[1:], right=True)
    source_row = keep - offsets[source_file]
    merged["mixture_source_file_index"] = source_file
    merged["mixture_source_row_index"] = source_row
    merged["metadata"] = {
        "schema_name": "sharpa_stable_grasp_mixture_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "poise.grasp.merge",
        "input_caches": [str(path) for path in input_paths],
        "input_row_counts": row_counts,
        "concatenated_row_count": int(sum(row_counts)),
        "duplicate_row_count": int(sum(row_counts) - merged_count),
        "mixture_row_count": merged_count,
        "dedup_signature": ["q", "q_cmd", "object_pose_palm"],
    }
    output = save_stable_grasp_cache(merged, resolve(args.output))
    report = {
        "output": str(output),
        "summary": summarize_stable_grasp_cache(merged),
        "metadata": merged["metadata"],
    }
    report_path = resolve(args.report_json or args.output.with_suffix(".json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
