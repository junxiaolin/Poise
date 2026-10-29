#!/usr/bin/env python3
"""Select a fixed number of deterministic, feature-diverse stable grasp parents.

The first parent is the highest stability-score row.  Remaining parents use
farthest-point sampling over standardized q, q_cmd, object translation, and a
quaternion-sign-canonicalized object orientation.  Row-wise metadata is kept.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=486)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a dictionary")
    for key in ("q", "q_cmd", "object_pose_palm"):
        if not isinstance(value.get(key), torch.Tensor):
            raise KeyError(f"{path} is missing tensor {key!r}")
    return value


def row_count(cache: dict[str, Any]) -> int:
    return int(cache["q"].shape[0])


def select_rows(cache: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    total = row_count(cache)
    output: dict[str, Any] = {}
    for key, value in cache.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == total
        ):
            output[key] = value[indices]
        elif key == "metadata" and isinstance(value, dict):
            output[key] = dict(value)
        else:
            output[key] = value
    return output


def normalized_features(cache: dict[str, Any]) -> torch.Tensor:
    pose = cache["object_pose_palm"].to(torch.float64)
    quat = pose[:, 3:7].clone()
    # The largest component also gives a deterministic sign at a 180-degree
    # rotation, where the scalar component is zero.
    pivot = quat.gather(1, quat.abs().argmax(1, keepdim=True))
    quat *= torch.where(pivot < 0.0, -torch.ones_like(pivot), torch.ones_like(pivot))
    features = torch.cat(
        (
            cache["q"].to(torch.float64),
            cache["q_cmd"].to(torch.float64),
            pose[:, :3],
            quat,
        ),
        dim=-1,
    )
    scale = features.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    return (features - features.mean(dim=0)) / scale


def deterministic_farthest_point_indices(
    features: torch.Tensor, scores: torch.Tensor, count: int
) -> torch.Tensor:
    total = int(features.shape[0])
    if not 0 < count <= total:
        raise ValueError(f"target count must lie in [1,{total}], got {count}")
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = int(torch.argmax(scores).item())
    min_distance = torch.sum((features - features[selected[0]]) ** 2, dim=-1)
    min_distance[selected[0]] = -1.0
    for offset in range(1, count):
        selected[offset] = int(torch.argmax(min_distance).item())
        distance = torch.sum((features - features[selected[offset]]) ** 2, dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance[selected[: offset + 1]] = -1.0
    return selected


def main() -> int:
    args = parse_args()
    cache = load(args.input)
    total = row_count(cache)
    scores_raw = cache.get("stability_score")
    scores = (
        torch.as_tensor(scores_raw, dtype=torch.float64)
        if isinstance(scores_raw, torch.Tensor) and tuple(scores_raw.shape) == (total,)
        else torch.zeros(total, dtype=torch.float64)
    )
    selected = deterministic_farthest_point_indices(
        normalized_features(cache), scores, int(args.target_count)
    )
    output = select_rows(cache, selected)
    # Parent IDs are contiguous in selected-row order: 0..count-1 in selected-row order.
    output["source_seed_index"] = torch.arange(selected.numel(), dtype=torch.long)
    metadata = dict(output.get("metadata", {}))
    metadata["poise_parent_selection"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_rows": total,
        "target_count": int(args.target_count),
        "method": "highest_stability_then_standardized_q_qcmd_object_pose_farthest_point",
        "selected_input_indices": selected.tolist(),
    }
    output["metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_rows": total,
        "selected_rows": int(selected.numel()),
        "selected_input_indices": selected.tolist(),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "selected_input_indices"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
