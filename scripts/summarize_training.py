"""Summarize local TensorBoard training signals without starting Isaac Lab."""

import argparse
import json
import math
from pathlib import Path
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Episode/episode_success_rate",
    "Episode/first_target_success_rate",
    "Episode/shared_rotation_curriculum_max_deg",
    "Episode/translation_delta_curriculum_max_m",
    "Episode/target_delta_frontier_success_rate",
    "Episode/translation_delta_frontier_success_rate",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=100, help="Number of logged scalar entries in each tail average")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    if not args.run.is_dir():
        parser.error(f"Training directory does not exist: {args.run}")
    events = EventAccumulator(str(args.run), size_guidance={"scalars": 0}).Reload()
    available = set(events.Tags()["scalars"])
    metrics = {}
    for name in METRICS:
        if name not in available:
            continue
        records = events.Scalars(name)
        if not records:
            continue
        values = [record.value for record in records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite training metric: {name}")
        tail = values[-args.window:]
        metrics[name] = {
            "first_step": records[0].step, "last_step": records[-1].step,
            "first": values[0], "last": values[-1], "maximum": max(values),
            "tail_entries": len(tail), "tail_mean": statistics.mean(tail),
        }
    if not metrics:
        raise ValueError("No POISE training scalars found in this directory")
    result = {
        "run": str(args.run.resolve()), "window": args.window, "metrics": metrics,
        "interpretation": "Tail means average logged scalar entries, not pooled episode outcomes. Training signals are not held-out reaching evaluation.",
    }
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
