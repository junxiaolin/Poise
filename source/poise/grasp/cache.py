"""Validate the existing Sharpa Wave reset-cache format without importing Isaac Lab.

The historical ``cube_keypoints_palm`` key stores eight bounding-box points
for any object. Its serialized name is retained until the generators migrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from poise.core.schema import ACTION_DIM, HAND_ID

SCHEMA_VERSION = 2


def validate_stable_grasp_cache(cache: dict[str, Any]) -> None:
    if not isinstance(cache, dict):
        raise TypeError("Stable grasp cache must be a dictionary.")
    if cache.get("schema_version") != SCHEMA_VERSION or cache.get("hand") != HAND_ID:
        raise ValueError("POISE requires schema_version=2 and hand='sharpa_wave_right'.")
    if cache.get("action_dim", ACTION_DIM) != ACTION_DIM:
        raise ValueError("Sharpa Wave reset states require 22 action channels.")
    if not isinstance(cache.get("object_id"), str) or not cache["object_id"].strip():
        raise ValueError("A nonempty object_id is required.")
    q = cache.get("q")
    if not isinstance(q, torch.Tensor) or q.ndim != 2 or q.shape[1] != ACTION_DIM:
        raise ValueError("q must be a tensor with shape (N, 22).")
    count = q.shape[0]
    shapes = {
        "q": (count, ACTION_DIM),
        "qdot": (count, ACTION_DIM),
        "q_cmd": (count, ACTION_DIM),
        "object_pose_palm": (count, 7),
        "cube_keypoints_palm": (count, 8, 3),
        "contact_bits": (count, 5),
        "stability_score": (count,),
    }
    for key, shape in shapes.items():
        value = cache.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{key} must be a tensor with shape {shape}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} contains NaN or Inf.")
    norms = torch.linalg.vector_norm(cache["object_pose_palm"][:, 3:7].float(), dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0):
        raise ValueError("object_pose_palm must contain unit wxyz quaternions.")


def load_stable_grasp_cache(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a trusted, versioned reset cache and validate its tensor interface."""
    cache = torch.load(Path(path), map_location=map_location, weights_only=False)
    validate_stable_grasp_cache(cache)
    return cache


def save_stable_grasp_cache(cache: dict[str, Any], path: str | Path) -> Path:
    validate_stable_grasp_cache(cache)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    return output


def summarize_stable_grasp_cache(cache: dict[str, Any]) -> dict[str, Any]:
    validate_stable_grasp_cache(cache)
    contacts = cache["contact_bits"].float().sum(-1)
    scores = cache["stability_score"].float()
    return {
        "num_grasps": len(scores),
        "mean_contact_fingers": contacts.mean().item() if len(scores) else 0.0,
        "min_contact_fingers": contacts.min().item() if len(scores) else 0.0,
        "max_contact_fingers": contacts.max().item() if len(scores) else 0.0,
        "mean_stability_score": scores.mean().item() if len(scores) else 0.0,
    }
