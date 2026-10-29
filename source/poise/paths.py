"""Resolve data independently of the current working directory."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    root = os.environ.get("POISE_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    # Source installations use the repository root. Wheel installations should
    # set POISE_ROOT to the directory holding their downloaded/generated data.
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else workspace_root() / path
