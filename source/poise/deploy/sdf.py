"""Immutable signed-distance fields for palm collision checks."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ForbiddenSdfGrid:
    """Validated regular XYZ SDF grid loaded from a compact ``.npz`` file."""

    path: Path
    sdf_m: np.ndarray
    origin_m: np.ndarray
    voxel_size_m: float
    metadata: dict[str, object]
    sha256: str

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.sdf_m.shape)

    @property
    def maximum_m(self) -> np.ndarray:
        return self.origin_m + self.voxel_size_m * (
            np.asarray(self.shape_xyz, dtype=np.float64) - 1.0
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def load_forbidden_sdf_grid(path: str | Path) -> ForbiddenSdfGrid:
    """Load and validate one SDF field, caching its immutable arrays."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Forbidden SDF grid does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as payload:
        required = {"sdf_m", "origin_m", "voxel_size_m", "metadata_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Forbidden SDF grid is missing fields: {sorted(missing)}")
        sdf_m = np.asarray(payload["sdf_m"], dtype=np.float32)
        origin_m = np.asarray(payload["origin_m"], dtype=np.float64)
        voxel_size_m = float(np.asarray(payload["voxel_size_m"]).item())
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if sdf_m.ndim != 3 or min(sdf_m.shape) < 2:
        raise ValueError(f"Forbidden SDF must be a 3D grid, got {sdf_m.shape}.")
    if origin_m.shape != (3,) or not np.all(np.isfinite(origin_m)):
        raise ValueError("Forbidden SDF origin_m must be finite XYZ.")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("Forbidden SDF voxel_size_m must be finite and positive.")
    if not np.all(np.isfinite(sdf_m)):
        raise ValueError("Forbidden SDF contains NaN or Inf.")
    if metadata.get("schema") != "sharpa_hand_collision_sdf_v1":
        raise ValueError("Unsupported forbidden SDF schema.")
    if metadata.get("sign") != "positive_inside_negative_outside":
        raise ValueError("Forbidden SDF sign convention is unsafe or unsupported.")
    if tuple(metadata.get("shape", ())) != tuple(int(v) for v in sdf_m.shape):
        raise ValueError("Forbidden SDF metadata shape does not match the field.")
    sdf_m.setflags(write=False)
    origin_m.setflags(write=False)
    return ForbiddenSdfGrid(
        path=resolved,
        sdf_m=sdf_m,
        origin_m=origin_m,
        voxel_size_m=voxel_size_m,
        metadata=metadata,
        sha256=sha256_file(resolved),
    )


def sample_forbidden_sdf_trilinear(
    points_m: np.ndarray,
    grid: ForbiddenSdfGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample XYZ points with trilinear interpolation.

    Returns ``(distance_m, valid)``. Values outside the padded field are set to
    ``-inf`` and therefore never become forbidden merely because of padding.
    """

    points = np.asarray(points_m, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("Forbidden SDF query points must be finite XYZ values.")
    flat = points.reshape(-1, 3)
    index = (flat - grid.origin_m[None, :]) / grid.voxel_size_m
    upper_index = np.asarray(grid.shape_xyz, dtype=np.float64) - 1.0
    valid = np.all((index >= 0.0) & (index <= upper_index[None, :]), axis=-1)
    safe = np.clip(index, 0.0, upper_index)
    lo = np.floor(safe).astype(np.int64)
    hi = np.minimum(lo + 1, np.asarray(grid.shape_xyz, dtype=np.int64) - 1)
    frac = safe - lo
    x0, y0, z0 = lo.T
    x1, y1, z1 = hi.T
    fx, fy, fz = frac.T
    field = grid.sdf_m
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    sampled = c0 * (1.0 - fz) + c1 * fz
    sampled = np.where(valid, sampled, -np.inf)
    return sampled.reshape(points.shape[:-1]), valid.reshape(points.shape[:-1])


def forbidden_sdf_hits(
    points_m: np.ndarray,
    *,
    grid: ForbiddenSdfGrid,
    clearance_m: float | np.ndarray,
) -> np.ndarray:
    """Return point hits against the mesh dilated by scalar/per-point clearance."""

    clearance = np.asarray(clearance_m, dtype=np.float64)
    if not np.all(np.isfinite(clearance)) or np.any(clearance < 0.0):
        raise ValueError("Forbidden SDF clearance must be finite and non-negative.")
    sampled, valid = sample_forbidden_sdf_trilinear(points_m, grid)
    try:
        threshold = np.broadcast_to(clearance, sampled.shape)
    except ValueError as exc:
        raise ValueError(
            "Forbidden SDF clearance must be scalar or broadcast to query points."
        ) from exc
    return valid & (sampled >= -threshold)
