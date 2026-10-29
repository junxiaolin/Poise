#!/usr/bin/env python3
"""Generate dedicated BPS64, mass properties, and 64^3 CSGWCD fields for manual hex prisms."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import trimesh


from poise.paths import workspace_root
ROOT = workspace_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-id",
        action="append",
        default=None,
        help="Object id to generate; repeat for multiple objects. Default: both manual hex prisms.",
    )
    parser.add_argument("--grid-resolution", type=int, default=64)
    parser.add_argument("--grid-half-extent-m", type=float, default=0.080)
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional combined output. By default, write the dedicated path declared by each asset spec.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bps_basis_points() -> np.ndarray:
    axis = np.linspace(-0.0225, 0.0225, 4, dtype=np.float64)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)


def load_canonical_mesh(spec) -> tuple[trimesh.Trimesh, Path]:
    from poise.sim.assets.sharpa_manual_hex_prisms import get_manual_hex_prism_obj_path

    obj_path = get_manual_hex_prism_obj_path(spec.object_id)
    got_sha = sha256_file(obj_path)
    if got_sha != spec.source_obj_sha256:
        raise RuntimeError(
            f"Canonical OBJ SHA mismatch for {spec.object_id}: expected {spec.source_obj_sha256}, got {got_sha}"
        )
    mesh = trimesh.load_mesh(obj_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(mesh).__name__}: {obj_path}")
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if not mesh.is_watertight:
        raise RuntimeError(f"Manual hex-prism OBJ is not watertight: {obj_path}")
    if not mesh.is_convex:
        raise RuntimeError(f"Manual hex-prism OBJ is not convex: {obj_path}")
    if not np.allclose(mesh.extents, np.asarray(spec.extents_xyz_m), atol=1.0e-7):
        raise RuntimeError(
            f"OBJ dimensions mismatch for {spec.object_id}: got {mesh.extents.tolist()}, "
            f"expected {spec.extents_xyz_m}"
        )
    return mesh, obj_path


def nearest_surface(
    mesh: trimesh.Trimesh, points: np.ndarray, *, chunk_size: int = 32768
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closest_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        stop = min(len(points), start + chunk_size)
        closest, distance, triangle_ids = trimesh.proximity.closest_point(mesh, points[start:stop])
        closest_parts.append(np.asarray(closest, dtype=np.float64))
        distance_parts.append(np.asarray(distance, dtype=np.float64))
        triangle_parts.append(np.asarray(triangle_ids, dtype=np.int64))
    return (
        np.concatenate(closest_parts, axis=0),
        np.concatenate(distance_parts, axis=0),
        np.concatenate(triangle_parts, axis=0),
    )


def build_payload(specs, grid_resolution: int, grid_half_extent_m: float) -> dict:
    basis = bps_basis_points()
    grid_axis = np.linspace(-grid_half_extent_m, grid_half_extent_m, grid_resolution, dtype=np.float64)
    gx, gy, gz = np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")
    grid_points = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)

    latents, volumes, coms, inertias, delta_grids, normal_grids = [], [], [], [], [], []
    obj_sha256s = []
    for spec in specs:
        mesh, obj_path = load_canonical_mesh(spec)
        _, bps_distance, _ = nearest_surface(mesh, basis)
        props = mesh.mass_properties
        volume = float(props["volume"])
        if volume <= 0.0:
            raise RuntimeError(f"Non-positive mesh volume for {spec.object_id}: {volume}")
        mass = float(props["mass"])
        inertia_unit_mass = np.asarray(props["inertia"], dtype=np.float64) / mass

        closest, _, triangle_ids = nearest_surface(mesh, grid_points)
        delta = closest - grid_points
        normals = np.asarray(mesh.face_normals[triangle_ids], dtype=np.float64)
        shape = (grid_resolution, grid_resolution, grid_resolution, 3)
        latents.append(bps_distance / 0.060)
        volumes.append(volume)
        coms.append(np.asarray(props["center_mass"], dtype=np.float64))
        inertias.append(inertia_unit_mass)
        delta_grids.append(delta.reshape(shape))
        normal_grids.append(normals.reshape(shape))
        obj_sha256s.append(sha256_file(obj_path))
        print(
            f"[OK] {spec.object_id}: extents={mesh.extents.tolist()} volume={volume:.9g} "
            f"inertia_eig={np.linalg.eigvalsh(inertia_unit_mass).tolist()}",
            flush=True,
        )

    return {
        "format": "sharpa_manual_hex_prism_bps64_mesh_data_64_v1",
        "object_ids": tuple(spec.object_id for spec in specs),
        "object_families": tuple("hex_prism" for _ in specs),
        "object_dims_m": torch.tensor([spec.extents_xyz_m for spec in specs], dtype=torch.float32),
        "source_obj_sha256": tuple(obj_sha256s),
        "geometry_latents": torch.tensor(np.stack(latents), dtype=torch.float32),
        "bps_basis_points_m": torch.tensor(basis, dtype=torch.float32),
        "bps_distance_normalization_m": 0.060,
        "mesh_volume_m3": torch.tensor(volumes, dtype=torch.float64),
        "mesh_com_m": torch.tensor(np.stack(coms), dtype=torch.float32),
        "mesh_inertia_unit_mass_kg_m2": torch.tensor(np.stack(inertias), dtype=torch.float32),
        "csgwcd_grid_resolution": int(grid_resolution),
        "csgwcd_grid_half_extent_m": float(grid_half_extent_m),
        "csgwcd_surface_delta_grid_m": torch.tensor(np.stack(delta_grids), dtype=torch.float32),
        "csgwcd_outward_normal_grid": torch.tensor(np.stack(normal_grids), dtype=torch.float32),
    }


def save_payload(payload: dict, path: Path) -> None:
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[WROTE] {path} ({path.stat().st_size / 1024**2:.1f} MiB, sha256={sha256_file(path)})")


def main() -> int:
    args = parse_args()
    if args.grid_resolution != 64:
        raise ValueError("Formal manual hex-prism training requires exactly a 64^3 CSGWCD field")
    if args.grid_half_extent_m <= 0.05:
        raise ValueError("grid-half-extent-m must leave margin around the complete object mesh")

    from poise.sim.assets.sharpa_manual_hex_prisms import SHARPA_MANUAL_HEX_PRISM_SPECS

    specs = SHARPA_MANUAL_HEX_PRISM_SPECS
    if args.object_id:
        requested = tuple(str(value) for value in args.object_id)
        by_id = {spec.object_id: spec for spec in specs}
        unknown = tuple(value for value in requested if value not in by_id)
        if unknown:
            raise ValueError(f"Unknown manual hex-prism object ids: {unknown}")
        specs = tuple(by_id[value] for value in requested)

    if args.combined_output is not None:
        save_payload(build_payload(specs, args.grid_resolution, args.grid_half_extent_m), args.combined_output)
    else:
        for spec in specs:
            save_payload(build_payload((spec,), args.grid_resolution, args.grid_half_extent_m), Path(spec.mesh_data_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
