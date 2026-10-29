"""Generate the palm collision SDF used by target feasibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Palm collision mesh in meters, in the hand-root frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.002)
    parser.add_argument("--padding", type=float, default=0.03)
    args = parser.parse_args()
    if not np.isfinite([args.voxel_size, args.padding]).all() or args.voxel_size <= 0 or args.padding <= 0:
        parser.error("voxel-size and padding must be finite and positive")
    mesh = trimesh.load_mesh(args.mesh, process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError("Signed distances require a watertight triangle collision mesh")
    if mesh.extents.max() > 0.5:
        raise ValueError("Palm mesh must be in meters, not millimeters")
    mesh.fix_normals()
    origin = np.floor((mesh.bounds[0] - args.padding) / args.voxel_size) * args.voxel_size
    maximum = np.ceil((mesh.bounds[1] + args.padding) / args.voxel_size) * args.voxel_size
    shape = np.rint((maximum - origin) / args.voxel_size).astype(int) + 1
    indices = np.indices(shape).reshape(3, -1).T
    points = origin + indices * args.voxel_size
    distance = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), 8192):
        distance[start:start + 8192] = trimesh.proximity.signed_distance(mesh, points[start:start + 8192])
    metadata = {"schema": "sharpa_hand_collision_sdf_v1", "sign": "positive_inside_negative_outside",
                "shape": shape.tolist(), "source_mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                "frame": "sharpa_hand_root", "voxel_size_m": args.voxel_size}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sdf_m=distance.reshape(shape), origin_m=origin,
                        voxel_size_m=np.asarray(args.voxel_size), metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps({"path": str(args.output), "shape": shape.tolist(), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
