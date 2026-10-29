"""Verify the complete asset contract for the D40 x L160 experiment."""

import json
from pathlib import Path

import torch

from poise.assets.hand import check_hand, sha256
from poise.grasp.cache import load_stable_grasp_cache, summarize_stable_grasp_cache
from poise.paths import workspace_root
from poise.sim.assets.sharpa_manual_hex_prisms import get_manual_hex_prism_spec


def main():
    root = workspace_root()
    spec = get_manual_hex_prism_spec("hexprism_d40_l160_original_m")
    directory = Path("data/objects/sharpa_manual_hex_prisms") / spec.object_id
    expected = {
        directory / f"{spec.object_id}.obj": spec.source_obj_sha256,
        directory / f"{spec.object_id}.usda": spec.rigid_usd_sha256,
        Path(spec.mesh_data_path): spec.mesh_data_sha256,
        Path(spec.stable_cache_path): spec.stable_cache_sha256,
    }
    for relative, digest in expected.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Asset hash mismatch: {relative}")
    cache = load_stable_grasp_cache(root / spec.stable_cache_path)
    if cache["object_id"] != spec.object_id or len(cache["q"]) != spec.stable_cache_row_count:
        raise ValueError("Wrong reset cache identity or row count")
    geometry = torch.load(root / spec.mesh_data_path, map_location="cpu", weights_only=False)
    if tuple(geometry["object_ids"]) != (spec.object_id,):
        raise ValueError("Wrong geometry object identity")
    half_extent = geometry["csgwcd_grid_half_extent_m"]
    if half_extent <= max(spec.extents_xyz_m) * .5 * 1.025:
        raise ValueError("Geometry grid does not contain the scaled long object")
    print(json.dumps({"status": "passed", "hand": check_hand(),
                      "assets": {str(k): v for k, v in expected.items()},
                      "cache": summarize_stable_grasp_cache(cache),
                      "geometry_half_extent_m": half_extent}, indent=2))


if __name__ == "__main__":
    main()
