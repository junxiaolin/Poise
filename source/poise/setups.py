"""Versioned reproduction setups. This module needs only the standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from poise.paths import workspace_root


def repository_root():
    """Code/config location, deliberately independent of POISE_ROOT (data)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs/setups").is_dir():
        raise ValueError("Setup commands require a source checkout: install POISE with pip install --no-deps -e . from the repository root")
    return root


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError(f"Expected a normalized relative path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"Expected a normalized relative path: {value!r}")
    return Path(*path.parts)


def data_path(root, value):
    root = Path(root).resolve()
    path = root / relative_path(value)
    # Prevent existing symlinks from redirecting installation outside the workspace.
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Asset escapes data workspace: {value}")
    return path


def read_pinned(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Configuration SHA256 mismatch: {path}")
    return json.loads(Path(path).read_text())


def recipe_path(name):
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid packaged recipe name: {name}")
    return Path(__file__).parent / "sim/recipes" / f"{name}.json"


def list_setups():
    return [load_setup(p.stem) for p in sorted((repository_root() / "configs/setups").glob("*.json"))]


def load_setup(name):
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid setup ID: {name}")
    root = repository_root()
    path = root / "configs/setups" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown setup {name!r}; run poise list-setups")
    setup = json.loads(path.read_text())
    if setup["schema_version"] != 1 or setup["id"] != name:
        raise ValueError("Invalid setup identity/version")
    recipe = read_pinned(recipe_path(setup["recipe"]["name"]), setup["recipe"]["sha256"])
    read_pinned(recipe_path(setup["ppo"]["name"]), setup["ppo"]["sha256"])
    manifest = read_pinned(root / relative_path(setup["assets_manifest"]), setup["assets_manifest_sha256"])
    if manifest["schema_version"] != 1 or manifest["setup_id"] != name:
        raise ValueError("Asset manifest belongs to another setup")
    rollout = read_pinned(root / relative_path(setup["rollout"]["config"]), setup["rollout"]["sha256"])
    unknown_overrides = rollout["overrides"].keys() - recipe.keys()
    if unknown_overrides:
        raise ValueError(f"Rollout overrides absent from training recipe: {sorted(unknown_overrides)}")
    objects = setup["objects"]
    if not objects or len({o["id"] for o in objects}) != len(objects):
        raise ValueError("Setup must contain distinct objects")
    for field, key in (("primitive_object_ids", "id"), ("primitive_dims_m", "dimensions_m"),
                       ("primitive_object_shape_names", "shape"), ("primitive_stable_cache_paths", "cache")):
        if recipe[field] != [o[key] for o in objects]:
            raise ValueError(f"Setup object order differs from recipe: {field}")
    if recipe["hand_type"] != setup["hand"]:
        raise ValueError("Setup hand differs from recipe")
    prefix = "multiobject_curriculum" if recipe.get("multiobject_per_object_curriculum_enabled") else "curriculum"
    if recipe[f"{prefix}_global_min_frontier_done_episodes"] != setup["training"]["global_frontier_min"]:
        raise ValueError("Setup curriculum gate differs from recipe")
    seen = set()
    for record in manifest["files"]:
        p = relative_path(record["path"])
        if p.parts[0] != "data" or record["path"] in seen:
            raise ValueError(f"Duplicate or non-data manifest path: {p}")
        seen.add(record["path"])
        digest = record["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or record["bytes"] < 0:
            raise ValueError(f"Invalid asset digest/size: {p}")
    required = {o[k] for o in objects for k in ("cache", "mesh", "usd")}
    required.update(recipe[k] for k in ("stable_cache_path", "object_mesh_data_path", "primitive_geometry_latent_path",
                                        "pose6d_feasible_target_forbidden_sdf_path"))
    required.add(recipe["robot_cfg"]["spawn"]["usd_path"])
    if not required <= seen:
        raise ValueError(f"Missing recipe assets in manifest: {sorted(required - seen)}")
    reference = setup["reference_checkpoint"]
    matches = [r for r in manifest["files"] if r["path"] == reference["path"]]
    if len(matches) != 1 or matches[0]["sha256"] != reference["sha256"] or matches[0]["role"] != "reference_checkpoint":
        raise ValueError("Reference checkpoint differs from asset manifest")
    return setup


def asset_records(setup, *, with_reference=False):
    manifest = read_pinned(repository_root() / relative_path(setup["assets_manifest"]), setup["assets_manifest_sha256"])
    return [r for r in manifest["files"] if with_reference or r["role"] != "reference_checkpoint"]


def setup_contract(setup):
    return {"schema_version": 1, "id": setup["id"], "recipe_sha256": setup["recipe"]["sha256"],
            "ppo_sha256": setup["ppo"]["sha256"], "assets_manifest_sha256": setup["assets_manifest_sha256"],
            "object_ids": [o["id"] for o in setup["objects"]]}


def check_setup(setup, *, root=None, with_reference=False, deep=False):
    """Hashes are checked before loading any trusted tensor serialization."""
    root = Path(root or workspace_root()).resolve()
    missing, changed = [], []
    records = asset_records(setup, with_reference=with_reference)
    for r in records:
        p = data_path(root, r["path"])
        if not p.is_file():
            missing.append(r["path"])
        elif p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            changed.append(r["path"])
    report = {"setup": setup["id"], "root": str(root), "valid": not (missing or changed),
              "files": len(records), "missing": missing, "changed": changed, "with_reference": with_reference}
    if not report["valid"] or not deep:
        return report
    import torch
    from poise.grasp.cache import load_stable_grasp_cache

    recipe = read_pinned(recipe_path(setup["recipe"]["name"]), setup["recipe"]["sha256"])
    geometry = torch.load(data_path(root, recipe["object_mesh_data_path"]), map_location="cpu", weights_only=False)
    ids = list(geometry["object_ids"])
    if ids != [o["id"] for o in setup["objects"]]:
        raise ValueError("Geometry object order differs from setup")
    if geometry["geometry_latents"].shape != (len(ids), 64) or not torch.isfinite(geometry["geometry_latents"]).all():
        raise ValueError("Invalid BPS64 descriptors")
    rows = {}
    for i, obj in enumerate(setup["objects"]):
        cache = load_stable_grasp_cache(data_path(root, obj["cache"]))
        if cache["object_id"] != obj["id"] or len(cache["q"]) != obj["cache_rows"]:
            raise ValueError(f"Cache identity/count differs: {obj['id']}")
        dims = torch.as_tensor(obj["dimensions_m"], dtype=torch.float64)
        for value in (cache["object_dims_m"], geometry["object_dims_m"][i]):
            if not torch.allclose(torch.as_tensor(value, dtype=torch.float64), dims, atol=1e-6, rtol=0):
                raise ValueError(f"Object dimensions differ: {obj['id']}")
        for key in ("q", "q_cmd", "object_pose_palm"):
            if not torch.isfinite(cache[key]).all():
                raise ValueError(f"Non-finite {key}: {obj['id']}")
        rows[obj["id"]] = len(cache["q"])
    report["cache_rows"] = rows
    if with_reference:
        report["checkpoint"] = check_checkpoint(setup, data_path(root, setup["reference_checkpoint"]["path"]))
    return report


def require_setup(setup, **kwargs):
    report = check_setup(setup, **kwargs)
    if not report["valid"]:
        raise ValueError(f"Setup assets incomplete or modified: {json.dumps(report)}. Run poise fetch-assets --setup {setup['id']} --source PATH")
    return report


def check_checkpoint(setup, path):
    import torch

    path = Path(path)
    digest = sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    info = (saved.get("infos") or {}).get("poise", {})
    reference = digest == setup["reference_checkpoint"]["sha256"]
    if not reference and info.get("setup") != setup_contract(setup):
        raise ValueError("Checkpoint has no matching setup contract; use the matching setup/reference or the legacy --recipe entry for an older custom run")
    states = info.get("rank_states", [info])
    if not states or any(s.get("object_ids") != [o["id"] for o in setup["objects"]] for s in states):
        raise ValueError("Checkpoint object order differs from setup")
    weights = saved["model_state_dict"]
    if weights["actor.0.weight"].shape[1] != 244 or weights["actor.6.weight"].shape[0] != 22:
        raise ValueError("Checkpoint must implement the 244D/22D interface")
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Checkpoint contains non-finite policy tensors")
    return {"sha256": digest, "reference": reference, "iteration": saved.get("iter"), "setup": setup["id"]}
