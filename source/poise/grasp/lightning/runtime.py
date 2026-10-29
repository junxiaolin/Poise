"""Check the LyGra interfaces and native extensions used by POISE."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys


REQUIRED_PARAMETERS = {
    "lygra.pipeline.module.collision.batch_object_hand_collision_check": ("default_q",),
    "lygra.pipeline.module.contact_optimization.search_contact_point": (
        "external_force_hand", "external_force_point_object", "return_wrench_info",
    ),
    "lygra.pipeline.module.kinematics.batch_ik": ("extra_result",),
    "lygra.pipeline.module.kinematics.batch_contact_adjustment": ("extra_result",),
    "lygra.batch_grasp_optimizer.compute_wrench_score_vectorized": (
        "external_force", "external_force_pos", "external_torque", "return_best_solution",
    ),
    "lygra.urdf_utils.override_urdf_joint_limits": ("joint_limits", "profile_name"),
}
REQUIRED_NATIVE_EXPORTS = {
    "gem": (
        "batch_random_mask_gather", "batch_point_to_link_mesh_collision",
        "batch_link_vertex_transform", "batch_link_mesh_gjk", "batch_solve_contact_with_normal",
    ),
    "lbvh": ("lux_build_s2bundle_bvh", "lux_traverse_s2bundle_bvh_with_buffer"),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_interfaces():
    for name, required in REQUIRED_PARAMETERS.items():
        module, attribute = name.rsplit(".", 1)
        function = getattr(importlib.import_module(module), attribute)
        missing = set(required) - set(inspect.signature(function).parameters)
        if missing:
            raise RuntimeError(f"{name} lacks POISE parameters: {sorted(missing)}")


def check_runtime(*, cuda_smoke=False):
    # Upstream prints native-loader errors to stdout; keep the JSON report clean.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            lygra = importlib.import_module("lygra")
            for name, exports in REQUIRED_NATIVE_EXPORTS.items():
                extension = getattr(lygra, name, None)
                if extension is None:
                    raise RuntimeError(f"LyGra native extension {name} did not load")
                missing = [export for export in exports if not callable(getattr(extension, export, None))]
                if missing:
                    raise RuntimeError(f"LyGra native extension {name} lacks exports: {missing}")
            check_interfaces()
            importlib.import_module("poise.grasp.lightning.generate")
            importlib.import_module("poise.grasp.lightning.qcmd")
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Lightning Grasp preflight failed: {error}. "
                "Install the pinned sources/patch/kernels in docs/lightning_grasp.md "
                "using the optimizer Python."
            ) from error
    root = Path(lygra.__file__).resolve().parent

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    report = {
        "package": str(root),
        "python": sys.executable,
        "revision": git("rev-parse", "HEAD"),
        "tracked_changes": git("diff", "HEAD", "--stat"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "trimesh", "urdfpy", "networkx", "open3d")},
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))},
        "kernel_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "cpp/build").rglob("*.so"))},
        "interfaces": "passed",
        "cuda_smoke": "not_requested",
    }
    receipt = root.parent / "poise-install.json"
    if receipt.is_file():
        report["installation_receipt_sha256"] = sha256(receipt)
    if cuda_smoke:
        import torch
        from lygra.batch_grasp_optimizer import compute_wrench_score_vectorized
        from lygra.bvh_s2bundle import ContactFieldS2BundleLBVH
        from lygra.pipeline.module.contact_collection import sample_from_mask_and_gather

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the optimizer environment")
        # Execute both native extensions, rather than only checking their imports.
        points = torch.tensor([[0., 0., 0.], [0.02, 0., 0.]], device="cuda")
        mask = torch.tensor([[[1, 0]]], device="cuda", dtype=torch.int32)
        gathered, indices = sample_from_mask_and_gather(mask, points, 1)
        if not torch.equal(gathered.reshape(-1, 3), points[:1]) or indices.item() != 0:
            raise RuntimeError("Geometry CUDA gather smoke returned the wrong point")
        bvh = ContactFieldS2BundleLBVH()
        directions = torch.tensor([[1., 0., 0.], [1., 0., 0.]], device="cuda")
        bvh.build(points - 0.005, points + 0.005,
                  points, directions, torch.arange(2, device="cuda"), leaf_size=1)
        matches = bvh.query(points, directions, bvh_buffer=torch.zeros(16, device="cuda", dtype=torch.int32))
        if not torch.isfinite(matches).all() or not torch.equal(matches.reshape(-1).long(), torch.arange(2, device="cuda")):
            raise RuntimeError("LBVH CUDA query smoke did not find its input points")
        # Two opposing contacts, with a small external force. This exercises the
        # extra return values that q_cmd needs, without generating hand fields.
        positions = torch.tensor([[[-0.02, 0., 0.], [0.02, 0., 0.]]], device="cuda")
        normals = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]], device="cuda")
        score, _, solution = compute_wrench_score_vectorized(
            positions, normals, external_force=torch.tensor([[0.05, 0., 0.]], device="cuda"),
            return_best_solution=True,
        )
        coefficients = solution["contact_force_coeff"]
        if coefficients.shape != (1, 2) or not torch.isfinite(coefficients).all() or not torch.isfinite(score).all():
            raise RuntimeError("CUDA wrench smoke returned invalid forces")
        report["cuda_smoke"] = {
            "device": torch.cuda.get_device_name(), "wrench_score": score.item(),
            "geometry_gather": "passed", "lbvh_query": "passed",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", action="store_true", help="Also exercise both native kernels and a two-contact CUDA wrench")
    args = parser.parse_args()
    print(json.dumps(check_runtime(cuda_smoke=args.cuda_smoke), indent=2))


if __name__ == "__main__":
    main()
