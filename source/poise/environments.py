"""Inspect stage dependencies without connecting to cameras or robot hardware."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import subprocess

from poise.setups import repository_root, sha256


def check_environment(stage, *, foundationpose_root=None, native=False):
    manifest = json.loads((repository_root() / "environments" / stage / "manifest.json").read_text())
    versions, missing, differences = {}, [], {}
    for name, expected in manifest["tested_packages"].items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if versions[name] != expected:
            differences[name] = {"installed": versions[name], "tested": expected}
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python != manifest["python"]:
        differences["python"] = {"installed": python, "tested": manifest["python"]}
    report = {"stage": stage, "python": sys.executable, "versions": versions,
              "missing": missing, "version_differences": differences, "valid": not missing and not differences,
              "native": "not_requested", "hardware_access": False}
    if native and not missing:
        with contextlib.redirect_stdout(sys.stderr):
            if stage == "lygra-poise":
                from poise.grasp.lightning.runtime import check_runtime
                report["native"] = check_runtime()
            elif stage == "foundationpose-poise":
                if foundationpose_root is None:
                    raise ValueError("--foundationpose-root is required for the native deployment check")
                root = Path(foundationpose_root).expanduser().resolve()
                pins = json.loads((repository_root() / "dependencies/foundationpose/manifest.json").read_text())
                revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
                if revision != pins["revision"] or changed:
                    raise ValueError("FoundationPose checkout differs from the pinned, unmodified source revision")
                report["source_revision"] = revision
                for relative, expected in pins["weights_sha256"].items():
                    path = root / relative
                    if not path.is_file() or sha256(path) != expected:
                        raise ValueError(f"FoundationPose weight/config missing or modified: {path}")
                sys.path.insert(0, str(root))
                try:
                    importlib.import_module("nvdiffrast.torch")
                    estimator = importlib.import_module("estimater")
                    if Path(estimator.__file__).resolve() != root / "estimater.py":
                        raise ValueError("FoundationPose imported from a different checkout")
                    cv2 = importlib.import_module("cv2")
                    if not hasattr(cv2, "aruco"):
                        raise ValueError("OpenCV ArUco is unavailable")
                    from poise.core.observation import SharpaPose6DObservationBuilder
                    from poise.deploy.tracker import FoundationPoseObjectTracker
                    report["native"] = "FoundationPose, rasterizer, ArUco and POISE imports passed; no model inference or device capture"
                finally:
                    sys.path.pop(0)
            else:
                importlib.import_module("isaaclab.app")
                report["native"] = "AppLauncher import passed; run setup smoke for simulation validation"
    return report
