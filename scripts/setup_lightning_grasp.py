"""Fetch pinned public LyGra sources, POISE's patch and upstream CUDA binaries.

Uses only the Python standard library and Git. Does not install Python packages
or modify existing checkouts. See docs/lightning_grasp.md for environment setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


DEPENDENCY = Path(__file__).resolve().parents[1] / "dependencies/lightning_grasp"
KERNEL_MEMBERS = {
    "build/geometry/geometry.so": "lygra/cpp/build/geometry/geometry.so",
    "build/lbvh/lbvh.so": "lygra/cpp/build/lbvh/lbvh.so",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def extract_kernels(archive, destination):
    # Never extract arbitrary archive paths or symlinks. Only these two payloads
    # are used; reject duplicate members instead of picking one silently.
    with zipfile.ZipFile(archive) as bundle:
        for member, relative in KERNEL_MEMBERS.items():
            if bundle.namelist().count(member) != 1:
                raise ValueError(f"Expected exactly one kernel archive member: {member}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def setup(destination, *, kernel_archive=None):
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination exists; choose a new directory: {destination}")
    manifest_path = DEPENDENCY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    patch = DEPENDENCY / manifest["patch"]["file"]
    verify(patch, manifest["patch"]["sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".poise-lygra-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        checkout = stage / "checkout"
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)

        def git(*args):
            return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()

        git("remote", "add", "origin", manifest["repository"])
        git("fetch", "--depth=1", "origin", manifest["revision"])
        git("checkout", "--quiet", "--detach", "FETCH_HEAD")
        if git("rev-parse", "HEAD") != manifest["revision"]:
            raise ValueError("Fetched revision differs from manifest")
        git("apply", "--check", str(patch))
        git("apply", str(patch))
        for relative, expected in manifest["patched_source_sha256"].items():
            verify(checkout / relative, expected)
        archive = Path(kernel_archive) if kernel_archive is not None else stage / "kernels.zip"
        if kernel_archive is None:
            with urllib.request.urlopen(manifest["kernels"]["url"], timeout=60) as source, archive.open("xb") as output:
                shutil.copyfileobj(source, output)
        verify(archive, manifest["kernels"]["sha256"])
        extract_kernels(archive, checkout)
        receipt = {
            "manifest": manifest,
            "manifest_sha256": sha256(manifest_path),
            "kernel_sha256": {relative: sha256(checkout / relative) for relative in KERNEL_MEMBERS.values()},
        }
        (checkout / "poise-install.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Only publish a complete checkout. A failed download/apply is discarded.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        checkout.rename(destination)
    return {"destination": str(destination), **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--kernel-archive", type=Path, help="Use an already downloaded, hash-verified release ZIP")
    args = parser.parse_args()
    print(json.dumps(setup(args.destination, kernel_archive=args.kernel_archive), indent=2))


if __name__ == "__main__":
    main()
