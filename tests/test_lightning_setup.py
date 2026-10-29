"""Dependency setup must reject changed downloads and leave checkouts intact."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "setup_lightning_grasp", Path(__file__).parents[1] / "scripts/setup_lightning_grasp.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def dependency(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args]).decode().strip()

    git("init", "--quiet")
    (upstream / "module.py").write_text("version = 1\n")
    git("add", "module.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", "base")
    revision = git("rev-parse", "HEAD")
    (upstream / "module.py").write_text("version = 2\n")
    bundle = tmp_path / "dependency"
    bundle.mkdir()
    patch = bundle / "poise.patch"
    patch.write_text(git("diff") + "\n")
    archive = tmp_path / "kernels.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in setup.KERNEL_MEMBERS:
            handle.writestr(name, "native payload")
        handle.writestr("../../outside", "must not extract")
    manifest = {
        "repository": upstream.as_uri(), "revision": revision,
        "patch": {"file": patch.name, "sha256": setup.sha256(patch)},
        "kernels": {"sha256": setup.sha256(archive)},
        "patched_source_sha256": {"module.py": setup.sha256(upstream / "module.py")},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(setup, "DEPENDENCY", bundle)
    return archive, tmp_path / "installed", bundle


def test_setup_publishes_patched_sources_and_only_named_kernels(dependency):
    archive, destination, _ = dependency
    result = setup.setup(destination, kernel_archive=archive)
    assert (destination / "module.py").read_text() == "version = 2\n"
    assert len(result["kernel_sha256"]) == 2
    assert json.loads((destination / "poise-install.json").read_text())["kernel_sha256"] == result["kernel_sha256"]
    assert not (destination.parent / "outside").exists()
    with pytest.raises(FileExistsError):
        setup.setup(destination, kernel_archive=archive)
    assert (destination / "module.py").read_text() == "version = 2\n"


@pytest.mark.parametrize("changed", ["archive", "patch", "source"])
def test_setup_failure_never_publishes_partial_checkout(dependency, changed):
    archive, destination, bundle = dependency
    if changed == "archive":
        archive.write_bytes(b"corrupted download")
    elif changed == "patch":
        (bundle / "poise.patch").write_text("corrupted patch")
    else:
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["patched_source_sha256"]["module.py"] = "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        setup.setup(destination, kernel_archive=archive)
    assert not destination.exists()
    assert not list(destination.parent.glob(".poise-lygra-*"))


def test_duplicate_native_member_is_rejected(tmp_path):
    archive = tmp_path / "duplicate.zip"
    member = next(iter(setup.KERNEL_MEMBERS))
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"a")
        with pytest.warns(UserWarning, match="Duplicate name"):
            handle.writestr(member, b"b")
    with pytest.raises(ValueError, match="exactly one"):
        setup.extract_kernels(archive, tmp_path / "destination")


def test_shipped_patch_matches_manifest():
    manifest = json.loads((setup.DEPENDENCY / "manifest.json").read_text())
    patch = setup.DEPENDENCY / manifest["patch"]["file"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest["patch"]["sha256"]
