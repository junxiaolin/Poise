"""Asset installation verifies bytes before replacing an existing hand tree."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from poise.assets import hand


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "model.usda").write_text("public asset")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest = hand.hand_manifest()
    manifest.update(revision=revision, files=[{"source": "model.usda", "destination": "upstream.usda",
                     "sha256": hand.sha256(root / "model.usda")}])
    monkeypatch.setattr(hand, "hand_manifest", lambda: manifest)
    return root, manifest


def test_stage_verify_replace_and_preserve_backup(source, tmp_path):
    root, _ = source
    dest = tmp_path / "hand"
    dest.mkdir()
    (dest / "old.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        hand.install_hand(dest, source=root)
    report = hand.install_hand(dest, source=root, replace=True)
    assert (Path(report["backup"]) / "old.txt").read_text() == "keep me"
    assert hand.check_hand(dest)["valid"]
    assert (dest / "poise_collision.usda").is_file()
    assert hand.install_hand(dest, source=root)["valid"]


def test_modified_source_does_not_touch_current_installation(source, tmp_path):
    root, _ = source
    dest = tmp_path / "hand"
    dest.mkdir()
    (dest / "old.txt").write_text("keep me")
    (root / "model.usda").write_text("tampered")
    with pytest.raises(ValueError, match="SHA256"):
        hand.install_hand(dest, source=root, replace=True)
    assert (dest / "old.txt").read_text() == "keep me"
    assert not list(tmp_path.glob("hand.backup-*"))


def test_wrong_revision_and_changed_dependency_rejected(source, tmp_path):
    root, manifest = source
    actual = manifest["revision"]
    manifest["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        hand.install_hand(tmp_path / "hand", source=root)
    manifest["revision"] = actual
    hand.install_hand(tmp_path / "hand", source=root)
    (tmp_path / "hand/upstream.usda").write_text("corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        hand.check_hand(tmp_path / "hand")


def test_archive_ignores_unselected_paths_and_rejects_links(tmp_path):
    record = {"source": "asset", "destination": "upstream.usda", "sha256": hashlib.sha256(b"ok").hexdigest()}
    archive = tmp_path / "source.tar.gz"
    def write_archive(link=False):
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("sharpa-urdf-usd-xml-pinned/asset")
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                tar.addfile(info)
            else:
                info.size = 2
                tar.addfile(info, io.BytesIO(b"ok"))
            bad = tarfile.TarInfo("../../outside")
            bad.size = 3
            tar.addfile(bad, io.BytesIO(b"bad"))
    write_archive()
    hand._read_archive(archive, [record], tmp_path / "hand", "pinned")
    assert (tmp_path / "hand/upstream.usda").read_bytes() == b"ok"
    assert not (tmp_path / "outside").exists()
    write_archive(link=True)
    with pytest.raises(ValueError, match="regular"):
        hand._read_archive(archive, [record], tmp_path / "hand", "pinned")


def test_packaged_manifest_checksums_match_overlays():
    manifest = hand.hand_manifest()
    for record in manifest["overlays"]:
        data = hand.files("poise.assets.sharpa_wave").joinpath(record["resource"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
