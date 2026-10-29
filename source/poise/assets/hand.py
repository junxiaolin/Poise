"""Install the pinned public Sharpa right hand and auditable collision overlays."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen
import uuid

from poise.paths import resolve_path


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hand_manifest():
    return json.loads(files("poise.assets.sharpa_wave").joinpath("manifest.json").read_text())


def _expected(manifest):
    return {**{r["destination"]: r["sha256"] for r in manifest["files"]},
            **{r["resource"]: r["sha256"] for r in manifest["overlays"]}}


def check_hand(destination=None):
    """Verify the full installed dependency tree, not only its entrypoint."""
    destination = Path(destination or resolve_path("data/robots/sharpa_wave")).expanduser().resolve()
    manifest = hand_manifest()
    expected = _expected(manifest)
    for rel, digest in expected.items():
        path = destination / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Sharpa asset missing or SHA256 mismatch: {path}; run poise fetch-hand")
    return {"valid": True, "directory": str(destination), "repository": manifest["repository"],
            "revision": manifest["revision"], "verified_files": len(expected),
            "entrypoint_sha256": expected["right_sharpa_wave.usda"],
            "collision_overlay_sha256": expected["poise_collision.usda"]}


def _read_archive(archive, records, destination, revision):
    """Read only named regular files; never extract archive paths or symlinks."""
    prefix = f"sharpa-urdf-usd-xml-{revision}"
    with tarfile.open(archive, "r:gz") as source:
        for record in records:
            name = f"{prefix}/{record['source']}"
            member = source.getmember(name)
            if not member.isfile():
                raise ValueError(f"Expected regular asset file: {name}")
            with source.extractfile(member) as handle:
                _write_verified(handle.read(), record, destination)


def _write_verified(content, record, destination):
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise ValueError(f"Upstream SHA256 mismatch: {record['source']}")
    rel = PurePosixPath(record["destination"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Invalid manifest destination: {rel}")
    output = destination / str(rel)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def install_hand(destination=None, *, source=None, replace=False):
    """Stage and verify before switching; preserve previous assets on replacement.

    ``source`` selects a local checkout at the pinned revision. Without it the
    public HTTPS archive is downloaded; no private research checkout is needed.
    """
    destination = Path(destination or resolve_path("data/robots/sharpa_wave")).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("Hand destination must not be a symlink")
    manifest = hand_manifest()
    if destination.exists():
        try:
            return check_hand(destination)
        except ValueError:
            if not replace:
                raise FileExistsError(f"Different hand assets exist at {destination}; use --replace to retain a backup and switch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".poise-hand-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "hand"
        staging.mkdir()
        if source is not None:
            source = Path(source).expanduser().resolve()
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if revision != manifest["revision"]:
                raise ValueError(f"Sharpa source revision {revision} differs from pinned {manifest['revision']}")
            for record in manifest["files"]:
                _write_verified((source / record["source"]).read_bytes(), record, staging)
        else:
            archive = Path(temporary) / "source.tar.gz"
            with urlopen(manifest["archive_url"], timeout=120) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            _read_archive(archive, manifest["files"], staging, manifest["revision"])
        resources = files("poise.assets.sharpa_wave")
        for record in manifest["overlays"]:
            name = record["resource"]
            _write_verified(resources.joinpath(name).read_bytes(),
                            {"source": name, "destination": name, "sha256": record["sha256"]}, staging)
        provenance = {"schema_version": 1, "repository": manifest["repository"],
                      "revision": manifest["revision"], "license": manifest["license"],
                      "files": _expected(manifest),
                      "modifications": "POISE wrapper plus self-collision and collision-filter overlay; upstream files unchanged"}
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        check_hand(staging)
        backup = None
        if destination.exists():
            backup = destination.with_name(destination.name + ".backup-" + uuid.uuid4().hex[:12])
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
            raise
    result = check_hand(destination)
    if backup is not None:
        result["backup"] = str(backup)
    return result
