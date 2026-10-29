"""Verified setup transport: shared packages and complete offline bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from poise.paths import workspace_root
from poise.setups import asset_records, data_path, repository_root, require_setup, sha256


def verify_file(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Asset size/SHA256 mismatch: {record['path']}")


def install_assets(setup, *, source=None, root=None, with_reference=False):
    """Reuse verified files; stage every missing dependency before installation."""
    from poise.assets.layers import extract, stage_layers
    root = Path(root or workspace_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = asset_records(setup, with_reference=with_reference)
    missing = []
    for record in records:
        target = data_path(root, record["path"])
        if target.exists():
            verify_file(target, record)
        else:
            missing.append(record)
    if not missing:
        return {**require_setup(setup, root=root, with_reference=with_reference),
                "installed": [], "reused": len(records), "packages": []}
    manifest = json.loads((repository_root() / setup["assets_manifest"]).read_text())
    distribution = manifest["distribution"]
    source = Path(source).expanduser().resolve() if source is not None else None
    layered = 'release_index' in distribution and (source is None or (source / 'release-index.json').is_file())
    packages = []
    with tempfile.TemporaryDirectory(prefix=".poise-assets-", dir=root) as temporary:
        stage = Path(temporary)
        staged = stage / "files"
        if layered:
            packages = stage_layers(setup, distribution, missing, staged, root, source)
        else:
            if source is None:
                if distribution["status"] != "published" or not distribution.get("archive_url") or not distribution.get("archive_sha256"):
                    raise ValueError("Assets are private/unpublished. Supply --source with a workspace, release directory or bundle.")
                if not distribution["archive_url"].startswith("https://"):
                    raise ValueError("Asset downloads require HTTPS")
                source = stage / "download.tar.gz"
                with urllib.request.urlopen(distribution["archive_url"], timeout=60) as response, source.open("xb") as stream:
                    if not response.geturl().startswith("https://"):
                        raise ValueError("Asset redirect must remain HTTPS")
                    shutil.copyfileobj(response, stream)
                if sha256(source) != distribution["archive_sha256"]:
                    raise ValueError("Asset archive SHA256 mismatch")
            if source.is_dir():
                for record in missing:
                    src = data_path(source, record["path"])
                    verify_file(src, record)
                    target = data_path(staged, record["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, target)
            else:
                extract(source, missing, staged)
        for record in missing:
            verify_file(data_path(staged, record["path"]), record)
        installed = []
        for record in missing:
            target = data_path(root, record["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(data_path(staged, record["path"]), target)
                installed.append(record["path"])
            except FileExistsError:
                verify_file(target, record)
    return {**require_setup(setup, root=root, with_reference=with_reference),
            "installed": installed, "reused": len(records) - len(missing), "packages": packages}


def package_assets(setup, output, *, root=None, with_reference=False):
    """Create a local archive only. No upload, publication or URL mutation."""
    root = Path(root or workspace_root()).resolve()
    require_setup(setup, root=root, with_reference=with_reference)
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = asset_records(setup, with_reference=with_reference)
    with tempfile.TemporaryDirectory(prefix=".poise-bundle-", dir=output.parent) as temporary:
        archive_path = Path(temporary) / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz", dereference=True) as archive:
            for r in records:
                snapshot = Path(temporary) / "asset"
                shutil.copyfile(data_path(root, r["path"]), snapshot)
                verify_file(snapshot, r)
                archive.add(snapshot, arcname=r["path"], recursive=False)
            metadata = Path(temporary) / "bundle.json"
            metadata.write_text(json.dumps({"setup": setup, "files": records, "published": False}, indent=2) + "\n")
            archive.add(metadata, arcname="bundle.json", recursive=False)
            code_root = repository_root()
            for notice in [code_root / "NOTICE", code_root / "LICENSE", code_root / "manifests/rights.json", *sorted((code_root / "licenses").glob("*.txt"))]:
                if notice.is_file():
                    archive.add(notice, arcname=str(notice.relative_to(code_root)), recursive=False)
        os.link(archive_path, output)
    return {"setup": setup["id"], "archive": str(output), "sha256": sha256(output),
            "files": len(records), "bytes": output.stat().st_size, "published": False}
