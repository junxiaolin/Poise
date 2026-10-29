"""Fetch pinned optional FoundationPose sources. No install, weights or device access."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "dependencies/foundationpose/manifest.json").read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".poise-foundationpose-", dir=destination.parent) as temporary:
        root = Path(temporary) / "checkout"
        for relative, spec in [(".", manifest), *[(f"poise-deps/{name}", value) for name, value in manifest["extensions"].items()]]:
            checkout = root / relative
            checkout.mkdir(parents=True, exist_ok=True)
            def git(*values):
                return subprocess.check_output(["git", "-C", str(checkout), *values], text=True).strip()
            git("init", "--quiet")
            git("remote", "add", "origin", spec["repository"])
            git("fetch", "--depth=1", "origin", spec["revision"])
            git("checkout", "--quiet", "--detach", "FETCH_HEAD")
            if git("rev-parse", "HEAD") != spec["revision"]:
                raise ValueError("Unexpected fetched revision")
            git("submodule", "update", "--init", "--recursive", "--depth=1")
        (root / "poise-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        root.rename(destination)
    print(json.dumps({"destination": str(destination), "revision": manifest["revision"],
                      "next": "Follow environments/foundationpose-poise/README.md; weights and native builds are separate"}, indent=2))


if __name__ == "__main__":
    main()
