"""Build and optionally bind a private, deduplicated release; never uploads."""
import argparse
import json
from pathlib import Path
import shutil

from poise.assets.layers import build_release
from poise.paths import workspace_root
from poise.setups import list_setups, repository_root, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bind', action='store_true', help='Pin this index in the four setup manifests')
    args = parser.parse_args()
    setups = list_setups()
    grasp = repository_root() / 'manifests/grasp-assets.json'
    setups.append(dict(id='grasp_hexprism40_v1', assets_manifest='manifests/grasp-assets.json', assets_manifest_sha256=sha256(grasp)))
    index = build_release(setups, args.output, root=workspace_root())
    if args.bind:
        code = repository_root()
        destination = code / 'manifests/releases/v0.1.0-rc1.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / 'release-index.json', destination)
        for setup in setups:
            path = code / setup['assets_manifest']
            manifest = json.loads(path.read_text())
            manifest['distribution'] = dict(status='private_staging', release_index=str(destination.relative_to(code)),
                                            release_index_sha256=sha256(destination))
            path.write_text(json.dumps(manifest, indent=2) + '\n')
            setup['assets_manifest_sha256'] = sha256(path)
            if setup['id'] != 'grasp_hexprism40_v1':
                (code / 'configs/setups' / (setup['id'] + '.json')).write_text(json.dumps(setup, indent=2) + '\n')
    print(json.dumps(dict(packages=len(index['packages']), setups=len(setups), published=False,
                         bytes=sum(p['bytes'] for p in index['packages'].values())), indent=2))


if __name__ == '__main__':
    main()
