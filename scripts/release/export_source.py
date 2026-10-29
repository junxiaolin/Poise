"""Export a reviewable public-source candidate without private Git history."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from poise.setups import list_setups, repository_root, sha256


def scrub(text):
    text = re.sub(r'/home/[^/]+/workspace/Poise\b', '<POISE_ROOT>', text)
    text = re.sub(r'/home/[^/]+/[^\s\x22\x27<>`),;]+', '<LOCAL_PATH>', text)
    text = re.sub(r'/mnt/[^/]+/[^\s\x22\x27<>`),;]+', '<TRAINING_PATH>', text)
    return re.sub(r'\b(?:1[3-9][0-9]|1[01][0-9]|12[0-6]|2[0-4][0-9]|25[0-5])(?:\.[0-9]{1,3}){3}\b', '<PRIVATE_NODE>', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = repository_root()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    setups = list_setups()
    origins = {s['recipe']['origin'] for s in setups}
    sources = {s['reference_checkpoint']['source_record'] for s in setups}
    paths = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines())
    excluded, selected = [], []
    for name in sorted(paths):
        if not (root / name).is_file():
            continue
        internal = name.startswith(('docs/codex_handoffs/', 'docs/training_runs/', 'docs/grasp_runs/',
                                    'docs/evaluation_runs/', 'docs/repository_validation/',
                                    'scripts/evaluation/', 'scripts/train_volcengine_', 'scripts/launch_hex'))
        internal |= name in ('docs/local_run_zh.md', 'tests/test_hexhammer_cloud_launch.py', 'tests/test_aug16_cloud_launch.py')
        if internal and name not in origins | sources:
            excluded.append(name); continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Redact documentation/evidence, never mutate executable source or assets.
        if name.startswith('docs/') and target.suffix in ('.md', '.json', '.yaml', '.yml', '.txt', '.csv'):
            target.write_text(scrub((root / name).read_text()))
        else:
            shutil.copy2(root / name, target)
        selected.append(name)
    for setup in setups:
        setup['recipe']['origin_sha256'] = sha256(output / setup['recipe']['origin'])
        (output / 'configs/setups' / (setup['id'] + '.json')).write_text(json.dumps(setup, indent=2) + '\n')
    # Snapshot digests must describe the exported, redacted evidence.
    for path in (output / 'docs/baselines').glob('*/manifest.json'):
        data = json.loads(path.read_text())
        if 'snapshots' in data:
            data['snapshots'] = {name: sha256(path.parent / name) for name in data['snapshots']}
            data['evidence_redacted_for_release'] = True
            path.write_text(json.dumps(data, indent=2) + '\n')
    record = dict(files={name:sha256(output/name) for name in selected}, excluded_private_records=excluded,
                  git_history_included=False, published=False)
    (output / 'source-export.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), files=len(selected), excluded=len(excluded), published=False)))


if __name__ == '__main__':
    main()
