"""Read-only source/history scan; report locations and fingerprints, never secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


RULES = {
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
    'aws_access_key': rb'\bAKIA[0-9A-Z]{16}\b',
    'assigned_secret': rb'(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[=:]\s*[\x22\x27][A-Za-z0-9_+/=-]{24,}[\x22\x27]',
    'internal_path': rb'/home/[^/\s]+/(?:workspace|miniconda3)|/mnt/[^/\s]+/',
    'private_node': rb'\b(?:1[01][0-9]|12[0-6]|1[3-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.[0-9]{1,3}){3}\b',
}


def scan(data, location):
    results = []
    for rule, pattern in RULES.items():
        for match in re.finditer(pattern, data):
            # This is regex syntax in the scanner/exporter, not an actual path.
            # Other values in these files are still scanned normally.
            if (rule == 'internal_path' and match.group() == b'/mnt/[^/'
                    and location.endswith(('scripts/release/audit.py', 'scripts/release/export_source.py'))):
                continue
            results.append(dict(location=location, rule=rule,
                line=data[:match.start()].count(b'\n') + 1,
                fingerprint=hashlib.sha256(match.group()).hexdigest()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    findings, count = [], 0
    if args.history:
        lines = subprocess.check_output(['git', '-C', str(root), 'rev-list', '--objects', '--all'], text=True).splitlines()
        objects = {line.split(' ', 1)[0]: line.split(' ', 1)[1] if ' ' in line else '' for line in lines}
        proc = subprocess.Popen(['git', '-C', str(root), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        for oid, path in objects.items():
            proc.stdin.write((oid + '\n').encode()); proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2]); data = proc.stdout.read(size); proc.stdout.read(1)
            if header[1] == b'blob':
                count += 1
                findings += scan(data, oid + ':' + path)
        proc.stdin.close(); proc.wait()
    else:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(part in ('.git', '__pycache__', '.pytest_cache', 'data', 'outputs', 'third_party', 'build', 'dist') for part in p.relative_to(root).parts):
                continue
            count += 1
            findings += scan(p.read_bytes(), str(p.relative_to(root)))
    secrets = [f for f in findings if f['rule'] not in ('internal_path', 'private_node')]
    report = dict(scope='all reachable Git blobs' if args.history else 'candidate tree', files_or_blobs=count,
                  secret_candidates=len(secrets), internal_findings=len(findings)-len(secrets), findings=findings,
                  limitations='Pattern scan; not a proof that no secrets exist. Unreachable Git objects and external credential stores are not scanned.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'findings'}))


if __name__ == '__main__':
    main()
