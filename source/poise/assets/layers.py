"""Content-pinned, shared release assets. Packaging never uploads or publishes."""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from poise.setups import asset_records, data_path, relative_path, repository_root, sha256


def package_id(record):
    path = record['path']
    if record['role'] == 'reference_checkpoint':
        return 'checkpoint-' + Path(path).parent.name
    if path.startswith('data/robots/sharpa_limits/'):
        return 'assets-generation-v1'
    if path.startswith(('data/robots/', 'data/hand_fields/')):
        return 'assets-common-v1'
    if path.startswith('data/geometry_latents/'):
        return 'geometry-' + Path(path).stem
    if '/sharpa_hex_prism_hammer/' in path:
        return 'assets-hexhammer-v1'
    if 'regular_hex_prism_af40mm_h40mm_chamfered' in path:
        return 'assets-hex40-v1'
    return 'assets-convex9-extra-v1'


def build_release(setups, output, *, root):
    """Write each unique asset exactly once, then an index; refuse old outputs."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, groups, dependencies = {}, {}, {}
    for setup in setups:
        deps = set()
        for record in asset_records(setup, with_reference=True):
            old = records.get(record['path'])
            if old and (old['sha256'], old['bytes']) != (record['sha256'], record['bytes']):
                raise ValueError('Conflicting setup asset: ' + record['path'])
            if not old:
                records[record['path']] = record
                groups.setdefault(package_id(record), []).append(record)
            deps.add(package_id(record))
        dependencies[setup['id']] = sorted(deps)
    index = dict(schema_version=1, status='private_staging', packages={}, setups=dependencies)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.release-') as temp:
        stage = Path(temp)
        for name, entries in sorted(groups.items()):
            entries.sort(key=lambda r: r['path'])
            raw = name.startswith(('checkpoint-', 'geometry-')) and len(entries) == 1
            filename = name + ('.pt' if raw else '.tar.gz')
            target = stage / filename
            if raw:
                src = data_path(root, entries[0]['path'])
                shutil.copyfile(src, target)
                _verify(target, entries[0])
            else:
                with target.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as gz:
                    with tarfile.open(fileobj=gz, mode='w') as tar:
                        for record in entries:
                            src = data_path(root, record['path'])
                            data = src.read_bytes()
                            import hashlib
                            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                                raise ValueError('Asset mismatch: ' + record['path'])
                            _add(tar, record['path'], data)
                        code = repository_root()
                        for p in [code / 'NOTICE', code / 'LICENSE', *sorted((code / 'licenses').glob('*.txt'))]:
                            if p.is_file():
                                _add(tar, str(p.relative_to(code)), p.read_bytes())
            index['packages'][name] = dict(filename=filename, format='raw' if raw else 'tar.gz',
                bytes=target.stat().st_size, sha256=sha256(target), url=None, files=entries)
        for notice in [repository_root() / 'NOTICE', repository_root() / 'LICENSE', repository_root() / 'manifests/rights.json', *sorted((repository_root() / 'licenses').glob('*.txt'))]:
            if notice.is_file():
                dest = stage / notice.relative_to(repository_root())
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(notice, dest)
        (stage / 'release-index.json').write_text(json.dumps(index, indent=2) + '\n')
        # The output must be absent, including empty pre-existing directories.
        output.mkdir()
        for p in stage.iterdir():
            os.replace(p, output / p.name)
    return index


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    tar.addfile(info, io.BytesIO(data))


def _verify(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
        raise ValueError('Asset size/SHA256 mismatch: ' + str(path))


def read_index(setup, distribution, source=None):
    code = repository_root()
    pinned = code / relative_path(distribution['release_index'])
    expected = distribution['release_index_sha256']
    if sha256(pinned) != expected:
        raise ValueError('Release index SHA256 mismatch')
    if source is not None and sha256(Path(source) / 'release-index.json') != expected:
        raise ValueError('Local release index differs from the pinned index')
    index = json.loads(pinned.read_text())
    if index['schema_version'] != 1:
        raise ValueError('Unsupported release index')
    ids = index['setups'][setup['id']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate package dependency')
    available = {}
    for name in ids:
        package = index['packages'][name]
        if Path(package['filename']).name != package['filename'] or package['format'] not in ('raw', 'tar.gz'):
            raise ValueError('Invalid package filename/format')
        if package['format'] == 'raw' and len(package['files']) != 1:
            raise ValueError('Raw package must contain one file')
        for record in package['files']:
            relative_path(record['path'])
            if record['path'] in available:
                raise ValueError('Asset appears in multiple dependencies: ' + record['path'])
            available[record['path']] = (record, name)
    for record in asset_records(setup, with_reference=True):
        candidate = available.get(record['path'])
        if candidate is None or any(candidate[0][k] != record[k] for k in ('sha256', 'bytes')):
            raise ValueError('Release index does not match setup asset: ' + record['path'])
    return index, available


def download(package, cache):
    url = package.get('url')
    if not url or not url.startswith('https://'):
        raise ValueError('Assets are private/unpublished; use --source with the release directory')
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (package['sha256'] + '-' + package['filename'])
    if destination.exists():
        _verify(destination, package)
        return destination
    fd, name = tempfile.mkstemp(dir=cache, prefix='.download-')
    pending = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream, urllib.request.urlopen(url, timeout=60) as response:
            if not response.geturl().startswith('https://'):
                raise ValueError('Asset redirect must remain HTTPS')
            remaining = package['bytes']
            while remaining:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Truncated asset download')
                stream.write(block)
                remaining -= len(block)
            if response.read(1):
                raise ValueError('Asset download exceeds pinned size')
        _verify(pending, package)
        try:
            os.link(pending, destination)
        except FileExistsError:
            _verify(destination, package)
        return destination
    finally:
        pending.unlink(missing_ok=True)


def extract(archive, records, staged):
    """Extract only pinned regular members; tar paths never control the target."""
    with tarfile.open(archive, 'r:*') as tar:
        members = {}
        for member in tar.getmembers():
            members.setdefault(member.name, []).append(member)
        for record in records:
            candidates = members.get(record['path'], [])
            if len(candidates) != 1 or not candidates[0].isfile():
                raise ValueError('Expected one regular archive member: ' + record['path'])
            member = candidates[0]
            if member.size != record['bytes']:
                raise ValueError('Archive member size mismatch: ' + record['path'])
            target = data_path(staged, record['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)


def stage_layers(setup, distribution, records, staged, root, source=None):
    index, available = read_index(setup, distribution, source)
    needed = {}
    for record in records:
        needed.setdefault(available[record['path']][1], []).append(record)
    if source is None and index['status'] != 'published':
        raise ValueError('Assets are private/unpublished; supply --source with the release directory')
    for name, selected in needed.items():
        package = index['packages'][name]
        path = (Path(source) / package['filename']) if source else download(package, root / '.poise/downloads')
        _verify(path, package)
        if package['format'] == 'raw':
            target = data_path(staged, selected[0]['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        else:
            extract(path, selected, staged)
    return sorted(needed)
