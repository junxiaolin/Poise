"""Local asset transport must preserve data and never extract arbitrary members."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from poise import setups
from poise.assets import setup_data


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    code, source = tmp_path / 'code', tmp_path / 'source'
    code.mkdir()
    (code / 'NOTICE').write_text('Private test bundle')
    (code / 'licenses').mkdir()
    records = []
    for name, data in [('a.bin', b'first'), ('b.bin', b'second'), ('reference.pt', b'checkpoint')]:
        path = source / 'data' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append({'path': 'data/' + name, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(),
                        'role': 'reference_checkpoint' if name == 'reference.pt' else 'object'})
    manifest = code / 'assets.json'
    manifest.write_text(json.dumps({'files': records, 'distribution': {'status': 'private_staging'}}))
    setup = {'id': 'fixture', 'assets_manifest': 'assets.json', 'assets_manifest_sha256': setups.sha256(manifest)}
    monkeypatch.setattr(setups, 'repository_root', lambda: code)
    monkeypatch.setattr(setup_data, 'repository_root', lambda: code)
    return setup, source, tmp_path / 'target', records


def test_package_import_idempotence_and_conflict(bundle, tmp_path):
    setup, source, target, records = bundle
    archive = tmp_path / 'bundle.tar.gz'
    report = setup_data.package_assets(setup, archive, root=source, with_reference=True)
    assert not report['published']
    first = setup_data.install_assets(setup, source=archive, root=target, with_reference=True)
    assert len(first['installed']) == 3
    assert setup_data.install_assets(setup, source=archive, root=target, with_reference=True)['installed'] == []
    with tarfile.open(archive) as tar:
        assert {'bundle.json', 'NOTICE'} <= set(tar.getnames())
    with pytest.raises(FileExistsError):
        setup_data.package_assets(setup, archive, root=source)
    (target / records[0]['path']).write_bytes(b'user data')
    with pytest.raises(ValueError, match='mismatch'):
        setup_data.install_assets(setup, source=source, root=target)
    assert (target / records[0]['path']).read_bytes() == b'user data'


def test_optional_checkpoint_and_private_no_network(bundle, monkeypatch):
    setup, source, target, records = bundle
    def network_forbidden(*args, **kwargs):
        pytest.fail('private assets must not initiate a download')
    monkeypatch.setattr(setup_data.urllib.request, 'urlopen', network_forbidden)
    with pytest.raises(ValueError, match='unpublished'):
        setup_data.install_assets(setup, root=target)
    setup_data.install_assets(setup, source=source, root=target)
    assert not (target / records[-1]['path']).exists()


@pytest.mark.parametrize('corruption', ['content', 'duplicate', 'symlink', 'missing'])
def test_bad_tar_publishes_no_partial_data(bundle, tmp_path, corruption):
    setup, source, target, records = bundle
    archive = tmp_path / 'bad.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(source / records[0]['path'], arcname=records[0]['path'])
        record = records[1]
        member = tarfile.TarInfo(record['path'])
        member.size = record['bytes']
        if corruption == 'content':
            tar.addfile(member, io.BytesIO(b'x' * member.size))
        elif corruption == 'duplicate':
            for _ in range(2):
                tar.add(source / record['path'], arcname=record['path'])
        elif corruption == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '../../outside'
            member.size = 0
            tar.addfile(member)
    with pytest.raises(ValueError):
        setup_data.install_assets(setup, source=archive, root=target)
    assert not list(target.rglob('*.bin'))


def test_unlisted_tar_members_not_extracted_and_symlink_escape_fails(bundle, tmp_path):
    setup, source, target, records = bundle
    archive = tmp_path / 'extra.tar'
    with tarfile.open(archive, 'w') as tar:
        for record in records:
            tar.add(source / record['path'], arcname=record['path'])
        evil = tarfile.TarInfo('../outside')
        evil.size = 3
        tar.addfile(evil, io.BytesIO(b'bad'))
    setup_data.install_assets(setup, source=archive, root=target)
    assert not (tmp_path / 'outside').exists()
    escaped = tmp_path / 'escaped'
    escaped.mkdir()
    (escaped / 'data').symlink_to(source / 'data', target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        setup_data.install_assets(setup, source=source, root=escaped)
