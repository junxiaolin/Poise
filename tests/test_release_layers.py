"""Shared downloads preserve setup identities, reuse, and all-or-nothing staging."""
import copy
import hashlib
import io
import json
from pathlib import Path

import pytest

from poise import setups
from poise.assets import layers, setup_data


@pytest.fixture
def release(tmp_path, monkeypatch):
    code, source = tmp_path / 'code', tmp_path / 'source'
    code.mkdir()
    (code / 'NOTICE').write_text('Fixture')
    for module in (setups, layers, setup_data):
        monkeypatch.setattr(module, 'repository_root', lambda: code)
    specs = []
    for name in ('one', 'two'):
        files = []
        for path, content, role in [
            ('data/robots/shared.bin', b'shared', 'hand'),
            (f'data/checkpoints/{name}/model.pt', name.encode(), 'reference_checkpoint')]:
            p = source / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            files.append(dict(path=path, bytes=len(content), sha256=setups.sha256(p), role=('shared' if name == 'two' and role == 'hand' else role)))
        manifest = code / (name + '.json')
        manifest.write_text(json.dumps(dict(files=files, distribution={})))
        specs.append(dict(id=name, assets_manifest=manifest.name, assets_manifest_sha256=setups.sha256(manifest)))
    output = tmp_path / 'release'
    index = layers.build_release(specs, output, root=source)

    def bind(index):
        text = json.dumps(index)
        (code / 'index.json').write_text(text)
        (output / 'release-index.json').write_text(text)
        for s in specs:
            p = code / s['assets_manifest']
            d = json.loads(p.read_text())
            d['distribution'] = dict(release_index='index.json', release_index_sha256=setups.sha256(code / 'index.json'))
            p.write_text(json.dumps(d))
            s['assets_manifest_sha256'] = setups.sha256(p)
    bind(index)
    return specs, output, tmp_path / 'target', index, bind


def test_dedup_reuses_assets_and_optional_model(release):
    specs, source, target, index, _ = release
    assert len(index['packages']) == 3
    one = setup_data.install_assets(specs[0], source=source, root=target)
    assert one['packages'] == ['assets-common-v1']
    common = target / 'data/robots/shared.bin'
    inode = common.stat().st_ino
    # A second setup needs only its own checkpoint; even the common package can be absent.
    (source / index['packages']['assets-common-v1']['filename']).unlink()
    two = setup_data.install_assets(specs[1], source=source, root=target, with_reference=True)
    assert two['packages'] == ['checkpoint-two'] and common.stat().st_ino == inode
    assert setup_data.install_assets(specs[1], root=target, with_reference=True)['installed'] == []


def test_bad_later_package_does_not_install_earlier_package(release):
    specs, source, target, index, _ = release
    model = source / index['packages']['checkpoint-one']['filename']
    model.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        setup_data.install_assets(specs[0], source=source, root=target, with_reference=True)
    assert not (target / 'data').exists()


def test_https_download_cache_and_unpublished_guard(release, monkeypatch):
    specs, source, target, index, bind = release
    requests = []
    class Response(io.BytesIO):
        def geturl(self):
            return 'https://example.org/download'
    def get(url, **kwargs):
        requests.append(url)
        return Response((source / url.rsplit('/', 1)[1]).read_bytes())
    monkeypatch.setattr(layers.urllib.request, 'urlopen', get)
    with pytest.raises(ValueError, match='unpublished'):
        setup_data.install_assets(specs[0], root=target)
    assert requests == []
    index['status'] = 'published'
    for p in index['packages'].values():
        p['url'] = 'https://example.org/' + p['filename']
    bind(index)
    setup_data.install_assets(specs[0], root=target, with_reference=True)
    assert len(requests) == 2
    (target / 'data/checkpoints/one/model.pt').unlink()
    setup_data.install_assets(specs[0], root=target, with_reference=True)
    assert len(requests) == 2  # restored from the hash-verified local download cache


def test_conflicting_index_and_insecure_url_rejected(release, monkeypatch):
    specs, source, target, index, bind = release
    changed = copy.deepcopy(index)
    changed['packages']['assets-common-v1']['files'][0]['sha256'] = '0' * 64
    bind(changed)
    with pytest.raises(ValueError, match='does not match'):
        setup_data.install_assets(specs[0], source=source, root=target)
    index['status'] = 'published'
    index['packages']['assets-common-v1']['url'] = 'http://example.org/unsafe'
    bind(index)
    with pytest.raises(ValueError, match='unpublished'):
        setup_data.install_assets(specs[0], root=target)
