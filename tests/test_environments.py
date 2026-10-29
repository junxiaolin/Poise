"""Dependency inspection distinguishes missing packages from version drift."""

import json
from poise import environments


def test_environment_reports_missing_and_mismatched_without_optional_imports(tmp_path, monkeypatch):
    directory = tmp_path / 'environments/train'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'python': '3.11', 'tested_packages': {'missing': '1', 'changed': '2'}}))
    monkeypatch.setattr(environments, 'repository_root', lambda: tmp_path)
    def version(name):
        if name == 'missing':
            raise environments.importlib.metadata.PackageNotFoundError(name)
        return '3'
    monkeypatch.setattr(environments.importlib.metadata, 'version', version)
    report = environments.check_environment('train')
    assert not report['valid']
    assert report['missing'] == ['missing']
    assert report['version_differences']['changed'] == {'installed': '3', 'tested': '2'}
    assert report['native'] == 'not_requested'
