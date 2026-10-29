"""Restore curriculum structure across devices without silently changing shape."""

import pytest
import torch

pytest.importorskip("rsl_rl")
from poise.training.runner import _cpu, _restore, _configuration_hash


def test_nested_curriculum_snapshot_is_independent_and_strict():
    state = {"rotation": torch.tensor([0.2, 0.3]), "steps": {"global": 57}}
    saved = _cpu(state)
    state["rotation"].fill_(1)
    assert saved["rotation"].tolist() == pytest.approx([0.2, 0.3])
    restored = _restore(state, saved)
    torch.testing.assert_close(restored["rotation"], saved["rotation"])
    assert restored["steps"]["global"] == 57
    with pytest.raises(ValueError, match="shape"):
        _restore(torch.zeros(3), saved["rotation"])


def test_single_worker_sync_switch_preserves_checkpoint_identity(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("WORLD_SIZE", "1")
    old = {"sim": {"device": "cuda:0"}, "seed": 42}
    new = {**old, "curriculum_distributed_sync_enabled": True}
    env = lambda config: SimpleNamespace(cfg=SimpleNamespace(to_dict=lambda: config))
    assert _configuration_hash(env(old)) == _configuration_hash(env(new))
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert _configuration_hash(env(old)) != _configuration_hash(env(new))


def test_resume_rejects_wrong_setup_before_applying_state(tmp_path):
    from poise.training.runner import PoiseRunner
    runner = object.__new__(PoiseRunner)
    runner.setup_contract = {'id': 'required-setup'}
    path = tmp_path / 'mismatched.pt'
    torch.save({'infos': {'poise': {'schema_version': 3, 'setup': {'id': 'other'}}}}, path)
    # No environment/optimizer is constructed: the contract must fail first.
    with pytest.raises(ValueError, match='Resume setup contract'):
        runner.load(path)
