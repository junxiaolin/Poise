import math
from dataclasses import asdict

import torch

from poise.core.tactile import TactileHistory, contact_frame
from poise.core.schema import TACTILE_OBSERVATION_CONTRACT, observation_contract_from_manifest
from poise.core.policy_bundle import export_fused_policy


def test_contact_location_is_finger_local_and_missing_contact_is_zero():
    pose = torch.zeros(1, 5, 7)
    pose[..., :3] = torch.tensor([1., 2., 3.])
    pose[..., 3] = math.sqrt(.5)
    pose[..., 6] = math.sqrt(.5)
    point = pose[..., :3] + torch.tensor([0., .02, 0.])
    force = torch.zeros(1, 2, 5, 3)
    force[:, 0, :, 2] = 2
    force[:, 1, :, 2] = 4
    point[:, 1] = float('nan')
    force[:, 0, 2] = 0
    frame = contact_frame(force, point, pose)
    torch.testing.assert_close(frame[0, :5], torch.tensor([3., 0., 0., 3., 3.]))
    local = frame[:, 5:].reshape(1, 5, 3)
    torch.testing.assert_close(local[0, 0], torch.tensor([.02, 0., 0.]), atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(local[0, 1:3]) == 0
    assert torch.isfinite(frame).all()


def test_history_deduplicates_steps_and_never_reuses_reset_contacts():
    h = TactileHistory(2, 'cpu', 42, hold_probability=0)
    h.reset(torch.arange(2), 0)
    assert not h.update(torch.ones(2, 20), 0).any()
    assert (h.update(torch.ones(2, 20), 1) == 1).all()
    assert (h.update(torch.full((2, 20), 9.), 1) == 1).all()
    h.update(torch.full((2, 20), 2.), 2)
    torch.testing.assert_close(h.frames[0, :, 0], torch.tensor([1., 1., 2.]))
    h.reset(torch.tensor([0]), 2)
    assert not h.update(torch.full((2, 20), 99.), 2)[0].any()
    h.update(torch.full((2, 20), 3.), 3)
    torch.testing.assert_close(h.frames[0, :, 0], torch.tensor([3., 3., 3.]))
    torch.testing.assert_close(h.frames[1, :, 0], torch.tensor([1., 2., 3.]))


def test_held_sample_keeps_force_and_position_and_reset_overrides_hold():
    h = TactileHistory(2, 'cpu', 42, hold_probability=1)
    first = torch.arange(40, dtype=torch.float).reshape(2, 20)
    h.update(first, 0)
    h.update(first + 100, 1)
    torch.testing.assert_close(h.frames[:, -1], first)
    h.reset(torch.tensor([1]), 1)
    h.update(first + 200, 2)
    torch.testing.assert_close(h.frames[0, -1], first[0])
    torch.testing.assert_close(h.frames[1, -1], first[1] + 200)


def test_tactile_contract_and_fused_policy(tmp_path):
    c = TACTILE_OBSERVATION_CONTRACT
    manifest = asdict(c)
    manifest['contract'] = manifest.pop('contract_id')
    assert observation_contract_from_manifest(manifest) == c
    assert c.fields[-1].stop == 304
    assert all(a.stop == b.start for a, b in zip(c.fields, c.fields[1:]))
    state = {
        'actor.0.weight': torch.randn(22, 304) * .01,
        'actor.0.bias': torch.zeros(22),
        'actor_obs_normalizer._mean': torch.randn(1, 304),
        'actor_obs_normalizer._std': torch.ones(1, 304),
    }
    cp = tmp_path / 'model.pt'
    torch.save({'model_state_dict': state}, cp)
    out = tmp_path / 'policy.ts'
    export_fused_policy(cp, out, activation='elu', observation_dim=304)
    x = torch.randn(5, 304)
    expected = torch.nn.functional.linear((x - state['actor_obs_normalizer._mean']) / 1.01,
                                          state['actor.0.weight'], state['actor.0.bias'])
    torch.testing.assert_close(torch.jit.load(str(out))(x), expected, atol=1e-6, rtol=1e-5)
