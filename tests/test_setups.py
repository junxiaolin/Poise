"""Reproduction binds inputs and rejects mixing experiments before simulation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poise import setups
from poise.launch import command

HEX = 'hexprism40_palmup_v1'
NINE = 'manualconvex9_palmup_v1'
OMNI = 'hexprism40_omniwrist_fullpose_v1'
HAMMER = 'hexhammer_palmup_fullpose_192k_v1'


def test_shipped_setups_preserve_baselines_and_objects():
    for setup in setups.list_setups():
        assert all(r['role'] != 'reference_checkpoint' for r in setups.asset_records(setup))
        recipe = setups.recipe_path(setup['recipe']['name'])
        origin = setups.repository_root() / setup['recipe']['origin']
        assert setups.sha256(origin) == setup['recipe']['origin_sha256']
        original, current = json.loads(origin.read_text()), json.loads(recipe.read_text())
        rollout = json.loads((setups.repository_root() / setup['rollout']['config']).read_text())
        assert not (rollout['overrides'].keys() - current.keys())
        original.pop('log_dir', None)
        current.pop('log_dir', None)
        # Only portable hand path differs in the Convex9 historical recipe.
        original['robot_cfg']['spawn']['usd_path'] = current['robot_cfg']['spawn']['usd_path']
        assert original == current
        assert len(setup['objects']) == (9 if setup['id'] == NINE else 1)
        assert all(o['cache_rows'] == (11630 if setup['id'] == HAMMER else 6842)
                   for o in setup['objects'])


@pytest.mark.parametrize('change', ['order', 'recipe_hash', 'manifest_hash'])
def test_setup_drift_fails(tmp_path, monkeypatch, change):
    setup = copy.deepcopy(setups.load_setup(NINE))
    real_root = setups.repository_root()
    (tmp_path / 'configs/setups').mkdir(parents=True)
    (tmp_path / 'configs/rollout').symlink_to(real_root / 'configs/rollout', target_is_directory=True)
    (tmp_path / 'manifests').symlink_to(real_root / 'manifests', target_is_directory=True)
    if change == 'order':
        setup['objects'].reverse()
    elif change == 'recipe_hash':
        setup['recipe']['sha256'] = '0' * 64
    else:
        setup['assets_manifest_sha256'] = '0' * 64
    (tmp_path / 'configs/setups' / (NINE + '.json')).write_text(json.dumps(setup))
    monkeypatch.setattr(setups, 'repository_root', lambda: tmp_path)
    with pytest.raises(ValueError, match='order|SHA256'):
        setups.load_setup(NINE)


def checkpoint(setup):
    return {'iter': 1, 'infos': {'poise': {'setup': setups.setup_contract(setup),
            'rank_states': [{'object_ids': [o['id'] for o in setup['objects']]}]}},
            'model_state_dict': {'actor.0.weight': torch.ones(4, 244), 'actor.6.weight': torch.ones(22, 4)}}


def test_checkpoint_contract_rejects_wrong_setup_order_interface_and_nan(tmp_path):
    setup = setups.load_setup(NINE)
    path = tmp_path / 'model.pt'
    saved = checkpoint(setup)
    torch.save(saved, path)
    assert not setups.check_checkpoint(setup, path)['reference']
    with pytest.raises(ValueError, match='contract'):
        setups.check_checkpoint(setups.load_setup(HEX), path)
    for change, match in [('order', 'order'), ('interface', '244D'), ('nan', 'non-finite')]:
        broken = copy.deepcopy(saved)
        if change == 'order':
            broken['infos']['poise']['rank_states'][0]['object_ids'].reverse()
        elif change == 'interface':
            broken['model_state_dict']['actor.0.weight'] = torch.ones(4, 243)
        else:
            broken['model_state_dict']['actor.0.weight'][0, 0] = float('nan')
        torch.save(broken, path)
        with pytest.raises(ValueError, match=match):
            setups.check_checkpoint(setup, path)


def args(**values):
    return SimpleNamespace(**({'command': 'train', 'setup': HEX, 'profile': 'smoke',
        'num_envs': None, 'iterations': None, 'seed': None, 'resume': None,
        'checkpoint': None, 'headless': True, 'steps': None, 'verify_interface': False, 'view_env': 0,
        'device': 'cuda:0', 'output': Path('outputs/test'), 'kit_args': None} | values))


def test_launch_profiles_and_rollout_use_pinned_defaults():
    smoke = command(args())
    assert smoke[smoke.index('--num-envs') + 1] == '8'
    assert smoke[smoke.index('--iterations') + 1] == '2'
    formal = command(args(profile='reference', setup=NINE))
    assert '--nproc_per_node=2' in formal
    assert formal[formal.index('--physical-gpus') + 1] == '1'
    rollout = command(args(command='rollout', setup=NINE))
    assert rollout[rollout.index('--num-envs') + 1] == '9'
    assert rollout[rollout.index('--steps') + 1] == '400'
    assert rollout[rollout.index('--checkpoint') + 1].endswith('model_23300.pt')
    gui = command(args(command='rollout', setup=NINE, headless=False, view_env=3))
    assert gui[gui.index('--metrics-env') + 1] == '3'
    assert '--camera-eye' in gui
    with pytest.raises(ValueError, match='custom'):
        command(args(num_envs=8))
    with pytest.raises(ValueError, match='environment'):
        command(args(setup=NINE, profile='custom', num_envs=8))
    with pytest.raises(ValueError, match='positive'):
        command(args(profile='custom', iterations=0))


def test_omniwrist_baseline_preserves_actual_training_and_scale():
    setup = setups.load_setup(OMNI)
    recipe = json.loads(setups.recipe_path(setup['recipe']['name']).read_text())
    saved = json.loads((setups.repository_root() / 'docs/baselines' / OMNI / 'env.source.json').read_text())
    saved['log_dir'] = None
    saved['robot_cfg']['spawn']['usd_path'] = recipe['robot_cfg']['spawn']['usd_path']
    assert recipe == saved
    assert recipe['dexndm_omni_wrist_ornt']
    assert recipe['pose6d_success_criterion'] == 'full_pose'
    assert recipe['observation_space'] == 244 and recipe['state_space'] == 492
    assert setup['training']['global_frontier_min'] == 160000
    formal = command(args(profile='reference', setup=OMNI))
    assert '--nproc_per_node=8' in formal
    assert formal[formal.index('--physical-gpus') + 1] == '4'
    assert formal[formal.index('--num-envs') + 1] == '24000'
    rollout = command(args(command='rollout', setup=OMNI))
    assert rollout[rollout.index('--checkpoint') + 1].endswith('model_35800.pt')


def test_hexhammer_baseline_preserves_saved_training_and_full_curriculum_rollout():
    setup = setups.load_setup(HAMMER)
    root = setups.repository_root()
    recipe = json.loads(setups.recipe_path(setup['recipe']['name']).read_text())
    saved = json.loads((root / 'docs/baselines' / HAMMER / 'env.source.json').read_text())
    saved['log_dir'] = None
    saved['robot_cfg']['spawn']['usd_path'] = recipe['robot_cfg']['spawn']['usd_path']
    assert recipe == saved
    assert not recipe['dexndm_omni_wrist_ornt']
    assert recipe['pose6d_success_criterion'] == 'full_pose'
    assert (recipe['observation_space'], recipe['state_space']) == (244, 492)
    formal = command(args(profile='reference', setup=HAMMER))
    assert '--nproc_per_node=8' in formal
    assert formal[formal.index('--physical-gpus') + 1] == '4'
    assert formal[formal.index('--num-envs') + 1] == '24000'
    assert setup['training']['global_frontier_min'] == 20000
    rollout = json.loads((root / setup['rollout']['config']).read_text())
    overrides = rollout['overrides']
    assert overrides['twist_target_curriculum_initial_max_rad'] == pytest.approx(3.1415926)
    assert overrides['pose6d_translation_curriculum_initial_max_m'] == pytest.approx(0.1)
    assert not any('noise' in key or 'randomization' in key for key in overrides)
    cmd = command(args(command='rollout', setup=HAMMER))
    assert cmd[cmd.index('--checkpoint') + 1].endswith('model_89800.pt')
