from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


from poise.grasp import finalize as MODULE


def test_quaternion_geodesic_is_sign_invariant() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(MODULE.quaternion_geodesic_angle(q, -q), torch.zeros(1))


def test_greedy_maximin_prefers_distant_pose() -> None:
    candidates = torch.tensor(
        [
            [0.001, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.010, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.020, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    selected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert MODULE.greedy_maximin_indices(candidates, selected, 1, 0.004, 0.08) == [2]


def test_select_rows_preserves_nonrow_fields() -> None:
    cache = {
        "q": torch.arange(12).reshape(3, 4),
        "q_cmd": torch.arange(12).reshape(3, 4),
        "object_pose_palm": torch.arange(21).reshape(3, 7),
        "source_seed_index": torch.arange(3),
        "object_id": "cube",
        "metadata": {"a": 1},
    }
    result = MODULE.select_rows(cache, torch.tensor([2, 0]))
    assert result["q"][:, 0].tolist() == [8, 0]
    assert result["object_id"] == "cube"
    assert result["metadata"] == {"a": 1}


def test_merge_concatenates_row_tensors(tmp_path: Path) -> None:
    paths = []
    for i, rows in enumerate((2, 3)):
        cache = {
            "q": torch.full((rows, 4), float(i)),
            "q_cmd": torch.full((rows, 4), float(i)),
            "object_pose_palm": torch.full((rows, 7), float(i)),
            "source_seed_index": torch.arange(rows),
            "object_id": "test",
            "metadata": {"source": i},
        }
        path = tmp_path / f"input_{i}.pt"
        torch.save(cache, path)
        paths.append(path)
    output = tmp_path / "merged.pt"
    args = type("Args", (), {"inputs": paths, "output": output})()
    assert MODULE.command_merge(args) == 0
    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert merged["q"].shape == (5, 4)
    assert merged["q"][:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert merged["object_id"] == "test"


def test_filter_contract_from_cache_preserves_joint_limit_provenance() -> None:
    cache = {
        "q": torch.zeros((1, 4)),
        "q_cmd": torch.zeros((1, 4)),
        "object_pose_palm": torch.zeros((1, 7)),
        "metadata": {
            "hold_filter": {
                "hand_root_orientation": "palm_up",
                "hand_usd_path": "/tmp/hand.usda",
                "hand_usd_sha256": "abc",
                "joint_limit_reference_usd_path": "/tmp/limits.usda",
                "joint_limit_reference_usd_sha256": "def",
                "joint_limit_profile": "usda09",
                "joint_projection": {
                    "rule": "signed_endpoints_times_0.9",
                    "limits_rad": [[-1.0, 1.0]],
                },
            }
        },
    }
    contract = MODULE.filter_contract_from_cache(cache)
    assert contract is not None
    assert contract["joint_limit_profile"] == "usda09"
    assert contract["hand_usd_sha256"] == "abc"
    assert contract["joint_limit_reference_usd_sha256"] == "def"
    assert contract["joint_projection"]["limits_rad"] == [[-1.0, 1.0]]


def test_exact_total_allocation_respects_small_parent_capacity():
    capacities = torch.tensor([10, 20, 20])
    quotas = MODULE.allocate_exact_per_seed_quotas(
        capacities, min_per_seed=10, max_per_seed=20, target_total_rows=42
    )
    assert quotas.tolist() == [10, 16, 16]
    import pytest

    with pytest.raises(RuntimeError, match="outside feasible"):
        MODULE.allocate_exact_per_seed_quotas(
            capacities, min_per_seed=10, max_per_seed=20, target_total_rows=51
        )


def test_formal_6842_quota():
    quotas = MODULE.allocate_exact_per_seed_quotas(
        torch.full((486,), 20), min_per_seed=10, max_per_seed=20, target_total_rows=6842
    )
    assert quotas.sum().item() == 6842
    assert torch.bincount(quotas).tolist()[14:] == [448, 38]


def test_filter_contract_rejects_different_wrist_trials():
    import pytest

    a = {
        "q": torch.zeros(2, 22),
        "metadata": {"hold_filter": {"omni_wrist": {"trials": 8}}},
    }
    b = {
        "q": torch.zeros(2, 22),
        "metadata": {"hold_filter": {"omni_wrist": {"trials": 4}}},
    }
    with pytest.raises(ValueError, match="omni_wrist"):
        MODULE.assert_matching_filter_contracts(a, b)


def test_parent_selection_is_unique_and_quaternion_sign_invariant():
    from poise.grasp.select_parents import (
        normalized_features,
        deterministic_farthest_point_indices,
    )

    torch.manual_seed(22)
    pose = torch.randn(24, 7)
    pose[:, 3:] /= pose[:, 3:].norm(dim=-1, keepdim=True)
    c = {
        "q": torch.randn(24, 22),
        "q_cmd": torch.randn(24, 22),
        "object_pose_palm": pose,
    }
    f = normalized_features(c)
    c["object_pose_palm"][:, 3:] *= -1
    torch.testing.assert_close(f, normalized_features(c))
    ids = deterministic_farthest_point_indices(f, torch.arange(24), 12)
    assert len(ids.unique()) == 12 and ids[0] == 23


def test_filter_contract_ignores_measured_trial_results():
    a = {
        "q": torch.zeros(2, 22),
        "metadata": {
            "hold_filter": {
                "omni_wrist": {"trials": 8, "trial_summaries": [{"stable_count": 2}]}
            }
        },
    }
    b = {
        "q": torch.zeros(2, 22),
        "metadata": {
            "hold_filter": {
                "omni_wrist": {"trials": 8, "trial_summaries": [{"stable_count": 1}]}
            }
        },
    }
    assert MODULE.assert_matching_filter_contracts(a, b)["omni_wrist"] == {"trials": 8}


def test_finalize_fills_deficient_parent_and_preserves_lineage(tmp_path):
    def cache(ids):
        n = len(ids)
        pose = torch.zeros(n, 7)
        pose[:, 0] = torch.arange(n) * 0.001
        pose[:, 3] = 1
        return {
            "q": torch.zeros(n, 22),
            "q_cmd": torch.zeros(n, 22),
            "object_pose_palm": pose,
            "source_seed_index": torch.tensor(ids),
            "object_id": "test",
            "object_dims_m": (0.04, 0.04, 0.04),
            "metadata": {
                "hold_filter": {"omni_wrist": {"trials": 8}, "hold_seconds": 2}
            },
        }

    seeds = tmp_path / "parents.pt"
    initial = tmp_path / "initial.pt"
    supplement = tmp_path / "supplement.pt"
    torch.save(cache([0, 1]), seeds)
    torch.save(cache([0] * 8 + [1] * 20), initial)
    torch.save(cache([0] * 4), supplement)
    args = SimpleNamespace(
        seed_cache=seeds,
        initial_filtered=initial,
        supplement_filtered=supplement,
        min_per_seed=10,
        max_per_seed=20,
        target_total_rows=24,
        position_scale_m=0.004,
        rotation_scale_rad=0.08,
        output=tmp_path / "final.pt",
        report_json=tmp_path / "final.json",
    )
    assert MODULE.command_finalize(args) == 0
    result = MODULE.load_cache(args.output)
    assert MODULE.seed_counts(result, 2).tolist() == [12, 12]
    assert torch.bincount(result["variant_source_file_index"]).tolist() == [20, 4]
    assert result["metadata"]["filter_contract"]["omni_wrist"]["trials"] == 8
