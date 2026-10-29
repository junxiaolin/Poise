import pytest
import torch

from poise.grasp import load_stable_grasp_cache, save_stable_grasp_cache, validate_stable_grasp_cache


@pytest.fixture
def cache():
    return {
        "schema_version": 2, "hand": "sharpa_wave_right", "action_dim": 22,
        "object_id": "new_custom_object",
        "q": torch.zeros(2, 22), "qdot": torch.zeros(2, 22), "q_cmd": torch.ones(2, 22),
        "object_pose_palm": torch.tensor([[0., 0, 0, 1, 0, 0, 0]]).repeat(2, 1),
        "cube_keypoints_palm": torch.zeros(2, 8, 3), "contact_bits": torch.ones(2, 5, dtype=torch.bool),
        "stability_score": torch.ones(2),
    }


def test_roundtrip_preserves_measured_and_commanded_joints(cache, tmp_path):
    result = load_stable_grasp_cache(save_stable_grasp_cache(cache, tmp_path / "grasp.pt"))
    torch.testing.assert_close(result["q"], cache["q"])
    torch.testing.assert_close(result["q_cmd"], cache["q_cmd"])


@pytest.mark.parametrize("field,value", [("hand", "wuji_right"), ("action_dim", 20), ("schema_version", 1), ("object_id", "")])
def test_reject_wrong_interface(cache, field, value):
    cache[field] = value
    with pytest.raises(ValueError):
        validate_stable_grasp_cache(cache)


def test_reject_nonfinite_and_invalid_rotation(cache):
    cache["q_cmd"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="q_cmd contains"):
        validate_stable_grasp_cache(cache)
    cache["q_cmd"][0, 0] = 0
    cache["object_pose_palm"][0, 3] = 0
    with pytest.raises(ValueError, match="unit wxyz"):
        validate_stable_grasp_cache(cache)
