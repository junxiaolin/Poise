"""Exercise real Gloo PPO updates and rank-specific checkpoint continuation."""

import json
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from poise.training.distributed import configure_launch, curriculum_launch_settings
from poise.training.runner import PoiseRunner


@pytest.mark.parametrize("per_object", [False, True])
def test_frontier_override_targets_active_curriculum(per_object):
    cfg = SimpleNamespace(
        multiobject_per_object_curriculum_enabled=per_object,
        curriculum_distributed_sync_enabled=not per_object,
        multiobject_curriculum_distributed_sync_enabled=per_object,
        curriculum_global_min_frontier_done_episodes=20000,
        multiobject_curriculum_global_min_frontier_done_episodes=10000,
    )
    settings = curriculum_launch_settings(cfg, 5000)
    assert settings == {"curriculum_sync": True,
                        "curriculum_scope": "per_object" if per_object else "shared",
                        "global_frontier_min": 5000}
    assert cfg.curriculum_global_min_frontier_done_episodes == (20000 if per_object else 5000)
    assert cfg.multiobject_curriculum_global_min_frontier_done_episodes == (5000 if per_object else 10000)
    assert curriculum_launch_settings(cfg) == settings


def test_shared_gpu_rank_mapping(monkeypatch):
    for key, value in {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="gloo"):
        configure_launch(True, 1, "nccl")
    assert configure_launch(True, 1, "gloo") == (1, 2, 0)
    assert os.environ["RANK"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"


def test_separate_gpu_rank_mapping(monkeypatch):
    for key, value in {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(key, value)
    assert configure_launch(True, 2, "gloo") == (1, 2, 1)
    assert os.environ["LOCAL_RANK"] == "1"


def test_eight_workers_on_four_gpus(monkeypatch):
    mapped = []
    for rank in range(8):
        for key, value in {"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8",
                           "RANK": str(rank), "LOCAL_RANK": str(rank)}.items():
            monkeypatch.setenv(key, value)
        logical, world, physical = configure_launch(True, 4, "gloo")
        assert (logical, world) == (rank, 8)
        assert os.environ["RANK"] == str(rank)
        mapped.append(physical)
    assert mapped == [0, 1, 2, 3, 0, 1, 2, 3]


class ToyEnv:
    def __init__(self, rank):
        self.device = "cpu"
        self.num_envs = 8
        self.num_actions = 2
        self.max_episode_length = 16
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
        self.unwrapped = self
        self.rank = rank
        self.common_step_counter = 0
        self.adaptive_count = torch.tensor(rank * 100)
        self.target_rng = torch.Generator().manual_seed(100 + rank)
        self.cfg = SimpleNamespace(
            primitive_object_ids=["toy"],
            to_dict=lambda: {"sim": {"device": "cpu"}, "seed": 42 + rank},
        )

    def get_observations(self):
        return TensorDict({"policy": torch.randn(8, 3) + self.rank * 10,
                           "critic": torch.randn(8, 4) + self.rank * 10}, batch_size=[8])

    def step(self, actions):
        self.common_step_counter += 1
        self.adaptive_count += 1
        return self.get_observations(), -actions.square().sum(-1), torch.zeros(8, dtype=torch.long), {}

    def reset(self):
        return self.get_observations(), {}


def _worker(rank, root, rendezvous, world_size):
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank), POISE_DISTRIBUTED_BACKEND="gloo")
    torch.set_num_threads(1)
    torch.manual_seed(42 + rank)
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    cfg = json.loads((Path(__file__).parents[1] / "source/poise/sim/recipes/ppo.json").read_text())
    cfg.update(num_steps_per_env=2, save_interval=1, device="cpu")
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    env = ToyEnv(rank)
    runner = PoiseRunner(env, copy.deepcopy(cfg), log_dir=root, device="cpu")
    runner.learn(2)
    torch.distributed.barrier()
    checkpoint = str(Path(root) / "model_1.pt")
    expected = torch.load(checkpoint, weights_only=False)["infos"]["poise"]["rank_states"][rank]
    env = ToyEnv(rank)
    runner = PoiseRunner(env, copy.deepcopy(cfg), log_dir=root, device="cpu")
    env.adaptive_count.fill_(-1)
    runner.load(checkpoint, map_location="cpu")
    assert env.adaptive_count.item() == rank * 100 + 4
    assert env.common_step_counter == 4
    assert runner.current_learning_iteration == 2
    assert runner.tot_timesteps == 32 * world_size
    for key, value in runner.alg.policy.named_buffers():
        torch.testing.assert_close(value, expected["policy_buffers"][key])
    runner.learn(1)
    torch.distributed.barrier()
    saved = torch.load(Path(root) / "model_2.pt", weights_only=False)
    assert saved["infos"]["poise"]["next_iteration"] == 3
    assert saved["infos"]["poise"]["tot_timesteps"] == 48 * world_size
    states = saved["infos"]["poise"]["rank_states"]
    assert [s["adaptive_count"].item() for s in [r["curriculum"] for r in states]] == [6 + 100 * r for r in range(world_size)]
    assert states[0]["configuration_sha256"] != states[1]["configuration_sha256"]
    assert not torch.equal(states[0]["torch_rng"], states[1]["torch_rng"])
    assert not torch.equal(states[0]["policy_buffers"]["actor_obs_normalizer._mean"],
                           states[1]["policy_buffers"]["actor_obs_normalizer._mean"])
    parameters = torch.cat([p.detach().flatten() for p in runner.alg.policy.parameters()])
    received = [torch.empty_like(parameters) for _ in range(world_size)]
    torch.distributed.all_gather(received, parameters)
    for other in received[1:]:
        torch.testing.assert_close(received[0], other, rtol=0, atol=0)
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 8])
def test_gloo_ppo_save_and_resume(tmp_path, world_size):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path), f"file://{tmp_path}/rendezvous", world_size),
                                nprocs=world_size, join=True)


def test_memory_reclamation_preserves_ppo_and_random_streams(tmp_path, monkeypatch):
    """An actual PPO run must produce the same checkpoint with/without trimming."""
    import random
    import numpy as np

    monkeypatch.setenv("WORLD_SIZE", "1")
    torch.set_num_threads(1)
    cfg = json.loads((Path(__file__).parents[1] / "source/poise/sim/recipes/ppo.json").read_text())
    cfg.update(num_steps_per_env=2, save_interval=1, device="cpu")
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
    snapshots = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("POISE_MEMORY_RECLAIM", enabled)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        runner = PoiseRunner(ToyEnv(0), copy.deepcopy(cfg), log_dir=str(tmp_path / enabled), device="cpu")
        runner.learn(3, init_at_random_ep_len=True)
        runner.writer.close()
        snapshot = torch.load(tmp_path / enabled / "model_2.pt", weights_only=False)
        snapshot["infos"]["poise"].pop("tot_time")
        snapshots.append(snapshot)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    equal(*snapshots)


def test_resume_only_explicit_global_gate_change(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    torch.set_num_threads(1)
    cfg = json.loads((Path(__file__).parents[1] / "source/poise/sim/recipes/ppo.json").read_text())
    cfg.update(num_steps_per_env=2, save_interval=1, device="cpu")
    cfg["policy"].update(actor_hidden_dims=[16], critic_hidden_dims=[16])
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)

    def environment(gate, reward=0.5):
        env = ToyEnv(0)
        env.cfg.curriculum_global_min_frontier_done_episodes = gate
        env.cfg.to_dict = lambda: {
            "sim": {"device": "cpu"}, "seed": 42, "reward_scale": reward,
            "curriculum_global_min_frontier_done_episodes": gate,
        }
        return env

    original = PoiseRunner(environment(20000), copy.deepcopy(cfg), log_dir=str(tmp_path / "old"), device="cpu")
    original.learn(2)
    original.writer.close()
    checkpoint = tmp_path / "old/model_1.pt"
    before = checkpoint.read_bytes()
    saved = torch.load(checkpoint, weights_only=False)
    runner = PoiseRunner(environment(160000), copy.deepcopy(cfg), log_dir=str(tmp_path / "new"), device="cpu")
    with pytest.raises(ValueError, match="Resume configuration"):
        runner.load(checkpoint, map_location="cpu")
    with pytest.raises(ValueError, match="Resume configuration"):
        runner.load(checkpoint, map_location="cpu", resume_global_frontier_from=40000)
    changed_reward = PoiseRunner(environment(160000, reward=0.7), copy.deepcopy(cfg),
                                log_dir=str(tmp_path / "reject"), device="cpu")
    with pytest.raises(ValueError, match="Resume configuration"):
        changed_reward.load(checkpoint, map_location="cpu", resume_global_frontier_from=20000)
    runner.load(checkpoint, map_location="cpu", resume_global_frontier_from=20000)
    assert checkpoint.read_bytes() == before
    assert runner.current_learning_iteration == 2
    assert runner.env.adaptive_count.item() == 4
    assert runner.env.common_step_counter == 4
    assert runner.alg.learning_rate == saved["infos"]["poise"]["learning_rate"]
    for key, value in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(value, saved["model_state_dict"][key], rtol=0, atol=0)
    for key, state in runner.alg.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, saved["optimizer_state_dict"]["state"][key][name], rtol=0, atol=0)
    runner.learn(1)
    runner.writer.close()
    # Future checkpoints carry the new configuration identity and resume normally.
    continued = PoiseRunner(environment(160000), copy.deepcopy(cfg), log_dir=str(tmp_path / "continued"), device="cpu")
    continued.load(tmp_path / "new/model_2.pt", map_location="cpu")
    assert continued.current_learning_iteration == 3
