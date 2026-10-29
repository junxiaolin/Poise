"""Exercise the policy's actual sampling path, including RSL-RL hook dispatch."""

import pytest
import torch

pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from poise.training.ppo import BoundedStdActorCritic


@pytest.mark.parametrize("noise_type", ["scalar", "log"])
def test_sampling_enforces_std_bounds_after_parameter_update_and_reload(noise_type):
    obs = TensorDict({"policy": torch.zeros(4096, 3), "critic": torch.zeros(4096, 4)},
                     batch_size=[4096])
    kwargs = dict(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
                  num_actions=3, actor_hidden_dims=[8], critic_hidden_dims=[8],
                  noise_std_type=noise_type, min_action_std=0.3, max_action_std=1.5)
    policy = BoundedStdActorCritic(**kwargs)
    parameter = policy.std if noise_type == "scalar" else policy.log_std
    raw_std = torch.tensor([0.1, 0.6, 2.0])
    with torch.no_grad():
        parameter.copy_(raw_std if noise_type == "scalar" else raw_std.log())
    restored = BoundedStdActorCritic(**kwargs)
    restored.load_state_dict(policy.state_dict())
    for model in (policy, restored):
        torch.manual_seed(42)
        actions = model.act(obs)
        expected = torch.tensor([0.3, 0.6, 1.5])
        torch.testing.assert_close(model.action_std, expected.expand_as(actions))
        torch.testing.assert_close((actions - model.action_mean).std(dim=0), expected,
                                   rtol=0.05, atol=0.01)
        loss = -model.get_actions_log_prob(actions.detach()).mean()
        loss.backward()
        std_parameter = model.std if noise_type == "scalar" else model.log_std
        assert torch.isfinite(std_parameter.grad).all()
        assert std_parameter.grad[0] == 0 and std_parameter.grad[2] == 0
        assert std_parameter.grad[1].abs() > 0
