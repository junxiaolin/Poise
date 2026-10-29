"""Small RSL-RL adapters used by POISE training scripts."""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def _optional_float_from_env(name: str) -> float | None:
    value = os.environ.get(name, "")
    if value == "":
        return None
    return float(value)


class BoundedStdActorCritic(ActorCritic):
    """Actor-critic that clamps the effective action std used by the policy distribution."""

    def __init__(
        self,
        *args,
        min_action_std: float | None = None,
        max_action_std: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_action_std = _optional_float_from_env("MIN_ACTION_STD") if min_action_std is None else min_action_std
        self.max_action_std = _optional_float_from_env("MAX_ACTION_STD") if max_action_std is None else max_action_std
        if self.min_action_std is not None and self.min_action_std <= 0.0:
            raise ValueError(f"min_action_std must be positive, got {self.min_action_std}.")
        if self.max_action_std is not None and self.max_action_std <= 0.0:
            raise ValueError(f"max_action_std must be positive, got {self.max_action_std}.")
        if (
            self.min_action_std is not None
            and self.max_action_std is not None
            and self.max_action_std < self.min_action_std
        ):
            raise ValueError(
                f"max_action_std must be >= min_action_std, got {self.max_action_std} < {self.min_action_std}."
            )
        print(
            "BoundedStdActorCritic action std bounds: "
            f"min={self.min_action_std if self.min_action_std is not None else 'none'}, "
            f"max={self.max_action_std if self.max_action_std is not None else 'none'}"
        )

    def _bounded_std(self, std: torch.Tensor) -> torch.Tensor:
        if self.min_action_std is None and self.max_action_std is None:
            return std
        return torch.clamp(std, min=self.min_action_std, max=self.max_action_std)

    def update_distribution(self, obs) -> None:
        # RSL-RL 3.0.1 calls the public hook from act(). Newer releases use
        # _update_distribution(); keep both entry points bounded.
        self._update_distribution(obs)

    def _update_distribution(self, obs) -> None:
        if getattr(self, "state_dependent_std", False):
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, self._bounded_std(std))


class PoisePPO(PPO):
    """PPO with DexNDM/HoRa-style adaptive KL scheduling.

    RSL-RL's built-in adaptive schedule updates the learning rate after every
    mini-batch. DexNDM accumulates KL values within a mini epoch and adjusts the
    learning rate once per epoch, which is a gentler feedback loop.
    """

    KL_EARLY_STOP_THRESHOLD = 0.04
    MAX_LEARNING_RATE = 1.0e-3

    def _dexndm_update_learning_rate(self, mean_kl: torch.Tensor) -> float | None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            if self.is_multi_gpu:
                if self._distributed_backend_is_gloo() and mean_kl.is_cuda:
                    mean_kl_cpu = mean_kl.detach().cpu()
                    torch.distributed.all_reduce(mean_kl_cpu, op=torch.distributed.ReduceOp.SUM)
                    mean_kl = mean_kl_cpu.to(device=self.device) / self.gpu_world_size
                else:
                    torch.distributed.all_reduce(mean_kl, op=torch.distributed.ReduceOp.SUM)
                    mean_kl /= self.gpu_world_size

            kl_value = float(mean_kl.item())
            if self.gpu_global_rank == 0:
                if kl_value > 2.0 * float(self.desired_kl):
                    self.learning_rate = max(self.learning_rate / 1.5, 1.0e-6)
                if kl_value < 0.5 * float(self.desired_kl):
                    self.learning_rate = min(self.learning_rate * 1.5, self.MAX_LEARNING_RATE)

            if self.is_multi_gpu:
                lr_device = "cpu" if self._distributed_backend_is_gloo() else self.device
                lr_tensor = torch.tensor(self.learning_rate, device=lr_device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = float(lr_tensor.item())

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

            return kl_value

    def _distributed_backend_is_gloo(self) -> bool:
        return torch.distributed.is_initialized() and str(torch.distributed.get_backend()).lower() == "gloo"

    def _broadcast_module_state(self, module: nn.Module, src: int = 0) -> None:
        state_dict = module.state_dict()
        use_cpu_collectives = self._distributed_backend_is_gloo()
        for value in state_dict.values():
            if not torch.is_tensor(value):
                continue
            if use_cpu_collectives:
                broadcast_value = value.detach().cpu().clone()
            elif value.is_cuda:
                broadcast_value = value
            else:
                broadcast_value = value.to(device=self.device)
            torch.distributed.broadcast(broadcast_value, src=src)
            if use_cpu_collectives:
                value.copy_(broadcast_value.to(device=value.device))
            elif not value.is_cuda:
                value.copy_(broadcast_value.cpu())
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(device=self.device)
        module.load_state_dict(state_dict)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters without Python object collectives.

        RSL-RL's default implementation broadcasts the whole state dict through
        ``broadcast_object_list``.  On some Isaac Sim multi-process runs this can
        stall after both ranks initialize.  All ranks construct the same module
        shape, so tensor-wise broadcast is enough and keeps the communication on
        normal tensor collectives.
        """

        self._broadcast_module_state(self.policy)
        if self.rnd:
            self._broadcast_module_state(self.rnd.predictor)

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        if self._distributed_backend_is_gloo() and all_grads.is_cuda:
            reduced_grads = all_grads.detach().cpu()
            torch.distributed.all_reduce(reduced_grads, op=torch.distributed.ReduceOp.SUM)
            all_grads = reduced_grads.to(device=self.device)
        else:
            torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            from itertools import chain

            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                offset += numel

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_kl_loss = 0.0
        kl_update_count = 0
        num_updates = 0
        kl_early_stopped = False

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        epoch_kls: list[torch.Tensor] = []
        batches_in_epoch = 0
        kl_reference_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_mini_batches

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1.0e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    cached_reference = kl_reference_cache[batches_in_epoch]
                    if cached_reference is None:
                        ref_mu_batch = old_mu_batch
                        ref_sigma_batch = old_sigma_batch
                    else:
                        ref_mu_batch, ref_sigma_batch = cached_reference
                    kl = torch.sum(
                        torch.log(ref_sigma_batch / sigma_batch + 1.0e-5)
                        + (torch.square(sigma_batch) + torch.square(ref_mu_batch - mu_batch))
                        / (2.0 * (torch.square(ref_sigma_batch) + 1.0e-5))
                        - 0.5,
                        axis=-1,
                    )
                    epoch_kls.append(torch.mean(kl.detach()))

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mse_loss = torch.nn.MSELoss()
                rnd_loss = mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            num_updates += 1
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

            if self.desired_kl is not None and self.schedule == "adaptive":
                kl_reference_cache[batches_in_epoch] = (mu_batch.detach().clone(), sigma_batch.detach().clone())

            batches_in_epoch += 1
            if batches_in_epoch >= self.num_mini_batches:
                if epoch_kls:
                    epoch_kl = torch.mean(torch.stack(epoch_kls))
                    global_epoch_kl = self._dexndm_update_learning_rate(epoch_kl)
                    mean_kl_loss += float(global_epoch_kl)
                    kl_update_count += 1
                    kl_early_stopped = float(global_epoch_kl) > self.KL_EARLY_STOP_THRESHOLD
                epoch_kls.clear()
                batches_in_epoch = 0
                if kl_early_stopped:
                    break

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if kl_update_count > 0:
            mean_kl_loss /= kl_update_count

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if kl_update_count > 0:
            loss_dict["dexndm_adaptive_kl"] = mean_kl_loss
            loss_dict["dexndm_kl_early_stop"] = float(kl_early_stopped)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
