"""Checkpoint PPO, adaptive curriculum and random streams together."""

import copy
import hashlib
import json
import os
import random

import numpy as np
import torch
import rsl_rl.runners.on_policy_runner as runner_module
from rsl_rl.runners import OnPolicyRunner

from poise.training.ppo import BoundedStdActorCritic, PoisePPO
from poise.training.distributed import initialize_process_group
from poise.training.memory import reclaim_host_memory, report_host_memory

runner_module.BoundedStdActorCritic = BoundedStdActorCritic
runner_module.PoisePPO = PoisePPO


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _restore(current, saved):
    if isinstance(current, torch.Tensor):
        if current.shape != saved.shape:
            raise ValueError(f"Curriculum shape mismatch: {current.shape} versus {saved.shape}")
        current.copy_(saved.to(current.device))
        return current
    if isinstance(current, dict):
        if current.keys() != saved.keys():
            raise ValueError("Curriculum dictionary keys differ")
        return {key: _restore(current[key], saved[key]) for key in current}
    return copy.deepcopy(saved)


def _configuration_hash(env, overrides=None):
    config = copy.deepcopy(env.cfg.to_dict())
    config.update(overrides or {})
    config.pop("log_dir", None)
    config.pop("viewer", None)
    config["sim"].pop("device", None)
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        # This switch has no effect without collectives; retain schema-v2
        # single-worker checkpoint compatibility after adding it to the recipe.
        config.pop("curriculum_distributed_sync_enabled", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


class PoiseRunner(OnPolicyRunner):
    """Resume optimization and curriculum; simulation episodes restart on load.

    PhysX contacts and episode trajectories are not serialized, so continuation
    is deliberately not described as a bitwise simulation restart.
    """

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        self.gpu_global_rank = int(os.environ.get("RANK", "0")) if self.is_distributed else 0
        self.gpu_local_rank = int(os.environ.get("LOCAL_RANK", "0")) if self.is_distributed else 0
        self.multi_gpu_cfg = None
        if self.is_distributed:
            if self.device != f"cuda:{self.gpu_local_rank}" and self.device != "cpu":
                raise ValueError("Runner device differs from the assigned physical GPU")
            initialize_process_group(self.device)
            self.multi_gpu_cfg = {
                "global_rank": self.gpu_global_rank,
                "local_rank": self.gpu_local_rank,
                "world_size": self.gpu_world_size,
            }

    def _rank_state(self):
        env = self.env.unwrapped
        curriculum = {key: _cpu(value) for key, value in vars(env).items() if key.startswith("adaptive_")}
        state = {
            "configuration_sha256": _configuration_hash(env),
            "object_ids": list(env.cfg.primitive_object_ids),
            "curriculum": curriculum,
            "common_step_counter": env.common_step_counter,
            # RSL-RL updates observation statistics locally on each worker.
            "policy_buffers": {key: _cpu(value) for key, value in self.alg.policy.named_buffers()},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "target_rng": env.target_rng.get_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        }
        if getattr(env, "tactile_enabled", False):
            state["tactile_rng"] = env.tactile.rng.get_state()
        return state

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # RSL-RL calls save() only on rank zero. Collect the worker snapshots at
        # the common post-update boundary, before that rank-only call, so save
        # never enters a collective that other workers cannot reach.
        update = self.alg.update
        broadcast = self.alg.broadcast_parameters
        iteration = self.current_learning_iteration
        final_iteration = iteration + num_learning_iterations - 1
        first_iteration = iteration
        # USD/PhysX construction and reset allocate large temporary CPU heaps.
        # Reclaim only after their initialization (including resume) is complete.
        reclaim_host_memory("before_learn")

        def update_and_snapshot():
            nonlocal iteration
            losses = update()
            if self.is_distributed and (iteration % self.save_interval == 0 or iteration == final_iteration):
                states = [None] * self.gpu_world_size if self.gpu_global_rank == 0 else None
                torch.distributed.gather_object(self._rank_state(), states, dst=0)
                self._checkpoint_rank_states = states
                self._checkpoint_iteration = iteration
            if iteration == first_iteration:
                # The first rollout/update may lazily initialize native buffers.
                # All ranks reach this boundary; later updates pay no trim cost.
                reclaim_host_memory("after_first_update")
            if iteration % 10 == 9:
                report_host_memory(f"update_{iteration + 1}")
            iteration += 1
            return losses

        def broadcast_with_restored_buffers():
            broadcast()
            buffers = getattr(self, "_resumed_policy_buffers", None)
            if buffers is not None:
                self._restore_policy_buffers(buffers)

        self.alg.update = update_and_snapshot
        self.alg.broadcast_parameters = broadcast_with_restored_buffers
        try:
            result = super().learn(num_learning_iterations, init_at_random_ep_len)
            report_host_memory("after_learn")
            return result
        finally:
            self.alg.update = update
            self.alg.broadcast_parameters = broadcast
            self._resumed_policy_buffers = None

    def save(self, path, infos=None):
        if self.is_distributed:
            if self._checkpoint_iteration != self.current_learning_iteration:
                raise RuntimeError("Distributed checkpoints must be saved at a synchronized update boundary")
            states = self._checkpoint_rank_states
        else:
            states = [self._rank_state()]
        infos = dict(infos or {})
        infos["poise"] = {
            "schema_version": 3,
            "world_size": self.gpu_world_size,
            "rank_states": states,
            "next_iteration": self.current_learning_iteration + 1,
            "learning_rate": self.alg.learning_rate,
            "tot_timesteps": self.tot_timesteps, "tot_time": self.tot_time,
        }
        if getattr(self, "setup_contract", None) is not None:
            infos["poise"]["setup"] = self.setup_contract
        env = self.env.unwrapped
        if getattr(env, "tactile_enabled", False):
            infos["poise"]["observation"] = {
                "profile": env.cfg.observation_profile,
                "actor_dim": 304, "critic_dim": 552,
                "tactile_history_shape": [3, 20],
                "hold_probability": env.tactile.hold_probability,
            }
        # A visible checkpoint is always complete, including all worker states.
        temporary = str(path) + ".tmp"
        super().save(temporary, infos)
        os.replace(temporary, path)

    @torch.inference_mode()
    def _restore_policy_buffers(self, buffers):
        current = dict(self.alg.policy.named_buffers())
        if current.keys() != buffers.keys():
            raise ValueError("Checkpoint policy buffer names differ")
        for key, value in buffers.items():
            _restore(current[key], value)

    def load(self, path, load_optimizer=True, map_location=None, *, resume_global_frontier_from=None):
        # Validate identity before applying any checkpoint state.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        common = (saved.get("infos") or {}).get("poise")
        if common is None or common.get("schema_version") not in (2, 3):
            raise ValueError("Resume requires a POISE checkpoint with curriculum state.")
        if getattr(self, "setup_contract", None) is not None and common.get("setup") != self.setup_contract:
            raise ValueError("Resume setup contract differs from checkpoint; older runs require their original --recipe entry")
        if common.get("world_size", 1) != self.gpu_world_size:
            raise ValueError("Resume requires the checkpoint's distributed world size")
        state = common if common["schema_version"] == 2 else common["rank_states"][self.gpu_global_rank]
        env = self.env.unwrapped
        overrides = None
        if resume_global_frontier_from is not None:
            if resume_global_frontier_from < 1:
                raise ValueError("Original global frontier gate must be positive")
            prefix = ("multiobject_curriculum" if getattr(env.cfg, "multiobject_per_object_curriculum_enabled", False)
                      else "curriculum")
            gate = f"{prefix}_global_min_frontier_done_episodes"
            if getattr(env.cfg, gate, -1) < 1:
                raise ValueError("New global frontier gate must be positive")
            overrides = {gate: int(resume_global_frontier_from)}
        # Reconstruct the old identity by changing only the explicitly allowed
        # gate. Every other field must still match the original checkpoint.
        if _configuration_hash(env, overrides) != state["configuration_sha256"]:
            raise ValueError("Resume configuration differs from checkpoint; use the same recipe, seed and environment count")
        if list(env.cfg.primitive_object_ids) != state["object_ids"]:
            raise ValueError("Resume object order differs from checkpoint")
        if overrides:
            print("POISE_RESUME_CONFIG_OVERRIDE " + json.dumps({
                "rank": self.gpu_global_rank, "checkpoint": str(path),
                "changes": {gate: [resume_global_frontier_from, getattr(env.cfg, gate)]},
                "original_configuration_sha256": state["configuration_sha256"],
                "new_configuration_sha256": _configuration_hash(env),
            }), flush=True)
        infos = super().load(path, load_optimizer, map_location)
        for key, value in state["curriculum"].items():
            setattr(env, key, _restore(getattr(env, key), value))
        env.common_step_counter = state["common_step_counter"]
        self.current_learning_iteration = common["next_iteration"]
        self.alg.learning_rate = common["learning_rate"]
        self.tot_timesteps, self.tot_time = common["tot_timesteps"], common["tot_time"]
        self._resumed_policy_buffers = state.get("policy_buffers")
        if self._resumed_policy_buffers is not None:
            self._restore_policy_buffers(self._resumed_policy_buffers)
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        env.target_rng.set_state(state["target_rng"])
        if getattr(env, "tactile_enabled", False):
            env.tactile.rng.set_state(state["tactile_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        # Regenerate reset targets using the restored curriculum, without treating
        # unfinished pre-checkpoint episodes as new curriculum successes/failures.
        self.env.reset()
        print(f"POISE_RESUMED next_iteration={self.current_learning_iteration} curriculum_fields={len(state['curriculum'])}", flush=True)
        return infos
