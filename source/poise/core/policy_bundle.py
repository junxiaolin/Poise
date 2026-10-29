"""TorchScript policy export and dependency-light runtime loading."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .schema import (
    ACTION_DIM,
    OBSERVATION_DIM,
    SCHEMA_VERSION,
    SHARPA_ACTION_TO_SDK_INDICES,
    SHARPA_SDK_TO_ACTION_INDICES,
    joint_interface_manifest,
    validate_observation_dimension,
    observation_contract_from_manifest,
)


NORMALIZER_EPS = 1.0e-2


class FusedActor(nn.Module):
    """Actor MLP with the frozen empirical observation normalizer in forward()."""

    def __init__(
        self,
        actor: nn.Sequential,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        normalizer_eps: float = NORMALIZER_EPS,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("observation_mean", observation_mean.reshape(1, -1).clone())
        self.register_buffer("observation_std", observation_std.reshape(1, -1).clone())
        self.normalizer_eps = float(normalizer_eps)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.observation_mean) / (self.observation_std + self.normalizer_eps)
        return self.actor(normalized)


def _activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    try:
        return activations[normalized]()
    except KeyError as exc:
        raise ValueError(f"Unsupported actor activation {name!r}; expected one of {tuple(activations)}.") from exc


def build_fused_actor(
    model_state_dict: Mapping[str, torch.Tensor],
    *,
    activation: str,
    normalizer_eps: float = NORMALIZER_EPS,
    observation_dim: int = OBSERVATION_DIM,
) -> tuple[FusedActor, dict[str, Any]]:
    """Reconstruct the feed-forward inference actor directly from a checkpoint."""

    validate_observation_dimension(observation_dim)

    layer_pattern = re.compile(r"^actor\.(\d+)\.weight$")
    layer_indices = sorted(
        int(match.group(1))
        for key in model_state_dict
        if (match := layer_pattern.fullmatch(key)) is not None
    )
    if not layer_indices:
        raise ValueError("Checkpoint does not contain actor.<index>.weight tensors.")

    modules: list[nn.Module] = []
    layer_shapes: list[list[int]] = []
    for layer_number, state_index in enumerate(layer_indices):
        weight_key = f"actor.{state_index}.weight"
        bias_key = f"actor.{state_index}.bias"
        if bias_key not in model_state_dict:
            raise ValueError(f"Checkpoint is missing {bias_key}.")
        weight = model_state_dict[weight_key].detach().to(device="cpu", dtype=torch.float32)
        bias = model_state_dict[bias_key].detach().to(device="cpu", dtype=torch.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"Invalid actor layer tensors for actor.{state_index}.")
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError(f"Actor layer actor.{state_index} contains NaN or Inf.")
        linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        modules.append(linear)
        layer_shapes.append([int(weight.shape[1]), int(weight.shape[0])])
        if layer_number + 1 < len(layer_indices):
            modules.append(_activation(activation))

    mean_key = "actor_obs_normalizer._mean"
    std_key = "actor_obs_normalizer._std"
    if mean_key not in model_state_dict or std_key not in model_state_dict:
        raise ValueError("Checkpoint is missing actor observation normalizer mean/std.")
    mean = model_state_dict[mean_key].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    std = model_state_dict[std_key].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if mean.shape != (observation_dim,) or std.shape != (observation_dim,):
        raise ValueError(
            f"Expected {observation_dim}D actor normalizer, got "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}."
        )
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Actor observation normalizer contains NaN or Inf.")
    if bool(torch.any(std < 0.0)):
        raise ValueError("Actor observation normalizer contains a negative standard deviation.")
    if layer_shapes[0][0] != observation_dim or layer_shapes[-1][1] != ACTION_DIM:
        raise ValueError(
            f"Actor dimensions must be {observation_dim}->{ACTION_DIM}, got "
            f"{layer_shapes[0][0]}->{layer_shapes[-1][1]}."
        )

    count_value = model_state_dict.get("actor_obs_normalizer.count")
    normalizer_count = int(count_value.detach().cpu().item()) if isinstance(count_value, torch.Tensor) else None
    module = FusedActor(nn.Sequential(*modules), mean, std, normalizer_eps).eval()
    metadata = {
        "input_dim": observation_dim,
        "output_dim": ACTION_DIM,
        "activation": activation.lower(),
        "linear_layer_shapes": layer_shapes,
        "normalizer_eps": float(normalizer_eps),
        "normalizer_count": normalizer_count,
        "observation_mean_min": float(mean.min().item()),
        "observation_mean_max": float(mean.max().item()),
        "observation_std_min": float(std.min().item()),
        "observation_std_max": float(std.max().item()),
    }
    return module, metadata


def export_fused_policy(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    activation: str,
    normalizer_eps: float = NORMALIZER_EPS,
    observation_dim: int = OBSERVATION_DIM,
) -> dict[str, Any]:
    """Export a checkpoint's deterministic actor as a fused TorchScript module."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    module, metadata = build_fused_actor(
        checkpoint["model_state_dict"],
        activation=activation,
        normalizer_eps=normalizer_eps,
        observation_dim=observation_dim,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.freeze(torch.jit.script(module))
    scripted.save(str(output_path))

    probe = torch.linspace(-1.0, 1.0, observation_dim, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        eager_output = module(probe)
        scripted_output = scripted(probe)
    max_error = float(torch.max(torch.abs(eager_output - scripted_output)).item())
    if max_error > 1.0e-6:
        raise RuntimeError(f"TorchScript export changed actor output (max_abs_error={max_error:.3e}).")
    metadata.update(
        {
            "checkpoint_iteration": int(checkpoint["iter"]) if "iter" in checkpoint else None,
            "torchscript_eager_max_abs_error": max_error,
        }
    )
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PolicyBundle:
    """Load a frozen deployment bundle without importing Isaac Lab or RSL-RL."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        module: torch.jit.ScriptModule,
        device: torch.device,
        observation_dim: int,
        include_gravity_palm: bool,
        object_descriptor: torch.Tensor | None,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.module = module
        self.device = device
        self.observation_dim = int(observation_dim)
        self.include_gravity_palm = bool(include_gravity_palm)
        self.object_descriptor = object_descriptor

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        verify_hash: bool = True,
    ) -> "PolicyBundle":
        root = Path(bundle_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Policy bundle schema is unsafe for the current real-hardware "
                "joint interface: expected "
                f"{SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}. "
                "Regenerate or explicitly migrate the bundle."
            )
        expected_joint_interface = joint_interface_manifest()
        observed_joint_interface = manifest.get("joint_interface")
        if observed_joint_interface != expected_joint_interface:
            raise ValueError(
                "Policy bundle joint_interface does not match the fail-closed "
                "Sharpa action-order == SDK-order contract. Legacy "
                "IsaacLab-articulation permutations cannot be used at this "
                "hardware boundary."
            )
        observation_contract = observation_contract_from_manifest(
            manifest.get("observation")
        )
        observation_record = manifest.get("observation", {})
        descriptor_record = observation_record.get("object_descriptor")
        object_descriptor: torch.Tensor | None = None
        if observation_contract.object_descriptor_dim:
            if not isinstance(descriptor_record, dict):
                raise ValueError(
                    "BPS64 policy bundle is missing observation.object_descriptor."
                )
            if descriptor_record.get("kind") != "mesh_bps64":
                raise ValueError("BPS64 object descriptor kind must be 'mesh_bps64'.")
            dimension = int(descriptor_record.get("dimension", -1))
            if dimension != observation_contract.object_descriptor_dim:
                raise ValueError(
                    "BPS64 object descriptor dimension does not match the observation contract."
                )
            object_id = str(descriptor_record.get("object_id", ""))
            cache_object_id = str(
                manifest.get("physical_object", {}).get(
                    "stable_cache_object_id",
                    "",
                )
            )
            if not object_id or object_id != cache_object_id:
                raise ValueError(
                    "BPS64 object descriptor object_id does not match "
                    "physical_object.stable_cache_object_id."
                )
            object_descriptor = torch.as_tensor(
                descriptor_record.get("values"),
                dtype=torch.float32,
            )
            if object_descriptor.shape != (dimension,):
                raise ValueError(
                    f"BPS64 object descriptor must have shape ({dimension},), "
                    f"got {tuple(object_descriptor.shape)}."
                )
            if not bool(torch.isfinite(object_descriptor).all()):
                raise ValueError("BPS64 object descriptor contains NaN or Inf.")
        elif descriptor_record is not None:
            raise ValueError(
                "observation.object_descriptor is forbidden by this observation contract."
            )
        policy_record = manifest.get("policy", {})
        policy_input_dim = int(
            policy_record.get("input_dim", observation_contract.dimension)
        )
        policy_output_dim = int(policy_record.get("output_dim", ACTION_DIM))
        if policy_input_dim != observation_contract.dimension or policy_output_dim != ACTION_DIM:
            raise ValueError(
                "Policy dimensions do not match the approved observation/action contract: "
                f"policy={policy_input_dim}->{policy_output_dim}, "
                f"contract={observation_contract.dimension}->{ACTION_DIM}."
            )
        policy_path = root / str(policy_record.get("filename", "policy.ts"))
        if verify_hash:
            expected = policy_record.get("sha256")
            actual = _sha256_file(policy_path)
            if expected != actual:
                raise ValueError(f"Policy SHA256 mismatch: expected {expected}, got {actual}.")
        target_device = torch.device(device)
        module = torch.jit.load(str(policy_path), map_location=target_device).eval()
        return cls(
            root,
            manifest,
            module,
            target_device,
            observation_contract.dimension,
            observation_contract.include_gravity_palm,
            object_descriptor,
        )

    def infer(
        self,
        observation: torch.Tensor,
        *,
        clip_action: bool = True,
    ) -> torch.Tensor:
        """Run deterministic inference; optionally apply the deployment action clip."""

        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        if observation.ndim not in (1, 2) or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation shape ({self.observation_dim},) or (batch, {self.observation_dim}), "
                f"got {tuple(observation.shape)}."
            )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("Policy input contains NaN or Inf.")
        single = observation.ndim == 1
        batch = observation.unsqueeze(0) if single else observation
        with torch.inference_mode():
            action = self.module(batch)
        if action.shape != (batch.shape[0], ACTION_DIM) or not bool(torch.isfinite(action).all()):
            raise ValueError("Policy output must be a finite (batch, 22) tensor")
        if clip_action:
            clip_value = float(self.manifest.get("policy", {}).get("clip_actions", 1.0))
            action = torch.clamp(action, -clip_value, clip_value)
        return action.squeeze(0) if single else action

    def next_joint_target(
        self,
        previous_joint_target: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the simulator's incremental, smoothed, joint-limited action contract."""

        contract = self.manifest.get("sim_contract", {})
        if "joint_lower" not in contract or "joint_upper" not in contract:
            raise RuntimeError("Bundle has no sim_contract joint limits; generate the sim golden reference first.")
        previous = (
            previous_joint_target.to(dtype=torch.float32)
            if isinstance(previous_joint_target, torch.Tensor)
            else torch.as_tensor(
                previous_joint_target,
                device=self.device,
                dtype=torch.float32,
            )
        )
        action = torch.as_tensor(action, device=previous.device, dtype=torch.float32)
        lower = torch.as_tensor(contract["joint_lower"], device=previous.device, dtype=torch.float32)
        upper = torch.as_tensor(contract["joint_upper"], device=previous.device, dtype=torch.float32)
        control = self.manifest.get("control", {})
        clip_value = float(self.manifest.get("policy", {}).get("clip_actions", 1.0))
        action_scale = float(control["action_scale_rad"])
        smoothing = float(control["action_smoothing"])
        raw_target = previous + action_scale * torch.clamp(action, -clip_value, clip_value)
        raw_target = torch.minimum(torch.maximum(raw_target, lower), upper)
        return smoothing * raw_target + (1.0 - smoothing) * previous

    @staticmethod
    def action_to_sdk(values: torch.Tensor) -> torch.Tensor:
        """Convert policy/action order to raw SDK order (identity contract)."""
        index = torch.tensor(
            SHARPA_ACTION_TO_SDK_INDICES,
            device=values.device,
            dtype=torch.long,
        )
        return values.index_select(dim=-1, index=index)

    @staticmethod
    def sdk_to_action(values: torch.Tensor) -> torch.Tensor:
        """Convert raw SDK order to policy/action order (identity contract)."""
        index = torch.tensor(
            SHARPA_SDK_TO_ACTION_INDICES,
            device=values.device,
            dtype=torch.long,
        )
        return values.index_select(dim=-1, index=index)
