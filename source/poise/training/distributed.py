"""Map torchrun workers to physical GPUs before Isaac Sim starts."""

import os
from datetime import timedelta


def curriculum_launch_settings(cfg, frontier_min=None):
    """Apply a gate override to the active curriculum and report its scope."""
    per_object = bool(getattr(cfg, "multiobject_per_object_curriculum_enabled", False))
    prefix = "multiobject_curriculum" if per_object else "curriculum"
    gate = f"{prefix}_global_min_frontier_done_episodes"
    if frontier_min is not None:
        if frontier_min < 1:
            raise ValueError("frontier_min must be positive")
        setattr(cfg, gate, int(frontier_min))
    return {
        "curriculum_sync": bool(getattr(cfg, f"{prefix}_distributed_sync_enabled", False)),
        "curriculum_scope": "per_object" if per_object else "shared",
        "global_frontier_min": int(getattr(cfg, gate, -1)),
    }


def configure_launch(distributed, physical_gpus=None, backend="gloo"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not distributed:
        raise ValueError("torchrun with multiple workers requires --distributed")
    if distributed and world_size <= 1:
        raise ValueError("--distributed requires torchrun with multiple workers")
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank")
    if physical_gpus is not None:
        if not distributed or physical_gpus < 1:
            raise ValueError("--physical-gpus requires distributed training and a positive count")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if local_world_size > physical_gpus and backend != "gloo":
            raise ValueError("Workers sharing a GPU require the gloo backend")
        local_rank %= physical_gpus
    os.environ["POISE_DISTRIBUTED_BACKEND"] = backend
    if distributed:
        # AppLauncher treats LOCAL_RANK as a CUDA ordinal. RANK stays logical.
        os.environ["LOCAL_RANK"] = str(local_rank)
    return rank, world_size, local_rank


def initialize_process_group(device):
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=os.environ.get("POISE_DISTRIBUTED_BACKEND", "gloo"),
            timeout=timedelta(minutes=30),
        )
