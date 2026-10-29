"""Object-filtered fingertip observations, independent of Isaac Lab.

Frames contain five force magnitudes (N), then five local contact XYZs (m),
in thumb/index/middle/ring/pinky order. History is oldest to newest.
"""

import torch

from poise.core.observation import quat_apply_wxyz, quat_conjugate_wxyz

TACTILE_PROFILE = "sharpa_pose6d_v1_tactile60_actor304_critic552"


def contact_frame(force_history_w, contact_pos_w, fingertip_pose_w):
    """Two physics-sample force average and current mean contact location.

    Forces and points must refer to the SAME filtered object/finger pair.
    Missing contact points (Isaac uses NaN) and absent current contact give
    zero features. The force is the magnitude of the normal resultant.
    """
    if force_history_w.shape[1:] != (2, 5, 3):
        raise ValueError("Expected two physics samples for five fingertip forces")
    if contact_pos_w.shape != fingertip_pose_w.shape[:-1] + (3,):
        raise ValueError("Contact points and fingertip poses must match")
    norms = torch.linalg.vector_norm(force_history_w, dim=-1)
    valid = (norms[:, 0] > 1e-6) & torch.isfinite(contact_pos_w).all(-1)
    valid &= torch.isfinite(force_history_w).all(dim=(1, 3))
    local = quat_apply_wxyz(
        quat_conjugate_wxyz(fingertip_pose_w[..., 3:7]),
        torch.nan_to_num(contact_pos_w) - fingertip_pose_w[..., :3],
    )
    local = torch.where(valid[..., None], local, torch.zeros_like(local))
    force = torch.where(valid, norms.mean(1), torch.zeros_like(norms[:, 0]))
    return torch.cat((force, local.flatten(1)), dim=-1)


class TactileHistory:
    """One shared actor/critic history, updated once per control step."""

    def __init__(self, num_envs, device, seed, hold_probability=0.005):
        if not 0 <= hold_probability <= 1:
            raise ValueError("Tactile hold probability must be in [0, 1]")
        self.frames = torch.zeros(num_envs, 3, 20, device=device)
        self.pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.reset_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.previous_step = -1
        self.hold_probability = hold_probability
        self.rng = torch.Generator(device=device).manual_seed(int(seed) + 90721)

    def reset(self, env_ids, step):
        self.frames[env_ids] = 0
        self.pending[env_ids] = True
        self.reset_step[env_ids] = step

    def update(self, frame, step):
        if step == self.previous_step:
            return self.frames.flatten(1)
        waiting = self.pending & (self.reset_step == step)
        first = self.pending & ~waiting
        hold = torch.rand(frame.shape[0], 5, device=frame.device, generator=self.rng)
        hold = (hold < self.hold_probability) & ~self.pending[:, None]
        # Keep each finger's force AND position together on a held sample.
        mask = torch.cat((hold, hold[..., None].expand(-1, -1, 3).flatten(1)), dim=1)
        frame = torch.where(mask, self.frames[:, -1], frame)
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = frame
        self.frames[first] = frame[first, None].expand(-1, 3, -1)
        self.frames[waiting] = 0
        self.pending[:] = waiting
        self.previous_step = step
        return self.frames.flatten(1)
