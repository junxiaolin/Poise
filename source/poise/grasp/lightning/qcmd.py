# Derived from Lightning Grasp by Zhao-Heng Yin; CC BY-NC 4.0.
# See licenses/Lightning-Grasp.txt. POISE changes: Sharpa-only, portable paths.
#!/usr/bin/env python3
"""Generate Sharpa Wave q_cmd from Lightning Grasp wrench metadata."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


from lygra.kinematics import batch_jacobian, build_kinematics_tree
from poise.grasp.lightning.hand import build_robot
from poise.grasp.lightning.hand import ACTIVE_JOINTS


def default_metadata_path(cache_path):
    return cache_path.with_suffix(".metadata.npz")


def default_output_path(cache_path, force_scale, max_delta_rad):
    force_tag = f"{force_scale:g}".replace(".", "p")
    delta_tag = f"{max_delta_rad:g}".replace(".", "p")
    return cache_path.with_suffix(f".qcmd_force{force_tag}_maxdelta{delta_tag}.npz")


def read_summary(metadata_path):
    summary_path = metadata_path.with_suffix(".json")
    if not summary_path.exists():
        return {}, summary_path
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f), summary_path


def parse_usd_joint_drive_stiffness(usd_path, joint_names):
    text = usd_path.read_text(encoding="utf-8")
    values = {}
    stiffness_pattern = re.compile(
        r"drive:angular:physics:stiffness\s*=\s*([-+0-9.eE]+)"
    )
    for joint_name in joint_names:
        block_pattern = re.compile(
            rf'(?:over|def\s+\w+)\s+"{re.escape(joint_name)}"\s*\{{(.*?)\n\s*\}}',
            re.DOTALL,
        )
        block_match = block_pattern.search(text)
        if block_match is None:
            continue
        body = block_match.group(1)
        stiffness_match = stiffness_pattern.search(body)
        if stiffness_match is None:
            continue
        values[joint_name] = float(stiffness_match.group(1))

    missing = [name for name in joint_names if name not in values]
    if missing:
        raise ValueError(f"Missing USD drive stiffness for joints: {missing}")
    return np.asarray([values[name] for name in joint_names], dtype=np.float32)


def require_metadata(metadata, keys):
    missing = [key for key in keys if key not in metadata.files]
    if missing:
        raise ValueError(
            "Metadata is missing fields required for q_cmd generation: "
            f"{missing}. Regenerate the Lightning cache with the updated "
            "poise.grasp.lightning.generate entry point so contact geometry and wrench "
            "coefficients are saved."
        )


def compute_qcmd(
    tree,
    q_np,
    contact_link_id_np,
    contact_pos_np,
    target_normal_np,
    contact_force_coeff_np,
    stiffness_np,
    joint_lower_np,
    joint_upper_np,
    *,
    force_scale,
    max_delta_rad,
    min_stiffness,
    torque_sign,
    device,
    batch_size,
):
    q_out = []
    q_cmd_out = []
    delta_out = []
    delta_unclamped_out = []
    tau_out = []
    force_out = []

    stiffness = torch.as_tensor(stiffness_np, device=device, dtype=torch.float32).clamp_min(min_stiffness)
    lower = torch.as_tensor(joint_lower_np, device=device, dtype=torch.float32)
    upper = torch.as_tensor(joint_upper_np, device=device, dtype=torch.float32)

    for start in range(0, len(q_np), batch_size):
        end = min(start + batch_size, len(q_np))
        q = torch.as_tensor(q_np[start:end], device=device, dtype=torch.float32)
        contact_link_id = torch.as_tensor(contact_link_id_np[start:end], device=device, dtype=torch.long)
        contact_pos = torch.as_tensor(contact_pos_np[start:end], device=device, dtype=torch.float32)
        target_normal = torch.as_tensor(target_normal_np[start:end], device=device, dtype=torch.float32)
        contact_force_coeff = torch.as_tensor(
            contact_force_coeff_np[start:end], device=device, dtype=torch.float32
        )

        target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
        contact_force = target_normal * contact_force_coeff.unsqueeze(-1) * float(force_scale)

        jac = batch_jacobian(tree, q, ret_fk=True)
        n_link = tree.n_link()
        n_dof = tree.n_actuated_dof()
        J = jac["J"].view(q.shape[0], n_link, 6, n_dof)
        link_pose = jac["fk"]["link"]

        batch_idx = torch.arange(q.shape[0], device=device).view(-1, 1).expand_as(contact_link_id)
        J_contact = J[batch_idx, contact_link_id]
        link_pose_contact = link_pose[batch_idx, contact_link_id]

        contact_pos_in_hand = torch.matmul(
            link_pose_contact[..., :3, :3],
            contact_pos.unsqueeze(-1),
        ).squeeze(-1)
        J_point = J_contact[..., :3, :] + torch.cross(
            J_contact[..., 3:, :],
            contact_pos_in_hand.unsqueeze(-1),
            dim=2,
        )
        tau_des = torch.einsum("bkcd,bkc->bd", J_point, contact_force) * float(torque_sign)
        delta_unclamped = tau_des / stiffness
        if max_delta_rad > 0:
            delta = torch.clamp(delta_unclamped, -float(max_delta_rad), float(max_delta_rad))
        else:
            delta = delta_unclamped
        q_cmd = torch.minimum(torch.maximum(q + delta, lower), upper)
        delta = q_cmd - q

        q_out.append(q.detach().cpu())
        q_cmd_out.append(q_cmd.detach().cpu())
        delta_out.append(delta.detach().cpu())
        delta_unclamped_out.append(delta_unclamped.detach().cpu())
        tau_out.append(tau_des.detach().cpu())
        force_out.append(contact_force.detach().cpu())

    return {
        "qpos": torch.cat(q_out, dim=0).numpy().astype(np.float32),
        "q_cmd": torch.cat(q_cmd_out, dim=0).numpy().astype(np.float32),
        "q_cmd_delta": torch.cat(delta_out, dim=0).numpy().astype(np.float32),
        "q_cmd_delta_unclamped": torch.cat(delta_unclamped_out, dim=0).numpy().astype(np.float32),
        "tau_des": torch.cat(tau_out, dim=0).numpy().astype(np.float32),
        "contact_force_object_hand": torch.cat(force_out, dim=0).numpy().astype(np.float32),
    }


def write_readme(path, output_path, cache_path, metadata_path, usd_path, summary):
    text = f"""# Sharpa Wave Lightning q_cmd Cache

This file was generated offline from Lightning Grasp wrench metadata.

## Formula

For each contact, Lightning provides an object-side contact direction `n_i` and
a non-negative wrench coefficient `alpha_i`. This tool builds

```text
f_i = force_scale * alpha_i * n_i
tau_des = sum_i J_contact_i(q).T f_i
q_cmd = clamp(q + tau_des / Kp, joint_limits, max_delta_rad)
```

`Kp` comes from `drive:angular:physics:stiffness` in:

`{usd_path}`

## Files

- q_cmd NPZ: `{output_path}`
- source cache: `{cache_path}`
- source metadata: `{metadata_path}`

## Important Fields

- `qpos`: Lightning/Sharpa 22-DoF joint state, radians.
- `q_cmd`: position target inferred from wrench preload, radians.
- `q_cmd_delta`: clamped `q_cmd - qpos`, radians.
- `tau_des`: joint torque implied by the contact Jacobian and scaled contact forces.
- `contact_force_object_hand`: object-side contact forces in the Lightning hand frame.
- `joint_stiffness`: per-joint USD drive stiffness in `joint_names` order.
- `contact_force_coeff`: raw Lightning NNLS coefficients.
- `force_scale`, `max_delta_rad`, `torque_sign`: generation parameters.

The force scale is a relative calibration for Lightning's unit-normal wrench
coefficients, not an independently measured object mass/friction parameter.

## Source Summary

```json
{json.dumps(summary, indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--output_npz_path", type=Path, default=None)
    parser.add_argument("--readme_path", type=Path, default=None)
    parser.add_argument("--robot", default="sharpa_wave", choices=("sharpa_wave",))
    parser.add_argument(
        "--joint_limit_profile",
        choices=("urdf", "deploy", "usda09"),
        default="deploy",
        help="Sharpa joint-limit profile used to clamp q_cmd.",
    )
    parser.add_argument("--usd_path", type=Path, required=True)
    parser.add_argument("--force_scale", type=float, default=0.05)
    parser.add_argument("--max_delta_rad", type=float, default=0.12)
    parser.add_argument("--min_stiffness", type=float, default=1e-6)
    parser.add_argument("--torque_sign", type=float, choices=(1.0, -1.0), default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    cache_path = args.cache_path
    metadata_path = args.metadata_path or default_metadata_path(cache_path)
    output_path = args.output_npz_path or default_output_path(
        cache_path, args.force_scale, args.max_delta_rad
    )
    readme_path = args.readme_path or output_path.with_suffix(".readme.md")

    cache = np.load(cache_path).astype(np.float32)
    if cache.ndim != 2 or cache.shape[1] != 29:
        raise ValueError(f"Expected native Sharpa cache shape (N, 29), got {cache.shape}.")

    metadata = np.load(metadata_path, allow_pickle=False)
    require_metadata(
        metadata,
        (
            "contact_link_id",
            "contact_pos",
            "target_normal",
            "contact_force_coeff",
        ),
    )

    robot = build_robot(
        args.robot,
        joint_limit_profile=args.joint_limit_profile,
    )
    tree = build_kinematics_tree(robot.urdf_path, robot.get_active_joints())
    joint_lower, joint_upper = tree.get_actuated_joint_limit()
    stiffness = parse_usd_joint_drive_stiffness(args.usd_path, ACTIVE_JOINTS)

    qcmd = compute_qcmd(
        tree,
        cache[:, :22],
        metadata["contact_link_id"],
        metadata["contact_pos"],
        metadata["target_normal"],
        metadata["contact_force_coeff"],
        stiffness,
        joint_lower,
        joint_upper,
        force_scale=args.force_scale,
        max_delta_rad=args.max_delta_rad,
        min_stiffness=args.min_stiffness,
        torque_sign=args.torque_sign,
        device=args.device,
        batch_size=args.batch_size,
    )

    summary, summary_path = read_summary(metadata_path)
    source_args = summary.get("args", {})
    hand_root_pos = np.asarray(source_args.get("sharpa_hand_pos", [0.0, 0.0, 0.5]), dtype=np.float32)
    hand_root_quat = np.asarray(
        source_args.get("sharpa_hand_quat_wxyz", [0.7071068, 0.0, 0.7071068, 0.0]),
        dtype=np.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        native_cache=cache,
        qpos=qcmd["qpos"],
        q_cmd=qcmd["q_cmd"],
        q_cmd_reset=qcmd["q_cmd"],
        q_cmd_delta=qcmd["q_cmd_delta"],
        q_cmd_delta_unclamped=qcmd["q_cmd_delta_unclamped"],
        tau_des=qcmd["tau_des"],
        contact_force_object_hand=qcmd["contact_force_object_hand"],
        object_pos_w=cache[:, 22:25],
        object_quat_wxyz=cache[:, 25:29],
        hand_root_pos_w=hand_root_pos,
        hand_root_quat_wxyz=hand_root_quat,
        object_pose_hand=metadata["object_pose_hand"] if "object_pose_hand" in metadata.files else np.empty((len(cache), 4, 4), dtype=np.float32),
        joint_names=np.asarray(ACTIVE_JOINTS, dtype="U64"),
        joint_stiffness=stiffness,
        joint_lower=joint_lower.astype(np.float32),
        joint_upper=joint_upper.astype(np.float32),
        contact_link_id=metadata["contact_link_id"],
        contact_pos=metadata["contact_pos"],
        contact_normal=metadata["contact_normal"] if "contact_normal" in metadata.files else np.empty((len(cache), 0, 3), dtype=np.float32),
        target_pos=metadata["target_pos"] if "target_pos" in metadata.files else np.empty((len(cache), 0, 3), dtype=np.float32),
        target_normal=metadata["target_normal"],
        contact_force_coeff=metadata["contact_force_coeff"],
        contact_force_coeff_full=metadata["contact_force_coeff_full"] if "contact_force_coeff_full" in metadata.files else metadata["contact_force_coeff"],
        wrench_best_pivot_idx=metadata["wrench_best_pivot_idx"] if "wrench_best_pivot_idx" in metadata.files else np.empty((len(cache),), dtype=np.int64),
        wrench_best_sol=metadata["wrench_best_sol"] if "wrench_best_sol" in metadata.files else np.empty((len(cache), 0), dtype=np.float32),
        wrench_score=metadata["wrench_score"] if "wrench_score" in metadata.files else np.empty((len(cache),), dtype=np.float32),
        source_cache_path=np.asarray(str(cache_path), dtype="U512"),
        source_metadata_path=np.asarray(str(metadata_path), dtype="U512"),
        source_metadata_summary_path=np.asarray(str(summary_path), dtype="U512"),
        source_usd_path=np.asarray(str(args.usd_path), dtype="U512"),
        force_scale=np.asarray(args.force_scale, dtype=np.float32),
        max_delta_rad=np.asarray(args.max_delta_rad, dtype=np.float32),
        torque_sign=np.asarray(args.torque_sign, dtype=np.float32),
    )

    delta_abs = np.abs(qcmd["q_cmd_delta"])
    tau_abs = np.abs(qcmd["tau_des"])
    readme_summary = {
        "num_entries": int(len(cache)),
        "force_scale": float(args.force_scale),
        "max_delta_rad": float(args.max_delta_rad),
        "torque_sign": float(args.torque_sign),
        "q_cmd_delta_abs_max": float(delta_abs.max(initial=0.0)),
        "q_cmd_delta_abs_mean": float(delta_abs.mean() if delta_abs.size else 0.0),
        "tau_des_abs_max": float(tau_abs.max(initial=0.0)),
        "source_num_cache_entries": summary.get("num_cache_entries"),
        "source_args": source_args,
    }
    write_readme(readme_path, output_path, cache_path, metadata_path, args.usd_path, readme_summary)

    print(f"Saved q_cmd NPZ: {output_path}")
    print(f"Saved README: {readme_path}")
    print(json.dumps(readme_summary, indent=2))


if __name__ == "__main__":
    main()
