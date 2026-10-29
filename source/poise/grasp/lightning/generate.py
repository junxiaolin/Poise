# Derived from Lightning Grasp by Zhao-Heng Yin; CC BY-NC 4.0.
# See licenses/Lightning-Grasp.txt. POISE changes: Sharpa-only, portable paths.
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


from poise.grasp.lightning.cache_export import get_external_force_hand, save_sharpa_rl_cache
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix, sample_independent_set
from lygra.kinematics import build_kinematics_tree
from lygra.memory import IKGPUBufferPool
from lygra.mesh import get_urdf_mesh, get_urdf_mesh_decomposed, get_urdf_mesh_for_projection
from lygra.mesh_analyzer import get_support_point_mask
from lygra.pipeline.module.collision import batch_object_hand_collision_check
from lygra.pipeline.module.contact_collection import sample_from_mask_and_gather
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
from lygra.pipeline.module.kinematics import batch_contact_adjustment, batch_ik
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter
from poise.grasp.lightning.hand import build_robot
from lygra.utils.geom_utils import MeshObject


def inclusive_axis_values(low, high, step):
    values = []
    value = low
    eps = abs(step) * 1e-5
    while value <= high + eps:
        values.append(value)
        value += step

    if values[-1] > high + eps:
        values.pop()
    if values[-1] < high - eps:
        values.append(high)
    return np.asarray(values, dtype=np.float32)


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def build_pose_grid(args):
    position_xyz = None
    if args.position_xyz_file is not None:
        position_path = Path(args.position_xyz_file)
        if position_path.suffix == ".npy":
            position_xyz = np.load(position_path)
        else:
            position_xyz = np.loadtxt(position_path, dtype=np.float32, ndmin=2)
        position_xyz = np.asarray(position_xyz, dtype=np.float32)
        if position_xyz.ndim != 2 or position_xyz.shape[1] != 3 or len(position_xyz) == 0:
            raise ValueError(f"--position_xyz_file must contain a non-empty Nx3 array, got {position_xyz.shape}")
        if not np.isfinite(position_xyz).all():
            raise ValueError("--position_xyz_file contains NaN or Inf")
        x_values = np.unique(position_xyz[:, 0])
        y_values = np.unique(position_xyz[:, 1])
        z_values = np.unique(position_xyz[:, 2])
    else:
        x_values = inclusive_axis_values(args.x_range[0], args.x_range[1], args.x_step)
        y_values = inclusive_axis_values(args.y_range[0], args.y_range[1], args.y_step)
        z_values = inclusive_axis_values(args.z_range[0], args.z_range[1], args.z_step)
    roll_values = np.deg2rad(np.asarray(args.roll_values_deg, dtype=np.float32))
    pitch_values = np.deg2rad(np.asarray(args.pitch_values_deg, dtype=np.float32))
    yaw_values = np.deg2rad(np.asarray(args.yaw_values_deg, dtype=np.float32))

    poses = []
    table = []
    grid_index = 0
    if position_xyz is None:
        position_rows = [
            (ix, iy, iz, (ix * len(y_values) + iy) * len(z_values) + iz, x, y, z)
            for ix, x in enumerate(x_values)
            for iy, y in enumerate(y_values)
            for iz, z in enumerate(z_values)
        ]
    else:
        # Explicit positions are intentionally not interpreted as a Cartesian product.
        # Physical xyz is preserved in metadata; downstream lineage reconstructs the
        # frozen workspace cell from xyz rather than relying on these local indices.
        position_rows = [(-1, -1, -1, i, *xyz.tolist()) for i, xyz in enumerate(position_xyz)]
    for ix, iy, iz, position_index, x, y, z in position_rows:
                for ir, roll in enumerate(roll_values):
                    for ip, pitch in enumerate(pitch_values):
                        for iyaw, yaw in enumerate(yaw_values):
                            rpy_index = (ir * len(pitch_values) + ip) * len(yaw_values) + iyaw
                            pose = np.eye(4, dtype=np.float32)
                            pose[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
                            pose[:3, 3] = (x, y, z)
                            poses.append(pose)
                            table.append(
                                (
                                    grid_index,
                                    position_index,
                                    rpy_index,
                                    ix,
                                    iy,
                                    iz,
                                    ir,
                                    ip,
                                    iyaw,
                                    x,
                                    y,
                                    z,
                                    roll,
                                    pitch,
                                    yaw,
                                )
                            )
                            grid_index += 1

    return np.stack(poses, axis=0), np.asarray(table, dtype=np.float32), {
        "x_values": x_values,
        "y_values": y_values,
        "z_values": z_values,
        "roll_values": roll_values,
        "pitch_values": pitch_values,
        "yaw_values": yaw_values,
    }


def strict_sample_pose_and_contact_from_interaction(
    n_contact,
    interaction_matrix,
    dependency_matrix,
    object_points,
    object_normals,
    object_poses,
    condition,
    contact_domain_resolution=500,
    min_contact_domain_size=5,
):
    interaction_score = interaction_matrix.sum(dim=-1)
    initial_mask = torch.zeros_like(interaction_score)
    initial_mask[torch.where(interaction_score > min_contact_domain_size)] = 1

    contact_ids = sample_independent_set(n_contact, initial_mask, dependency_matrix)
    valid_batch_idx = torch.where((contact_ids >= 0).all(dim=-1))[0]
    if valid_batch_idx.numel() == 0:
        return None

    contact_ids = contact_ids[valid_batch_idx]
    interaction_matrix = interaction_matrix[valid_batch_idx]
    object_poses = object_poses[valid_batch_idx]
    filtered_condition = {k: v[valid_batch_idx] for k, v in condition.items()}

    batch_indices = torch.arange(contact_ids.size(0), device=contact_ids.device).unsqueeze(1)
    batch_indices = batch_indices.expand(-1, contact_ids.size(1))
    contact_domain_mask = interaction_matrix[batch_indices, contact_ids]
    if not contact_domain_mask.any(dim=-1).all():
        return None

    gathered, contact_domain_idx = sample_from_mask_and_gather(
        contact_domain_mask,
        torch.cat((object_points, object_normals), dim=-1),
        contact_domain_resolution,
    )

    return (
        gathered[..., :3],
        gathered[..., 3:],
        contact_domain_idx,
        object_poses,
        contact_ids,
        filtered_condition,
        valid_batch_idx,
    )


def subset_tensor_result(result, keep_idx):
    subset = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == len(result["q"]):
            subset[k] = v[keep_idx]
        else:
            subset[k] = v
    return subset


def append_result(collected, result):
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            collected.setdefault(k, []).append(v.detach().cpu())


def cap_per_grid_pose(result, keep_per_grid_pose):
    if len(result["q"]) == 0 or keep_per_grid_pose is None or keep_per_grid_pose <= 0:
        return result

    grid_np = result["grid_index"].detach().cpu().numpy()
    order = np.argsort(grid_np, kind="stable")
    counts = {}
    keep = []
    for idx in order:
        grid_index = int(grid_np[idx])
        count = counts.get(grid_index, 0)
        if count < keep_per_grid_pose:
            keep.append(idx)
            counts[grid_index] = count + 1

    keep_idx = torch.as_tensor(keep, device=result["q"].device, dtype=torch.long)
    return subset_tensor_result(result, keep_idx)


def sort_results_by_grid(result):
    if len(result["q"]) == 0:
        return result

    grid_np = result["grid_index"].detach().cpu().numpy()
    if "pass_index" in result:
        pass_np = result["pass_index"].detach().cpu().numpy()
        order = np.lexsort((pass_np, grid_np))
    else:
        order = np.argsort(grid_np, kind="stable")
    order_t = torch.from_numpy(order).long()
    return subset_tensor_result(result, order_t)


def balance_results_by_position(
    result,
    grid_table,
    max_per_position=None,
    target_total_grasps=None,
    seed=0,
):
    if len(result["q"]) == 0:
        return result
    if max_per_position is None and target_total_grasps is None:
        return result

    grid_np = result["grid_index"].detach().cpu().numpy().astype(np.int64)
    position_np = grid_table[grid_np, 1].astype(np.int64)
    rng = np.random.default_rng(seed)

    groups = {}
    for result_idx, position_idx in enumerate(position_np):
        groups.setdefault(int(position_idx), []).append(result_idx)

    for indices in groups.values():
        rng.shuffle(indices)

    selected = []
    selected_per_position = {position_idx: 0 for position_idx in groups}
    position_order = sorted(groups)
    while True:
        progressed = False
        for position_idx in position_order:
            if target_total_grasps is not None and len(selected) >= target_total_grasps:
                break
            if max_per_position is not None and selected_per_position[position_idx] >= max_per_position:
                continue
            if not groups[position_idx]:
                continue

            selected.append(groups[position_idx].pop(0))
            selected_per_position[position_idx] += 1
            progressed = True

        if not progressed:
            break
        if target_total_grasps is not None and len(selected) >= target_total_grasps:
            break

    selected_idx = torch.as_tensor(selected, dtype=torch.long)
    return subset_tensor_result(result, selected_idx)


def merge_results(collected, n_dof):
    if not collected:
        return {
            "q": torch.empty((0, n_dof), dtype=torch.float32),
            "object_pose": torch.empty((0, 4, 4), dtype=torch.float32),
            "grid_index": torch.empty((0,), dtype=torch.long),
        }

    return {k: torch.cat(v, dim=0) for k, v in collected.items()}


def write_metadata(metadata_path, result, grid_table, axis_values, output_path, args, cache):
    grid_index = result["grid_index"].detach().cpu().numpy().astype(np.int64)
    rows = grid_table[grid_index].copy() if len(grid_index) > 0 else np.empty((0, grid_table.shape[1]))
    names = [
        "grid_index",
        "position_index",
        "rpy_index",
        "x_index",
        "y_index",
        "z_index",
        "roll_index",
        "pitch_index",
        "yaw_index",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    ]
    metadata = {name: rows[:, i] for i, name in enumerate(names)}
    metadata["grid_index"] = metadata["grid_index"].astype(np.int64)
    for name in names[1:9]:
        metadata[name] = metadata[name].astype(np.int64)

    object_pose_hand = result["object_pose"].detach().cpu().numpy().astype(np.float32)
    metadata["object_pose_hand"] = object_pose_hand
    metadata["cache_object_pos"] = cache[:, 22:25].astype(np.float32)
    metadata["cache_object_quat_wxyz"] = cache[:, 25:29].astype(np.float32)
    metadata["x_values"] = axis_values["x_values"]
    metadata["y_values"] = axis_values["y_values"]
    metadata["z_values"] = axis_values["z_values"]
    metadata["roll_values"] = axis_values["roll_values"]
    metadata["pitch_values"] = axis_values["pitch_values"]
    metadata["yaw_values"] = axis_values["yaw_values"]

    if "pass_index" in result:
        metadata["pass_index"] = result["pass_index"].detach().cpu().numpy().astype(np.int64)
    if "candidate_contact_ids" in result:
        metadata["candidate_contact_ids"] = (
            result["candidate_contact_ids"].detach().cpu().numpy().astype(np.int64)
        )
    int_metadata_keys = (
        "contact_link_id",
        "wrench_best_pivot_idx",
    )
    float_metadata_keys = (
        "q_mask",
        "contact_pos",
        "contact_normal",
        "target_pos",
        "target_normal",
        "contact_force_coeff",
        "contact_force_coeff_full",
        "wrench_best_sol",
        "wrench_score",
    )
    for key in int_metadata_keys:
        if key in result:
            metadata[key] = result[key].detach().cpu().numpy().astype(np.int64)
    for key in float_metadata_keys:
        if key in result:
            metadata[key] = result[key].detach().cpu().numpy().astype(np.float32)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metadata_path, **metadata)

    summary_path = metadata_path.with_suffix(".json")
    summary = {
        "cache_path": str(output_path),
        "metadata_path": str(metadata_path),
        "num_cache_entries": int(len(cache)),
        "num_grid_poses": int(len(grid_table)),
        "x_values": axis_values["x_values"].tolist(),
        "y_values": axis_values["y_values"].tolist(),
        "z_values": axis_values["z_values"].tolist(),
        "roll_values_deg": np.rad2deg(axis_values["roll_values"]).tolist(),
        "pitch_values_deg": np.rad2deg(axis_values["pitch_values"]).tolist(),
        "yaw_values_deg": np.rad2deg(axis_values["yaw_values"]).tolist(),
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Sharpa RL cache from a deterministic object pose grid.")
    parser.add_argument("--robot", default="sharpa_wave", choices=("sharpa_wave",))
    parser.add_argument(
        "--joint_limit_profile",
        choices=("urdf", "deploy", "usda09"),
        default="deploy",
        help=(
            "Sharpa joint-limit profile used by contact sampling, IK, and filtering. "
            "'usda09' uses sharpa-rl-lab USDA limits with each endpoint scaled by 0.9."
        ),
    )
    parser.add_argument("--object_mesh_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path", default=None)

    parser.add_argument(
        "--position_xyz_file",
        default=None,
        help="Optional .npy or whitespace-delimited Nx3 xyz list. When set, positions are used exactly and are not expanded as a Cartesian product.",
    )
    parser.add_argument("--x_range", type=float, nargs=2, default=(0.030, 0.110))
    parser.add_argument("--y_range", type=float, nargs=2, default=(-0.050, 0.050))
    parser.add_argument("--z_range", type=float, nargs=2, default=(0.050, 0.140))
    parser.add_argument("--x_step", type=float, default=0.020)
    parser.add_argument("--y_step", type=float, default=0.025)
    parser.add_argument("--z_step", type=float, default=0.020)
    parser.add_argument("--roll_values_deg", type=float, nargs="+", default=(-45.0, 0.0, 45.0))
    parser.add_argument("--pitch_values_deg", type=float, nargs="+", default=(-45.0, 0.0, 45.0))
    parser.add_argument("--yaw_values_deg", type=float, nargs="+", default=(-45.0, 0.0, 45.0))
    parser.add_argument("--max_grid_poses", type=int, default=None, help="Limit processed grid poses for smoke tests.")

    parser.add_argument("--pose_batch_size", type=int, default=96)
    parser.add_argument("--batch_size_inner", type=int, default=64)
    parser.add_argument("--n_contact", type=int, default=5)
    parser.add_argument("--n_sample_point", type=int, default=1536)
    parser.add_argument("--ik_finetune_iter", type=int, default=3)
    parser.add_argument("--contact_domain_resolution", type=int, default=500)
    parser.add_argument("--min_contact_domain_size", type=int, default=5)
    parser.add_argument("--zo_lr_sigma", type=float, default=5.0)
    parser.add_argument("--cf_accel", default="lbvhs2")
    parser.add_argument("--n_assign_retry", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_passes", type=int, default=1, help="Repeat the grid with different random seeds.")
    parser.add_argument("--keep_per_grid_pose", type=int, default=1, help="Maximum grasps to keep per exact grid pose.")
    parser.add_argument("--balance_by_position", action="store_true", help="Select final grasps round-robin over xyz position bins.")
    parser.add_argument("--max_per_position", type=int, default=None, help="Maximum final grasps per xyz position bin.")
    parser.add_argument("--target_total_grasps", type=int, default=None, help="Stop balanced final selection at this many grasps.")
    parser.add_argument("--balanced_selection_seed", type=int, default=0)

    parser.add_argument("--sharpa_hand_pos", type=float, nargs=3, default=(0.0, 0.0, 0.5))
    parser.add_argument("--sharpa_hand_quat_wxyz", type=float, nargs=4, default=(0.819152, 0.0, -0.5735764, 0.0))
    parser.add_argument("--sharpa_palm_down", action="store_true", default=True)
    parser.add_argument("--no_sharpa_palm_down", dest="sharpa_palm_down", action="store_false")

    parser.add_argument("--gravity_aware", action="store_true", default=True)
    parser.add_argument("--no_gravity_aware", dest="gravity_aware", action="store_false")
    parser.add_argument("--gravity_world", type=float, nargs=3, default=(0.0, 0.0, -1.0))
    parser.add_argument("--gravity_force_scale", type=float, default=0.05)
    parser.add_argument("--gravity_force_point_object", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()

    if args.sharpa_palm_down:
        args.sharpa_hand_quat_wxyz = (0.5735764, 0.0, 0.819152, 0.0)
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_suffix(".metadata.npz")

    grid_poses_np, grid_table, axis_values = build_pose_grid(args)
    if args.max_grid_poses is not None:
        grid_poses_np = grid_poses_np[: args.max_grid_poses]
        grid_table = grid_table[: args.max_grid_poses]

    print(f"Grid poses: {len(grid_poses_np)}")
    print(f"x: {axis_values['x_values']}")
    print(f"y: {axis_values['y_values']}")
    print(f"z: {axis_values['z_values']}")
    print(f"roll/pitch/yaw deg: {args.roll_values_deg} {args.pitch_values_deg} {args.yaw_values_deg}")

    robot = build_robot(
        args.robot,
        joint_limit_profile=args.joint_limit_profile,
    )
    tree = build_kinematics_tree(robot.urdf_path, robot.get_active_joints())
    default_collision_q = robot.get_resting_q(tree)

    mesh_data_for_ik = get_urdf_mesh_for_projection(
        urdf_path=robot.urdf_path,
        tree=tree,
        config=robot.get_contact_field_config(),
        mesh_scale=robot.get_mesh_scale(),
    )
    decomposed_static_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        override_link_names=robot.get_static_links(),
        mesh_scale=robot.get_mesh_scale(),
    )
    decomposed_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        mesh_scale=robot.get_mesh_scale(),
    )

    self_collision_link_pairs = tree.get_self_collision_check_link_pairs(
        link_body_id=decomposed_mesh_data["link_body_id"],
        whitelist_link=[],
        whitelist_pairs=robot.get_white_list_pairs(),
    )
    self_collision_link_pairs = torch.from_numpy(self_collision_link_pairs).cuda().int()

    contact_field = robot.get_contact_field()
    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])
    contact_parent_links = contact_field.get_all_parent_link_names()
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()
    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets).cuda()
    accel_structure = contact_field.generate_acceleration_structure(method=args.cf_accel)

    obj = MeshObject(args.object_mesh_path)
    object_area = obj.get_area()
    zo_lr = ((object_area / args.n_sample_point) ** 0.5) * args.zo_lr_sigma
    points_np, normals_np = obj.sample_point_and_normal(count=args.n_sample_point)
    points_all = torch.from_numpy(points_np).cuda().float()
    normals_all = torch.from_numpy(normals_np).cuda().float()
    support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0]
    points = points_all[torch.where(support_point_mask)]
    normals = normals_all[torch.where(support_point_mask)]

    gpu_memory_pool = IKGPUBufferPool(
        n_dof=tree.n_dof(),
        n_link=tree.n_link(),
        n_actuated_dof=tree.n_actuated_dof(),
        max_batch=min([args.pose_batch_size * args.batch_size_inner, 65536]),
        retry=10,
    )

    external_force_hand = get_external_force_hand(args)
    collected = {}
    stats = {
        "processed": 0,
        "collision_free": 0,
        "contact_domain_pose": 0,
        "contact_candidate": 0,
        "ik_success": 0,
        "finetune_success": 0,
        "postprocess_success": 0,
        "grid_cap_success": 0,
        "balanced_selected": 0,
    }

    with torch.no_grad():
        for pass_idx in range(args.n_passes):
            pass_seed = args.seed + pass_idx
            np.random.seed(pass_seed)
            torch.manual_seed(pass_seed)
            print(f"Pass {pass_idx + 1}/{args.n_passes}, seed={pass_seed}")

            for start in tqdm(range(0, len(grid_poses_np), args.pose_batch_size), desc="Grid batches"):
                end = min(start + args.pose_batch_size, len(grid_poses_np))
                object_poses = torch.from_numpy(grid_poses_np[start:end]).cuda().float()
                grid_indices = torch.arange(start, end, device=object_poses.device, dtype=torch.long)
                stats["processed"] += len(object_poses)

                points_expanded = points_all.unsqueeze(0).expand(object_poses.size(0), -1, -1)
                rotation = object_poses[:, :3, :3]
                translation = object_poses[:, :3, 3]
                point_transformed = torch.bmm(points_expanded, rotation.transpose(-1, -2)) + translation.unsqueeze(1)
                no_collision = batch_object_hand_collision_check(
                    tree=tree,
                    mesh=decomposed_static_mesh_data,
                    object_point=point_transformed,
                    default_q=default_collision_q,
                ).bool()
                keep = torch.where(no_collision)[0]
                if keep.numel() == 0:
                    continue

                object_poses = object_poses[keep]
                grid_indices = grid_indices[keep]
                stats["collision_free"] += len(object_poses)

                condition = {
                    "extra_contact_pos": torch.zeros((len(object_poses), 1, 3), device=object_poses.device),
                    "extra_contact_normal": torch.zeros((len(object_poses), 1, 3), device=object_poses.device),
                    "extra_contact_mask": torch.zeros((len(object_poses), 1), device=object_poses.device),
                }

                interaction_matrix_hand_point_idx = batch_object_all_contact_fields_interaction(
                    object_pos=points,
                    object_normal=normals,
                    object_pose=object_poses,
                    accel_structure=accel_structure,
                )
                interaction_matrix = (interaction_matrix_hand_point_idx >= 0).int()
                link_interaction_matrix = contact_field.reduce_link_interaction(interaction_matrix)

                sampled = strict_sample_pose_and_contact_from_interaction(
                    n_contact=args.n_contact,
                    interaction_matrix=link_interaction_matrix,
                    dependency_matrix=dependency_matrix,
                    object_points=points,
                    object_normals=normals,
                    object_poses=object_poses,
                    condition=condition,
                    contact_domain_resolution=args.contact_domain_resolution,
                    min_contact_domain_size=args.min_contact_domain_size,
                )
                if sampled is None:
                    continue

                (
                    contact_domain_pos,
                    contact_domain_normal,
                    contact_domain_point_idx,
                    valid_object_poses,
                    contact_link_ids,
                    valid_condition,
                    valid_outer_idx,
                ) = sampled
                valid_grid_indices = grid_indices[valid_outer_idx]
                stats["contact_domain_pose"] += len(valid_object_poses)

                target_contact_pos, target_contact_normal, target_contact_point_idx, \
                    valid_object_poses, target_contact_link_ids, target_batch_outer_ids, wrench_info = search_contact_point(
                        contact_domain_pos=contact_domain_pos,
                        contact_domain_normal=contact_domain_normal,
                        contact_domain_point_idx=contact_domain_point_idx,
                        object_poses=valid_object_poses,
                        contact_ids=contact_link_ids,
                        batch_size=args.batch_size_inner,
                        return_hand_frame=True,
                        condition=valid_condition,
                        zo_lr=zo_lr,
                        external_force_hand=external_force_hand,
                        external_force_point_object=args.gravity_force_point_object,
                        return_wrench_info=True,
                    )
                if len(target_contact_pos) == 0:
                    continue

                candidate_grid_indices = valid_grid_indices[target_batch_outer_ids]
                stats["contact_candidate"] += len(target_contact_pos)

                strict_interaction_matrix = interaction_matrix[valid_outer_idx]
                strict_interaction_matrix_hand_point_idx = interaction_matrix_hand_point_idx[valid_outer_idx]
                contact_ids, local_contact_ids = contact_field.sample_contact_ids(
                    interaction_matrix=strict_interaction_matrix,
                    interaction_matrix_hand_point_idx=strict_interaction_matrix_hand_point_idx,
                    target_batch_outer_ids=target_batch_outer_ids,
                    target_contact_link_ids=target_contact_link_ids,
                    target_contact_point_idx=target_contact_point_idx,
                )
                contact_pos_in_linkf, contact_normal_in_linkf = contact_field.sample_contact_geometry(
                    contact_ids, local_contact_ids
                )

                result = batch_ik(
                    tree=tree,
                    contact_ids=contact_ids,
                    contact_parent_ids=contact_parent_ids,
                    contact_pos_in_linkf=contact_pos_in_linkf.float(),
                    contact_normal_in_linkf=contact_normal_in_linkf.float(),
                    target_contact_pos=target_contact_pos.float(),
                    target_contact_normal=target_contact_normal.float(),
                    object_pose=valid_object_poses.float(),
                    gpu_memory_pool=gpu_memory_pool,
                    extra_result={
                        "grid_index": candidate_grid_indices,
                        "pass_index": torch.full_like(candidate_grid_indices, pass_idx),
                        "candidate_contact_ids": contact_ids,
                        "contact_force_coeff": wrench_info["contact_force_coeff"],
                        "contact_force_coeff_full": wrench_info["contact_force_coeff_full"],
                        "wrench_best_pivot_idx": wrench_info["wrench_best_pivot_idx"],
                        "wrench_best_sol": wrench_info["wrench_best_sol"],
                        "wrench_score": wrench_info["wrench_score"],
                    },
                )
                if len(result["q"]) == 0:
                    continue
                stats["ik_success"] += len(result["q"])

                result = batch_contact_adjustment(
                    tree=tree,
                    mesh=mesh_data_for_ik,
                    q_init=result["q"],
                    q_mask=result["q_mask"],
                    contact_ids=result["candidate_contact_ids"],
                    contact_link_ids=result["contact_link_id"],
                    contact_pos_in_linkf=result["contact_pos"],
                    contact_normal_in_linkf=result["contact_normal"],
                    target_contact_pos=result["target_pos"],
                    target_contact_normal=result["target_normal"],
                    object_pose=result["object_pose"],
                    n_iter=args.ik_finetune_iter,
                    gpu_memory_pool=gpu_memory_pool,
                    ret_mesh_buffer=True,
                    extra_result={
                        key: result[key]
                        for key in (
                            "grid_index",
                            "pass_index",
                            "candidate_contact_ids",
                            "contact_force_coeff",
                            "contact_force_coeff_full",
                            "wrench_best_pivot_idx",
                            "wrench_best_sol",
                            "wrench_score",
                        )
                        if key in result
                    },
                )
                if result is None or len(result["q"]) == 0:
                    continue
                stats["finetune_success"] += len(result["q"])

                result = batch_assign_free_finger_and_filter(
                    tree=tree,
                    result=result,
                    object_point=points_all,
                    self_collision_link_pairs=self_collision_link_pairs,
                    decomposed_mesh_data=decomposed_mesh_data,
                    n_assign_retry=args.n_assign_retry,
                )
                if len(result["q"]) == 0:
                    continue
                stats["postprocess_success"] += len(result["q"])

                result = cap_per_grid_pose(result, args.keep_per_grid_pose)
                stats["grid_cap_success"] += len(result["q"])
                append_result(collected, result)

    merged = merge_results(collected, tree.n_dof())
    merged = sort_results_by_grid(merged)
    if args.balance_by_position:
        merged = balance_results_by_position(
            merged,
            grid_table=grid_table,
            max_per_position=args.max_per_position,
            target_total_grasps=args.target_total_grasps,
            seed=args.balanced_selection_seed,
        )
        merged = sort_results_by_grid(merged)
    stats["balanced_selected"] = len(merged["q"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_sharpa_rl_cache(merged, str(output_path), args.sharpa_hand_pos, args.sharpa_hand_quat_wxyz)
    cache = np.load(output_path)
    summary_path = write_metadata(metadata_path, merged, grid_table, axis_values, output_path, args, cache)

    print("Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved metadata summary: {summary_path}")


if __name__ == "__main__":
    main()
