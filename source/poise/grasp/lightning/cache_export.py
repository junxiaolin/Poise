# Derived from Lightning Grasp by Zhao-Heng Yin; CC BY-NC 4.0.
# See licenses/Lightning-Grasp.txt. POISE changes: Sharpa-only, portable paths.
import os
import numpy as np

def quat_wxyz_to_matrix(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat_wxyz(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    quat = np.empty((matrix.shape[0], 4), dtype=np.float64)

    for i, rot in enumerate(matrix):
        trace = np.trace(rot)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quat[i, 0] = 0.25 * s
            quat[i, 1] = (rot[2, 1] - rot[1, 2]) / s
            quat[i, 2] = (rot[0, 2] - rot[2, 0]) / s
            quat[i, 3] = (rot[1, 0] - rot[0, 1]) / s
        else:
            axis = np.argmax(np.diag(rot))
            if axis == 0:
                s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                quat[i, 0] = (rot[2, 1] - rot[1, 2]) / s
                quat[i, 1] = 0.25 * s
                quat[i, 2] = (rot[0, 1] + rot[1, 0]) / s
                quat[i, 3] = (rot[0, 2] + rot[2, 0]) / s
            elif axis == 1:
                s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                quat[i, 0] = (rot[0, 2] - rot[2, 0]) / s
                quat[i, 1] = (rot[0, 1] + rot[1, 0]) / s
                quat[i, 2] = 0.25 * s
                quat[i, 3] = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                quat[i, 0] = (rot[1, 0] - rot[0, 1]) / s
                quat[i, 1] = (rot[0, 2] + rot[2, 0]) / s
                quat[i, 2] = (rot[1, 2] + rot[2, 1]) / s
                quat[i, 3] = 0.25 * s

    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


def save_sharpa_rl_cache(result, cache_path, hand_pos, hand_quat_wxyz):
    q = result["q"].detach().cpu().numpy()
    object_pose = result["object_pose"].detach().cpu().numpy()

    if q.shape[1] != 22:
        raise ValueError(f"Sharpa RL cache export expects 22 hand DOFs, got {q.shape[1]}.")

    world_from_hand = np.eye(4, dtype=np.float64)
    world_from_hand[:3, :3] = quat_wxyz_to_matrix(hand_quat_wxyz)
    world_from_hand[:3, 3] = np.asarray(hand_pos, dtype=np.float64)

    world_from_object = world_from_hand[None] @ object_pose
    object_pos = world_from_object[:, :3, 3]
    object_quat = matrix_to_quat_wxyz(world_from_object[:, :3, :3])
    cache = np.concatenate([q, object_pos, object_quat], axis=1).astype(np.float32)

    output_dir = os.path.dirname(cache_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(cache_path, cache)
    print(f"Saved Sharpa RL cache: {cache_path} {cache.shape}")


def get_external_force_hand(args):
    if not args.gravity_aware:
        return None

    gravity_world = np.asarray(args.gravity_world, dtype=np.float64)
    norm = np.linalg.norm(gravity_world)
    if norm < 1e-8:
        raise ValueError("gravity_world must be non-zero when --gravity_aware is set.")

    gravity_world = gravity_world / norm * args.gravity_force_scale
    world_from_hand_rot = quat_wxyz_to_matrix(args.sharpa_hand_quat_wxyz)
    return world_from_hand_rot.T @ gravity_world
