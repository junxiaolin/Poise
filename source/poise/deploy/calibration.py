"""Rigid-transform averaging and uncertainty helpers for hand-camera calibration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def as_transform(value: np.ndarray) -> np.ndarray:
    """Return a validated float64 homogeneous transform."""
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}.")
    if not np.all(np.isfinite(transform)):
        raise ValueError("Transform contains non-finite values.")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-9):
        raise ValueError("Transform has an invalid homogeneous last row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("Transform rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError("Transform rotation determinant is not +1.")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    transform = as_transform(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert an SO(3) matrix to a normalized, positive-w quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation, got {rotation.shape}.")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        raise ValueError("Cannot convert a degenerate rotation to a quaternion.")
    quat /= norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle logarithm of an SO(3) matrix in radians."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cosine))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    sine = math.sin(angle)
    if abs(sine) < 1.0e-8:
        # Calibration residuals should never approach pi, but this keeps the
        # failure explicit instead of silently returning an unstable vector.
        raise ValueError("Rotation logarithm is unstable near 180 degrees.")
    return angle * vee / (2.0 * sine)


def mean_transform(transforms: np.ndarray) -> np.ndarray:
    """Compute a translation mean and SVD-projected chordal rotation mean."""
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or transforms.shape[0] == 0:
        raise ValueError(f"Expected non-empty Nx4x4 transforms, got {transforms.shape}.")
    for transform in transforms:
        as_transform(transform)
    rotation_sum = transforms[:, :3, :3].sum(axis=0)
    left, _, right_t = np.linalg.svd(rotation_sum)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(left @ right_t)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = left @ correction @ right_t
    result[:3, 3] = transforms[:, :3, 3].mean(axis=0)
    return result


def transform_dispersion(transforms: np.ndarray, average: np.ndarray | None = None) -> dict[str, Any]:
    """Summarize translational and SO(3) residual dispersion around a mean."""
    transforms = np.asarray(transforms, dtype=np.float64)
    average = mean_transform(transforms) if average is None else as_transform(average)
    sample_count = int(transforms.shape[0])
    ddof = 1 if sample_count > 1 else 0

    translation_residuals_m = transforms[:, :3, 3] - average[:3, 3]
    translation_std_axis_m = np.std(translation_residuals_m, axis=0, ddof=ddof)
    translation_norms_m = np.linalg.norm(translation_residuals_m, axis=1)

    rotation_residuals_rad = np.stack(
        [rotation_log_vector(average[:3, :3].T @ transform[:3, :3]) for transform in transforms],
        axis=0,
    )
    rotation_std_axis_rad = np.std(rotation_residuals_rad, axis=0, ddof=ddof)
    rotation_norms_rad = np.linalg.norm(rotation_residuals_rad, axis=1)

    return {
        "translation_std_axis_mm": (translation_std_axis_m * 1000.0).tolist(),
        "translation_std_3d_mm": float(np.linalg.norm(translation_std_axis_m) * 1000.0),
        "translation_rms_3d_mm": float(np.sqrt(np.mean(translation_norms_m**2)) * 1000.0),
        "translation_max_3d_mm": float(np.max(translation_norms_m) * 1000.0),
        "rotation_std_axis_deg": np.degrees(rotation_std_axis_rad).tolist(),
        "rotation_std_3d_deg": float(np.linalg.norm(np.degrees(rotation_std_axis_rad))),
        "rotation_rms_3d_deg": float(np.degrees(np.sqrt(np.mean(rotation_norms_rad**2)))),
        "rotation_max_3d_deg": float(np.degrees(np.max(rotation_norms_rad))),
    }
