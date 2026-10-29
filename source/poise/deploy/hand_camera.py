"""AprilTag hand-camera estimation shared by deployment startup and monitoring."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .calibration import (
    invert_transform,
    mean_transform,
    rotation_log_vector,
    transform_dispersion,
)


CALIBRATION_SCHEMA_VERSION = "sharpa_hand_camera_apriltag_v2"

# Confirmed physical installation. T_A_B maps coordinates in B into frame A.
# +x_tag = +y_hand, +y_tag = +z_hand, +z_tag = +x_hand.
DEFAULT_T_HAND_TAG = np.array(
    [
        [0.0, 0.0, 1.0, 0.0050],
        [1.0, 0.0, 0.0, 0.0000],
        [0.0, 1.0, 0.0, -0.0795],
        [0.0, 0.0, 0.0, 1.0000],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TagPoseEstimate:
    timestamp_ms: float
    corners_px: np.ndarray
    t_camera_tag: np.ndarray
    t_hand_camera: np.ndarray
    reprojection_rms_px: float
    incidence_angle_deg: float


@dataclass(frozen=True)
class CalibrationThresholds:
    camera_tag_translation_std_mm: float = 0.30
    rotation_std_deg: float = 0.50
    reprojection_rms_px: float = 0.40
    first_last_translation_drift_mm: float = 0.50
    first_last_window_samples: int = 50
    minimum_detection_rate: float = 0.95

    def __post_init__(self) -> None:
        positive = (
            self.camera_tag_translation_std_mm,
            self.rotation_std_deg,
            self.reprojection_rms_px,
            self.first_last_translation_drift_mm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Calibration quality thresholds must be finite and positive.")
        if self.first_last_window_samples < 1:
            raise ValueError("first_last_window_samples must be positive.")
        if (
            not np.isfinite(self.minimum_detection_rate)
            or not 0.0 < self.minimum_detection_rate <= 1.0
        ):
            raise ValueError("minimum_detection_rate must be within (0, 1].")


@dataclass(frozen=True)
class HandCameraCalibration:
    t_hand_camera: np.ndarray
    t_camera_hand: np.ndarray
    mean_t_camera_tag: np.ndarray
    estimates: tuple[TagPoseEstimate, ...]
    predicted_tag_corners_px: np.ndarray
    metrics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.metrics["passed"])


def tag_object_points(tag_size_m: float) -> np.ndarray:
    """Return ArUco-ordered corners with +y_tag toward the printed top."""
    if tag_size_m <= 0.0:
        raise ValueError("tag_size_m must be positive.")
    half = tag_size_m * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def rvec_tvec_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(np.asarray(transform[:3, :3], dtype=np.float64))
    return rvec.reshape(3, 1), np.asarray(transform[:3, 3], dtype=np.float64).reshape(3, 1)


def reprojection_rms_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    residual = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def solve_tag_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Select the physically valid, lowest-error IPPE square solution."""
    result = cv2.solvePnPGeneric(
        object_points,
        np.asarray(image_points, dtype=np.float64).reshape(4, 2),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return None
    candidates: list[tuple[float, float, np.ndarray]] = []
    for rvec, tvec in zip(result[1], result[2], strict=True):
        transform = transform_from_rvec_tvec(rvec, tvec)
        if transform[2, 3] <= 0.0:
            continue
        tag_to_camera = -transform[:3, 3]
        tag_to_camera /= np.linalg.norm(tag_to_camera)
        normal_alignment = float(np.dot(transform[:3, 2], tag_to_camera))
        if normal_alignment <= 0.0:
            continue
        error = reprojection_rms_px(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        incidence_deg = math.degrees(math.acos(np.clip(normal_alignment, -1.0, 1.0)))
        candidates.append((error, incidence_deg, transform))
    if not candidates:
        return None
    error, incidence_deg, transform = min(candidates, key=lambda item: item[0])
    return transform, error, incidence_deg


class AprilTagHandCameraEstimator:
    """Estimate T_camera_tag and T_hand_camera from one BGR color frame."""

    def __init__(
        self,
        *,
        tag_id: int = 0,
        tag_size_m: float = 0.060,
        t_hand_tag: np.ndarray,
    ) -> None:
        if tag_id < 0:
            raise ValueError("tag_id must be non-negative.")
        self.tag_id = int(tag_id)
        self.tag_size_m = float(tag_size_m)
        self.t_hand_tag = np.asarray(t_hand_tag, dtype=np.float64).copy()
        self.object_points = tag_object_points(self.tag_size_m)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 1.0e-3
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def estimate(
        self,
        color_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        *,
        timestamp_ms: float,
    ) -> TagPoseEstimate | None:
        gray = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        id_list = ids.reshape(-1).tolist()
        if self.tag_id not in id_list:
            return None
        selected = corners[id_list.index(self.tag_id)].reshape(4, 2).astype(np.float64)
        solved = solve_tag_pose(
            self.object_points,
            selected,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        if solved is None:
            return None
        t_camera_tag, error_px, incidence_deg = solved
        return TagPoseEstimate(
            timestamp_ms=float(timestamp_ms),
            corners_px=selected,
            t_camera_tag=t_camera_tag,
            t_hand_camera=self.t_hand_tag @ invert_transform(t_camera_tag),
            reprojection_rms_px=error_px,
            incidence_angle_deg=incidence_deg,
        )


def aggregate_hand_camera_estimates(
    estimates: list[TagPoseEstimate] | tuple[TagPoseEstimate, ...],
    *,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    estimator: AprilTagHandCameraEstimator,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
    capture_attempts: int | None = None,
) -> HandCameraCalibration:
    """Average startup estimates and apply the deployment calibration gate."""
    estimates = tuple(estimates)
    if len(estimates) < 2:
        raise ValueError("At least two estimates are required.")
    if capture_attempts is None:
        capture_attempts = len(estimates)
    if capture_attempts < len(estimates):
        raise ValueError("capture_attempts cannot be smaller than accepted estimates.")
    camera_tag_samples = np.stack([item.t_camera_tag for item in estimates])
    hand_camera_samples = np.stack([item.t_hand_camera for item in estimates])
    corners = np.stack([item.corners_px for item in estimates])
    pnp_errors = np.asarray([item.reprojection_rms_px for item in estimates], dtype=np.float64)
    incidence_angles = np.asarray([item.incidence_angle_deg for item in estimates], dtype=np.float64)

    mean_t_camera_tag = mean_transform(camera_tag_samples)
    t_hand_camera = estimator.t_hand_tag @ invert_transform(mean_t_camera_tag)
    t_camera_hand = invert_transform(t_hand_camera)
    camera_tag_dispersion = transform_dispersion(
        camera_tag_samples,
        mean_t_camera_tag,
    )
    hand_camera_dispersion = transform_dispersion(
        hand_camera_samples,
        t_hand_camera,
    )
    first_last_window_samples = min(
        thresholds.first_last_window_samples,
        len(estimates) // 2,
    )
    first_translation = camera_tag_samples[
        :first_last_window_samples, :3, 3
    ].mean(axis=0)
    last_translation = camera_tag_samples[
        -first_last_window_samples:, :3, 3
    ].mean(axis=0)
    first_last_translation_drift_mm = float(
        np.linalg.norm(last_translation - first_translation) * 1000.0
    )
    detection_rate = float(len(estimates) / capture_attempts)

    rvec, tvec = rvec_tvec_from_transform(mean_t_camera_tag)
    predicted, _ = cv2.projectPoints(
        estimator.object_points,
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    predicted = predicted.reshape(4, 2)
    residuals = corners - predicted[None]
    frame_errors = np.sqrt(np.mean(np.sum(residuals**2, axis=2), axis=1))
    mean_pose_reprojection = float(np.sqrt(np.mean(np.sum(residuals**2, axis=2))))
    checks = {
        "camera_tag_translation_std": (
            camera_tag_dispersion["translation_std_3d_mm"]
            < thresholds.camera_tag_translation_std_mm
        ),
        "rotation_std": (
            camera_tag_dispersion["rotation_std_3d_deg"]
            < thresholds.rotation_std_deg
        ),
        "reprojection": mean_pose_reprojection < thresholds.reprojection_rms_px,
        "first_last_translation_drift": (
            first_last_translation_drift_mm
            < thresholds.first_last_translation_drift_mm
        ),
        "detection_rate": detection_rate >= thresholds.minimum_detection_rate,
    }
    metrics = {
        "sample_count": len(estimates),
        "capture_attempts": capture_attempts,
        "detection_rate": detection_rate,
        "camera_tag_dispersion": camera_tag_dispersion,
        "hand_camera_dispersion": hand_camera_dispersion,
        "first_last_translation_drift_mm": first_last_translation_drift_mm,
        "first_last_window_samples": first_last_window_samples,
        "pnp_reprojection_rms_px": {
            "mean": float(pnp_errors.mean()),
            "std": float(pnp_errors.std(ddof=1)),
            "max": float(pnp_errors.max()),
        },
        "mean_pose_reprojection_rms_px": mean_pose_reprojection,
        "mean_pose_reprojection_frame_std_px": float(frame_errors.std(ddof=1)),
        "mean_pose_reprojection_frame_max_px": float(frame_errors.max()),
        "tag_incidence_angle_deg": {
            "mean": float(incidence_angles.mean()),
            "std": float(incidence_angles.std(ddof=1)),
            "max": float(incidence_angles.max()),
        },
        "thresholds": {
            "camera_tag_translation_std_3d_mm": (
                thresholds.camera_tag_translation_std_mm
            ),
            "rotation_std_3d_deg": thresholds.rotation_std_deg,
            "mean_pose_reprojection_rms_px": thresholds.reprojection_rms_px,
            "first_last_translation_drift_mm": (
                thresholds.first_last_translation_drift_mm
            ),
            "first_last_window_samples": (
                thresholds.first_last_window_samples
            ),
            "minimum_detection_rate": thresholds.minimum_detection_rate,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    return HandCameraCalibration(
        t_hand_camera=t_hand_camera,
        t_camera_hand=t_camera_hand,
        mean_t_camera_tag=mean_t_camera_tag,
        estimates=estimates,
        predicted_tag_corners_px=predicted,
        metrics=metrics,
    )


def transform_delta(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return translation millimetres and geodesic rotation degrees."""
    translation_mm = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]) * 1000.0)
    rotation_rad = float(
        np.linalg.norm(rotation_log_vector(reference[:3, :3].T @ current[:3, :3]))
    )
    return translation_mm, math.degrees(rotation_rad)


@dataclass(frozen=True)
class DriftStatus:
    state: str
    translation_delta_mm: float | None
    rotation_delta_deg: float | None
    consecutive_violations: int
    missing_observations: int
    should_pause: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "translation_delta_mm": self.translation_delta_mm,
            "rotation_delta_deg": self.rotation_delta_deg,
            "consecutive_violations": self.consecutive_violations,
            "missing_observations": self.missing_observations,
            "should_pause": self.should_pause,
        }


class HandCameraDriftWatchdog:
    """Low-pass direct tag-pose drift checks with fail-closed gating."""

    def __init__(
        self,
        reference_t_camera_tag: np.ndarray,
        *,
        translation_threshold_mm: float = 2.0,
        rotation_threshold_deg: float = 0.5,
        window_size: int = 15,
        consecutive_limit: int = 5,
        missing_limit: int = 30,
    ) -> None:
        if window_size < 2 or consecutive_limit < 1 or missing_limit < 1:
            raise ValueError("Invalid watchdog window or limit.")
        self.reference = np.asarray(reference_t_camera_tag, dtype=np.float64).copy()
        self.translation_threshold_mm = float(translation_threshold_mm)
        self.rotation_threshold_deg = float(rotation_threshold_deg)
        self.window: deque[np.ndarray] = deque(maxlen=int(window_size))
        self.consecutive_limit = int(consecutive_limit)
        self.missing_limit = int(missing_limit)
        self.consecutive_violations = 0
        self.missing_observations = 0

    def update(self, estimate: TagPoseEstimate | None) -> DriftStatus:
        if estimate is None:
            self.missing_observations += 1
            should_pause = self.missing_observations >= self.missing_limit
            return DriftStatus(
                state="tag_missing" if should_pause else "waiting_for_tag",
                translation_delta_mm=None,
                rotation_delta_deg=None,
                consecutive_violations=self.consecutive_violations,
                missing_observations=self.missing_observations,
                should_pause=should_pause,
            )

        self.missing_observations = 0
        self.window.append(estimate.t_camera_tag)
        if len(self.window) < self.window.maxlen:
            return DriftStatus(
                state="warming_up",
                translation_delta_mm=None,
                rotation_delta_deg=None,
                consecutive_violations=0,
                missing_observations=0,
                should_pause=False,
            )

        smoothed = mean_transform(np.stack(tuple(self.window)))
        translation_mm, rotation_deg = transform_delta(self.reference, smoothed)
        violated = (
            translation_mm > self.translation_threshold_mm
            or rotation_deg > self.rotation_threshold_deg
        )
        self.consecutive_violations = self.consecutive_violations + 1 if violated else 0
        should_pause = self.consecutive_violations >= self.consecutive_limit
        return DriftStatus(
            state="recalibrate" if should_pause else ("warning" if violated else "ok"),
            translation_delta_mm=translation_mm,
            rotation_delta_deg=rotation_deg,
            consecutive_violations=self.consecutive_violations,
            missing_observations=0,
            should_pause=should_pause,
        )
