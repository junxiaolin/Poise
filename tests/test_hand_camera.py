from __future__ import annotations

import unittest

import cv2
import numpy as np

from poise.deploy.hand_camera import (
    DEFAULT_T_HAND_TAG,
    AprilTagHandCameraEstimator,
    CalibrationThresholds,
    HandCameraDriftWatchdog,
    TagPoseEstimate,
    aggregate_hand_camera_estimates,
)
from poise.deploy.calibration import invert_transform


class SharpaPose6DHandCameraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [
                [900.0, 0.0, 640.0],
                [0.0, 900.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion = np.zeros(5, dtype=np.float64)
        self.estimator = AprilTagHandCameraEstimator(t_hand_tag=DEFAULT_T_HAND_TAG)

    def make_estimates(self) -> list[TagPoseEstimate]:
        base = np.eye(4, dtype=np.float64)
        base[:3, :3] = np.diag([1.0, -1.0, -1.0])
        base[:3, 3] = np.array([0.02, 0.08, 0.45])
        estimates = []
        for index in range(30):
            phase = 2.0 * np.pi * index / 30.0
            current = base.copy()
            current[:3, 3] += np.array(
                [np.sin(phase), np.cos(phase), np.sin(2.0 * phase)]
            ) * 5.0e-5
            rvec, _ = cv2.Rodrigues(current[:3, :3])
            corners, _ = cv2.projectPoints(
                self.estimator.object_points,
                rvec,
                current[:3, 3],
                self.camera_matrix,
                self.distortion,
            )
            estimates.append(
                TagPoseEstimate(
                    timestamp_ms=float(index),
                    corners_px=corners.reshape(4, 2),
                    t_camera_tag=current,
                    t_hand_camera=DEFAULT_T_HAND_TAG @ invert_transform(current),
                    reprojection_rms_px=0.05,
                    incidence_angle_deg=0.0,
                )
            )
        return estimates

    def test_aggregate_passes_quality_gate(self) -> None:
        self.assertEqual(CalibrationThresholds().rotation_std_deg, 0.50)
        result = aggregate_hand_camera_estimates(
            self.make_estimates(),
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            estimator=self.estimator,
            thresholds=CalibrationThresholds(),
        )
        self.assertTrue(result.passed)
        self.assertLess(
            result.metrics["camera_tag_dispersion"]["translation_std_3d_mm"],
            0.30,
        )
        self.assertLess(result.metrics["mean_pose_reprojection_rms_px"], 0.40)

    def test_gate_does_not_count_rotation_lever_arm_as_translation_noise(
        self,
    ) -> None:
        estimates = self.make_estimates()
        for index, estimate in enumerate(estimates):
            angle_rad = np.deg2rad(
                0.15 * np.sin(2.0 * np.pi * index / len(estimates))
            )
            rotation_delta, _ = cv2.Rodrigues(
                np.array([angle_rad, 0.0, 0.0], dtype=np.float64)
            )
            estimate.t_camera_tag[:3, :3] = (
                rotation_delta @ estimate.t_camera_tag[:3, :3]
            )
            estimate.t_hand_camera[:] = (
                DEFAULT_T_HAND_TAG
                @ invert_transform(estimate.t_camera_tag)
            )

        result = aggregate_hand_camera_estimates(
            estimates,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            estimator=self.estimator,
        )

        self.assertTrue(result.passed)
        self.assertLess(
            result.metrics["camera_tag_dispersion"]["translation_std_3d_mm"],
            0.30,
        )
        self.assertGreater(
            result.metrics["hand_camera_dispersion"]["translation_std_3d_mm"],
            0.30,
        )

    def test_gate_rejects_first_last_translation_drift(self) -> None:
        estimates = self.make_estimates()
        for index, estimate in enumerate(estimates):
            estimate.t_camera_tag[0, 3] += (
                -0.0003 if index < 5 else 0.0003
            )
            estimate.t_hand_camera[:] = (
                DEFAULT_T_HAND_TAG
                @ invert_transform(estimate.t_camera_tag)
            )

        result = aggregate_hand_camera_estimates(
            estimates,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            estimator=self.estimator,
            thresholds=CalibrationThresholds(
                first_last_window_samples=5,
            ),
        )

        self.assertTrue(result.metrics["checks"]["camera_tag_translation_std"])
        self.assertFalse(
            result.metrics["checks"]["first_last_translation_drift"]
        )
        self.assertFalse(result.passed)

    def test_gate_rejects_low_detection_rate(self) -> None:
        estimates = self.make_estimates()
        result = aggregate_hand_camera_estimates(
            estimates,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            estimator=self.estimator,
            capture_attempts=40,
        )
        self.assertAlmostEqual(result.metrics["detection_rate"], 0.75)
        self.assertFalse(result.metrics["checks"]["detection_rate"])
        self.assertFalse(result.passed)

    def test_drift_watchdog_pauses_after_consecutive_shift(self) -> None:
        reference_estimate = self.make_estimates()[0]
        watchdog = HandCameraDriftWatchdog(
            reference_estimate.t_camera_tag,
            translation_threshold_mm=2.0,
            rotation_threshold_deg=0.5,
            window_size=3,
            consecutive_limit=2,
            missing_limit=2,
        )
        self.assertEqual(watchdog.update(reference_estimate).state, "warming_up")
        self.assertEqual(watchdog.update(reference_estimate).state, "warming_up")
        self.assertEqual(watchdog.update(reference_estimate).state, "ok")

        shifted_transform = reference_estimate.t_camera_tag.copy()
        shifted_transform[0, 3] += 0.004
        shifted = TagPoseEstimate(
            timestamp_ms=100.0,
            corners_px=reference_estimate.corners_px,
            t_camera_tag=shifted_transform,
            t_hand_camera=(
                DEFAULT_T_HAND_TAG @ invert_transform(shifted_transform)
            ),
            reprojection_rms_px=0.05,
            incidence_angle_deg=0.0,
        )
        watchdog.update(shifted)
        warning = watchdog.update(shifted)
        self.assertEqual(warning.state, "warning")
        paused = watchdog.update(shifted)
        self.assertEqual(paused.state, "recalibrate")
        self.assertTrue(paused.should_pause)

    def test_drift_watchdog_ignores_hand_camera_lever_diagnostic(self) -> None:
        reference_estimate = self.make_estimates()[0]
        watchdog = HandCameraDriftWatchdog(
            reference_estimate.t_camera_tag,
            translation_threshold_mm=2.0,
            rotation_threshold_deg=0.5,
            window_size=2,
            consecutive_limit=2,
        )
        noisy_hand_camera = reference_estimate.t_hand_camera.copy()
        noisy_hand_camera[0, 3] += 0.004
        diagnostic_only_shift = TagPoseEstimate(
            timestamp_ms=100.0,
            corners_px=reference_estimate.corners_px,
            t_camera_tag=reference_estimate.t_camera_tag,
            t_hand_camera=noisy_hand_camera,
            reprojection_rms_px=0.05,
            incidence_angle_deg=0.0,
        )

        watchdog.update(diagnostic_only_shift)
        self.assertEqual(watchdog.update(diagnostic_only_shift).state, "ok")
        self.assertEqual(watchdog.update(diagnostic_only_shift).state, "ok")

    def test_missing_tag_is_fail_closed(self) -> None:
        reference = self.make_estimates()[0].t_camera_tag
        watchdog = HandCameraDriftWatchdog(
            reference,
            window_size=2,
            missing_limit=2,
        )
        self.assertFalse(watchdog.update(None).should_pause)
        status = watchdog.update(None)
        self.assertEqual(status.state, "tag_missing")
        self.assertTrue(status.should_pause)


if __name__ == "__main__":
    unittest.main()
