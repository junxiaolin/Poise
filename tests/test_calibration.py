from __future__ import annotations

import math
import unittest

import numpy as np

from poise.deploy.calibration import (
    invert_transform,
    mean_transform,
    rotation_matrix_to_quat_wxyz,
    transform_dispersion,
)


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class SharpaPose6DCalibrationTest(unittest.TestCase):
    def test_inverse_round_trip(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_z(math.radians(35.0))
        transform[:3, 3] = np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(transform @ invert_transform(transform), np.eye(4), atol=1.0e-12)

    def test_mean_and_dispersion(self) -> None:
        translations = (
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            * 1.0e-4
        )
        angles = np.radians(np.array([-0.1, 0.1, -0.1, 0.1]))
        transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
        for index in range(4):
            transforms[index, :3, :3] = rotation_z(float(angles[index]))
            transforms[index, :3, 3] = np.array([0.2, 0.1, -0.3]) + translations[index]

        average = mean_transform(transforms)
        np.testing.assert_allclose(average, np.array(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, 0.1],
                [0.0, 0.0, 1.0, -0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ), atol=1.0e-12)
        dispersion = transform_dispersion(transforms, average)
        self.assertLess(dispersion["translation_std_3d_mm"], 0.2)
        self.assertLess(dispersion["rotation_std_3d_deg"], 0.2)

    def test_confirmed_hand_tag_transform_quaternion(self) -> None:
        transform = np.array(
            [
                [0.0, 0.0, 1.0, 0.0050],
                [1.0, 0.0, 0.0, 0.0000],
                [0.0, 1.0, 0.0, -0.0795],
                [0.0, 0.0, 0.0, 1.0000],
            ],
            dtype=np.float64,
        )
        quaternion = rotation_matrix_to_quat_wxyz(transform[:3, :3])
        np.testing.assert_allclose(quaternion, np.full(4, 0.5), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
