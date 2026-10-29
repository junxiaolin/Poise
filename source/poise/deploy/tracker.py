"""FoundationPose adapter for a metric mesh; dependencies are loaded on use."""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from .calibration import as_transform
from .realsense_runtime import RealSenseFrame

class FoundationPoseObjectTracker:
    """Thin lazy-import wrapper for a metric mesh in a declared object frame."""

    def __init__(
        self,
        *,
        foundationpose_root: Path,
        mesh_file: Path,
        debug_dir: Path,
        expected_bounds_m: np.ndarray | None = None,
        t_mesh_object_sim: np.ndarray | None = None,
        object_label: str = "Object",
        require_texture: bool = False,
    ) -> None:
        foundationpose_root = foundationpose_root.expanduser().resolve()
        mesh_file = mesh_file.expanduser().resolve()
        if not foundationpose_root.is_dir():
            raise FileNotFoundError(f"FoundationPose root not found: {foundationpose_root}")
        if not mesh_file.is_file():
            raise FileNotFoundError(f"{object_label} mesh not found: {mesh_file}")
        if str(foundationpose_root) not in sys.path:
            sys.path.insert(0, str(foundationpose_root))

        import nvdiffrast.torch as dr
        import torch
        import trimesh
        from estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
            set_logging_format,
            set_seed,
        )

        set_logging_format()
        logging.getLogger().setLevel(logging.WARNING)
        set_seed(0)
        mesh = trimesh.load(mesh_file, process=False, maintain_order=True)
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError(f"Expected a mesh, got {type(mesh).__name__}.")
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if expected_bounds_m is not None:
            expected_bounds = np.asarray(expected_bounds_m, dtype=np.float64)
            if expected_bounds.shape != (2, 3):
                raise ValueError("expected_bounds_m must have shape (2, 3).")
            if not np.allclose(bounds, expected_bounds, atol=1.0e-6):
                raise ValueError(
                    f"{object_label} mesh bounds are wrong: bounds={bounds.tolist()}, "
                    f"expected={expected_bounds.tolist()}."
                )
        if require_texture and not isinstance(
            mesh.visual,
            trimesh.visual.texture.TextureVisuals,
        ):
            raise ValueError(f"{object_label} mesh must load with TextureVisuals.")
        mesh_to_object_sim = (
            np.eye(4, dtype=np.float64)
            if t_mesh_object_sim is None
            else as_transform(np.asarray(t_mesh_object_sim, dtype=np.float64))
        )

        debug_dir.mkdir(parents=True, exist_ok=True)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self._estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=str(debug_dir),
        )
        self._torch = torch
        mesh_center_mass = np.asarray(mesh.center_mass, dtype=np.float64)
        if (
            mesh_center_mass.shape != (3,)
            or not np.all(np.isfinite(mesh_center_mass))
        ):
            raise ValueError(
                f"{object_label} mesh volume center must be a finite 3-vector."
            )

        self.mesh_file = mesh_file
        # Retain immutable CPU geometry for the lightweight paper-video pose
        # ghost.  The display path never reloads or reprocesses the mesh.
        self.mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self.mesh_faces = np.asarray(mesh.faces, dtype=np.int32).copy()
        self.mesh_vertices.setflags(write=False)
        self.mesh_faces.setflags(write=False)
        self.mesh_bounds = bounds
        self.mesh_center = bounds.mean(axis=0)
        self.mesh_center_mass = mesh_center_mass
        self.t_mesh_object_sim = mesh_to_object_sim
        self.object_label = object_label

    def _synchronize(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    @staticmethod
    def _sanitize_pose(pose: np.ndarray) -> np.ndarray:
        """Project small float32 drift back to SO(3), rejecting real failures."""
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4).copy()
        if not np.all(np.isfinite(pose)):
            raise ValueError("FoundationPose returned non-finite values.")
        if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-5):
            raise ValueError(f"FoundationPose returned invalid last row: {pose[3].tolist()}.")
        rotation = pose[:3, :3]
        determinant = float(np.linalg.det(rotation))
        orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        if determinant <= 0.0 or abs(determinant - 1.0) > 1.0e-3 or orthogonality_error > 1.0e-3:
            raise ValueError(
                "FoundationPose returned a non-rigid rotation: "
                f"det={determinant:.9f}, orthogonality_error={orthogonality_error:.3e}."
            )
        left, _, right_t = np.linalg.svd(rotation)
        projected = left @ right_t
        if np.linalg.det(projected) <= 0.0:
            raise ValueError("FoundationPose SO(3) projection produced a reflection.")
        pose[:3, :3] = projected
        pose[3] = np.array([0.0, 0.0, 0.0, 1.0])
        return as_transform(pose)

    def register(
        self,
        frame: RealSenseFrame,
        *,
        camera_matrix: np.ndarray,
        mask: np.ndarray,
        iterations: int,
    ) -> tuple[np.ndarray, float]:
        # cvtColor performs the BGR->RGB copy in optimized native code.  The
        # previous negative-stride slice plus ascontiguousarray cost several
        # milliseconds on every 1280x720 frame.
        rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
        self._synchronize()
        start = time.perf_counter()
        pose = self._estimator.register(
            K=camera_matrix,
            rgb=rgb,
            depth=frame.depth_m,
            ob_mask=mask,
            iteration=iterations,
        )
        self._synchronize()
        return self._sanitize_pose(pose), time.perf_counter() - start

    def track(
        self,
        frame: RealSenseFrame,
        *,
        camera_matrix: np.ndarray,
        iterations: int,
    ) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
        # The previous track call already synchronizes after returning.  A
        # second pre-sync here only stalls the 20 Hz host loop without adding
        # a safety guarantee; the post-sync remains to make the measured
        # tracking duration and returned CPU pose authoritative.
        start = time.perf_counter()
        pose = self._estimator.track_one(
            rgb=rgb,
            depth=frame.depth_m,
            K=camera_matrix,
            iteration=iterations,
        )
        self._synchronize()
        return self._sanitize_pose(pose), time.perf_counter() - start
