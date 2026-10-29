"""User-authored regular hex-prism rigid assets, including the original D40 x L160 mesh."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


SHARPA_MANUAL_HEX_PRISM_DEFAULT_MASS_KG = 0.05
SHARPA_MANUAL_HEX_PRISM_DEFAULT_FRICTION = 1.0
SHARPA_MANUAL_HEX_PRISM_FAMILY = "hex_prism"
SHARPA_HEXPRISM40_SDF_PROXY_MODE = "compound_sphere_cover_v1"
SHARPA_HEXPRISM40_SDF_PROXY_SUBDIVISIONS = (3, 3, 3)


def _bbox_cell_sphere_cover(
    extents_xyz_m: tuple[float, float, float],
    subdivisions_xyz: tuple[int, int, int],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, ...]]:
    """Conservatively cover an axis-aligned bounding box with cell spheres."""

    if any(int(count) < 1 for count in subdivisions_xyz):
        raise ValueError("Sphere-cover subdivisions must be positive.")
    cell = tuple(
        float(extent) / int(count)
        for extent, count in zip(extents_xyz_m, subdivisions_xyz, strict=True)
    )
    axes = tuple(
        tuple(-0.5 * float(extent) + (index + 0.5) * width for index in range(count))
        for extent, count, width in zip(
            extents_xyz_m, subdivisions_xyz, cell, strict=True
        )
    )
    points = tuple(
        (float(x), float(y), float(z))
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    )
    radius = math.sqrt(sum((0.5 * width) ** 2 for width in cell))
    return points, tuple(radius for _ in points)


@dataclass(frozen=True)
class ManualHexPrismSpec:
    object_id: str
    extents_xyz_m: tuple[float, float, float]
    stable_cache_path: str
    stable_cache_sha256: str
    stable_cache_row_count: int
    mesh_data_path: str
    mesh_data_sha256: str
    source_obj_sha256: str
    mesh_data_source_obj_sha256: str
    rigid_usd_sha256: str


SHARPA_MANUAL_HEX_PRISM_SPECS = (
    ManualHexPrismSpec(
        object_id="regular_hex_prism_af35mm_h35mm_chamfered",
        extents_xyz_m=(0.03983716965, 0.035, 0.035),
        stable_cache_path=(
            "data/stable_grasps/sharpa_manual_hex_prisms/"
            "regular_hex_prism_af35mm_h35mm_chamfered_griddiverse310_"
            "variants10to20_palmup_omniwrist8_hold2s_training.pt"
        ),
        stable_cache_sha256="0c23fa71e7507b3dcd6f5e3bff0590622521366afdddd07db6c3f8b964ee19a7",
        stable_cache_row_count=4324,
        mesh_data_path="data/geometry_latents/sharpa_hexprism35_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="5aed8ca98ceaf4c176582a81be70c16a7ef9f86e25bd61ce79c15c8886787c60",
        source_obj_sha256="3d9c159318912c1a579a29acdbb4d2fe05299a347db58466f577b7cdf6d5823c",
        # The pinned BPS64 payload predates the canonical OBJ text re-export.
        # Geometry and mass properties are unchanged, but retain both hashes.
        mesh_data_source_obj_sha256="94f77d83d71d3e64c4889631ef85dfe70246b84126669238788d440709ad0e90",
        rigid_usd_sha256="97741ea8b80b33321590aef4ee835f6f3bab65f32901f5c80059472f121c95e1",
    ),
    ManualHexPrismSpec(
        object_id="regular_hex_prism_af40mm_h40mm_chamfered",
        extents_xyz_m=(0.04561066818, 0.040, 0.040),
        stable_cache_path=(
            "data/stable_grasps/sharpa_manual_hex_prisms/"
            "regular_hex_prism_af40mm_h40mm_chamfered_griddiverse410_"
            "variants10to20_palmup_omniwrist8_hold2s_training.pt"
        ),
        stable_cache_sha256="121c1b2bbca4b24f8e5b582ddd13ffaa795e1c0e34a6513d597e29a2684cad6a",
        stable_cache_row_count=5969,
        mesh_data_path="data/geometry_latents/sharpa_hexprism40_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="8aadb99540dabefc67eacd41f90cf9c02d8825ac85f46ab0f0ed1add955055ff",
        source_obj_sha256="79506c0d45249506260a44c00150fa92efc85eb1878cdc529302cc1675dea158",
        # The pinned BPS64 payload predates the canonical OBJ text re-export.
        # Geometry and mass properties are unchanged, but retain both hashes.
        mesh_data_source_obj_sha256="d0d90be76046db7d7becbdee98d0cc02415dd0adf23dd623e17be490e0e4b87c",
        rigid_usd_sha256="f30d4061289c7ef32b33adb51c7f599c48d3696a402b9189556fdcb5fdecdd6d",
    ),
)
SHARPA_MANUAL_HEX_PRISM_SPECS += (
    ManualHexPrismSpec(
        object_id="hexprism_d40_l160_original_m",
        extents_xyz_m=(0.03999999910593033, 0.03464101627469063, 0.1599999964237213),
        stable_cache_path="data/stable_grasps/sharpa_manual_hex_prisms/hexprism_d40_l160_fast_omniwrist8.pt",
        stable_cache_sha256="e5b649fdf58348c82495ae9d7c85a3ff5f5ac58a0b4c8bf15d63c4f627c9c725",
        stable_cache_row_count=480,
        mesh_data_path="data/geometry_latents/hexprism_d40_l160_bps64_mesh_data_64_v1.pt",
        mesh_data_sha256="61bdf2aaa53b5314340050b8f77d74278e2662e78c7dc3f1495bf5a609a95bf5",
        source_obj_sha256="8c36363b5cc81388513c8d3c64a67e25bccad63c3e2ed19819ced0c478d5d638",
        mesh_data_source_obj_sha256="8c36363b5cc81388513c8d3c64a67e25bccad63c3e2ed19819ced0c478d5d638",
        rigid_usd_sha256="986c2849ae3aede993042d9397eaf352ae58c940f80509d92b646a252f6f17c8",
    ),
)
SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS = tuple(spec.object_id for spec in SHARPA_MANUAL_HEX_PRISM_SPECS)
SHARPA_MANUAL_HEX_PRISM_DIMS_M = tuple(spec.extents_xyz_m for spec in SHARPA_MANUAL_HEX_PRISM_SPECS)
SHARPA_MANUAL_HEX_PRISM_STABLE_CACHE_PATHS = tuple(spec.stable_cache_path for spec in SHARPA_MANUAL_HEX_PRISM_SPECS)
SHARPA_MANUAL_HEX_PRISM_MESH_DATA_PATHS = tuple(spec.mesh_data_path for spec in SHARPA_MANUAL_HEX_PRISM_SPECS)

SHARPA_HEXPRISM35_SPEC = SHARPA_MANUAL_HEX_PRISM_SPECS[0]
SHARPA_HEXPRISM40_SPEC = SHARPA_MANUAL_HEX_PRISM_SPECS[1]
(
    SHARPA_HEXPRISM40_SDF_PROXY_POINTS_LOCAL_M,
    SHARPA_HEXPRISM40_SDF_PROXY_RADII_M,
) = _bbox_cell_sphere_cover(
    SHARPA_HEXPRISM40_SPEC.extents_xyz_m,
    SHARPA_HEXPRISM40_SDF_PROXY_SUBDIVISIONS,
)


def workspace_root() -> Path:
    from poise.paths import workspace_root as resolve_root
    return resolve_root()


def get_manual_hex_prism_spec(object_id: str) -> ManualHexPrismSpec:
    for spec in SHARPA_MANUAL_HEX_PRISM_SPECS:
        if spec.object_id == object_id:
            return spec
    raise ValueError(f"Unknown manual hex-prism object_id={object_id!r}; expected {SHARPA_MANUAL_HEX_PRISM_OBJECT_IDS}")


def get_manual_hex_prism_asset_dir(object_id: str) -> Path:
    get_manual_hex_prism_spec(object_id)
    return workspace_root() / "data/objects/sharpa_manual_hex_prisms" / object_id


def get_manual_hex_prism_obj_path(object_id: str) -> Path:
    path = get_manual_hex_prism_asset_dir(object_id) / f"{object_id}.obj"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical OBJ asset: {path}")
    return path


def get_manual_hex_prism_usd_path(object_id: str) -> Path:
    path = get_manual_hex_prism_asset_dir(object_id) / f"{object_id}.usda"
    if not path.is_file():
        raise FileNotFoundError(f"Missing converted rigid asset: {path}")
    return path


def get_sharpa_manual_hex_prism_cfg(
    *,
    object_id: str,
    prim_path: str = "/World/envs/env_.*/Object",
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.58),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    mass_kg: float = SHARPA_MANUAL_HEX_PRISM_DEFAULT_MASS_KG,
    friction: float = SHARPA_MANUAL_HEX_PRISM_DEFAULT_FRICTION,
):
    """Create an Isaac Lab rigid-object config for either converted manual prism.

    Friction is randomized through the environment's material contract after
    spawning, matching the Wedge50/Cube02 training path.  The argument remains
    explicit for asset-call compatibility and validation.
    """

    if float(friction) <= 0.0:
        raise ValueError(f"friction must be positive, got {friction}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(get_manual_hex_prism_usd_path(object_id)),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(mass_kg)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )
