"""Generic canonical cuboid rigid-object configuration."""

from __future__ import annotations


CUBOID_DEFAULT_MASS_KG = 0.05
"""Default cuboid mass in kilograms."""

CUBOID_DEFAULT_FRICTION = 1.0
"""Default static/dynamic friction coefficient."""


def get_cuboid_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Object",
    dims_m: tuple[float, float, float] = (0.04, 0.04, 0.04),
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.58),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    mass_kg: float = CUBOID_DEFAULT_MASS_KG,
    friction: float = CUBOID_DEFAULT_FRICTION,
    visual_color: tuple[float, float, float] = (0.24, 0.72, 0.64),
    visual_material_path: str = "material",
    physics_material_path: str = "material",
):
    """Create an Isaac Lab rigid object config for an axis-aligned cuboid primitive."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    dims = tuple(float(v) for v in dims_m)
    if len(dims) != 3 or any(v <= 0.0 for v in dims):
        raise ValueError(f"dims_m must contain three positive values, got {dims_m!r}.")

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=dims,
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(friction),
                dynamic_friction=float(friction),
                restitution=0.0,
            ),
            physics_material_path=str(physics_material_path),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=visual_color,
                roughness=0.55,
                metallic=0.0,
            ),
            visual_material_path=str(visual_material_path),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )
