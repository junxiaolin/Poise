"""Generic canonical sphere rigid-object configuration."""

from __future__ import annotations


SPHERE_DEFAULT_MASS_KG = 0.05
"""Default sphere mass in kilograms."""

SPHERE_DEFAULT_FRICTION = 1.0
"""Default static/dynamic friction coefficient."""


def get_sphere_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Object",
    radius_m: float = 0.02,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
    mass_kg: float = SPHERE_DEFAULT_MASS_KG,
    friction: float = SPHERE_DEFAULT_FRICTION,
    visual_color: tuple[float, float, float] = (0.92, 0.58, 0.22),
    visual_material_path: str = "material",
    physics_material_path: str = "material",
):
    """Create an Isaac Lab rigid object config for a sphere primitive."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    radius = float(radius_m)
    if radius <= 0.0:
        raise ValueError(f"radius_m must be positive, got radius={radius_m!r}.")

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=radius,
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
