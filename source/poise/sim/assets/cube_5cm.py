"""5 cm canonical cube rigid-object configuration."""

from __future__ import annotations


CUBE_5CM_SIDE_LENGTH_M = 0.05
"""Cube side length in meters."""

CUBE_5CM_MASS_KG = 0.05
"""Initial cube mass in kilograms."""

CUBE_5CM_FRICTION = 1.0
"""Initial static/dynamic friction coefficient."""


def get_cube_5cm_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Object",
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.58),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    disable_gravity: bool = False,
):
    """Create an Isaac Lab rigid object config for the 5 cm cube."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_5CM_SIDE_LENGTH_M, CUBE_5CM_SIDE_LENGTH_M, CUBE_5CM_SIDE_LENGTH_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_5CM_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=CUBE_5CM_FRICTION,
                dynamic_friction=CUBE_5CM_FRICTION,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.86, 0.38, 0.16),
                roughness=0.55,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


class _LazyCube5cmCfg:
    """Lazy cube config proxy."""

    def __init__(self):
        self._cfg = None

    def _resolve(self):
        if self._cfg is None:
            self._cfg = get_cube_5cm_cfg()
        return self._cfg

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return repr(self._resolve())


CUBE_5CM_CFG = _LazyCube5cmCfg()
"""Lazy default 5 cm cube rigid-object config."""
