"""Sharpa Wave Isaac Lab articulation configuration."""

from __future__ import annotations

import math
import os
from pathlib import Path


SHARPA_DEFAULT_ROOT_POS = (0.0, 0.0, 0.5)
"""Default world position for the Sharpa Wave root."""

SHARPA_DEFAULT_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)
"""Default Sharpa Wave root orientation in Isaac Lab `(w, x, y, z)` convention."""

SHARPA_PALM_NORMAL_LOCAL_AXIS = (1.0, 0.0, 0.0)
"""Assumed Sharpa root-frame axis pointing out of the palm face for POISE v1 calibration."""

SHARPA_PALM_DOWN_ROOT_ROT = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
"""Initial POISE v1 palmdown root orientation, mapping local `+x` to world `-z`."""

SHARPA_PALM_UP_ROOT_ROT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
"""Initial POISE v1 palmup root orientation, mapping local `+x` to world `+z`."""

SHARPA_HAND_ROOT_ORIENTATIONS = ("palm_down", "palm_up")
"""Supported fixed Sharpa Wave root orientations for POISE tasks."""

SHARPA_DEFAULT_JOINT_POS = {
    "right_thumb_CMC_FE": math.radians(95.12771),
    "right_thumb_CMC_AA": math.radians(-3.11244),
    "right_thumb_MCP_FE": math.radians(14.81626),
    "right_thumb_MCP_AA": math.radians(-1.03493),
    "right_thumb_IP": math.radians(12.23986),
    "right_index_MCP_FE": math.radians(65.21091),
    "right_index_MCP_AA": math.radians(6.1133),
    "right_index_PIP": math.radians(15.58495),
    "right_index_DIP": math.radians(5.90325),
    "right_middle_MCP_FE": math.radians(31.74149),
    "right_middle_MCP_AA": math.radians(-0.95812),
    "right_middle_PIP": math.radians(41.88173),
    "right_middle_DIP": math.radians(12.844),
    "right_ring_MCP_FE": math.radians(31.72383),
    "right_ring_MCP_AA": math.radians(9.84458),
    "right_ring_PIP": math.radians(35.22366),
    "right_ring_DIP": math.radians(18.02839),
    "right_pinky_CMC": math.radians(10.9712),
    "right_pinky_MCP_FE": math.radians(68.30895),
    "right_pinky_MCP_AA": math.radians(7.99151),
    "right_pinky_PIP": math.radians(5.89626),
    "right_pinky_DIP": math.radians(5.89875),
}
"""Official Sharpa Wave sim starting pose from the public IsaacLab environment."""

SHARPA_DEPLOY_DOF_LIMITS_DEG = {
    "right_thumb_CMC_FE": (-10.0, 110.0),
    "right_thumb_CMC_AA": (-20.0, 7.5),
    "right_thumb_MCP_FE": (-30.0, 80.0),
    "right_thumb_MCP_AA": (-20.0, 20.0),
    "right_thumb_IP": (0.0, 100.0),
    "right_index_MCP_FE": (-10.0, 90.0),
    "right_index_MCP_AA": (-15.0, 5.0),
    "right_index_PIP": (0.0, 100.0),
    "right_index_DIP": (0.0, 80.0),
    "right_middle_MCP_FE": (-10.0, 90.0),
    "right_middle_MCP_AA": (-10.0, 10.0),
    "right_middle_PIP": (0.0, 100.0),
    "right_middle_DIP": (0.0, 80.0),
    "right_ring_MCP_FE": (-10.0, 90.0),
    "right_ring_MCP_AA": (-2.0, 10.0),
    "right_ring_PIP": (0.0, 100.0),
    "right_ring_DIP": (0.0, 80.0),
    "right_pinky_CMC": (0.0, 15.0),
    "right_pinky_MCP_FE": (-10.0, 90.0),
    "right_pinky_MCP_AA": (-2.0, 18.0),
    "right_pinky_PIP": (0.0, 100.0),
    "right_pinky_DIP": (0.0, 80.0),
}
"""Sharpa deployment joint-limit profile, in degrees."""


def resolve_sharpa_root_rot(orientation: str) -> tuple[float, float, float, float]:
    """Return a Sharpa root quaternion for a named fixed palm orientation."""

    if orientation == "palm_down":
        return SHARPA_PALM_DOWN_ROOT_ROT
    if orientation == "palm_up":
        return SHARPA_PALM_UP_ROOT_ROT
    raise ValueError(
        f"Unsupported Sharpa hand_root_orientation={orientation!r}. "
        f"Expected one of {SHARPA_HAND_ROOT_ORIENTATIONS}."
    )


def get_sharpa_wave_usd_path() -> Path:
    from poise.paths import resolve_path
    return resolve_path("data/robots/sharpa_wave/right_sharpa_wave.usda")


def get_sharpa_wave_hand_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    disable_gravity: bool = True,
    activate_contact_sensors: bool = False,
    init_pos: tuple[float, float, float] = SHARPA_DEFAULT_ROOT_POS,
    init_rot: tuple[float, float, float, float] = SHARPA_PALM_DOWN_ROOT_ROT,
    usd_path: str | Path | None = None,
):
    """Create an Isaac Lab articulation config for the Sharpa Wave right hand."""

    usd_path = Path(usd_path).resolve() if usd_path is not None else get_sharpa_wave_usd_path()
    if not usd_path.is_file():
        raise FileNotFoundError(f"Sharpa Wave USD not found: {usd_path}")
    if usd_path == get_sharpa_wave_usd_path().resolve():
        from poise.assets.hand import check_hand
        check_hand(usd_path.parent)

    import isaaclab.sim as sim_utils
    from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=SHARPA_DEFAULT_JOINT_POS,
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


class _LazySharpaWaveHandCfg:
    """Lazy Sharpa Wave config proxy."""

    def __init__(self):
        self._cfg = None

    def _resolve(self):
        if self._cfg is None:
            self._cfg = get_sharpa_wave_hand_cfg()
        return self._cfg

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return repr(self._resolve())


SHARPA_WAVE_RIGHT_HAND_CFG = _LazySharpaWaveHandCfg()
"""Lazy default right-hand Sharpa Wave articulation config."""
