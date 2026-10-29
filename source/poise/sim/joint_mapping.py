"""Joint and body-name conventions for POISE dexterous hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SHARPA_ACTION_DIM = 22
SHARPA_JOINT_ORDER = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
SHARPA_FINGERTIP_BODIES = (
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
)
SHARPA_ELASTOMER_BODIES = (
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
)
SHARPA_DP_BODIES = (
    "right_thumb_DP",
    "right_index_DP",
    "right_middle_DP",
    "right_ring_DP",
    "right_pinky_DP",
)
SHARPA_CONTACT_BIT_BODIES = SHARPA_ELASTOMER_BODIES
SHARPA_CONTACT_SENSOR_BODIES = (*SHARPA_ELASTOMER_BODIES, *SHARPA_DP_BODIES)
SHARPA_ALL_CONTACT_BODIES = SHARPA_CONTACT_SENSOR_BODIES
SHARPA_FORCE_SENSOR_BODIES = SHARPA_ELASTOMER_BODIES

# CSGWCD body contract for the pinned Wave1.4 HB1 USDA. The primary tuple is
# exactly the historical d432d40 10-body order. Secondary bodies are every
# other rigid body with a collision Mesh in the composed asset. Bodies in the
# excluded tuple are rigid transforms without collision prims and therefore
# cannot contribute a cube-filtered physical contact.
SHARPA_CSGWCD_PRIMARY_BODIES = SHARPA_ALL_CONTACT_BODIES
SHARPA_CSGWCD_SECONDARY_BODIES = (
    "right_hand_C_MC",
    "right_thumb_MC",
    "right_thumb_PP",
    "right_index_MCP_VL",
    "right_index_PP",
    "right_index_MP",
    "right_middle_MCP_VL",
    "right_middle_PP",
    "right_middle_MP",
    "right_ring_MCP_VL",
    "right_ring_PP",
    "right_ring_MP",
    "right_pinky_MC",
    "right_pinky_MCP_VL",
    "right_pinky_PP",
    "right_pinky_MP",
)
SHARPA_CSGWCD_COLLISION_BODIES = (
    *SHARPA_CSGWCD_PRIMARY_BODIES,
    *SHARPA_CSGWCD_SECONDARY_BODIES,
)
SHARPA_CSGWCD_EXCLUDED_NO_COLLISION_BODIES = (
    "right_thumb_CMC_VL",
    "right_thumb_MCP_VL",
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
)
SHARPA_CSGWCD_GROUP_NAMES = ("palm", "thumb", "index", "middle", "ring", "pinky")


def sharpa_csgwcd_group_name(body_name: str) -> str:
    """Return the conservative contribution-cap group for one Sharpa body."""

    if body_name == "right_hand_C_MC":
        return "palm"
    for finger in SHARPA_CSGWCD_GROUP_NAMES[1:]:
        if body_name.startswith(f"right_{finger}_"):
            return finger
    return "excluded"


SHARPA_CSGWCD_GROUP_IDS = tuple(
    SHARPA_CSGWCD_GROUP_NAMES.index(sharpa_csgwcd_group_name(body_name))
    for body_name in SHARPA_CSGWCD_COLLISION_BODIES
)

# Stable-grasp tensors and policy observations/actions use SHARPA_JOINT_ORDER,
# which is also the raw SDK order.  IsaacLab's articulation-internal order is
# resolved separately by joint name and must never be used at the SDK boundary.
SHARPA_ACTION_TO_SDK_INDICES = tuple(range(SHARPA_ACTION_DIM))
SHARPA_SDK_TO_ACTION_INDICES = tuple(range(SHARPA_ACTION_DIM))


@dataclass(frozen=True)
class HandSpec:
    """Hand-specific action and body naming conventions."""

    hand_id: str
    action_dim: int
    joint_order: tuple[str, ...]
    palm_body: str
    fingertip_bodies: tuple[str, ...]
    contact_bit_bodies: tuple[str, ...]
    contact_sensor_bodies: tuple[str, ...]
    all_contact_bodies: tuple[str, ...]
    force_sensor_bodies: tuple[str, ...]
    surface_delta_bodies: tuple[str, ...]
    action_to_sdk_indices: tuple[int, ...] = ()
    sdk_to_action_indices: tuple[int, ...] = ()


SHARPA_WAVE_RIGHT_HAND_SPEC = HandSpec(
    hand_id="sharpa_wave_right",
    action_dim=SHARPA_ACTION_DIM,
    joint_order=SHARPA_JOINT_ORDER,
    palm_body="",
    fingertip_bodies=SHARPA_FINGERTIP_BODIES,
    contact_bit_bodies=SHARPA_CONTACT_BIT_BODIES,
    contact_sensor_bodies=SHARPA_CONTACT_SENSOR_BODIES,
    all_contact_bodies=SHARPA_ALL_CONTACT_BODIES,
    force_sensor_bodies=SHARPA_FORCE_SENSOR_BODIES,
    surface_delta_bodies=SHARPA_CONTACT_SENSOR_BODIES,
    action_to_sdk_indices=SHARPA_ACTION_TO_SDK_INDICES,
    sdk_to_action_indices=SHARPA_SDK_TO_ACTION_INDICES,
)

HAND_SPECS = {
    SHARPA_WAVE_RIGHT_HAND_SPEC.hand_id: SHARPA_WAVE_RIGHT_HAND_SPEC,
}


@dataclass(frozen=True)
class ResolvedHandMapping:
    """Resolved joint/body indices inside an Isaac Lab articulation."""

    hand_id: str
    joint_names: tuple[str, ...]
    action_joint_indices: tuple[int, ...]
    palm_body_index: int | None
    fingertip_body_indices: tuple[int, ...]





def missing_names(expected: Iterable[str], available: Iterable[str]) -> tuple[str, ...]:
    """Return expected names that are absent from an available-name sequence."""

    available_set = set(available)
    return tuple(name for name in expected if name and name not in available_set)


def resolve_hand_spec(hand_id: str) -> HandSpec:
    """Return a configured hand spec by id."""

    try:
        return HAND_SPECS[str(hand_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported hand_type={hand_id!r}. Expected one of {tuple(HAND_SPECS)}.") from exc


def resolve_hand_mapping(
    hand_spec: HandSpec,
    joint_names: list[str] | tuple[str, ...],
    body_names: list[str] | tuple[str, ...],
) -> ResolvedHandMapping:
    """Resolve action joint indices and key body indices for a hand spec."""

    required_bodies = (*(() if not hand_spec.palm_body else (hand_spec.palm_body,)), *hand_spec.fingertip_bodies)
    missing_joints = missing_names(hand_spec.joint_order, joint_names)
    missing_bodies = missing_names(required_bodies, body_names)
    if missing_joints or missing_bodies:
        details = []
        if missing_joints:
            details.append(f"missing joints={missing_joints}")
        if missing_bodies:
            details.append(f"missing bodies={missing_bodies}")
        raise ValueError(f"{hand_spec.hand_id} articulation naming mismatch: " + "; ".join(details))

    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}

    return ResolvedHandMapping(
        hand_id=hand_spec.hand_id,
        joint_names=hand_spec.joint_order,
        action_joint_indices=tuple(joint_name_to_idx[name] for name in hand_spec.joint_order),
        palm_body_index=body_name_to_idx[hand_spec.palm_body] if hand_spec.palm_body else None,
        fingertip_body_indices=tuple(body_name_to_idx[name] for name in hand_spec.fingertip_bodies),
    )




def format_hand_action_table(hand_spec: HandSpec, action_joint_indices: Iterable[int] | None = None) -> str:
    """Return a compact action-index mapping table."""

    if action_joint_indices is None:
        action_joint_indices = ("?" for _ in hand_spec.joint_order)
    lines = ["action_index  sim_joint_index  joint_name"]
    for action_idx, (joint_name, joint_idx) in enumerate(zip(hand_spec.joint_order, action_joint_indices, strict=True)):
        lines.append(f"{action_idx:>12}  {str(joint_idx):>15}  {joint_name}")
    return "\n".join(lines)
