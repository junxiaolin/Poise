"""Sharpa Wave right-hand BPS64 interface shared by training and deployment."""
from __future__ import annotations
from dataclasses import asdict, dataclass

# Serialized IDs are retained for compatibility with verified research bundles.
SCHEMA_VERSION = "sharpa_pose6d_deployment_v2"
OBSERVATION_PROFILE = "sharpa_pose6d_deployable_v1_notactile"
OBSERVATION_CONTRACT_ID = "sharpa_pose6d_v1_notactile_244_bps64_gravity"
JOINT_INTERFACE_CONTRACT = "sharpa_action_sdk_identity_v1"
POSE_LAYOUT = "xyz_qwqxqyqz"
QUATERNION_CONVENTION = "wxyz"
HAND_ID = "sharpa_wave_right"
ACTION_DIM = 22
HISTORY_LENGTH = 3
HISTORY_FRAME_DIM = 44
OBJECT_DESCRIPTOR_DIM = 64
OBSERVATION_DIM = 244
TASK_OBSERVATION_DIM = 112

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
SHARPA_SDK_JOINT_ORDER = SHARPA_JOINT_ORDER
SHARPA_ACTION_TO_SDK_INDICES = tuple(range(ACTION_DIM))
SHARPA_SDK_TO_ACTION_INDICES = tuple(range(ACTION_DIM))

def joint_interface_manifest() -> dict[str, str | list[str] | list[int]]:
    """Return the fail-closed real-hardware joint-order contract."""
    return {
        "contract": JOINT_INTERFACE_CONTRACT,
        "policy_action_order": list(SHARPA_JOINT_ORDER),
        "sdk_order": list(SHARPA_SDK_JOINT_ORDER),
        "action_to_sdk_gather_indices": list(SHARPA_ACTION_TO_SDK_INDICES),
        "sdk_to_action_gather_indices": list(SHARPA_SDK_TO_ACTION_INDICES),
    }

@dataclass(frozen=True)
class ObservationField:
    """One contiguous field in the flat policy observation."""

    name: str
    start: int
    stop: int
    units: str

OBSERVATION_FIELDS = (
    ObservationField("history.t_minus_2.q", 0, 22, "scaled_joint_position"),
    ObservationField("history.t_minus_2.q_cmd", 22, 44, "scaled_joint_position"),
    ObservationField("history.t_minus_1.q", 44, 66, "scaled_joint_position"),
    ObservationField("history.t_minus_1.q_cmd", 66, 88, "scaled_joint_position"),
    ObservationField("history.t.q", 88, 110, "scaled_joint_position"),
    ObservationField("history.t.q_cmd", 110, 132, "scaled_joint_position"),
    ObservationField("object_pose_palm", 132, 139, POSE_LAYOUT),
    ObservationField("target_pose_palm", 139, 146, POSE_LAYOUT),
    ObservationField("pose_error", 146, 153, "xyz_qwqxqyqz"),
    ObservationField("current_anchor_pos_palm", 153, 156, "m"),
    ObservationField("target_anchor_pos_palm", 156, 159, "m"),
    ObservationField("anchor_delta", 159, 162, "m"),
    ObservationField("target_twist_axis_palm", 162, 165, "unit_vector"),
    ObservationField("target_twist_angle_over_pi", 165, 166, "normalized"),
    ObservationField("target_translation_axis_palm", 166, 169, "unit_vector"),
    ObservationField("target_translation_delta_scaled", 169, 170, "normalized"),
    ObservationField("object_linear_velocity_palm", 170, 173, "m_per_s"),
    ObservationField("object_angular_velocity_palm", 173, 176, "rad_per_s"),
    ObservationField("object_mesh_bps64", 176, 240, "normalized_bps_distance"),
    ObservationField("pose_tracker_confidence", 240, 241, "unit_interval"),
    ObservationField("gravity_direction_palm", 241, 244, "unit_vector"),
)

@dataclass(frozen=True)
class ObservationContract:
    """One fail-closed actor observation layout supported by deployment."""

    contract_id: str
    profile: str
    dimension: int
    history_length: int
    history_frame_dimension: int
    task_dimension: int
    include_gravity_palm: bool
    object_descriptor_dim: int = 0

    @property
    def fields(self) -> tuple[ObservationField, ...]:
        if self.dimension == 304:
            return OBSERVATION_FIELDS + tuple(
                field
                for frame, label in enumerate(("t_minus_2", "t_minus_1", "t"))
                for field in (
                    ObservationField(f"tactile.{label}.force", 244 + 20 * frame, 249 + 20 * frame, "N"),
                    ObservationField(f"tactile.{label}.contact_local_xyz", 249 + 20 * frame, 264 + 20 * frame, "m"),
                )
            )
        return OBSERVATION_FIELDS

OBSERVATION_CONTRACT = ObservationContract(
    contract_id=OBSERVATION_CONTRACT_ID,
    profile=OBSERVATION_PROFILE,
    dimension=OBSERVATION_DIM,
    history_length=HISTORY_LENGTH,
    history_frame_dimension=HISTORY_FRAME_DIM,
    task_dimension=TASK_OBSERVATION_DIM,
    include_gravity_palm=True,
    object_descriptor_dim=OBJECT_DESCRIPTOR_DIM,
)
TACTILE_OBSERVATION_CONTRACT = ObservationContract(
    contract_id="sharpa_pose6d_v1_tactile60_304_bps64_gravity",
    profile="sharpa_pose6d_v1_tactile60_actor304_critic552",
    dimension=304, history_length=3, history_frame_dimension=44,
    task_dimension=112, include_gravity_palm=True, object_descriptor_dim=64,
)
SUPPORTED_OBSERVATION_CONTRACTS = {
    item.contract_id: item for item in (OBSERVATION_CONTRACT, TACTILE_OBSERVATION_CONTRACT)
}

def observation_contract_for_dimension(dimension: int) -> ObservationContract:
    """Return the unique approved observation contract for ``dimension``.

    Object-specific contracts can intentionally share a tensor layout. Callers
    that only validate an actor width should use ``validate_observation_dimension``.
    """

    matches = [
        contract
        for contract in SUPPORTED_OBSERVATION_CONTRACTS.values()
        if contract.dimension == int(dimension)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unsupported Sharpa pose6D observation dimension {dimension}; "
            f"expected one of {sorted(item.dimension for item in SUPPORTED_OBSERVATION_CONTRACTS.values())}."
        )
    return matches[0]

def validate_observation_dimension(dimension: int) -> int:
    """Validate that ``dimension`` belongs to at least one approved contract."""

    value = int(dimension)
    if not any(contract.dimension == value for contract in SUPPORTED_OBSERVATION_CONTRACTS.values()):
        expected = sorted({item.dimension for item in SUPPORTED_OBSERVATION_CONTRACTS.values()})
        raise ValueError(
            f"Unsupported Sharpa pose6D observation dimension {dimension}; "
            f"expected one of {expected}."
        )
    return value

def observation_contract_from_manifest(value: object) -> ObservationContract:
    """Require an explicit BPS64 layout; legacy 177D/180D bundles are unsupported."""
    if not isinstance(value, dict):
        raise ValueError("Bundle requires an explicit observation manifest.")
    contract = SUPPORTED_OBSERVATION_CONTRACTS.get(value.get("contract"))
    if contract is None:
        raise ValueError("Unknown observation contract")
    expected = {"contract": contract.contract_id, **asdict(contract)}
    expected.pop("contract_id")
    for key, wanted in expected.items():
        if key in ("contract", "dimension") and key not in value:
            raise ValueError(f"Observation manifest is missing {key}.")
        if key in value and value[key] != wanted:
            raise ValueError(f"Observation manifest {key}={value[key]!r}; expected {wanted!r}.")
    return contract

def observation_layout_dicts(
    contract: ObservationContract = OBSERVATION_CONTRACT,
) -> list[dict[str, str | int]]:
    """Return a JSON-serializable observation layout."""

    return [asdict(field) for field in contract.fields]
