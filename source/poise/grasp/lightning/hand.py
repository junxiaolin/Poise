# Derived from Lightning Grasp by Zhao-Heng Yin; CC BY-NC 4.0.
# See licenses/Lightning-Grasp.txt. POISE changes: Sharpa-only, portable paths.
# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import combinations
import math
import os
import re
from pathlib import Path

from lygra.robot.base import RobotInterface
from lygra.urdf_utils import override_urdf_joint_limits
import numpy as np


from poise.paths import resolve_path

EXTERNAL_SHARPA_WAVE_URDF = resolve_path("data/robots/sharpa_wave/right_sharpa_wave.urdf")
SHARPA_RL_LAB_USDA_CANDIDATES = [resolve_path("data/robots/sharpa_limits/right_sharpa_wave.usda")]

USDA_DOF_LIMITS_SCALE = 0.9
JOINT_LIMIT_PROFILES = ("urdf", "deploy", "usda09")


CONTACT_LINKS = [
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
]


ACTIVE_JOINTS = [
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
]


DEPLOY_DOF_LIMITS = {
    "right_thumb_CMC_FE": (-0.1745, 1.9199),
    "right_thumb_CMC_AA": (-0.3491, 0.1309),
    "right_thumb_MCP_FE": (-0.5236, 1.3963),
    "right_thumb_MCP_AA": (-0.3491, 0.3491),
    "right_thumb_IP": (0.0, 1.7453),
    "right_index_MCP_FE": (-0.17453293, 1.5708),
    "right_index_MCP_AA": (-0.26179939, 0.08726646),
    "right_index_PIP": (0.0, 1.7453),
    "right_index_DIP": (0.0, 1.3963),
    "right_middle_MCP_FE": (-0.17453293, 1.5708),
    "right_middle_MCP_AA": (-0.17453293, 0.17453293),
    "right_middle_PIP": (0.0, 1.7453),
    "right_middle_DIP": (0.0, 1.3963),
    "right_ring_MCP_FE": (-0.17453293, 1.5708),
    "right_ring_MCP_AA": (-0.03490659, 0.17453293),
    "right_ring_PIP": (0.0, 1.7453),
    "right_ring_DIP": (0.0, 1.3963),
    "right_pinky_CMC": (0.0, 0.2618),
    "right_pinky_MCP_FE": (-0.17453293, 1.5708),
    "right_pinky_MCP_AA": (-0.03490659, 0.31415927),
    "right_pinky_PIP": (0.0, 1.7453),
    "right_pinky_DIP": (0.0, 1.3963),
}


def resolve_sharpa_rl_lab_usda_path():
    for candidate in SHARPA_RL_LAB_USDA_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate Sharpa Wave USDA among: "
        f"{[str(path) for path in SHARPA_RL_LAB_USDA_CANDIDATES]}"
    )


def parse_usda_joint_limits_deg(usda_path, joint_names):
    text = Path(usda_path).read_text(encoding="utf-8")
    lower_pattern = re.compile(r"float physics:lowerLimit = ([^\n]+)")
    upper_pattern = re.compile(r"float physics:upperLimit = ([^\n]+)")
    limits = {}

    for joint_name in joint_names:
        block_pattern = re.compile(
            rf'over "{re.escape(joint_name)}"\s*\{{(.*?)\n\s*\}}',
            re.DOTALL,
        )
        block_match = block_pattern.search(text)
        if block_match is None:
            raise ValueError(f"Missing USDA joint block for {joint_name!r} in {usda_path}")
        body = block_match.group(1)
        lower_match = lower_pattern.search(body)
        upper_match = upper_pattern.search(body)
        if lower_match is None or upper_match is None:
            raise ValueError(
                f"Missing USDA limits for {joint_name!r} in {usda_path}"
            )
        limits[joint_name] = (
            float(lower_match.group(1)),
            float(upper_match.group(1)),
        )

    return limits


def build_usda_scaled_joint_limits(usda_path=None, scale=USDA_DOF_LIMITS_SCALE):
    usda_path = Path(usda_path) if usda_path is not None else resolve_sharpa_rl_lab_usda_path()
    limits_deg = parse_usda_joint_limits_deg(usda_path, ACTIVE_JOINTS)
    scaled = {}
    for joint_name, (lower_deg, upper_deg) in limits_deg.items():
        lower_rad = math.radians(lower_deg) * scale
        upper_rad = math.radians(upper_deg) * scale
        if lower_rad > upper_rad:
            raise ValueError(
                f"Scaled USDA limits invalid for {joint_name}: "
                f"{lower_rad} > {upper_rad}"
            )
        scaled[joint_name] = (lower_rad, upper_rad)
    return scaled


OPEN_REST_JOINT_VALUES = {
    "right_thumb_CMC_FE": -0.1745,
    "right_thumb_CMC_AA": -0.3491,
    "right_thumb_MCP_FE": -0.5236,
    "right_thumb_MCP_AA": 0.0,
    "right_thumb_IP": 0.0,
    "right_index_MCP_FE": -0.17453294,
    "right_index_MCP_AA": -0.3491,
    "right_index_PIP": 0.0,
    "right_index_DIP": 0.0,
    "right_middle_MCP_FE": -0.17453294,
    "right_middle_MCP_AA": 0.0,
    "right_middle_PIP": 0.0,
    "right_middle_DIP": 0.0,
    "right_ring_MCP_FE": -0.17453294,
    "right_ring_MCP_AA": 0.2,
    "right_ring_PIP": 0.0,
    "right_ring_DIP": 0.0,
    "right_pinky_CMC": 0.0,
    "right_pinky_MCP_FE": -0.17453294,
    "right_pinky_MCP_AA": 0.3491,
    "right_pinky_PIP": 0.0,
    "right_pinky_DIP": 0.0,
}


FINGER_LINK_CHAINS = [
    [
        "right_thumb_CMC_VL",
        "right_thumb_MC",
        "right_thumb_MCP_VL",
        "right_thumb_PP",
        "right_thumb_DP",
        "right_thumb_elastomer",
        "right_thumb_fingertip",
    ],
    [
        "right_index_MCP_VL",
        "right_index_PP",
        "right_index_MP",
        "right_index_DP",
        "right_index_elastomer",
        "right_index_fingertip",
    ],
    [
        "right_middle_MCP_VL",
        "right_middle_PP",
        "right_middle_MP",
        "right_middle_DP",
        "right_middle_elastomer",
        "right_middle_fingertip",
    ],
    [
        "right_ring_MCP_VL",
        "right_ring_PP",
        "right_ring_MP",
        "right_ring_DP",
        "right_ring_elastomer",
        "right_ring_fingertip",
    ],
    [
        "right_pinky_MC",
        "right_pinky_MCP_VL",
        "right_pinky_PP",
        "right_pinky_MP",
        "right_pinky_DP",
        "right_pinky_elastomer",
        "right_pinky_fingertip",
    ],
]


class SharpaWave(RobotInterface):
    def __init__(self, urdf_path=None, joint_limit_profile="deploy"):
        super().__init__(urdf_path=urdf_path)
        self.source_urdf_path = self.urdf_path
        self.joint_limit_profile = joint_limit_profile
        if joint_limit_profile == "deploy":
            self.urdf_path = override_urdf_joint_limits(
                self.source_urdf_path,
                DEPLOY_DOF_LIMITS,
                profile_name="deploy_dof_limit",
            )
        elif joint_limit_profile == "usda09":
            self.urdf_path = override_urdf_joint_limits(
                self.source_urdf_path,
                build_usda_scaled_joint_limits(),
                profile_name="usda09",
            )
        elif joint_limit_profile != "urdf":
            raise ValueError(
                "Sharpa joint_limit_profile must be one of "
                f"{JOINT_LIMIT_PROFILES}, got {joint_limit_profile!r}"
            )

    def get_canonical_space(self):
        """
            Canonical space for placing objects in the hand frame.
        """
        box_min = np.array([0.04, -0.05, 0.09], dtype=np.float32)
        box_max = np.array([0.12, 0.05, 0.17], dtype=np.float32)
        return box_min, box_max

    def get_default_urdf_path(self):
        configured_path = os.environ.get("SHARPA_WAVE_URDF_PATH")
        if configured_path:
            return str(Path(configured_path).expanduser())
        if EXTERNAL_SHARPA_WAVE_URDF.is_file():
            # The upstream package resolver assumes the folder is named exactly
            # right_sharpa_wave. Stage a derived URDF using explicit mesh paths.
            import hashlib
            import tempfile
            import xml.etree.ElementTree as ET
            source = EXTERNAL_SHARPA_WAVE_URDF.read_text()
            source = source.replace("package://right_sharpa_wave/", str(EXTERNAL_SHARPA_WAVE_URDF.parent) + "/")
            root = ET.fromstring(source)
            for joint in root.findall("joint"):
                axis = joint.find("axis")
                if joint.get("type") == "fixed" and axis is not None and axis.get("xyz") == "0 0 0":
                    axis.set("xyz", "0 0 1")
            source = ET.tostring(root, encoding="unicode")
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            output = Path(tempfile.gettempdir()) / f"poise_sharpa_{digest}.urdf"
            output.write_text(source)
            return str(output)
        raise FileNotFoundError(EXTERNAL_SHARPA_WAVE_URDF)

    def get_contact_field_config(self):
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {},
            "patch_pos_dist_tol": 0.008,
            "patch_rot_dist_tol": 3.1415926 * 0.2,
        }

        for link in CONTACT_LINKS:
            config["movable_link"][link] = {
                "disabled_normal": []
            }

        return config

    def get_active_joints(self):
        return ACTIVE_JOINTS

    def get_resting_q(self, tree=None):
        if tree is None:
            tree = self.get_kinematics_tree()

        lower, upper = tree.get_actuated_joint_limit()
        q = np.zeros_like(lower, dtype=np.float32)
        joint_to_id = {name: idx for idx, name in enumerate(tree.actuated_joints)}

        for joint_name, joint_value in OPEN_REST_JOINT_VALUES.items():
            if joint_name in joint_to_id:
                q[joint_to_id[joint_name]] = joint_value

        return np.clip(q, lower, upper).astype(np.float32)

    def get_base_link(self):
        return "right_hand_C_MC"

    def get_static_links(self):
        return ["right_hand_C_MC"]

    def get_mesh_scale(self):
        return 1.0

    def get_white_list_pairs(self):
        pairs = []
        for chain in FINGER_LINK_CHAINS:
            pairs.extend([list(pair) for pair in combinations(chain, 2)])

        finger_links = sorted({link for chain in FINGER_LINK_CHAINS for link in chain})
        for link in finger_links:
            pairs.append(["right_hand_C_MC", link])

        non_thumb_links = sorted({link for chain in FINGER_LINK_CHAINS[1:] for link in chain})
        for thumb_base_link in ["right_thumb_CMC_VL", "right_thumb_MC"]:
            for link in non_thumb_links:
                pairs.append([thumb_base_link, link])

        return pairs


def build_robot(name="sharpa_wave", urdf_path=None, **kwargs):
    if name != "sharpa_wave":
        raise ValueError("Only Sharpa Wave is supported.")
    return SharpaWave(urdf_path=urdf_path, **kwargs)
