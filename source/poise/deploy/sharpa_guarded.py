"""Guarded Sharpa Wave position-control bridge with a latched emergency stop."""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sdk_runtime import sdk_command

from .sharpa_readonly import SharpaJointState


PROTOCOL_PREFIX = "__SHARPA_GUARDED__"

# The SDK is available only in the Python 3.10 container.  The helper owns the
# final safety boundary: every target is checked again inside the container,
# loss of the host heartbeat disables the hand, and ESTOP is latched.
GUARDED_HELPER_PROGRAM = r"""
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.join(os.environ["SHARPA_SDK_ROOT"], "python"))
from sharpa import ControlMode, ControlSource, SharpaWaveConfig, SharpaWaveManager

PREFIX = "__SHARPA_GUARDED__"
command_queue = queue.Queue()
stop_event = threading.Event()
stdin_closed = threading.Event()

def emit(kind, **payload):
    print(PREFIX + json.dumps({"kind": kind, **payload}, separators=(",", ":")), flush=True)

def request_stop(*_):
    stop_event.set()

def watch_stdin():
    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                command_queue.put(json.loads(stripped))
            except Exception as error:
                emit("protocol_error", message=f"invalid command JSON: {error}")
    finally:
        stdin_closed.set()

def check_error(error, operation):
    if int(error.code) != 0:
        raise RuntimeError(f"{operation} failed: code={error.code}, message={error.message}")

def query_value(operation):
    error, value = getattr(hand, operation)()
    check_error(error, operation)
    return value

def enum_name(value):
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    return str(value).split(".")[-1].upper()

def validate_vector(values, name):
    if not isinstance(values, list) or len(values) != 22:
        raise ValueError(f"{name} must contain 22 values.")
    result = [float(value) for value in values]
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"{name} contains NaN or Inf.")
    return result

def current_angles():
    state = hand.get_states()
    values = [float(value) for value in state.angles]
    if len(values) != 22:
        raise RuntimeError(f"SDK returned {len(values)} joint angles instead of 22.")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("SDK returned non-finite joint angles.")
    velocities = [float(value) for value in state.velocities]
    if len(velocities) != 22:
        raise RuntimeError(
            f"SDK returned {len(velocities)} joint velocities instead of 22."
        )
    if not all(math.isfinite(value) for value in velocities):
        raise RuntimeError("SDK returned non-finite joint velocities.")
    torques = [float(value) for value in state.torques]
    if len(torques) != 22:
        raise RuntimeError(
            f"SDK returned {len(torques)} joint torques instead of 22."
        )
    if not all(math.isfinite(value) for value in torques):
        raise RuntimeError("SDK returned non-finite joint torques.")
    return state, values

def disable_hand(reason, command_id=None):
    global armed, arm_transaction_started, estop_latched
    try:
        error = hand.set_enable_state(False)
        check_error(error, "set_enable_state(False)")
        disabled = True
        message = ""
    except Exception as error:
        disabled = False
        message = str(error)
    armed = False
    arm_transaction_started = False
    estop_latched = True
    emit(
        "estop",
        command_id=command_id,
        reason=reason,
        hardware_disable_succeeded=disabled,
        disable_error=message,
    )

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
threading.Thread(target=watch_stdin, daemon=True).start()

manager = None
hand = None
armed = False
arm_transaction_started = False
estop_latched = False
normal_stop = False
last_target = None
arm_angles = None
pregrasp_target = None
# Helper-owned start of the currently resumable pregrasp corridor.  It begins
# at the arm-time hold, but is advanced to the measured HOLD pose whenever a
# pregrasp attempt pauses.  This lets a bounded resume recover from small
# physical overshoot without reopening the original startup corridor.
pregrasp_resume_start = None
motion_phase = None
policy_pulse_used = False
policy_burst_started = False
policy_burst_completed = False
policy_burst_command_count = 0
exact_policy_started = False
exact_policy_completed = False
exact_policy_command_count = 0
last_heartbeat_s = None
commands_accepted = 0

try:
    read_rate_hz = float(os.environ.get("SHARPA_READ_RATE_HZ", "60"))
    heartbeat_timeout_s = float(os.environ.get("SHARPA_HEARTBEAT_TIMEOUT_S", "0.35"))
    max_step_rad = float(os.environ.get("SHARPA_MAX_STEP_RAD", "0.015"))
    policy_pulse_max_step_rad = float(
        os.environ.get("SHARPA_POLICY_PULSE_MAX_STEP_RAD", "0.010")
    )
    policy_burst_max_step_rad = float(
        os.environ.get("SHARPA_POLICY_BURST_MAX_STEP_RAD", "0.010")
    )
    policy_burst_cumulative_limit_rad = float(
        os.environ.get("SHARPA_POLICY_BURST_CUMULATIVE_LIMIT_RAD", "0.025")
    )
    policy_burst_max_commands = int(
        os.environ.get("SHARPA_POLICY_BURST_MAX_COMMANDS", "5")
    )
    exact_policy_enabled = (
        os.environ.get("SHARPA_EXACT_POLICY_ENABLED", "0") == "1"
    )
    exact_policy_max_commands = int(
        os.environ.get("SHARPA_EXACT_POLICY_MAX_COMMANDS", "20")
    )
    if exact_policy_max_commands <= 0:
        raise ValueError(
            "The helper exact policy command count must be positive."
        )
    exact_policy_confirmation = "EXACT_POLICY_CONTRACT"
    pregrasp_fraction = float(os.environ.get("SHARPA_PREGRASP_FRACTION", "0.70"))
    max_speed_coeff = float(os.environ.get("SHARPA_MAX_SPEED_COEFF", "0.30"))
    max_current_coeff = float(os.environ.get("SHARPA_MAX_CURRENT_COEFF", "0.30"))
    scalar_config = (
        read_rate_hz,
        heartbeat_timeout_s,
        max_step_rad,
        policy_pulse_max_step_rad,
        policy_burst_max_step_rad,
        policy_burst_cumulative_limit_rad,
        pregrasp_fraction,
        max_speed_coeff,
        max_current_coeff,
    )
    if not all(math.isfinite(value) for value in scalar_config):
        raise ValueError("Guarded helper scalar configuration contains NaN or Inf.")
    if max_step_rad > 0.015:
        raise ValueError("The helper absolute max step is 0.015 rad.")
    if policy_pulse_max_step_rad > 0.010:
        raise ValueError("The helper absolute policy pulse cap is 0.010 rad.")
    if policy_pulse_max_step_rad > max_step_rad:
        raise ValueError(
            "The policy pulse cap cannot exceed the helper max step."
        )
    if policy_burst_max_step_rad > 0.010:
        raise ValueError("The helper absolute policy burst step cap is 0.010 rad.")
    if policy_burst_max_step_rad > max_step_rad:
        raise ValueError(
            "The policy burst step cap cannot exceed the helper max step."
        )
    if policy_burst_cumulative_limit_rad > 0.025:
        raise ValueError(
            "The helper absolute policy burst cumulative cap is 0.025 rad."
        )
    if policy_burst_cumulative_limit_rad < policy_burst_max_step_rad:
        raise ValueError(
            "The policy burst cumulative cap must cover one burst step."
        )
    if policy_burst_max_commands not in (5, 10, 20):
        raise ValueError(
            "The helper policy burst command count must be 5, 10, or 20."
        )
    policy_burst_confirmation = (
        "FIVE_CYCLES_0.010_TOTAL_0.025"
        if policy_burst_max_commands == 5
        else (
            "TEN_CYCLES_0.010_DRIVE_0.018_HARD_0.025"
            if policy_burst_max_commands == 10
            else "TWENTY_CYCLES_0.010_DRIVE_0.018_HARD_0.025"
        )
    )
    if not 0.0 < pregrasp_fraction < 1.0:
        raise ValueError("The helper pregrasp fraction must be in (0, 1).")
    if max_speed_coeff > 0.30:
        raise ValueError("The helper absolute speed coefficient cap is 0.30.")
    if max_current_coeff > 0.30:
        raise ValueError("The helper absolute current coefficient cap is 0.30.")
    if heartbeat_timeout_s > 0.50:
        raise ValueError("The helper absolute heartbeat timeout cap is 0.50 seconds.")
    lower = validate_vector(json.loads(os.environ["SHARPA_HARD_LOWER_SDK"]), "hard lower limits")
    upper = validate_vector(json.loads(os.environ["SHARPA_HARD_UPPER_SDK"]), "hard upper limits")
    exact_lower_json = os.environ.get("SHARPA_EXACT_POLICY_LOWER_SDK", "").strip()
    exact_upper_json = os.environ.get("SHARPA_EXACT_POLICY_UPPER_SDK", "").strip()
    exact_lower = (
        validate_vector(json.loads(exact_lower_json), "exact policy lower limits")
        if exact_lower_json
        else None
    )
    exact_upper = (
        validate_vector(json.loads(exact_upper_json), "exact policy upper limits")
        if exact_upper_json
        else None
    )
    session_target_json = os.environ.get("SHARPA_SESSION_TARGET_SDK", "").strip()
    session_target = (
        validate_vector(json.loads(session_target_json), "session target")
        if session_target_json
        else None
    )
    session_measured_target_json = os.environ.get(
        "SHARPA_SESSION_MEASURED_TARGET_SDK", ""
    ).strip()
    session_measured_target = (
        validate_vector(
            json.loads(session_measured_target_json),
            "session measured target",
        )
        if session_measured_target_json
        else session_target
    )
    close_corridor_tolerance_rad = float(
        os.environ.get("SHARPA_CLOSE_CORRIDOR_TOLERANCE_RAD", "0.020")
    )
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError("Each hard lower limit must be below its upper limit.")
    if exact_policy_enabled and (exact_lower is None or exact_upper is None):
        raise ValueError(
            "Exact policy mode requires an independent 22D policy limit envelope."
        )
    if exact_lower is not None and (
        exact_upper is None
        or any(lo >= hi for lo, hi in zip(exact_lower, exact_upper))
    ):
        raise ValueError("Exact policy joint limits are invalid.")
    if exact_lower is not None and any(
        policy_lo < hard_lo or policy_hi > hard_hi
        for policy_lo, policy_hi, hard_lo, hard_hi in zip(
            exact_lower, exact_upper, lower, upper
        )
    ):
        raise ValueError(
            "Exact policy limits must remain inside the helper hard limits."
        )
    if session_target is not None and any(
        target < lo or target > hi
        for target, lo, hi in zip(session_target, lower, upper)
    ):
        raise ValueError("Session target lies outside the hard joint envelope.")
    if session_measured_target is not None and any(
        target < lo or target > hi
        for target, lo, hi in zip(session_measured_target, lower, upper)
    ):
        raise ValueError(
            "Session measured target lies outside the hard joint envelope."
        )
    if session_target is None:
        raise ValueError(
            "A session target is required for guarded pregrasp/close phases."
        )
    if session_measured_target is None:
        raise ValueError(
            "A session measured target is required for guarded close phases."
        )
    if (
        not math.isfinite(close_corridor_tolerance_rad)
        or close_corridor_tolerance_rad < 0.0
        or close_corridor_tolerance_rad > 0.020
    ):
        raise ValueError(
            "Close corridor tolerance must be finite and within [0, 0.020] rad."
        )
    if (
        heartbeat_timeout_s <= 0.0
        or max_step_rad <= 0.0
        or policy_pulse_max_step_rad <= 0.0
        or policy_burst_max_step_rad <= 0.0
        or policy_burst_cumulative_limit_rad <= 0.0
    ):
        raise ValueError(
            "Heartbeat timeout and all command-step limits must be positive."
        )

    manager = SharpaWaveManager.get_instance()
    requested_serial = os.environ.get("SHARPA_READ_SERIAL", "").strip()
    discovery_timeout_s = float(os.environ.get("SHARPA_DISCOVERY_TIMEOUT_S", "12"))
    if not math.isfinite(discovery_timeout_s) or discovery_timeout_s <= 0.0:
        raise ValueError("Discovery timeout must be finite and positive.")
    deadline = time.monotonic() + discovery_timeout_s
    serials = []
    while not stop_event.is_set() and time.monotonic() < deadline:
        serials = list(manager.get_all_device_sn())
        if requested_serial:
            serials = [serial for serial in serials if str(serial) == requested_serial]
        if serials:
            break
        time.sleep(0.1)
    if not serials:
        raise RuntimeError("No matching Sharpa Wave discovered.")

    config = SharpaWaveConfig()
    config.disable_tactile = True
    config.disable_sync_time = True
    hand = manager.connect(serials[0], config)
    if hand is None:
        raise RuntimeError("SharpaWaveManager.connect returned None.")
    hand.start()

    ready_deadline = time.monotonic() + 3.0
    state = None
    while time.monotonic() < ready_deadline:
        state, angles = current_angles()
        if int(state.sequence) > 0:
            break
        time.sleep(0.02)
    if state is None or int(state.sequence) <= 0:
        raise RuntimeError("Sharpa Wave did not produce a valid joint state.")

    info = hand.get_device_info()
    control_mode = query_value("get_control_mode")
    control_source = query_value("get_control_source")
    enable_state = bool(query_value("get_enable_state"))
    speed_coeff = float(query_value("get_speed_coeff"))
    current_coeff = float(query_value("get_current_coeff"))
    if not all(
        math.isfinite(value)
        for value in (speed_coeff, current_coeff)
    ):
        raise RuntimeError("Sharpa device coefficients contain NaN or Inf.")
    emit(
        "ready",
        serial=str(info.sn),
        hand_side=str(info.hand_side),
        control_mode=enum_name(control_mode),
        control_source=enum_name(control_source),
        enable_state=enable_state,
        speed_coeff=speed_coeff,
        current_coeff=current_coeff,
        max_step_rad=max_step_rad,
        policy_pulse_max_step_rad=policy_pulse_max_step_rad,
        policy_burst_max_step_rad=policy_burst_max_step_rad,
        policy_burst_cumulative_limit_rad=policy_burst_cumulative_limit_rad,
        policy_burst_max_commands=policy_burst_max_commands,
        exact_policy_enabled=exact_policy_enabled,
        exact_policy_max_commands=exact_policy_max_commands,
        pregrasp_fraction=pregrasp_fraction,
        close_corridor_tolerance_rad=close_corridor_tolerance_rad,
        heartbeat_timeout_s=heartbeat_timeout_s,
    )

    period_s = 1.0 / max(read_rate_hz, 1.0)
    next_state_s = time.monotonic()
    while not stop_event.is_set():
        now = time.monotonic()
        if armed and (
            stdin_closed.is_set()
            or last_heartbeat_s is None
            or now - last_heartbeat_s > heartbeat_timeout_s
        ):
            disable_hand("host_heartbeat_timeout_or_disconnect")

        try:
            command = command_queue.get(timeout=0.005)
        except queue.Empty:
            command = None
        if command is not None:
            command_id = str(command.get("command_id", ""))
            kind = str(command.get("kind", ""))
            try:
                if kind == "heartbeat":
                    if armed and not estop_latched:
                        last_heartbeat_s = time.monotonic()
                    emit("ack", command_id=command_id, operation=kind)
                elif kind == "arm":
                    if estop_latched:
                        raise RuntimeError("ESTOP is latched; restart after inspecting the hand.")
                    if armed:
                        raise RuntimeError("Controller is already armed.")
                    arm_transaction_started = True
                    if command.get("confirmation") != "POISE":
                        raise RuntimeError("Arm confirmation token is invalid.")
                    speed = float(command["speed_coeff"])
                    current = float(command["current_coeff"])
                    if not (0.0 < speed <= max_speed_coeff):
                        raise ValueError(f"speed_coeff must be in (0, {max_speed_coeff}].")
                    if not (0.0 < current <= max_current_coeff):
                        raise ValueError(f"current_coeff must be in (0, {max_current_coeff}].")
                    observed_current = float(query_value("get_current_coeff"))
                    if not math.isfinite(observed_current):
                        raise RuntimeError(
                            "Observed current coefficient is NaN or Inf."
                        )
                    if observed_current > current + 1.0e-3:
                        raise RuntimeError(
                            "Observed current coefficient exceeds the requested "
                            f"guarded maximum: maximum={current:.4f}, "
                            f"observed={observed_current:.4f}. Preconfigure it with "
                            "the trusted deployment setup."
                        )
                    state, angles = current_angles()
                    for index, value in enumerate(angles):
                        if value < lower[index] or value > upper[index]:
                            raise RuntimeError(
                                f"Measured joint {index}={value:.5f} is outside the hard "
                                f"envelope [{lower[index]:.5f}, {upper[index]:.5f}]."
                            )
                    check_error(
                        hand.set_control_mode(ControlMode.POSITION),
                        "set_control_mode(POSITION)",
                    )
                    check_error(hand.set_speed_coeff(speed), "set_speed_coeff")
                    check_error(
                        hand.set_control_source(ControlSource.SDK),
                        "set_control_source(SDK)",
                    )
                    if stop_event.is_set() or stdin_closed.is_set():
                        raise RuntimeError("Host disconnected during arm transaction.")
                    mode = enum_name(query_value("get_control_mode"))
                    source = enum_name(query_value("get_control_source"))
                    if mode != "POSITION" or source != "SDK":
                        raise RuntimeError(
                            f"POSITION/SDK reconfiguration did not persist: "
                            f"observed {mode}/{source}."
                        )
                    enabled = bool(query_value("get_enable_state"))
                    if not enabled:
                        check_error(hand.set_enable_state(True), "set_enable_state(True)")
                    check_error(hand.set_joint_position(angles, False), "initial hold")
                    last_target = angles
                    arm_angles = angles
                    pregrasp_resume_start = angles.copy()
                    pregrasp_target = [
                        start + pregrasp_fraction * (target - start)
                        for start, target in zip(arm_angles, session_target)
                    ]
                    motion_phase = "pregrasp"
                    last_heartbeat_s = time.monotonic()
                    armed = True
                    arm_transaction_started = False
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        enabled=True,
                        held_angles_rad=angles,
                        # Return the helper-owned endpoint so the host planner
                        # does not independently recompute a corridor boundary
                        # from separately serialized floating-point values.
                        pregrasp_target_rad=pregrasp_target,
                    )
                elif kind == "target":
                    if not armed or estop_latched:
                        raise RuntimeError("Target rejected because controller is not armed.")
                    target = validate_vector(command.get("angles_sdk_rad"), "target")
                    if last_target is None:
                        raise RuntimeError("No previous guarded target is available.")
                    requested_phase = str(command.get("motion_phase", ""))
                    if requested_phase not in ("pregrasp", "close"):
                        raise ValueError(
                            "Target motion_phase must be 'pregrasp' or 'close'."
                        )
                    if motion_phase == "close":
                        if requested_phase != "close":
                            raise RuntimeError(
                                "Guarded close phase is latched; pregrasp cannot resume."
                            )
                    elif motion_phase == "policy_pulse":
                        raise RuntimeError(
                            "Guarded motion is locked after the one-shot "
                            "policy pulse; restart the controller."
                        )
                    elif motion_phase == "policy_burst":
                        raise RuntimeError(
                            "Guarded motion is locked to the policy burst "
                            "protocol; regular targets are forbidden."
                        )
                    elif motion_phase == "exact_policy":
                        raise RuntimeError(
                            "Guarded motion is locked to exact policy mode; "
                            "regular targets are forbidden."
                        )
                    elif motion_phase != "pregrasp":
                        raise RuntimeError("No guarded motion phase is active.")
                    if requested_phase == "pregrasp":
                        if pregrasp_target is None:
                            raise RuntimeError(
                                "No guarded pregrasp target is available."
                            )
                        projected_target = []
                        maximum_projection = 0.0
                        if pregrasp_resume_start is None:
                            raise RuntimeError(
                                "No guarded pregrasp resume start is available."
                            )
                        for value, start, endpoint in zip(
                            target,
                            pregrasp_resume_start,
                            pregrasp_target,
                        ):
                            corridor_lower = min(start, endpoint)
                            corridor_upper = max(start, endpoint)
                            projected = min(max(value, corridor_lower), corridor_upper)
                            maximum_projection = max(
                                maximum_projection,
                                abs(projected - value),
                            )
                            projected_target.append(projected)
                        if maximum_projection > 5.0e-4:
                            raise ValueError(
                                "Pregrasp target disagrees with the helper-owned "
                                "corridor by "
                                f"{maximum_projection:.6f} rad."
                            )
                        # Resolve only sub-milliradian arm-state/serialization
                        # skew by projecting inward to the helper-owned safety
                        # corridor. Larger disagreements remain fail-closed.
                        target = projected_target
                    for index, (value, previous) in enumerate(zip(target, last_target)):
                        if value < lower[index] or value > upper[index]:
                            raise ValueError(
                                f"Target joint {index}={value:.5f} is outside "
                                f"[{lower[index]:.5f}, {upper[index]:.5f}]."
                            )
                        if abs(value - previous) > max_step_rad + 1.0e-7:
                            raise ValueError(
                                f"Target joint {index} step {value - previous:+.5f} exceeds "
                                f"{max_step_rad:.5f} rad."
                            )
                        if requested_phase == "close":
                            corridor_lower = min(
                                arm_angles[index],
                                session_target[index],
                                session_measured_target[index],
                            ) - close_corridor_tolerance_rad
                            corridor_upper = max(
                                arm_angles[index],
                                session_target[index],
                                session_measured_target[index],
                            ) + close_corridor_tolerance_rad
                            if value < corridor_lower or value > corridor_upper:
                                raise ValueError(
                                    f"Target joint {index}={value:.5f} leaves the "
                                    "startup-to-poise command corridor "
                                    f"[{corridor_lower:.5f}, {corridor_upper:.5f}]."
                                )
                        else:
                            if pregrasp_target is None:
                                raise RuntimeError(
                                    "No guarded pregrasp target is available."
                                )
                            if pregrasp_resume_start is None:
                                raise RuntimeError(
                                    "No guarded pregrasp resume start is available."
                                )
                            pregrasp_lower = min(
                                pregrasp_resume_start[index],
                                pregrasp_target[index],
                            )
                            pregrasp_upper = max(
                                pregrasp_resume_start[index],
                                pregrasp_target[index],
                            )
                            if (
                                value < pregrasp_lower - 1.0e-7
                                or value > pregrasp_upper + 1.0e-7
                            ):
                                raise ValueError(
                                    f"Pregrasp target joint {index}={value:.5f} "
                                    "leaves the startup-to-pregrasp corridor "
                                    f"[{pregrasp_lower:.5f}, "
                                    f"{pregrasp_upper:.5f}]."
                                )
                    check_error(hand.set_joint_position(target, False), "set_joint_position")
                    last_target = target
                    motion_phase = requested_phase
                    last_heartbeat_s = time.monotonic()
                    commands_accepted += 1
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        commands_accepted=commands_accepted,
                        # Echo the exact helper-accepted target. The host uses
                        # this value to keep its planner state synchronized
                        # with the final motor safety boundary.
                        accepted_angles_sdk_rad=target,
                    )
                elif kind == "policy_pulse":
                    if exact_policy_enabled:
                        raise RuntimeError(
                            "This helper session is reserved for exact policy mode."
                        )
                    if not armed or estop_latched:
                        raise RuntimeError(
                            "Policy pulse rejected because controller is not armed."
                        )
                    if command.get("confirmation") != "ONE_SHOT_0.010_RAD":
                        raise RuntimeError(
                            "Policy pulse confirmation token is invalid."
                        )
                    if policy_pulse_used:
                        raise RuntimeError(
                            "The one-shot policy pulse was already used."
                        )
                    if policy_burst_started:
                        raise RuntimeError(
                            "Policy pulse is forbidden after a policy burst."
                        )
                    if exact_policy_started:
                        raise RuntimeError(
                            "Policy pulse is forbidden after exact policy mode."
                        )
                    if motion_phase != "close":
                        raise RuntimeError(
                            "Policy pulse requires the latched guarded close phase."
                        )
                    if last_target is None or session_target is None:
                        raise RuntimeError(
                            "Policy pulse requires the initial grasp target."
                        )
                    target = validate_vector(
                        command.get("angles_sdk_rad"),
                        "policy pulse target",
                    )
                    for index, (value, previous, row_target) in enumerate(
                        zip(target, last_target, session_target)
                    ):
                        if abs(previous - row_target) > 1.0e-6:
                            raise RuntimeError(
                                f"Policy pulse base joint {index} is not the "
                                "helper-owned initial grasp q_cmd."
                            )
                        if value < lower[index] or value > upper[index]:
                            raise ValueError(
                                f"Policy pulse joint {index}={value:.5f} is "
                                "outside the hard joint envelope."
                            )
                        if (
                            abs(value - previous)
                            > policy_pulse_max_step_rad + 1.0e-7
                        ):
                            raise ValueError(
                                f"Policy pulse joint {index} step "
                                f"{value - previous:+.5f} exceeds "
                                f"{policy_pulse_max_step_rad:.5f} rad."
                            )
                    check_error(
                        hand.set_joint_position(target, False),
                        "one-shot policy pulse",
                    )
                    last_target = target
                    motion_phase = "policy_pulse"
                    policy_pulse_used = True
                    last_heartbeat_s = time.monotonic()
                    commands_accepted += 1
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        commands_accepted=commands_accepted,
                        policy_pulse_used=True,
                    )
                elif kind == "policy_burst":
                    if exact_policy_enabled:
                        raise RuntimeError(
                            "This helper session is reserved for exact policy mode."
                        )
                    if not armed or estop_latched:
                        raise RuntimeError(
                            "Policy burst rejected because controller is not armed."
                        )
                    if command.get("confirmation") != policy_burst_confirmation:
                        raise RuntimeError(
                            "Policy burst confirmation token is invalid."
                        )
                    if policy_pulse_used:
                        raise RuntimeError(
                            "Policy burst is forbidden after a policy pulse."
                        )
                    if exact_policy_started:
                        raise RuntimeError(
                            "Policy burst is forbidden after exact policy mode."
                        )
                    if policy_burst_completed:
                        raise RuntimeError(
                            "The policy burst is complete and permanently locked."
                        )
                    burst_index = int(command.get("burst_index", -1))
                    if burst_index != policy_burst_command_count:
                        raise RuntimeError(
                            f"Policy burst index {burst_index} does not match "
                            f"expected {policy_burst_command_count}."
                        )
                    if not policy_burst_started:
                        if motion_phase != "close":
                            raise RuntimeError(
                                "Policy burst requires the latched guarded close phase."
                            )
                        if last_target is None or session_target is None:
                            raise RuntimeError(
                                "Policy burst requires the initial grasp target."
                            )
                        for index, (previous, row_target) in enumerate(
                            zip(last_target, session_target)
                        ):
                            if abs(previous - row_target) > 1.0e-6:
                                raise RuntimeError(
                                    f"Policy burst base joint {index} is not "
                                    "the helper-owned initial grasp q_cmd."
                                )
                    elif motion_phase != "policy_burst":
                        raise RuntimeError(
                            "Policy burst phase was interrupted and is locked."
                        )
                    if policy_burst_command_count >= policy_burst_max_commands:
                        raise RuntimeError(
                            "Policy burst command count exceeds its hard cap."
                        )
                    target = validate_vector(
                        command.get("angles_sdk_rad"),
                        "policy burst target",
                    )
                    for index, (value, previous, row_target) in enumerate(
                        zip(target, last_target, session_target)
                    ):
                        if value < lower[index] or value > upper[index]:
                            raise ValueError(
                                f"Policy burst joint {index}={value:.5f} is "
                                "outside the hard joint envelope."
                            )
                        if (
                            abs(value - previous)
                            > policy_burst_max_step_rad + 1.0e-7
                        ):
                            raise ValueError(
                                f"Policy burst joint {index} step "
                                f"{value - previous:+.5f} exceeds "
                                f"{policy_burst_max_step_rad:.5f} rad."
                            )
                        if (
                            abs(value - row_target)
                            > policy_burst_cumulative_limit_rad + 1.0e-7
                        ):
                            raise ValueError(
                                f"Policy burst joint {index} leaves the "
                                "initial grasp cumulative envelope."
                            )
                    check_error(
                        hand.set_joint_position(target, False),
                        "bounded policy burst",
                    )
                    last_target = target
                    motion_phase = "policy_burst"
                    policy_burst_started = True
                    policy_burst_command_count += 1
                    if policy_burst_command_count >= policy_burst_max_commands:
                        policy_burst_completed = True
                    last_heartbeat_s = time.monotonic()
                    commands_accepted += 1
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        commands_accepted=commands_accepted,
                        policy_burst_command_count=policy_burst_command_count,
                        policy_burst_completed=policy_burst_completed,
                    )
                elif kind == "exact_policy":
                    if not exact_policy_enabled:
                        raise RuntimeError(
                            "Exact policy protocol is disabled for this helper session."
                        )
                    if not armed or estop_latched:
                        raise RuntimeError(
                            "Exact policy rejected because controller is not armed."
                        )
                    if command.get("confirmation") != exact_policy_confirmation:
                        raise RuntimeError(
                            "Exact policy confirmation token is invalid."
                        )
                    if policy_pulse_used or policy_burst_started:
                        raise RuntimeError(
                            "Exact policy is forbidden after another policy mode."
                        )
                    if exact_policy_completed:
                        raise RuntimeError(
                            "Exact policy is complete and permanently locked."
                        )
                    exact_index = int(command.get("exact_index", -1))
                    if exact_index != exact_policy_command_count:
                        raise RuntimeError(
                            f"Exact policy index {exact_index} does not match "
                            f"expected {exact_policy_command_count}."
                        )
                    if not exact_policy_started:
                        if motion_phase != "close":
                            raise RuntimeError(
                                "Exact policy requires the latched guarded close phase."
                            )
                        if last_target is None or session_target is None:
                            raise RuntimeError(
                                "Exact policy requires the initial grasp target."
                            )
                        for index, (previous, row_target) in enumerate(
                            zip(last_target, session_target)
                        ):
                            if abs(previous - row_target) > 1.0e-6:
                                raise RuntimeError(
                                    f"Exact policy base joint {index} is not "
                                    "the helper-owned initial grasp q_cmd."
                                )
                    elif motion_phase != "exact_policy":
                        raise RuntimeError(
                            "Exact policy phase was interrupted and is locked."
                        )
                    if (
                        exact_policy_command_count
                        >= exact_policy_max_commands
                    ):
                        raise RuntimeError(
                            "Exact policy command count exceeds its configured "
                            "hard cap."
                        )
                    target = validate_vector(
                        command.get("angles_sdk_rad"),
                        "exact q_policy target",
                    )
                    for index, value in enumerate(target):
                        if (
                            value < exact_lower[index] - 1.0e-7
                            or value > exact_upper[index] + 1.0e-7
                        ):
                            raise ValueError(
                                f"Exact q_policy joint {index}={value:.5f} "
                                "violates the policy joint limits "
                                f"[{exact_lower[index]:.5f},"
                                f"{exact_upper[index]:.5f}]."
                            )
                    # Exact simulator-contract mode intentionally has no
                    # per-cycle step or row-local envelope. Policy joint
                    # limits and the configured command count remain
                    # independently enforced here.
                    check_error(
                        hand.set_joint_position(target, False),
                        "exact policy contract target",
                    )
                    last_target = target
                    motion_phase = "exact_policy"
                    exact_policy_started = True
                    exact_policy_command_count += 1
                    if (
                        exact_policy_command_count
                        >= exact_policy_max_commands
                    ):
                        exact_policy_completed = True
                    last_heartbeat_s = time.monotonic()
                    commands_accepted += 1
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        commands_accepted=commands_accepted,
                        exact_policy_command_count=exact_policy_command_count,
                        exact_policy_completed=exact_policy_completed,
                    )
                elif kind == "refresh_target":
                    if not armed or estop_latched:
                        raise RuntimeError(
                            "Target refresh rejected because controller is not armed."
                        )
                    if last_target is None:
                        raise RuntimeError(
                            "No previously validated guarded target is available to refresh."
                        )
                    if motion_phase not in ("pregrasp", "close"):
                        raise RuntimeError("No guarded motion phase is active.")
                    # Refresh only the helper-owned target.  The host cannot
                    # supply angles through this operation, so this is a
                    # zero-step resend and cannot bypass the target guards.
                    check_error(
                        hand.set_joint_position(last_target, False),
                        "refresh last target",
                    )
                    last_heartbeat_s = time.monotonic()
                    emit("ack", command_id=command_id, operation=kind)
                elif kind == "hold":
                    if not armed or estop_latched:
                        raise RuntimeError("Hold rejected because controller is not armed.")
                    state, angles = current_angles()
                    check_error(hand.set_joint_position(angles, False), "hold current position")
                    last_target = angles
                    if motion_phase == "pregrasp":
                        # A measured joint can settle slightly beyond the old
                        # arm-time endpoint.  Make this exact helper-measured
                        # HOLD pose the start of the next bounded pregrasp
                        # resume corridor.  Hard limits and max_step_rad are
                        # still checked independently for every command.
                        pregrasp_resume_start = angles.copy()
                    if policy_burst_started:
                        policy_burst_completed = True
                    if exact_policy_started:
                        exact_policy_completed = True
                    last_heartbeat_s = time.monotonic()
                    emit(
                        "ack",
                        command_id=command_id,
                        operation=kind,
                        held_angles_rad=angles,
                        pregrasp_resume_start_rad=(
                            pregrasp_resume_start
                            if motion_phase == "pregrasp"
                            else None
                        ),
                    )
                elif kind == "estop":
                    disable_hand(str(command.get("reason", "host_estop")), command_id)
                elif kind == "normal_stop":
                    if armed:
                        state, angles = current_angles()
                        check_error(hand.set_joint_position(angles, False), "normal-stop hold")
                        last_target = angles
                    armed = False
                    normal_stop = True
                    emit("ack", command_id=command_id, operation=kind)
                    stop_event.set()
                else:
                    raise ValueError(f"Unknown command kind {kind!r}.")
            except BaseException as error:
                emit(
                    "command_error",
                    command_id=command_id,
                    operation=kind,
                    message=str(error),
                    traceback=traceback.format_exc(),
                )
                if armed or kind == "arm":
                    disable_hand(f"command_failure:{kind}", command_id)

        now = time.monotonic()
        if now >= next_state_s:
            state, angles = current_angles()
            emit(
                "state",
                sequence=int(state.sequence),
                device_timestamp=int(state.timestamp),
                angles_rad=angles,
                velocities_rad_s=[float(value) for value in state.velocities],
                torques_nm=[float(value) for value in state.torques],
                armed=armed,
                estop_latched=estop_latched,
                commands_accepted=commands_accepted,
                motion_phase=motion_phase,
                policy_pulse_used=policy_pulse_used,
                policy_burst_started=policy_burst_started,
                policy_burst_completed=policy_burst_completed,
                policy_burst_command_count=policy_burst_command_count,
                exact_policy_started=exact_policy_started,
                exact_policy_completed=exact_policy_completed,
                exact_policy_command_count=exact_policy_command_count,
            )
            next_state_s = now + period_s

    if (armed or arm_transaction_started) and not normal_stop:
        disable_hand("helper_shutdown_while_armed")
except BaseException as error:
    emit("error", message=str(error), traceback=traceback.format_exc())
    if hand is not None and (armed or arm_transaction_started):
        disable_hand("unhandled_helper_exception")
finally:
    try:
        if hand is not None:
            hand.stop()
    finally:
        if manager is not None:
            manager.disconnect_all()
    emit(
        "stopped",
        normal_stop=normal_stop,
        estop_latched=estop_latched,
        commands_accepted=commands_accepted,
    )
"""


@dataclass(frozen=True)
class GuardedControlStatus:
    armed: bool
    estop_latched: bool
    commands_accepted: int
    motion_phase: str | None = None


class GuardedSharpaControlClient:
    """Host client for the independently watchdog-protected SDK helper."""

    def __init__(
        self,
        hard_lower_sdk_rad: np.ndarray,
        hard_upper_sdk_rad: np.ndarray,
        *,
        session_target_sdk_rad: np.ndarray | None = None,
        session_measured_target_sdk_rad: np.ndarray | None = None,
        container: str = "poise-sharpa",
        sdk_root: str = "/opt/sharpa-wave-sdk",
        sdk_python: str = "python3",
        serial: str | None = None,
        read_rate_hz: float = 60.0,
        startup_timeout_s: float = 15.0,
        stale_after_s: float = 0.20,
        heartbeat_timeout_s: float = 0.35,
        max_step_rad: float = 0.015,
        policy_pulse_max_step_rad: float = 0.010,
        policy_burst_max_step_rad: float = 0.010,
        policy_burst_cumulative_limit_rad: float = 0.025,
        policy_burst_max_commands: int = 5,
        exact_policy_enabled: bool = False,
        exact_policy_max_commands: int = 20,
        exact_policy_lower_sdk_rad: np.ndarray | None = None,
        exact_policy_upper_sdk_rad: np.ndarray | None = None,
        pregrasp_fraction: float = 0.70,
        close_corridor_tolerance_rad: float = 0.020,
        max_speed_coeff: float = 0.30,
        max_current_coeff: float = 0.30,
    ) -> None:
        lower = np.asarray(hard_lower_sdk_rad, dtype=np.float64)
        upper = np.asarray(hard_upper_sdk_rad, dtype=np.float64)
        if lower.shape != (22,) or upper.shape != (22,):
            raise ValueError("Guarded hard limits must both be 22D.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Guarded hard limits contain NaN or Inf.")
        if np.any(lower >= upper):
            raise ValueError("Each guarded lower limit must be below its upper limit.")
        session_target = (
            None
            if session_target_sdk_rad is None
            else np.asarray(session_target_sdk_rad, dtype=np.float64)
        )
        if session_target is not None and (
            session_target.shape != (22,)
            or not np.all(np.isfinite(session_target))
        ):
            raise ValueError("session_target_sdk_rad must be finite and 22D.")
        if session_target is not None and (
            np.any(session_target < lower) or np.any(session_target > upper)
        ):
            raise ValueError(
                "session_target_sdk_rad lies outside the guarded hard limits."
            )
        session_measured_target = (
            None
            if session_target is None
            else (
                session_target.copy()
                if session_measured_target_sdk_rad is None
                else np.asarray(
                    session_measured_target_sdk_rad,
                    dtype=np.float64,
                )
            )
        )
        if session_measured_target is not None and (
            session_measured_target.shape != (22,)
            or not np.all(np.isfinite(session_measured_target))
        ):
            raise ValueError(
                "session_measured_target_sdk_rad must be finite and 22D."
            )
        if session_measured_target is not None and (
            np.any(session_measured_target < lower)
            or np.any(session_measured_target > upper)
        ):
            raise ValueError(
                "session_measured_target_sdk_rad lies outside the guarded hard limits."
            )
        exact_lower = (
            None
            if exact_policy_lower_sdk_rad is None
            else np.asarray(exact_policy_lower_sdk_rad, dtype=np.float64)
        )
        exact_upper = (
            None
            if exact_policy_upper_sdk_rad is None
            else np.asarray(exact_policy_upper_sdk_rad, dtype=np.float64)
        )
        if exact_policy_enabled and (
            exact_lower is None or exact_upper is None
        ):
            raise ValueError(
                "Exact policy mode requires explicit 22D policy limits."
            )
        if exact_policy_max_commands <= 0:
            raise ValueError(
                "exact_policy_max_commands must be positive."
            )
        if exact_lower is not None and (
            exact_upper is None
            or exact_lower.shape != (22,)
            or exact_upper.shape != (22,)
            or not np.all(np.isfinite(exact_lower))
            or not np.all(np.isfinite(exact_upper))
            or np.any(exact_lower >= exact_upper)
        ):
            raise ValueError("Exact policy limits must be finite valid 22D arrays.")
        if exact_lower is not None and (
            np.any(exact_lower < lower)
            or np.any(exact_upper > upper)
        ):
            raise ValueError(
                "Exact policy limits must remain inside guarded hard limits."
            )
        scalar_config = (
            read_rate_hz,
            startup_timeout_s,
            stale_after_s,
            heartbeat_timeout_s,
            max_step_rad,
            policy_pulse_max_step_rad,
            policy_burst_max_step_rad,
            policy_burst_cumulative_limit_rad,
            pregrasp_fraction,
            close_corridor_tolerance_rad,
            max_speed_coeff,
            max_current_coeff,
        )
        if not all(math.isfinite(float(value)) for value in scalar_config):
            raise ValueError(
                "Guarded client scalar configuration contains NaN or Inf."
            )
        if (
            read_rate_hz <= 0.0
            or startup_timeout_s <= 0.0
            or stale_after_s <= 0.0
            or heartbeat_timeout_s <= 0.0
            or max_step_rad <= 0.0
            or policy_pulse_max_step_rad <= 0.0
            or policy_burst_max_step_rad <= 0.0
            or policy_burst_cumulative_limit_rad <= 0.0
        ):
            raise ValueError(
                "Rates, timeouts, and command-step limits must be positive."
            )
        if max_step_rad > 0.015:
            raise ValueError(
                "max_step_rad cannot exceed the absolute 0.015 rad cap."
            )
        if not 0.0 <= close_corridor_tolerance_rad <= 0.020:
            raise ValueError(
                "close_corridor_tolerance_rad must be within [0, 0.020] rad."
            )
        if policy_pulse_max_step_rad > 0.010:
            raise ValueError(
                "policy_pulse_max_step_rad cannot exceed the absolute "
                "0.010 rad cap."
            )
        if policy_pulse_max_step_rad > max_step_rad:
            raise ValueError(
                "policy_pulse_max_step_rad cannot exceed max_step_rad."
            )
        if policy_burst_max_step_rad > 0.010:
            raise ValueError(
                "policy_burst_max_step_rad cannot exceed the absolute "
                "0.010 rad cap."
            )
        if policy_burst_max_step_rad > max_step_rad:
            raise ValueError(
                "policy_burst_max_step_rad cannot exceed max_step_rad."
            )
        if policy_burst_cumulative_limit_rad > 0.025:
            raise ValueError(
                "policy_burst_cumulative_limit_rad cannot exceed the "
                "absolute 0.025 rad cap."
            )
        if policy_burst_cumulative_limit_rad < policy_burst_max_step_rad:
            raise ValueError(
                "policy_burst_cumulative_limit_rad must cover one step."
            )
        if policy_burst_max_commands not in (5, 10, 20):
            raise ValueError(
                "policy_burst_max_commands must be 5, 10, or 20."
            )
        if not 0.0 < pregrasp_fraction < 1.0:
            raise ValueError("pregrasp_fraction must be in (0, 1).")
        if heartbeat_timeout_s > 0.50:
            raise ValueError(
                "heartbeat_timeout_s cannot exceed the absolute 0.50-second cap."
            )
        if max_speed_coeff > 0.30:
            raise ValueError(
                "Guarded speed coefficient cap cannot exceed 0.30."
            )
        if max_current_coeff > 0.30:
            raise ValueError(
                "Guarded current coefficient cap cannot exceed 0.30."
            )
        self.hard_lower_sdk_rad = lower.copy()
        self.hard_upper_sdk_rad = upper.copy()
        self.session_target_sdk_rad = (
            None if session_target is None else session_target.copy()
        )
        self.session_measured_target_sdk_rad = (
            None
            if session_measured_target is None
            else session_measured_target.copy()
        )
        sdk_command(container, sdk_root, sdk_python)  # Validate before starting any process.
        self.container = container
        self.sdk_root = sdk_root
        self.sdk_python = sdk_python
        self.serial = serial
        self.read_rate_hz = float(read_rate_hz)
        self.startup_timeout_s = float(startup_timeout_s)
        self.stale_after_s = float(stale_after_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.max_step_rad = float(max_step_rad)
        self.policy_pulse_max_step_rad = float(
            policy_pulse_max_step_rad
        )
        self.policy_burst_max_step_rad = float(
            policy_burst_max_step_rad
        )
        self.policy_burst_cumulative_limit_rad = float(
            policy_burst_cumulative_limit_rad
        )
        self.policy_burst_max_commands = int(
            policy_burst_max_commands
        )
        self.exact_policy_enabled = bool(exact_policy_enabled)
        self.exact_policy_max_commands = int(
            exact_policy_max_commands
        )
        self.exact_policy_lower_sdk_rad = (
            None if exact_lower is None else exact_lower.copy()
        )
        self.exact_policy_upper_sdk_rad = (
            None if exact_upper is None else exact_upper.copy()
        )
        self.pregrasp_fraction = float(pregrasp_fraction)
        self.close_corridor_tolerance_rad = float(
            close_corridor_tolerance_rad
        )
        self.max_speed_coeff = float(max_speed_coeff)
        self.max_current_coeff = float(max_current_coeff)
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._latest: SharpaJointState | None = None
        self._ready: dict[str, Any] | None = None
        self._status = GuardedControlStatus(False, False, 0)
        self._responses: dict[str, dict[str, Any]] = {}
        self._error: str | None = None
        self._stopped = False
        self._normal_stop_requested = False
        # This is the helper-owned start of the currently resumable
        # pregrasp corridor.  It is initialized by arm and refreshed by HOLD.
        self._armed_hold_sdk_rad: np.ndarray | None = None
        self._armed_pregrasp_target_sdk_rad: np.ndarray | None = None
        self._diagnostic_lines: deque[str] = deque(maxlen=60)

    @property
    def started(self) -> bool:
        return self._process is not None and self._ready is not None

    @property
    def device_info(self) -> dict[str, Any]:
        if self._ready is None:
            raise RuntimeError("Guarded Sharpa client is not ready.")
        return dict(self._ready)

    @property
    def status(self) -> GuardedControlStatus:
        with self._condition:
            return self._status

    @property
    def armed_pregrasp_target_sdk_rad(self) -> np.ndarray:
        """Return the exact pregrasp endpoint owned by the armed helper."""
        if self._armed_pregrasp_target_sdk_rad is None:
            raise RuntimeError(
                "No helper-owned pregrasp target is available before arm."
            )
        return self._armed_pregrasp_target_sdk_rad.copy()

    def _set_response(self, record: dict[str, Any]) -> None:
        command_id = record.get("command_id")
        if command_id:
            # Keep the first terminal response. A rejected command is followed
            # by an ESTOP record with the same ID; the rejection must not be
            # overwritten and accidentally reported as command success.
            self._responses.setdefault(str(command_id), record)

    def _handle_protocol_record(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        with self._condition:
            if kind == "ready":
                self._ready = {
                    key: record[key]
                    for key in (
                        "serial",
                        "hand_side",
                        "control_mode",
                        "control_source",
                        "enable_state",
                        "speed_coeff",
                        "current_coeff",
                        "max_step_rad",
                        "policy_pulse_max_step_rad",
                        "policy_burst_max_step_rad",
                        "policy_burst_cumulative_limit_rad",
                        "policy_burst_max_commands",
                        "pregrasp_fraction",
                        "heartbeat_timeout_s",
                    )
                }
                self._ready["exact_policy_enabled"] = bool(
                    record.get("exact_policy_enabled", False)
                )
                self._ready["exact_policy_max_commands"] = int(
                    record.get("exact_policy_max_commands", 20)
                )
            elif kind == "state":
                angles = np.asarray(record.get("angles_rad"), dtype=np.float64)
                velocities = np.asarray(record.get("velocities_rad_s"), dtype=np.float64)
                torques = np.asarray(record.get("torques_nm"), dtype=np.float64)
                if (
                    angles.shape != (22,)
                    or velocities.shape != (22,)
                    or torques.shape != (22,)
                ):
                    self._error = (
                        "Sharpa guarded SDK state dimensions are invalid: "
                        f"angles={angles.shape}, velocities={velocities.shape}, "
                        f"torques={torques.shape}."
                    )
                elif not (
                    np.all(np.isfinite(angles))
                    and np.all(np.isfinite(velocities))
                    and np.all(np.isfinite(torques))
                ):
                    self._error = "Sharpa guarded SDK state contains NaN or Inf."
                else:
                    self._latest = SharpaJointState(
                        angles_sdk_rad=angles,
                        velocities_sdk_rad_s=velocities,
                        torques_nm=torques,
                        sequence=int(record["sequence"]),
                        device_timestamp=int(record["device_timestamp"]),
                        received_monotonic_s=time.monotonic(),
                    )
                    self._status = GuardedControlStatus(
                        armed=bool(record.get("armed", False)),
                        estop_latched=bool(record.get("estop_latched", False)),
                        commands_accepted=int(record.get("commands_accepted", 0)),
                        motion_phase=(
                            None
                            if record.get("motion_phase") is None
                            else str(record["motion_phase"])
                        ),
                    )
            elif kind in ("ack", "command_error", "estop"):
                self._set_response(record)
                if kind == "estop":
                    self._status = GuardedControlStatus(
                        armed=False,
                        estop_latched=True,
                        commands_accepted=self._status.commands_accepted,
                        motion_phase=self._status.motion_phase,
                    )
            elif kind == "protocol_error":
                self._error = str(record.get("message", "guarded protocol error"))
            elif kind == "error":
                self._error = (
                    f"{record.get('message', 'unknown guarded SDK error')}\n"
                    f"{record.get('traceback', '')}"
                )
            elif kind == "stopped":
                self._stopped = True
                self._status = GuardedControlStatus(
                    armed=False,
                    estop_latched=bool(record.get("estop_latched", False)),
                    commands_accepted=int(record.get("commands_accepted", 0)),
                    motion_phase=self._status.motion_phase,
                )
            self._condition.notify_all()

    def _read_output(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                stripped = line.rstrip()
                if stripped.startswith(PROTOCOL_PREFIX):
                    try:
                        record = json.loads(stripped[len(PROTOCOL_PREFIX) :])
                        self._handle_protocol_record(record)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                        with self._condition:
                            self._error = (
                                f"Invalid Sharpa guarded protocol record: {error}"
                            )
                            self._condition.notify_all()
                elif stripped:
                    self._diagnostic_lines.append(stripped)
        finally:
            with self._condition:
                if not self._stopped and self._error is None:
                    return_code = self._process.poll()
                    self._error = (
                        "Sharpa guarded subprocess ended unexpectedly "
                        f"({return_code=})."
                    )
                self._condition.notify_all()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Guarded Sharpa client has already been started.")
        command = sdk_command(self.container, self.sdk_root, self.sdk_python) + [
            "-e",
            f"SHARPA_READ_RATE_HZ={self.read_rate_hz}",
            "-e",
            f"SHARPA_DISCOVERY_TIMEOUT_S={self.startup_timeout_s}",
            "-e",
            f"SHARPA_HEARTBEAT_TIMEOUT_S={self.heartbeat_timeout_s}",
            "-e",
            f"SHARPA_MAX_STEP_RAD={self.max_step_rad}",
            "-e",
            "SHARPA_POLICY_PULSE_MAX_STEP_RAD="
            f"{self.policy_pulse_max_step_rad}",
            "-e",
            "SHARPA_POLICY_BURST_MAX_STEP_RAD="
            f"{self.policy_burst_max_step_rad}",
            "-e",
            "SHARPA_POLICY_BURST_CUMULATIVE_LIMIT_RAD="
            f"{self.policy_burst_cumulative_limit_rad}",
            "-e",
            "SHARPA_POLICY_BURST_MAX_COMMANDS="
            f"{self.policy_burst_max_commands}",
            "-e",
            "SHARPA_EXACT_POLICY_ENABLED="
            f"{int(self.exact_policy_enabled)}",
            "-e",
            "SHARPA_EXACT_POLICY_MAX_COMMANDS="
            f"{self.exact_policy_max_commands}",
            "-e",
            f"SHARPA_PREGRASP_FRACTION={self.pregrasp_fraction}",
            "-e",
            "SHARPA_CLOSE_CORRIDOR_TOLERANCE_RAD="
            f"{self.close_corridor_tolerance_rad}",
            "-e",
            f"SHARPA_MAX_SPEED_COEFF={self.max_speed_coeff}",
            "-e",
            f"SHARPA_MAX_CURRENT_COEFF={self.max_current_coeff}",
            "-e",
            f"SHARPA_HARD_LOWER_SDK={json.dumps(self.hard_lower_sdk_rad.tolist())}",
            "-e",
            f"SHARPA_HARD_UPPER_SDK={json.dumps(self.hard_upper_sdk_rad.tolist())}",
        ]
        if self.serial:
            command.extend(("-e", f"SHARPA_READ_SERIAL={self.serial}"))
        if self.session_target_sdk_rad is not None:
            command.extend(
                (
                    "-e",
                    "SHARPA_SESSION_TARGET_SDK="
                    f"{json.dumps(self.session_target_sdk_rad.tolist())}",
                )
            )
        if self.session_measured_target_sdk_rad is not None:
            command.extend(
                (
                    "-e",
                    "SHARPA_SESSION_MEASURED_TARGET_SDK="
                    f"{json.dumps(self.session_measured_target_sdk_rad.tolist())}",
                )
            )
        if self.exact_policy_lower_sdk_rad is not None:
            command.extend(
                (
                    "-e",
                    "SHARPA_EXACT_POLICY_LOWER_SDK="
                    f"{json.dumps(self.exact_policy_lower_sdk_rad.tolist())}",
                    "-e",
                    "SHARPA_EXACT_POLICY_UPPER_SDK="
                    f"{json.dumps(self.exact_policy_upper_sdk_rad.tolist())}",
                )
            )
        command.extend((self.container, self.sdk_python, "-u", "-c", GUARDED_HELPER_PROGRAM))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name="sharpa-guarded-output",
            daemon=True,
        )
        self._reader.start()
        deadline = time.monotonic() + self.startup_timeout_s
        startup_error: str | None = None
        with self._condition:
            while self._ready is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if self._ready is None:
                startup_error = self._error or "Timed out waiting for guarded Sharpa SDK."
        if startup_error is not None:
            diagnostics = "\n".join(self._diagnostic_lines)
            self._terminate_process()
            raise RuntimeError(f"{startup_error}\nRecent SDK output:\n{diagnostics}")
        if (
            self.device_info["control_mode"] != "POSITION"
            or self.device_info["control_source"] != "SDK"
        ):
            self._terminate_process()
            raise RuntimeError(
                "Guarded controller requires the preconfigured POSITION/SDK mode; "
                f"observed {self.device_info['control_mode']}/"
                f"{self.device_info['control_source']}."
            )
        self.latest(wait_timeout_s=2.0)

    def latest(
        self,
        *,
        wait_timeout_s: float = 0.0,
        max_age_s: float | None = None,
    ) -> SharpaJointState:
        max_age_s = self.stale_after_s if max_age_s is None else float(max_age_s)
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ValueError("max_age_s must be finite and positive.")
        if not math.isfinite(float(wait_timeout_s)):
            raise ValueError("wait_timeout_s must be finite.")
        deadline = time.monotonic() + max(0.0, float(wait_timeout_s))
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError(self._error)
            if self._latest is None:
                raise RuntimeError("No guarded Sharpa joint state is available.")
            state = self._latest
        age_s = time.monotonic() - state.received_monotonic_s
        if age_s > max_age_s:
            raise RuntimeError(
                f"Sharpa guarded joint state is stale: "
                f"age={age_s:.3f}s > {max_age_s:.3f}s."
            )
        return state

    def _send(
        self,
        kind: str,
        *,
        wait_for_ack: bool,
        timeout_s: float = 1.0,
        **payload: Any,
    ) -> dict[str, Any] | None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Guarded Sharpa subprocess is not running.")
        command_id = uuid.uuid4().hex if wait_for_ack else ""
        record = {"kind": kind, "command_id": command_id, **payload}
        with self._write_lock:
            process.stdin.write(json.dumps(record, separators=(",", ":")) + "\n")
            process.stdin.flush()
        if not wait_for_ack:
            return None
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while command_id not in self._responses and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError(self._error)
            response = self._responses.pop(command_id, None)
        if response is None:
            raise TimeoutError(f"Timed out waiting for Sharpa {kind!r} acknowledgement.")
        if response.get("kind") == "command_error":
            raise RuntimeError(
                f"Sharpa {kind} rejected: {response.get('message', 'unknown error')}"
            )
        return response

    def arm(
        self,
        *,
        speed_coeff: float,
        current_coeff: float,
        confirmation: str,
    ) -> np.ndarray:
        if confirmation != "POISE":
            raise ValueError("Explicit arm confirmation must equal 'POISE'.")
        if not all(
            math.isfinite(float(value))
            for value in (speed_coeff, current_coeff)
        ):
            raise ValueError("Arm coefficients must be finite.")
        try:
            response = self._send(
                "arm",
                wait_for_ack=True,
                timeout_s=5.0,
                speed_coeff=float(speed_coeff),
                current_coeff=float(current_coeff),
                confirmation=confirmation,
            )
        except BaseException:
            try:
                self._send(
                    "estop",
                    wait_for_ack=True,
                    timeout_s=3.0,
                    reason="host_arm_handshake_failure",
                )
            except BaseException:
                self._terminate_process()
            raise
        assert response is not None
        held_angles = self._held_angles_from_response(response, "arm")
        pregrasp_target = np.asarray(
            response.get("pregrasp_target_rad"),
            dtype=np.float64,
        )
        if (
            pregrasp_target.shape != (22,)
            or not np.all(np.isfinite(pregrasp_target))
        ):
            try:
                self._send(
                    "estop",
                    wait_for_ack=True,
                    timeout_s=3.0,
                    reason="invalid_helper_pregrasp_target",
                )
            finally:
                self._armed_pregrasp_target_sdk_rad = None
            raise RuntimeError(
                "Sharpa arm response has invalid helper-owned pregrasp target."
            )
        self._armed_hold_sdk_rad = held_angles.copy()
        self._armed_pregrasp_target_sdk_rad = pregrasp_target.copy()
        with self._condition:
            self._status = GuardedControlStatus(
                armed=True,
                estop_latched=False,
                commands_accepted=self._status.commands_accepted,
                motion_phase=self._status.motion_phase,
            )
        return held_angles

    def heartbeat(self) -> None:
        self._send("heartbeat", wait_for_ack=False)

    def command_target(
        self,
        angles_sdk_rad: np.ndarray,
        *,
        motion_phase: str,
    ) -> np.ndarray:
        values = np.asarray(angles_sdk_rad, dtype=np.float64)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError("Guarded Sharpa target must be finite and 22D.")
        if motion_phase not in ("pregrasp", "close"):
            raise ValueError(
                "Guarded target motion_phase must be 'pregrasp' or 'close'."
            )
        if motion_phase == "pregrasp":
            if (
                self._armed_hold_sdk_rad is None
                or self._armed_pregrasp_target_sdk_rad is None
            ):
                raise RuntimeError(
                    "No helper-owned pregrasp corridor is available before arm."
                )
            lower = np.minimum(
                self._armed_hold_sdk_rad,
                self._armed_pregrasp_target_sdk_rad,
            )
            upper = np.maximum(
                self._armed_hold_sdk_rad,
                self._armed_pregrasp_target_sdk_rad,
            )
            projected = np.minimum(np.maximum(values, lower), upper)
            projection = float(np.max(np.abs(projected - values)))
            # A separately sampled SDK state can differ at the 1e-4-rad level.
            # Project only that tiny serialization/state-skew mismatch inward;
            # a larger planner disagreement remains fail-closed.
            if projection > 5.0e-4:
                raise RuntimeError(
                    "Host pregrasp target disagrees with the helper-owned "
                    f"corridor by {projection:.6f} rad."
                )
            values = projected
        response = self._send(
            "target",
            wait_for_ack=True,
            timeout_s=0.5,
            angles_sdk_rad=values.tolist(),
            motion_phase=motion_phase,
        )
        assert response is not None
        accepted = np.asarray(
            response.get("accepted_angles_sdk_rad"),
            dtype=np.float64,
        )
        if accepted.shape != (22,) or not np.all(np.isfinite(accepted)):
            raise RuntimeError(
                "Sharpa target response has invalid accepted joint angles."
            )
        if not np.allclose(accepted, values, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                "Sharpa helper accepted a target different from the host request."
            )
        return accepted.copy()

    def refresh_target(self) -> None:
        """Resend the helper-owned last target without accepting new angles."""
        self._send("refresh_target", wait_for_ack=True, timeout_s=0.5)

    def command_policy_pulse(
        self,
        angles_sdk_rad: np.ndarray,
        *,
        confirmation: str,
    ) -> int:
        values = np.asarray(angles_sdk_rad, dtype=np.float64)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Guarded policy pulse target must be finite and 22D."
            )
        if confirmation != "ONE_SHOT_0.010_RAD":
            raise ValueError(
                "Policy pulse confirmation must equal "
                "'ONE_SHOT_0.010_RAD'."
            )
        response = self._send(
            "policy_pulse",
            wait_for_ack=True,
            timeout_s=0.5,
            angles_sdk_rad=values.tolist(),
            confirmation=confirmation,
        )
        assert response is not None
        return int(response["commands_accepted"])

    def command_policy_burst(
        self,
        angles_sdk_rad: np.ndarray,
        *,
        burst_index: int,
        confirmation: str,
    ) -> int:
        values = np.asarray(angles_sdk_rad, dtype=np.float64)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Guarded policy burst target must be finite and 22D."
            )
        if not 0 <= burst_index < self.policy_burst_max_commands:
            raise ValueError(
                "Policy burst index is outside the configured command count."
            )
        expected_confirmation = (
            "FIVE_CYCLES_0.010_TOTAL_0.025"
            if self.policy_burst_max_commands == 5
            else (
                "TEN_CYCLES_0.010_DRIVE_0.018_HARD_0.025"
                if self.policy_burst_max_commands == 10
                else "TWENTY_CYCLES_0.010_DRIVE_0.018_HARD_0.025"
            )
        )
        if confirmation != expected_confirmation:
            raise ValueError(
                "Policy burst confirmation does not match its configured "
                "command count."
            )
        response = self._send(
            "policy_burst",
            wait_for_ack=True,
            timeout_s=0.5,
            angles_sdk_rad=values.tolist(),
            burst_index=int(burst_index),
            confirmation=confirmation,
        )
        assert response is not None
        observed_index = int(response["policy_burst_command_count"])
        if observed_index != burst_index + 1:
            raise RuntimeError(
                "Sharpa helper acknowledged an unexpected policy burst index."
            )
        expected_completed = (
            observed_index >= self.policy_burst_max_commands
        )
        if bool(response.get("policy_burst_completed", False)) != (
            expected_completed
        ):
            raise RuntimeError(
                "Sharpa helper returned an inconsistent burst completion latch."
            )
        return int(response["commands_accepted"])

    def command_exact_policy(
        self,
        angles_sdk_rad: np.ndarray,
        *,
        exact_index: int,
        confirmation: str,
    ) -> int:
        if not self.exact_policy_enabled:
            raise RuntimeError(
                "Exact policy protocol was not enabled for this helper session."
            )
        values = np.asarray(angles_sdk_rad, dtype=np.float64)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Exact q_policy target must be finite and 22D."
            )
        if not 0 <= exact_index < self.exact_policy_max_commands:
            raise ValueError(
                "Exact policy index is outside the configured command count."
            )
        expected_confirmation = "EXACT_POLICY_CONTRACT"
        if confirmation != expected_confirmation:
            raise ValueError(
                "Exact policy confirmation does not match the q_policy contract."
            )
        response = self._send(
            "exact_policy",
            wait_for_ack=True,
            timeout_s=0.5,
            angles_sdk_rad=values.tolist(),
            exact_index=int(exact_index),
            confirmation=confirmation,
        )
        assert response is not None
        observed_index = int(response["exact_policy_command_count"])
        if observed_index != exact_index + 1:
            raise RuntimeError(
                "Sharpa helper acknowledged an unexpected exact policy index."
            )
        if bool(response.get("exact_policy_completed", False)) != (
            observed_index >= self.exact_policy_max_commands
        ):
            raise RuntimeError(
                "Sharpa helper returned an inconsistent exact completion latch."
            )
        return int(response["commands_accepted"])

    @staticmethod
    def _held_angles_from_response(
        response: dict[str, Any],
        operation: str,
    ) -> np.ndarray:
        values = np.asarray(response.get("held_angles_rad"), dtype=np.float64)
        if values.shape != (22,) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Sharpa {operation} response has invalid held joint angles."
            )
        return values

    def hold(self) -> np.ndarray:
        response = self._send("hold", wait_for_ack=True, timeout_s=0.5)
        assert response is not None
        held = self._held_angles_from_response(response, "hold")
        resume_raw = response.get("pregrasp_resume_start_rad")
        if resume_raw is not None:
            resume = np.asarray(resume_raw, dtype=np.float64)
            if resume.shape != (22,) or not np.all(np.isfinite(resume)):
                raise RuntimeError(
                    "Sharpa hold response has invalid pregrasp resume start."
                )
            if not np.allclose(resume, held, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    "Sharpa hold response pregrasp resume start differs from "
                    "the helper-held joint angles."
                )
            if np.any(resume < self.hard_lower_sdk_rad) or np.any(
                resume > self.hard_upper_sdk_rad
            ):
                raise RuntimeError(
                    "Sharpa hold response pregrasp resume start is outside "
                    "the guarded hard limits."
                )
            self._armed_hold_sdk_rad = resume.copy()
        return held

    def estop(self, reason: str = "host_estop") -> dict[str, Any]:
        response = self._send(
            "estop",
            wait_for_ack=True,
            timeout_s=1.0,
            reason=str(reason),
        )
        assert response is not None
        if not response.get("hardware_disable_succeeded", False):
            raise RuntimeError(
                "Sharpa ESTOP was requested but hardware disable failed: "
                f"{response.get('disable_error', 'unknown error')}"
            )
        return response

    def normal_stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._normal_stop_requested = True
        self._send("normal_stop", wait_for_ack=True, timeout_s=2.0)
        with self._condition:
            self._status = GuardedControlStatus(
                armed=False,
                estop_latched=self._status.estop_latched,
                commands_accepted=self._status.commands_accepted,
                motion_phase=self._status.motion_phase,
            )

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self._process = None
        self._reader = None

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            status = self.status
            try:
                if status.armed and not status.estop_latched:
                    self.normal_stop()
                elif not self._normal_stop_requested:
                    self._send("normal_stop", wait_for_ack=True, timeout_s=2.0)
            except (BrokenPipeError, OSError, RuntimeError, TimeoutError):
                # Closing stdin or terminating an armed helper triggers its
                # independent fail-closed hardware-disable path.
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._terminate_process()
            return
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self._process = None
        self._reader = None

    def __enter__(self) -> "GuardedSharpaControlClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None and self.status.armed and not self.status.estop_latched:
            try:
                self.estop(f"host_exception:{exc_type.__name__}")
            except BaseException:
                pass
        self.close()
