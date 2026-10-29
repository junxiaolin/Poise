"""Read-only Sharpa Wave joint-state bridge through the Python 3.10 SDK container."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sdk_runtime import sdk_command


PROTOCOL_PREFIX = "__SHARPA_READONLY__"

# The host FoundationPose environment is Python 3.11, while the installed
# Sharpa native extension is CPython 3.10. Keep all SDK access inside this
# helper and expose state only. Deliberately do not call any set_* API.
READ_ONLY_HELPER_PROGRAM = r"""
import json
import math
import os
import signal
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.join(os.environ["SHARPA_SDK_ROOT"], "python"))
from sharpa import SharpaWaveConfig, SharpaWaveManager

PREFIX = "__SHARPA_READONLY__"
stop_event = threading.Event()

def emit(kind, **payload):
    print(PREFIX + json.dumps({"kind": kind, **payload}, separators=(",", ":")), flush=True)

def request_stop(*_):
    stop_event.set()

def state_vectors(state):
    angles = [float(value) for value in state.angles]
    velocities = [float(value) for value in state.velocities]
    torques = [float(value) for value in state.torques]
    dimensions = (len(angles), len(velocities), len(torques))
    if dimensions != (22, 22, 22):
        raise RuntimeError(
            "Sharpa Wave joint state must contain 22 angles, velocities, and torques; "
            f"got {dimensions}."
        )
    if not all(
        math.isfinite(value)
        for values in (angles, velocities, torques)
        for value in values
    ):
        raise RuntimeError("Sharpa Wave joint state contains NaN or Inf.")
    return angles, velocities, torques

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

def watch_stdin():
    try:
        for line in sys.stdin:
            if line.strip() == "STOP":
                stop_event.set()
                return
    except Exception:
        stop_event.set()

threading.Thread(target=watch_stdin, daemon=True).start()
manager = None
hand = None
try:
    manager = SharpaWaveManager.get_instance()
    requested_serial = os.environ.get("SHARPA_READ_SERIAL", "").strip()
    discovery_timeout_s = float(os.environ.get("SHARPA_DISCOVERY_TIMEOUT_S", "12"))
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
    while not stop_event.is_set() and time.monotonic() < ready_deadline:
        state = hand.get_states()
        state_vectors(state)
        if int(state.sequence) > 0:
            break
        time.sleep(0.02)
    if state is None or int(state.sequence) <= 0:
        raise RuntimeError("Sharpa Wave did not produce a 22D joint state.")

    info = hand.get_device_info()
    emit(
        "ready",
        serial=str(info.sn),
        hand_side=str(info.hand_side),
        control_mode=str(info.control_mode),
        control_source=str(info.control_source),
    )
    rate_hz = float(os.environ.get("SHARPA_READ_RATE_HZ", "60"))
    period_s = 1.0 / max(rate_hz, 1.0)
    next_deadline = time.monotonic()
    while not stop_event.is_set():
        state = hand.get_states()
        angles, velocities, torques = state_vectors(state)
        emit(
            "state",
            sequence=int(state.sequence),
            device_timestamp=int(state.timestamp),
            angles_rad=angles,
            velocities_rad_s=velocities,
            torques_nm=torques,
        )
        next_deadline += period_s
        delay = next_deadline - time.monotonic()
        if delay > 0.0:
            stop_event.wait(delay)
        else:
            next_deadline = time.monotonic()
except BaseException as error:
    emit("error", message=str(error), traceback=traceback.format_exc())
finally:
    try:
        if hand is not None:
            hand.stop()
    finally:
        if manager is not None:
            manager.disconnect_all()
    emit("stopped")
"""


@dataclass(frozen=True)
class SharpaJointState:
    angles_sdk_rad: np.ndarray
    velocities_sdk_rad_s: np.ndarray
    torques_nm: np.ndarray
    sequence: int
    device_timestamp: int
    received_monotonic_s: float


class SharpaReadOnlyStateClient:
    """Own a state-only SDK subprocess and retain its newest joint sample."""

    def __init__(
        self,
        *,
        container: str = "poise-sharpa",
        sdk_root: str = "/opt/sharpa-wave-sdk",
        sdk_python: str = "python3",
        serial: str | None = None,
        read_rate_hz: float = 60.0,
        startup_timeout_s: float = 15.0,
        stale_after_s: float = 0.20,
    ) -> None:
        if read_rate_hz <= 0.0:
            raise ValueError("read_rate_hz must be positive.")
        if startup_timeout_s <= 0.0 or stale_after_s <= 0.0:
            raise ValueError("Timeouts must be positive.")
        sdk_command(container, sdk_root, sdk_python)  # Validate before starting any process.
        self.container = container
        self.sdk_root = sdk_root
        self.sdk_python = sdk_python
        self.serial = serial
        self.read_rate_hz = float(read_rate_hz)
        self.startup_timeout_s = float(startup_timeout_s)
        self.stale_after_s = float(stale_after_s)
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest: SharpaJointState | None = None
        self._ready: dict[str, Any] | None = None
        self._error: str | None = None
        self._stopped = False
        self._diagnostic_lines: deque[str] = deque(maxlen=40)

    @property
    def started(self) -> bool:
        return self._process is not None and self._ready is not None

    @property
    def device_info(self) -> dict[str, Any]:
        if self._ready is None:
            raise RuntimeError("Sharpa read-only client is not ready.")
        return dict(self._ready)

    def _handle_protocol_record(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        with self._condition:
            if kind == "ready":
                self._ready = {
                    key: record[key]
                    for key in ("serial", "hand_side", "control_mode", "control_source")
                }
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
                        f"Sharpa SDK state dimensions are invalid: "
                        f"angles={angles.shape}, velocities={velocities.shape}, "
                        f"torques={torques.shape}."
                    )
                elif not (
                    np.all(np.isfinite(angles))
                    and np.all(np.isfinite(velocities))
                    and np.all(np.isfinite(torques))
                ):
                    self._error = "Sharpa SDK state contains NaN or Inf."
                else:
                    self._latest = SharpaJointState(
                        angles_sdk_rad=angles,
                        velocities_sdk_rad_s=velocities,
                        torques_nm=torques,
                        sequence=int(record["sequence"]),
                        device_timestamp=int(record["device_timestamp"]),
                        received_monotonic_s=time.monotonic(),
                    )
            elif kind == "error":
                self._error = f"{record.get('message', 'unknown SDK error')}\n{record.get('traceback', '')}"
            elif kind == "stopped":
                self._stopped = True
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
                            self._error = f"Invalid Sharpa read-only protocol record: {error}"
                            self._condition.notify_all()
                elif stripped:
                    self._diagnostic_lines.append(stripped)
        finally:
            with self._condition:
                if not self._stopped and self._error is None:
                    return_code = self._process.poll()
                    self._error = f"Sharpa read-only subprocess ended unexpectedly ({return_code=})."
                self._condition.notify_all()

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Sharpa read-only client has already been started.")
        command = sdk_command(self.container, self.sdk_root, self.sdk_python) + [
            "-e",
            f"SHARPA_READ_RATE_HZ={self.read_rate_hz}",
            "-e",
            f"SHARPA_DISCOVERY_TIMEOUT_S={self.startup_timeout_s}",
        ]
        if self.serial:
            command.extend(("-e", f"SHARPA_READ_SERIAL={self.serial}"))
        command.extend((self.container, self.sdk_python, "-u", "-c", READ_ONLY_HELPER_PROGRAM))
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
            name="sharpa-readonly-output",
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
                startup_error = self._error or "Timed out waiting for Sharpa SDK state."
        if startup_error is not None:
            diagnostics = "\n".join(self._diagnostic_lines)
            self.close()
            raise RuntimeError(f"{startup_error}\nRecent SDK output:\n{diagnostics}")
        # Require one valid state in addition to the ready record.
        self.latest(wait_timeout_s=2.0)

    def latest(
        self,
        *,
        wait_timeout_s: float = 0.0,
        max_age_s: float | None = None,
    ) -> SharpaJointState:
        max_age_s = self.stale_after_s if max_age_s is None else float(max_age_s)
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
                raise RuntimeError("No Sharpa joint state is available.")
            state = self._latest
        age_s = time.monotonic() - state.received_monotonic_s
        if age_s > max_age_s:
            raise RuntimeError(
                f"Sharpa joint state is stale: age={age_s:.3f}s > {max_age_s:.3f}s."
            )
        return state

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
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

    def __enter__(self) -> "SharpaReadOnlyStateClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
