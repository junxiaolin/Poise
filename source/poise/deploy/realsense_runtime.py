"""Single-pipeline RealSense RGB-D runtime for calibration and deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass
import threading
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RealSenseStreamConfig:
    serial: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: float = 100.0
    gain: float = 12.0
    reset_on_start: bool = True
    reconnect_wait_s: float = 6.0
    frame_timeout_ms: int = 15000
    # ``None`` preserves the legacy behavior and leaves white balance under
    # the sensor's current control mode.  A numeric value disables automatic
    # white balance and locks the color sensor to that value in kelvin.
    white_balance: float | None = None


@dataclass(frozen=True)
class RealSenseFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    timestamp_ms: float
    frame_number: int


@dataclass(frozen=True)
class CapturedRealSenseFrame:
    """A frame published by the capture worker's single-slot buffer.

    ``publish_sequence`` is a host-side monotonic sequence.  It is used for
    freshness checks instead of relying on the camera's frame number, which
    can reset when a device reconnects.
    """

    frame: RealSenseFrame
    captured_monotonic_s: float
    publish_sequence: int


class RealSenseCaptureWorker:
    """Continuously capture RGB-D frames into a latest-frame buffer.

    The worker owns calls to :meth:`RealSenseSession.next_frame`; consumers
    never wait on the RealSense pipeline.  A single-slot buffer deliberately
    drops intermediate frames when the downstream tracker is slower than the
    camera, keeping control latency bounded.
    """

    def __init__(
        self,
        session: "RealSenseSession",
        *,
        on_frame: Callable[[CapturedRealSenseFrame], None] | None = None,
    ) -> None:
        self.session = session
        self.on_frame = on_frame
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CapturedRealSenseFrame | None = None
        self._sequence = 0
        self._error: BaseException | None = None
        self._callback_error: BaseException | None = None
        self._captured_count = 0
        self._dropped_count = 0

    @property
    def latest(self) -> CapturedRealSenseFrame | None:
        with self._condition:
            return self._latest

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def callback_error(self) -> BaseException | None:
        with self._condition:
            return self._callback_error

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def captured_count(self) -> int:
        with self._condition:
            return self._captured_count

    @property
    def dropped_count(self) -> int:
        with self._condition:
            return self._dropped_count

    def start(self) -> "RealSenseCaptureWorker":
        if self._thread is not None:
            raise RuntimeError("RealSenseCaptureWorker is already started.")
        if not self.session.started:
            raise RuntimeError("RealSenseSession must be started first.")
        self._thread = threading.Thread(
            target=self._run,
            name="sharpa-realsense-capture",
            daemon=True,
        )
        self._thread.start()
        return self

    def __enter__(self) -> "RealSenseCaptureWorker":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self.session.next_frame()
                captured = CapturedRealSenseFrame(
                    frame=frame,
                    captured_monotonic_s=time.monotonic(),
                    publish_sequence=self._sequence + 1,
                )
                with self._condition:
                    if self._latest is not None:
                        self._dropped_count += 1
                    self._latest = captured
                    self._sequence = captured.publish_sequence
                    self._captured_count += 1
                    self._condition.notify_all()
                if self.on_frame is not None:
                    try:
                        self.on_frame(captured)
                    except BaseException as error:
                        # Diagnostics are best effort and must never terminate
                        # capture or affect the authoritative safety path.
                        with self._condition:
                            self._callback_error = error
                            self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                if not self._stop_event.is_set():
                    self._error = error
                self._condition.notify_all()

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        timeout_s: float | None = None,
    ) -> CapturedRealSenseFrame | None:
        """Return the newest frame newer than ``after_sequence``.

        ``None`` means the timeout elapsed.  A capture exception is raised so
        the control loop can fail closed rather than silently reusing a dead
        camera stream.
        """
        if timeout_s is not None and timeout_s < 0.0:
            raise ValueError("timeout_s cannot be negative.")
        deadline = (
            None if timeout_s is None else time.monotonic() + float(timeout_s)
        )
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(
                        "RealSense capture worker failed"
                    ) from self._error
                latest = self._latest
                if (
                    latest is not None
                    and latest.publish_sequence > int(after_sequence)
                ):
                    return latest
                if self._stop_event.is_set() or not self.running:
                    return None
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    self._condition.wait(remaining)

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(join_timeout_s)))
            if thread.is_alive():
                # ``wait_for_frames`` may be inside the SDK's long timeout.
                # Interrupt the pipeline only during worker shutdown; the
                # owning controller is already leaving the camera context.
                stop_session = getattr(self.session, "stop", None)
                if callable(stop_session):
                    stop_session()
                thread.join(timeout=max(0.0, float(join_timeout_s)))
                if thread.is_alive():
                    raise RuntimeError("RealSense capture worker did not stop after pipeline interruption")




class RealSenseSession:
    """Own one aligned RGB-D pipeline from startup calibration through policy."""

    def __init__(self, config: RealSenseStreamConfig = RealSenseStreamConfig()) -> None:
        self.config = config
        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._profile: Any | None = None
        self.depth_scale_m: float | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion_coefficients: np.ndarray | None = None
        self.distortion_model: str | None = None
        self.camera_name: str | None = None
        self.serial: str | None = None
        self.manual_controls: dict[str, Any] | None = None
        self._color_sensor: Any | None = None

    @property
    def started(self) -> bool:
        return self._pipeline is not None

    def __enter__(self) -> "RealSenseSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _matching_devices(self, rs: Any) -> list[Any]:
        devices = list(rs.context().query_devices())
        if self.config.serial is None:
            return devices
        return [
            device
            for device in devices
            if device.get_info(rs.camera_info.serial_number) == self.config.serial
        ]

    def _hardware_reset(self, rs: Any) -> None:
        devices = self._matching_devices(rs)
        if len(devices) != 1:
            raise RuntimeError(
                f"Expected one matching RealSense before reset, found {len(devices)}."
            )
        devices[0].hardware_reset()
        del devices
        time.sleep(self.config.reconnect_wait_s)
        for _ in range(20):
            if len(self._matching_devices(rs)) == 1:
                return
            time.sleep(0.5)
        raise RuntimeError("RealSense did not reconnect after hardware reset.")

    @staticmethod
    def _set_sensor_option(sensor: Any, option: Any, value: float, name: str) -> dict[str, float]:
        if not sensor.supports(option):
            raise RuntimeError(f"RealSense color sensor does not support {name}.")
        value_range = sensor.get_option_range(option)
        if value < value_range.min or value > value_range.max:
            raise ValueError(
                f"{name}={value} is outside [{value_range.min}, {value_range.max}]."
            )
        sensor.set_option(option, float(value))
        return {
            "requested": float(value),
            "actual": float(sensor.get_option(option)),
            "minimum": float(value_range.min),
            "maximum": float(value_range.max),
            "step": float(value_range.step),
        }

    @staticmethod
    def _sensor_option_record(sensor: Any, option: Any) -> dict[str, float]:
        """Return the current value and supported range without changing it."""

        value_range = sensor.get_option_range(option)
        return {
            "actual": float(sensor.get_option(option)),
            "minimum": float(value_range.min),
            "maximum": float(value_range.max),
            "step": float(value_range.step),
            "default": float(value_range.default),
        }

    def start(self) -> "RealSenseSession":
        if self.started:
            raise RuntimeError("RealSenseSession is already started.")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is unavailable; install the POISE deploy dependencies."
            ) from exc
        if self.config.reset_on_start:
            self._hardware_reset(rs)

        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial:
            stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        try:
            profile = pipeline.start(stream_config)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not start RealSense RGB-D. Close other viewers and retry."
            ) from exc

        try:
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            if not color_sensor.supports(rs.option.enable_auto_exposure):
                raise RuntimeError("RealSense color sensor cannot disable auto exposure.")
            color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            exposure = self._set_sensor_option(
                color_sensor,
                rs.option.exposure,
                self.config.exposure,
                "exposure",
            )
            gain = self._set_sensor_option(
                color_sensor,
                rs.option.gain,
                self.config.gain,
                "gain",
            )
            white_balance: dict[str, Any]
            if self.config.white_balance is not None:
                if not color_sensor.supports(rs.option.enable_auto_white_balance):
                    raise RuntimeError(
                        "RealSense color sensor cannot disable auto white balance."
                    )
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
                white_balance = {
                    "enable_auto_white_balance": False,
                    "value": self._set_sensor_option(
                        color_sensor,
                        rs.option.white_balance,
                        self.config.white_balance,
                        "white_balance",
                    ),
                    "mode": "explicit_manual",
                }
            elif color_sensor.supports(rs.option.enable_auto_white_balance) and color_sensor.supports(
                rs.option.white_balance
            ):
                white_balance = {
                    "enable_auto_white_balance": bool(
                        round(
                            float(
                                color_sensor.get_option(
                                    rs.option.enable_auto_white_balance
                                )
                            )
                        )
                    ),
                    "value": self._sensor_option_record(
                        color_sensor,
                        rs.option.white_balance,
                    ),
                    "mode": "sensor_current",
                }
            else:
                white_balance = {
                    "supported": False,
                    "mode": "unsupported",
                }
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_profile.get_intrinsics()
            self.camera_matrix = np.array(
                [
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.distortion_coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64)
            self.distortion_model = str(intrinsics.model)
            self.depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
            self.camera_name = device.get_info(rs.camera_info.name)
            self.serial = device.get_info(rs.camera_info.serial_number)
            self.manual_controls = {
                "enable_auto_exposure": False,
                "exposure": exposure,
                "gain": gain,
                "white_balance": white_balance,
            }
            self._rs = rs
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color)
            self._profile = profile
            self._color_sensor = color_sensor
            return self
        except BaseException:
            pipeline.stop()
            raise

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._profile = None
        self._color_sensor = None
        self._rs = None

    def lock_auto_white_balance(self, frame_count: int = 90) -> dict[str, Any]:
        """Converge automatic white balance, then freeze the observed value.

        This method is intended for appearance-capture sessions.  It must be
        called before another thread starts consuming frames from the same
        pipeline.  The returned record is also written into :meth:`metadata`.
        """

        if not self.started or self._color_sensor is None or self._rs is None:
            raise RuntimeError("RealSenseSession is not started.")
        if frame_count < 1:
            raise ValueError("White-balance auto-lock frame_count must be positive.")
        sensor = self._color_sensor
        rs = self._rs
        for option, name in (
            (rs.option.enable_auto_white_balance, "enable_auto_white_balance"),
            (rs.option.white_balance, "white_balance"),
        ):
            if not sensor.supports(option):
                raise RuntimeError(
                    f"RealSense color sensor does not support {name}."
                )

        original_auto_white_balance = float(
            sensor.get_option(rs.option.enable_auto_white_balance)
        )
        sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
        try:
            self.warmup(frame_count)
        except BaseException:
            sensor.set_option(
                rs.option.enable_auto_white_balance,
                original_auto_white_balance,
            )
            raise
        stabilized = float(sensor.get_option(rs.option.white_balance))
        sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
        locked = self._set_sensor_option(
            sensor,
            rs.option.white_balance,
            stabilized,
            "white_balance",
        )
        record: dict[str, Any] = {
            "enable_auto_white_balance": False,
            "value": locked,
            "mode": "auto_converged_then_locked",
            "auto_convergence_frames": int(frame_count),
        }
        if self.manual_controls is None:
            self.manual_controls = {}
        self.manual_controls["white_balance"] = record
        return record

    def next_frame(self) -> RealSenseFrame:
        if not self.started or self._align is None or self.depth_scale_m is None:
            raise RuntimeError("RealSenseSession is not started.")
        frames = self._pipeline.wait_for_frames(self.config.frame_timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete aligned RGB-D frame.")
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_m = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale_m
        )
        return RealSenseFrame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
        )

    def warmup(self, frame_count: int) -> RealSenseFrame:
        if frame_count < 1:
            raise ValueError("frame_count must be positive.")
        frame: RealSenseFrame | None = None
        for _ in range(frame_count):
            frame = self.next_frame()
        assert frame is not None
        return frame

    def metadata(self) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("RealSenseSession is not started.")
        assert self.camera_matrix is not None
        assert self.distortion_coefficients is not None
        return {
            "name": self.camera_name,
            "serial": self.serial,
            "stream": {
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "color_format": "bgr8",
                "depth_format": "z16_aligned_to_color",
            },
            "manual_color_controls": self.manual_controls,
            "depth_scale_m": self.depth_scale_m,
            "intrinsics": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_model": self.distortion_model,
                "distortion_coefficients": self.distortion_coefficients.tolist(),
            },
        }
