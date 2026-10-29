"""Camera failures cannot delay robot shutdown behind pipeline/thread cleanup."""

from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest

from poise.deploy.lifecycle import deployment_resources
from poise.deploy.realsense_runtime import RealSenseCaptureWorker
from poise.deploy.sharpa_guarded import GuardedSharpaControlClient
from poise.deploy.sharpa_readonly import SharpaReadOnlyStateClient


@pytest.mark.parametrize("failed_stop", [False, True])
def test_estop_and_close_precede_capture_and_camera_exit(failed_stop):
    events = []
    def estop(reason):
        events.append("estop")
        if failed_stop:
            raise RuntimeError("SDK disconnected")
    client = SimpleNamespace(started=True, status=SimpleNamespace(armed=True),
                             estop=estop, close=lambda: events.append("client_close"))
    @contextmanager
    def resource(name):
        try:
            yield
        finally:
            events.append(name)
    with pytest.raises(ValueError, match="tracking failure"):
        with deployment_resources(client) as resources:
            resources.enter_context(resource("camera_stop"))
            resources.enter_context(resource("capture_stop"))
            raise ValueError("tracking failure")
    assert events == ["estop", "client_close", "capture_stop", "camera_stop"]


def test_blocked_capture_is_interrupted_and_thread_joined():
    waiting, stopped = threading.Event(), threading.Event()
    class Camera:
        started = True
        def next_frame(self):
            waiting.set()
            assert stopped.wait(2)
            raise RuntimeError("pipeline interrupted")
        def stop(self):
            stopped.set()
    worker = RealSenseCaptureWorker(Camera()).start()
    assert waiting.wait(1)
    worker.stop(join_timeout_s=0.1)
    assert not worker.running
    assert worker.error is None


def test_camera_error_propagates_without_reusing_frame():
    class Camera:
        started = True
        def next_frame(self):
            raise ValueError("device disconnected")
    with RealSenseCaptureWorker(Camera()) as worker:
        with pytest.raises(RuntimeError, match="capture worker failed"):
            worker.wait_for_frame(timeout_s=1)


def test_cleanup_failure_preserves_original_tracking_error():
    closed = []
    client = SimpleNamespace(started=False, close=lambda: closed.append("client"))
    def broken_stop():
        raise RuntimeError("camera shutdown failed")
    with pytest.raises(ValueError, match="tracking failure"):
        with deployment_resources(client) as resources:
            resources.callback(broken_stop)
            raise ValueError("tracking failure")
    assert closed == ["client"]


@pytest.mark.parametrize("guarded", [False, True])
def test_sdk_launch_passes_custom_paths_as_arguments(monkeypatch, guarded):
    captured = []
    class LaunchIntercepted(Exception):
        pass
    def capture(command, **kwargs):
        captured.extend(command)
        raise LaunchIntercepted
    monkeypatch.setattr("subprocess.Popen", capture)
    kwargs = {"container": "poise-test", "sdk_root": "/opt/vendor sdk", "sdk_python": "/opt/py/bin/python"}
    client = GuardedSharpaControlClient([-1] * 22, [1] * 22, **kwargs) if guarded else SharpaReadOnlyStateClient(**kwargs)
    with pytest.raises(LaunchIntercepted):
        client.start()
    assert "SHARPA_SDK_ROOT=/opt/vendor sdk" in captured
    assert "LD_LIBRARY_PATH=/opt/vendor sdk/lib" in captured
    assert captured[-5:-1] == ["poise-test", "/opt/py/bin/python", "-u", "-c"]
    compile(captured[-1], "SDK helper", "exec")
