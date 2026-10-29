"""Live scalar mirroring must recover from records observed during a write."""

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.record_writer import RecordWriter

SCRIPT = Path(__file__).parents[1] / "scripts/tensorboard_view.py"
spec = importlib.util.spec_from_file_location("tensorboard_view", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(step):
    stream = io.BytesIO()
    RecordWriter(stream).write(Event(
        wall_time=time.time(), step=step,
        summary=Summary(value=[Summary.Value(
            tag="Episode/target_delta_frontier_success_rate", simple_value=0.25)]),
    ).SerializeToString())
    return stream.getvalue()


@pytest.mark.parametrize("split", [1, 8, 10, 12, 20, -1])
def test_partial_record_retried_without_loss_or_duplicates(tmp_path, split):
    path = tmp_path / "events"
    first, second = record(1), record(2)
    path.write_bytes(first + second[:split])
    reader = module.AppendingEventReader(path)
    assert [event.step for event in reader.Load()] == [1]
    assert list(reader.Load()) == []
    with path.open("ab") as stream:
        stream.write(second[split:] + record(3))
    assert [event.step for event in reader.Load()] == [2, 3]
    assert list(reader.Load()) == []


def test_transient_bad_tail_is_reread_from_disk(tmp_path):
    path = tmp_path / "events"
    first, second = record(1), record(2)
    bad = bytearray(second)
    bad[-1] ^= 1
    path.write_bytes(first + bad)
    reader = module.AppendingEventReader(path)
    assert [event.step for event in reader.Load()] == [1]
    # Repair in place, keeping inode and size unchanged, as with a stale tail.
    with path.open("r+b") as stream:
        stream.seek(len(first))
        stream.write(second)
    assert [event.step for event in reader.Load()] == [2]


@pytest.mark.parametrize("damage", ["garbage", "truncated", "checksum"])
def test_persistent_damage_recovers_to_checked_records(tmp_path, damage):
    path = tmp_path / "events"
    first = record(1)
    second = record(2)
    corruption = {"garbage": b"abcdefgh", "truncated": second[:-7],
                  "checksum": second[:-1] + bytes([second[-1] ^ 1])}[damage]
    path.write_bytes(first + corruption + record(3) + record(4))
    original = path.read_bytes()
    reader = module.AppendingEventReader(path)
    assert [event.step for event in reader.Load()] == [1]
    assert [event.step for event in reader.Load()] == [3, 4]
    assert list(reader.Load()) == []
    assert reader.recoveries[0]["skipped_bytes"] == len(corruption)
    assert path.read_bytes() == original


def test_recovery_waits_for_two_valid_records(tmp_path):
    path = tmp_path / "events"
    path.write_bytes(record(1) + b"damaged" + record(3))
    original = path.read_bytes()
    reader = module.AppendingEventReader(path)
    assert [event.step for event in reader.Load()] == [1]
    assert list(reader.Load()) == []
    assert path.read_bytes() == original
    with path.open("ab") as stream:
        stream.write(record(4))
    assert [event.step for event in reader.Load()] == [3, 4]


def test_live_mirror_and_restart_preserve_values_once(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    path = source / "events.out.tfevents.test"
    path.write_bytes(record(1))
    command = [sys.executable, str(SCRIPT), str(source), str(output)]
    process = subprocess.Popen(command + ["--interval", "0.05"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    tag = "03_Curriculum/Rotation_frontier_success_rate_pct"

    def wait_for(step):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert process.poll() is None
            if list(output.glob("events.out.tfevents.*")):
                accumulator = EventAccumulator(str(output), size_guidance={"scalars": 0}).Reload()
                if tag in accumulator.Tags()["scalars"]:
                    points = accumulator.Scalars(tag)
                    if points[-1].step == step:
                        return points
            time.sleep(0.05)
        pytest.fail(f"Mirror did not reach step {step}")

    try:
        wait_for(1)
        second = record(2)
        with path.open("ab") as stream:
            stream.write(second[:10])
        time.sleep(0.15)
        with path.open("ab") as stream:
            stream.write(second[10:] + record(3))
        assert [(p.step, p.value) for p in wait_for(3)] == [(1, 25.0), (2, 25.0), (3, 25.0)]
        with path.open("ab") as stream:
            stream.write(b"persistent damage" + record(4) + record(5))
        assert [p.step for p in wait_for(5)] == [1, 2, 3, 4, 5]
    finally:
        process.terminate()
        process.communicate(timeout=10)
    subprocess.run(command + ["--once"], check=True, capture_output=True, timeout=10)
    points = EventAccumulator(str(output), size_guidance={"scalars": 0}).Reload().Scalars(tag)
    assert [p.step for p in points] == [1, 2, 3, 4, 5]
