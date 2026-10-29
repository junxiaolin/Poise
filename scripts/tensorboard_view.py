"""Mirror core metrics and optional per-object curricula into compact TensorBoard logs.

Reads the original events without changing the trainer or its full logs.
Only TensorBoard is required; this process does not import Torch or Isaac Sim.
"""

import argparse
import fcntl
import json
import logging
import os
import struct
import time
from pathlib import Path

from google.protobuf.message import DecodeError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.scalar.metadata import create_summary_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class AppendingEventReader:
    """Read local TFRecords, committing offsets only after a complete valid event.

    Reopen at each poll and reread incomplete tails instead of keeping their
    bytes buffered. Network-mounted live logs can expose a transient partial
    record that otherwise leaves TensorBoard's long-lived reader stuck.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.warned_offset = None
        self.bad_offset = None
        self.recoveries = []

    def Load(self):
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.offset:
                self.offset = 0
                self.identity = identity
                self.bad_offset = None
            stream.seek(self.offset)
            while True:
                header = stream.read(12)
                if len(header) < 12:
                    return
                length, header_crc = struct.unpack("<QI", header)
                if masked_crc32c(header[:8]) != header_crc:
                    if self._recover(stream, stat.st_size):
                        continue
                    return
                # Bound reads by this poll's snapshot, including the data CRC.
                if length + 4 > stat.st_size - stream.tell():
                    return
                data = stream.read(length)
                footer = stream.read(4)
                if len(data) != length or len(footer) != 4:
                    return
                if masked_crc32c(data) != struct.unpack("<I", footer)[0]:
                    if self._recover(stream, stat.st_size):
                        continue
                    return
                try:
                    event = Event.FromString(data)
                except DecodeError:
                    if self._recover(stream, stat.st_size):
                        continue
                    return
                self.offset = stream.tell()
                self.warned_offset = None
                self.bad_offset = None
                yield event

    def _recover(self, stream, size):
        # First retry from disk on the next poll: a live tail may be transient.
        self._warn_checksum()
        if self.bad_offset != self.offset:
            self.bad_offset = self.offset
            return False
        # A corrupt record must not hide every later update. Require two whole
        # consecutive records with valid header/data CRCs and Event protobufs
        # before moving past it. Never modify or fabricate the source events.
        start = self.offset + 1
        stream.seek(start)
        window = stream.read(min(1024 * 1024, max(0, size - start)))
        for index in range(max(0, len(window) - 15)):
            length, crc = struct.unpack_from("<QI", window, index)
            position = start + index
            if length > size - position - 16:
                continue
            if masked_crc32c(window[index:index + 8]) != crc:
                continue
            next_position = self._valid_record_end(stream, position, size)
            if next_position is None or self._valid_record_end(stream, next_position, size) is None:
                continue
            recovery = dict(path=str(self.path), start=self.offset, end=position,
                            skipped_bytes=position - self.offset, checked_unix=time.time())
            self.recoveries.append(recovery)
            logging.warning("Recovered event stream: %s", json.dumps(recovery))
            self.offset = position
            self.bad_offset = None
            self.warned_offset = None
            stream.seek(position)
            return True
        return False

    @staticmethod
    def _valid_record_end(stream, position, size):
        if position + 16 > size:
            return None
        stream.seek(position)
        header = stream.read(12)
        if len(header) != 12:
            return None
        length, crc = struct.unpack("<QI", header)
        if length > size - position - 16 or masked_crc32c(header[:8]) != crc:
            return None
        data, footer = stream.read(length), stream.read(4)
        if len(data) != length or len(footer) != 4:
            return None
        if masked_crc32c(data) != struct.unpack("<I", footer)[0]:
            return None
        try:
            event = Event.FromString(data)
        except DecodeError:
            return None
        if event.wall_time <= 0 or event.WhichOneof("what") is None:
            return None
        return position + length + 16

    def _warn_checksum(self):
        if self.warned_offset != self.offset:
            logging.warning("Retrying event checksum at %s offset %d", self.path, self.offset)
            self.warned_offset = self.offset


# Original scalar -> (concise display name, unit conversion, interpretation).
METRICS = {
    "Train/mean_reward": ("01_Performance/Mean_reward", 1.0, "Mean episode return from rank zero."),
    "Episode/episode_success_rate": (
        "01_Performance/Episode_success_rate_pct", 100.0,
        "Ended episodes with at least one reached target, rank-zero logged mean; not held-out success."),
    "Episode/drop_rate": (
        "01_Performance/Per_step_drop_rate_pct", 100.0,
        "Mean fraction of environments meeting the drop condition per control step, rank zero."),
    "Episode/basic_pose_position_error_m": (
        "02_Target_error/Position_error_mm", 1000.0, "Mean target position error, rank zero."),
    "Episode/basic_pose_orientation_error_deg": (
        "02_Target_error/Orientation_error_deg", 1.0, "Mean target orientation error, rank zero."),
    "Episode/shared_rotation_curriculum_max_deg": (
        "03_Curriculum/Rotation_range_deg", 1.0, "Current synchronized curriculum maximum: 5 to 180 degrees."),
    "Episode/translation_delta_curriculum_max_m": (
        "03_Curriculum/Translation_range_mm", 1000.0, "Current synchronized translation curriculum maximum (mm); final limit is set by the run recipe."),
    "Episode/target_delta_frontier_success_rate": (
        "03_Curriculum/Rotation_frontier_success_rate_pct", 100.0,
        "Logged success rate at the rotation curriculum frontier."),
    "Episode/translation_delta_frontier_success_rate": (
        "03_Curriculum/Translation_frontier_success_rate_pct", 100.0,
        "Logged success rate at the translation curriculum frontier."),
    "Policy/mean_noise_std": (
        "04_Training/Action_std", 1.0, "Mean action noise standard deviation, bounded to [0.3, 1.5]."),
    "Perf/total_fps": (
        "04_Training/Sampling_steps_per_s", 1.0, "Global transitions divided by rollout and update time."),
}


def metric_map(source):
    metrics = dict(METRICS)
    config = source / "env.json"
    if config.is_file():
        cfg = json.loads(config.read_text())
        if cfg.get("multiobject_per_object_curriculum_enabled"):
            for object_id in cfg["primitive_object_ids"]:
                suffix = object_id.replace("/", "_")
                for original, label, scale in (
                    ("shared_rotation_curriculum_max_deg", "Rotation_range_deg", 1.0),
                    ("translation_delta_curriculum_max_m", "Translation_range_mm", 1000.0),
                    ("target_delta_frontier_success_rate", "Rotation_frontier_success_rate_pct", 100.0),
                    ("translation_delta_frontier_success_rate", "Translation_frontier_success_rate_pct", 100.0),
                ):
                    metrics[f"Episode/{original}_{suffix}"] = (
                        f"05_Objects/{suffix}/{label}", scale,
                        f"Synchronized per-object curriculum: {object_id}. Global gate is per object.",
                    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Original training event directory")
    parser.add_argument("output", type=Path, help="Separate compact event directory")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    if not source.is_dir() or source == output or source in output.parents:
        parser.error("Use an existing source and a separate output outside the source tree")
    if args.interval <= 0:
        parser.error("interval must be positive")
    metrics = metric_map(source)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".mirror.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_steps = {}
        if list(output.glob("events.out.tfevents.*")):
            existing = EventAccumulator(str(output), size_guidance={"scalars": 1}).Reload()
            for tag in existing.Tags()["scalars"]:
                records = existing.Scalars(tag)
                if records:
                    last_steps[tag] = records[-1].step
        (output / "metrics.json").write_text(json.dumps(
            {"source": str(source), "metrics": metrics}, indent=2, ensure_ascii=False))
        writer = EventFileWriter(str(output), flush_secs=10)
        loaders = {}
        try:
            while True:
                count = 0
                for path in sorted(source.glob("events.out.tfevents.*")):
                    if path not in loaders:
                        loaders[path] = AppendingEventReader(path)
                    for event in loaders[path].Load():
                        values = []
                        for value in event.summary.value:
                            if value.tag not in metrics or not value.HasField("simple_value"):
                                continue
                            tag, scale, description = metrics[value.tag]
                            if event.step <= last_steps.get(tag, -1):
                                continue
                            values.append(Summary.Value(tag=tag, simple_value=value.simple_value * scale,
                                metadata=create_summary_metadata(display_name="", description=description)))
                            last_steps[tag] = event.step
                        if values:
                            writer.add_event(Event(wall_time=event.wall_time, step=event.step,
                                                   summary=Summary(value=values)))
                            count += len(values)
                writer.flush()
                status = dict(checked_unix=time.time(), last_steps=last_steps,
                              sources=[dict(path=str(path), offset=reader.offset,
                                            size=path.stat().st_size,
                                            recoveries=reader.recoveries)
                                       for path, reader in loaders.items()])
                pending = output / "mirror-status.tmp"
                pending.write_text(json.dumps(status, indent=2) + "\n")
                pending.replace(output / "mirror-status.json")
                if count:
                    print(f"Mirrored {count} scalar points; latest update index {max(last_steps.values())}", flush=True)
                if args.once:
                    break
                time.sleep(args.interval)
        finally:
            writer.close()


if __name__ == "__main__":
    main()
