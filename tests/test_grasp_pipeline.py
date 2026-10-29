"""Failure and resume contracts for the multi-environment cache builder."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import pytest

spec = importlib.util.spec_from_file_location(
    "build_grasp_cache", Path(__file__).parents[1] / "scripts/build_grasp_cache.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resume_checks_inputs_outputs_and_does_not_repeat_stage(tmp_path):
    src = tmp_path / "input"
    src.write_text("fresh")
    out = tmp_path / "output"
    counter = tmp_path / "count"
    cmd = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(out)!r}).write_text(Path({str(src)!r}).read_text()); Path({str(counter)!r}).write_text('ran')",
    ]
    stage = module.Stages(tmp_path, os.environ.copy(), False, "contract")
    stage.run("build", cmd, [src], [out])
    assert out.read_text() == "fresh"
    counter.unlink()
    stage.resume = True
    stage.run("build", cmd, [src], [out])
    assert not counter.exists()
    src.write_text("different")
    with pytest.raises(RuntimeError, match="differs"):
        stage.run("build", cmd, [src], [out])


def test_failed_stage_never_has_completion_receipt(tmp_path):
    stage = module.Stages(tmp_path, os.environ.copy(), True, "contract")
    with pytest.raises(RuntimeError, match="exited 7"):
        stage.run(
            "broken",
            [sys.executable, "-c", "raise SystemExit(7)"],
            [],
            [tmp_path / "missing"],
        )
    assert not (tmp_path / "broken.stage.json").exists()
    assert (
        json.loads((tmp_path / "broken.running.json").read_text())["status"] == "failed"
    )


def test_unreceipted_outputs_are_not_silently_reused(tmp_path):
    out = tmp_path / "partial"
    out.write_text("partial")
    stage = module.Stages(tmp_path, os.environ.copy(), True, "contract")
    with pytest.raises(FileExistsError, match="Unreceipted"):
        stage.run("partial", [sys.executable, "-c", "pass"], [], [out])


def test_interrupted_stage_stops_its_child_group(tmp_path, monkeypatch):
    import signal

    class Child:
        pid = 123456
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    child = Child()
    launches = []
    stopped = []

    def launch(*args, **kwargs):
        launches.append(kwargs)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(
        module.os, "killpg", lambda pid, sig: stopped.append((pid, sig))
    )
    stage = module.Stages(tmp_path, os.environ.copy(), False, "contract")
    with pytest.raises(KeyboardInterrupt):
        stage.run("interrupted", ["worker"], [], [tmp_path / "result"])
    assert launches[0]["start_new_session"] is True
    assert stopped == [(child.pid, signal.SIGTERM)]
    assert child.waits == 2
    assert not (tmp_path / "interrupted.stage.json").exists()
    assert (
        json.loads((tmp_path / "interrupted.running.json").read_text())["status"]
        == "interrupted"
    )


def test_resume_rejects_modified_completed_output(tmp_path):
    out = tmp_path / "result"
    cmd = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(out)!r}).write_text('complete')",
    ]
    stage = module.Stages(tmp_path, os.environ.copy(), False, "contract")
    stage.run("build", cmd, [], [out])
    out.write_text("changed")
    stage.resume = True
    with pytest.raises(RuntimeError, match="differs"):
        stage.run("build", cmd, [], [out])


def test_fast_parent_shortfall_is_explicit_and_formal_remains_strict():
    cfg = dict(parents=32, rows=320, min_variants=5, max_variants=15)
    with pytest.raises(RuntimeError, match="need 32"):
        module.parent_targets(cfg, 20)
    cfg["minimum_parents"] = 8
    assert module.parent_targets(cfg, 20) == (20, 200)
    assert module.parent_targets(cfg, 40) == (32, 320)
    with pytest.raises(RuntimeError, match="need 8"):
        module.parent_targets(cfg, 7)


def test_time_budget_terminates_child_without_completion_receipt(tmp_path):
    pid_path = tmp_path / "pid"
    out = tmp_path / "result"
    cmd = [sys.executable, "-c", (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        f"time.sleep(60); Path({str(out)!r}).write_text('too late')"
    )]
    stage = module.Stages(tmp_path, os.environ.copy(), False, "contract",
                          deadline=time.monotonic() + 1)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        stage.run("slow", cmd, [], [out])
    assert not out.exists()
    assert not (tmp_path / "slow.stage.json").exists()
    record = json.loads((tmp_path / "slow.running.json").read_text())
    assert record["status"] == "timed_out"
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def test_exhausted_budget_never_launches_next_stage(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("launched work after deadline")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    stage = module.Stages(tmp_path, os.environ.copy(), False, "contract",
                          deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        stage.run("next", ["worker"], [], [tmp_path / "result"])
    assert not (tmp_path / "next.running.json").exists()
