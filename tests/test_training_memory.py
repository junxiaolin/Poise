"""Reclamation remains optional on unsupported allocators/platforms."""

import json

from poise.training import memory


def test_disabled_reclamation_does_not_touch_allocator(monkeypatch, capsys):
    monkeypatch.setenv("POISE_MEMORY_RECLAIM", "0")
    monkeypatch.setenv("POISE_MEMORY_REPORT", "1")

    def unexpected():
        raise AssertionError("Disabled reclamation must not resolve an allocator or run GC")

    monkeypatch.setattr(memory, "_glibc_trim", unexpected)
    monkeypatch.setattr(memory.gc, "collect", unexpected)
    memory.reclaim_host_memory("test")
    report = json.loads(capsys.readouterr().out.removeprefix("POISE_MEMORY "))
    assert report["enabled"] is False
    assert report["trim_released"] is False


def test_unsupported_allocator_is_safe(monkeypatch, capsys):
    monkeypatch.setenv("POISE_MEMORY_RECLAIM", "1")
    monkeypatch.setenv("POISE_MEMORY_REPORT", "1")
    monkeypatch.setattr(memory, "_glibc_trim", lambda: None)
    memory.reclaim_host_memory("unsupported")
    report = json.loads(capsys.readouterr().out.removeprefix("POISE_MEMORY "))
    assert report["glibc_trim_supported"] is False
    assert report["trim_released"] is False
