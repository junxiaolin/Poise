"""Reject stock LyGra interfaces and swallowed native-loader failures early."""

from types import SimpleNamespace

import pytest

from poise.grasp.lightning import runtime


def test_stock_optimizer_interface_fails_with_named_parameter(monkeypatch):
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: SimpleNamespace(
        batch_object_hand_collision_check=lambda tree, object_point: None,
    ))
    with pytest.raises(RuntimeError, match="default_q"):
        runtime.check_interfaces()


def test_missing_native_extension_fails_with_install_instructions(monkeypatch):
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: SimpleNamespace(gem=None))
    with pytest.raises(RuntimeError, match="docs/lightning_grasp.md"):
        runtime.check_runtime()


def test_incompatible_native_extension_fails_before_search(monkeypatch):
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: SimpleNamespace(gem=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="batch_random_mask_gather"):
        runtime.check_runtime()
