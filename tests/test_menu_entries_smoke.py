"""TOOL-015a: every top-level menu must OPEN and EXIT cleanly (no crash).

Non-interactive smoke harness. Constructs each sub-menu and calls its entry
method(s) with ALL interactive prompts stubbed to "cancel" and ALL
script/subprocess execution stubbed out, asserting NO unhandled exception is
raised. This catches render / option-building crashes — e.g. the BUG-117
"No datasets found" class where a menu blew up just building its option list —
across the whole menu tree, without running anything real (no training, no
downloads, no nvidia-smi).

Deeper per-leaf coverage (driving each individual menu option through its
handler) is tracked as TOOL-015b/c/d.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from cola_coder.cli import cli
from cola_coder.features.master_menu import MasterMenu
from cola_coder.features.menus import (
    DataMenu,
    EvalMenu,
    PipelineMenu,
    ToolsMenu,
    TrainingMenu,
)


class _Canceler:
    """Stub for cli.choose: always 'cancel' (None). Raises if a menu fails to
    exit on cancel (would otherwise hang the test in an infinite loop)."""

    def __init__(self, limit: int = 50) -> None:
        self.n = 0
        self.limit = limit

    def __call__(self, *args, **kwargs):
        self.n += 1
        if self.n > self.limit:
            raise AssertionError(
                "menu did not exit when choose() returned None — likely an "
                "infinite loop / missing cancel handling"
            )
        return None


@pytest.fixture
def stubbed(monkeypatch):
    # Every interactive prompt resolves to cancel/empty so menu loops exit
    # on the first pass instead of blocking on input.
    monkeypatch.setattr(cli, "choose", _Canceler())
    monkeypatch.setattr(cli, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(cli, "multi_select", lambda *a, **k: [])
    monkeypatch.setattr(cli, "pick_languages", lambda *a, **k: [])
    monkeypatch.setattr(cli, "weight_editor", lambda *a, **k: {})
    # Never actually run a script or shell out during a smoke test.
    monkeypatch.setattr(MasterMenu, "_run_script", lambda *a, **k: None)
    monkeypatch.setattr(MasterMenu, "_pause", lambda *a, **k: None)
    _ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok)


# (sub-menu factory, entry-method name)
_ENTRIES = [
    (DataMenu, "menu"),
    (TrainingMenu, "menu"),
    (EvalMenu, "menu"),
    (ToolsMenu, "menu"),
    (ToolsMenu, "settings_menu"),
    (ToolsMenu, "training_status_menu"),
    (PipelineMenu, "menu"),
]


@pytest.mark.parametrize(
    "factory,method", _ENTRIES, ids=[f"{f.__name__}.{m}" for f, m in _ENTRIES]
)
def test_menu_opens_and_exits_cleanly(stubbed, tmp_path, factory, method):
    master = MasterMenu(project_root=tmp_path)
    sub = factory(master)
    # Must render its options once and return on cancel without raising.
    getattr(sub, method)()


def test_all_submenus_constructible(tmp_path):
    """Each sub-menu constructs from a MasterMenu without error."""
    master = MasterMenu(project_root=tmp_path)
    for factory in (DataMenu, TrainingMenu, EvalMenu, ToolsMenu, PipelineMenu):
        assert factory(master) is not None
