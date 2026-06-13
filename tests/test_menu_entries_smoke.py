"""TOOL-015a/b: menus and their leaf options must run cleanly (no crash).

Non-interactive smoke harness. Constructs each sub-menu and calls its entry
method(s) with ALL interactive prompts stubbed to "cancel" and ALL
script/subprocess execution stubbed out, asserting NO unhandled exception is
raised. This catches render / option-building crashes — e.g. the BUG-117
"No datasets found" class where a menu blew up just building its option list —
across the whole menu tree, without running anything real (no training, no
downloads, no nvidia-smi).

TOOL-015a covers the menu-OPEN level (each top-level sub-menu opens + exits).
TOOL-015b extends this to every individual LEAF option of the DataMenu: each
leaf handler is invoked directly with all prompts cancelled/defaulted, so the
handler is exercised past option-list construction down to its early-return
cancel path. This catches leaf-level render/wiring crashes that the menu-open
test cannot reach.

Deeper per-leaf coverage of the other sub-menus is tracked as TOOL-015c/d.
"""

from __future__ import annotations

import builtins
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


# ── TOOL-015b: drive every DataMenu LEAF option through its handler ──────────


@pytest.fixture
def stubbed_leaf(stubbed, monkeypatch):
    """`stubbed` plus a stubbed builtins.input.

    Several DataMenu leaves prompt with bare `input()` (not via cli). Under
    pytest stdin is captured, so a real `input()` raises OSError rather than
    EOFError. The leaf handlers all wrap `input()` in `try/except (EOFError,
    KeyboardInterrupt)` to mean "cancel", so we raise EOFError to drive every
    text prompt down that same cancel path — keeping the smoke test purely
    interaction-free.
    """
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))


# Every DataMenu LEAF reachable from the 5 group menus. The group menus
# (_collect_data_menu, _modify_data_menu, ...) are loops that dispatch by the
# cli.choose() index; with prompts stubbed to cancel they exit immediately, so
# we invoke each leaf handler directly to exercise it past option construction.
# Inline `_run_script(...)` dispatch arms have no dedicated handler (they are
# covered by the menu-open test) and are intentionally omitted here.
_DATA_LEAVES = [
    # Collect Data
    "_huggingface_wizard",
    "_software_heritage_info",
    "_scrape_docs_menu",
    "_collect_text_data_menu",
    "_collect_math_data_menu",
    "_collect_github_artifacts_menu",
    "_download_instruction_datasets_menu",
    # Modify Data
    "_combine_datasets_menu",
    # Score & Filter
    "_score_quality_menu",
    "_score_hf_samples",
    "_score_repos_menu",
    "_train_quality_classifier_menu",
    "_run_scoring_pipeline",
    "_llm_judge_annotation",
    "_train_judge_classifier",
    "_apply_curriculum_ordering",
    "_scan_malware_menu",
    "_advanced_filters_info",
    # Inspect & View
    "_inspect_dataset",
    # Prepare for Training
    "_prepare_data_menu",
    "_prepare_training_wizard",
    "_prepare_mixed_data_menu",
    "_prepare_repo_level_data_menu",
]


@pytest.mark.parametrize("leaf", _DATA_LEAVES)
def test_data_menu_leaf_runs_cleanly(stubbed_leaf, tmp_path, leaf):
    """Each DataMenu leaf option reaches its handler and returns without
    raising when every prompt is cancelled/defaulted (BUG-117 leaf coverage)."""
    master = MasterMenu(project_root=tmp_path)
    data = DataMenu(master)
    getattr(data, leaf)()


def test_data_menu_group_handlers_exist():
    """Guard: the leaf list stays in sync with the dispatch table — every
    handler named here is an attribute of DataMenu."""
    for leaf in _DATA_LEAVES:
        assert callable(getattr(DataMenu, leaf, None)), leaf
