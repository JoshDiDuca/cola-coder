"""TOOL-015a/b/c/d: menus and their leaf options must run cleanly (no crash).

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

TOOL-015c extends the same per-leaf coverage to every TrainingMenu leaf (the
6 groups: Pipeline Manager, Foundation, Pre-Training, Post-Training, Alignment
& Reasoning, Monitoring & Tools).

TOOL-015d extends it to every EvalMenu leaf and every PipelineMenu leaf. The
PipelineMenu leaves dispatch real stages via ``_run_stage_script`` (which raises
on non-zero exit, unlike ``_run_script``); the pipeline leaf fixture additionally
stubs ``_run_stage_script`` so no stage ever executes/trains even if a leaf were
to reach dispatch.
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
from cola_coder.features.menus.pipeline_menu import PipelineMenu as _PipelineMenuCls


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

    Several leaves prompt with bare `input()` (not via cli). Under pytest stdin
    is captured, so a real `input()` raises OSError rather than EOFError. The
    leaf handlers all wrap `input()` in `try/except (EOFError,
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


# ── TOOL-015c: drive every TrainingMenu LEAF option through its handler ──────


# Every TrainingMenu leaf reachable from the 6 group menus. The group menus
# (_foundation_menu, _pretraining_menu, _post_training_menu, _alignment_menu,
# _monitoring_menu) are loops that dispatch by cli.choose() index; with prompts
# stubbed to cancel they exit on the first pass, so we invoke each group menu
# AND each terminal leaf handler directly to exercise them past option-list
# construction. _background_training_menu is itself a dispatch loop (covered),
# and its individual actions (start/stop/status/schedule/remove) are listed as
# leaves too. _train_size_menu has a `resume` default so it is callable bare.
# Pipeline Manager (group 0) delegates to PipelineMenu, covered by TOOL-015d.
_TRAINING_LEAVES = [
    # Group menus (dispatch loops)
    "_foundation_menu",
    "_pretraining_menu",
    "_post_training_menu",
    "_alignment_menu",
    "_monitoring_menu",
    # Foundation (Stage 1-2)
    "_train_tokenizer",
    # Pre-Training (Stage 3)
    "_train_size_menu",
    "_resume_training_menu",
    "_background_training_menu",
    "_start_background_training",
    "_stop_background_training",
    "_show_background_status",
    "_schedule_overnight_training",
    "_remove_overnight_schedule",
    # Post-Training (Stage 4-7)
    "_extend_context_menu",
    "_generate_instructions_menu",
    "_instruction_tuning_menu",
    "_moe_upcycling_menu",
    "_moe_finetune_menu",
    # Alignment & Reasoning (Stage 8-9)
    "_train_router_menu",
    "_train_reasoning",
    "_self_play_training_menu",
    # Monitoring & Tools
    "_vram_estimate_menu",
    "_lr_finder_menu",
    "_training_dashboard",
    "_eval_history_menu",
    # Legacy full-pipeline launcher (reachable via Pipeline Manager)
    "_full_pipeline_menu",
]


@pytest.mark.parametrize("leaf", _TRAINING_LEAVES)
def test_training_menu_leaf_runs_cleanly(stubbed_leaf, tmp_path, leaf):
    """Each TrainingMenu leaf option reaches its handler and returns without
    raising when every prompt is cancelled/defaulted (TOOL-015c)."""
    master = MasterMenu(project_root=tmp_path)
    training = TrainingMenu(master)
    getattr(training, leaf)()


def test_training_menu_group_handlers_exist():
    """Guard: the leaf list stays in sync with the dispatch table — every
    handler named here is an attribute of TrainingMenu."""
    for leaf in _TRAINING_LEAVES:
        assert callable(getattr(TrainingMenu, leaf, None)), leaf


# ── TOOL-015d: drive every EvalMenu LEAF option through its handler ──────────


# Every EvalMenu leaf reachable from the 5 group menus and the top-level menu.
# Group menus (_benchmarks_menu, _router_eval_menu, _quality_menu,
# _compare_menu) are dispatch loops; the top-level menu() also dispatches to
# _safety_eval_menu / _routing_accuracy_menu / _contamination_menu directly.
# The top-level Training Status entry delegates to ToolsMenu (covered by 015a).
_EVAL_LEAVES = [
    # Group menus (dispatch loops)
    "_benchmarks_menu",
    "_router_eval_menu",
    "_quality_menu",
    "_compare_menu",
    # Benchmarks
    "_ts_benchmark_menu",
    "_ts_quick_benchmark",
    "_ts_react_benchmark",
    "_python_humaneval_menu",
    "_python_completion_benchmark",
    "_benchmark_menu",
    "_run_eval_suite_menu",
    "_inference_profiler_menu",
    # Router Evaluation
    "_domain_detection_test",
    "_router_accuracy",
    "_router_specialist_benchmark",
    # Quality & Regression
    "_smoke_test_menu",
    "_regression_test_menu",
    "_quality_report_menu",
    "_model_card_menu",
    # Compare
    "_compare_checkpoints_menu",
    "_compare_models_menu",
    "_checkpoint_diff_menu",
    "_checkpoint_info_menu",
    # Top-level direct entries
    "_safety_eval_menu",
    "_routing_accuracy_menu",
    "_contamination_menu",
]


@pytest.mark.parametrize("leaf", _EVAL_LEAVES)
def test_eval_menu_leaf_runs_cleanly(stubbed_leaf, tmp_path, leaf):
    """Each EvalMenu leaf option reaches its handler and returns without
    raising when every prompt is cancelled/defaulted (TOOL-015d)."""
    master = MasterMenu(project_root=tmp_path)
    evalm = EvalMenu(master)
    getattr(evalm, leaf)()


def test_eval_menu_group_handlers_exist():
    """Guard: the leaf list stays in sync with the dispatch table — every
    handler named here is an attribute of EvalMenu."""
    for leaf in _EVAL_LEAVES:
        assert callable(getattr(EvalMenu, leaf, None)), leaf


# ── TOOL-015d: drive every PipelineMenu LEAF option through its handler ──────


@pytest.fixture
def stubbed_pipeline_leaf(stubbed_leaf, monkeypatch):
    """`stubbed_leaf` plus a stubbed PipelineMenu._run_stage_script.

    PipelineMenu stage handlers run real scripts via `_run_stage_script`, which
    (unlike `_run_script`) RAISES on non-zero exit. With every prompt cancelled
    the leaves return before dispatching a stage, but we stub `_run_stage_script`
    defensively so that even if a leaf reached `_execute_stage`/`_dispatch_stage`
    no actual training/collection script would ever run.
    """
    monkeypatch.setattr(_PipelineMenuCls, "_run_stage_script", lambda *a, **k: None)


# Every PipelineMenu leaf reachable from the top-level Pipeline Manager menu.
# All are dispatch handlers that either prompt (and cancel) or short-circuit on
# an empty runs list. _full_auto profiles hardware (read-only) then cancels at
# the mode prompt; _legacy_pipeline delegates to TrainingMenu._full_pipeline_menu.
_PIPELINE_LEAVES = [
    "_full_auto",
    "_create_run",
    "_resume_run",
    "_view_runs",
    "_run_single_stage",
    "_reset_to_stage",
    "_delete_run",
    "_legacy_pipeline",
]


@pytest.mark.parametrize("leaf", _PIPELINE_LEAVES)
def test_pipeline_menu_leaf_runs_cleanly(stubbed_pipeline_leaf, tmp_path, leaf):
    """Each PipelineMenu leaf option reaches its handler and returns without
    raising when every prompt is cancelled/defaulted, and without executing any
    real pipeline stage (TOOL-015d)."""
    master = MasterMenu(project_root=tmp_path)
    pipeline = PipelineMenu(master)
    getattr(pipeline, leaf)()


def test_pipeline_menu_group_handlers_exist():
    """Guard: the leaf list stays in sync with the dispatch table — every
    handler named here is an attribute of PipelineMenu."""
    for leaf in _PIPELINE_LEAVES:
        assert callable(getattr(PipelineMenu, leaf, None)), leaf
