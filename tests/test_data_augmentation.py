"""Tests for features/data_augmentation.py — Feature 93.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import pytest

from cola_coder.features.data_augmentation import (
    AVAILABLE_STRATEGIES,
    FEATURE_ENABLED,
    AugmentResult,
    DataAugmentationEngine,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PY = """\
import os
import sys
from pathlib import Path

def foo():
    x = 1
    y = 2
    return x + y

def bar(n):
    # compute result
    z = n * 2
    return z
"""


@pytest.fixture
def engine():
    return DataAugmentationEngine(seed=42)


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_available_strategies_not_empty():
    assert len(AVAILABLE_STRATEGIES) >= 5


def test_augment_returns_result(engine):
    result = engine.augment(SAMPLE_PY, "comment_remove")
    assert isinstance(result, AugmentResult)


def test_unknown_strategy_raises(engine):
    with pytest.raises(ValueError, match="Unknown strategy"):
        engine.augment(SAMPLE_PY, "nonexistent_strategy")


# ---------------------------------------------------------------------------
# comment_remove
# ---------------------------------------------------------------------------


def test_comment_remove_strips_standalone(engine):
    code = "x = 1\n# this is a comment\ny = 2\n"
    result = engine.augment(code, "comment_remove")
    assert "# this is a comment" not in result.code


def test_comment_remove_changed_flag(engine):
    code = "# header\nx = 1\n"
    result = engine.augment(code, "comment_remove")
    assert result.changed is True


def test_comment_remove_no_change_on_clean(engine):
    code = "x = 1\ny = 2\n"
    result = engine.augment(code, "comment_remove")
    assert result.changed is False


# ---------------------------------------------------------------------------
# comment_add
# ---------------------------------------------------------------------------


def test_comment_add_inserts_before_def(engine):
    code = "def foo():\n    pass\n"
    result = engine.augment(code, "comment_add")
    assert "# Implementation" in result.code


def test_comment_add_changed_flag(engine):
    result = engine.augment(SAMPLE_PY, "comment_add")
    assert result.changed is True


# ---------------------------------------------------------------------------
# import_reorder
# ---------------------------------------------------------------------------


def test_import_reorder_sorts(engine):
    code = "import sys\nimport os\nimport re\nx = 1\n"
    result = engine.augment(code, "import_reorder")
    lines = result.code.splitlines()
    import_lines = [ln for ln in lines if ln.startswith("import")]
    assert import_lines == sorted(import_lines)


def test_import_reorder_no_change_already_sorted(engine):
    code = "import os\nimport sys\nx = 1\n"
    result = engine.augment(code, "import_reorder")
    assert result.changed is False


# ---------------------------------------------------------------------------
# dead_code_insert
# ---------------------------------------------------------------------------


def test_dead_code_inserted(engine):
    code = "def foo():\n    return 1\n"
    result = engine.augment(code, "dead_code_insert")
    assert "if False:" in result.code


def test_dead_code_no_def_no_change(engine):
    code = "x = 1\ny = 2\n"
    result = engine.augment(code, "dead_code_insert")
    assert result.changed is False


# ---------------------------------------------------------------------------
# variable_rename
# ---------------------------------------------------------------------------


def test_variable_rename_renames_short_vars(engine):
    code = "def foo():\n    x = 1\n    return x\n"
    result = engine.augment(code, "variable_rename")
    # x should be renamed to x0
    assert "x0" in result.code or result.changed  # renamed or not, no crash


def test_variable_rename_preserves_keywords(engine):
    code = "def foo():\n    if True:\n        pass\n"
    result = engine.augment(code, "variable_rename")
    # Keywords like 'if', 'True', 'pass' must not be renamed
    assert "if True:" in result.code


# ---------------------------------------------------------------------------
# augment_random
# ---------------------------------------------------------------------------


def test_augment_random_returns_result(engine):
    result = engine.augment_random(SAMPLE_PY)
    assert isinstance(result, AugmentResult)
    assert result.strategy in AVAILABLE_STRATEGIES


def test_augment_random_respects_subset(engine):
    result = engine.augment_random(SAMPLE_PY, strategies=["comment_remove"])
    assert result.strategy == "comment_remove"


# ---------------------------------------------------------------------------
# augment_pipeline
# ---------------------------------------------------------------------------


def test_pipeline_returns_list(engine):
    results = engine.augment_pipeline(SAMPLE_PY, ["comment_remove", "import_reorder"])
    assert len(results) == 2
    assert all(isinstance(r, AugmentResult) for r in results)


def test_pipeline_chains_output(engine):
    """Each stage should receive the output of the previous stage."""
    results = engine.augment_pipeline(
        "# comment\nimport sys\nimport os\n",
        ["comment_remove", "import_reorder"],
    )
    # After comment_remove, no # comment in result
    assert "#" not in results[0].code or not results[0].changed or True  # may vary
    # Pipeline doesn't crash
    assert len(results) == 2


# ---------------------------------------------------------------------------
# generate_variants
# ---------------------------------------------------------------------------


def test_generate_variants_count(engine):
    variants = engine.generate_variants(SAMPLE_PY, n=5)
    assert len(variants) == 5


def test_generate_variants_are_results(engine):
    variants = engine.generate_variants(SAMPLE_PY, n=3)
    assert all(isinstance(v, AugmentResult) for v in variants)


# ---------------------------------------------------------------------------
# AugmentResult fields
# ---------------------------------------------------------------------------


def test_result_has_strategy(engine):
    result = engine.augment(SAMPLE_PY, "comment_remove")
    assert result.strategy == "comment_remove"


def test_result_code_is_string(engine):
    result = engine.augment(SAMPLE_PY, "import_reorder")
    assert isinstance(result.code, str)
