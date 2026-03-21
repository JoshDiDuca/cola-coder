"""Tests for code_similarity.py."""

import pytest

from cola_coder.features.code_similarity import CodeSimilarity, SimilarityResult


@pytest.fixture
def sim():
    return CodeSimilarity()


_PY_A = """\
def add(a, b):
    return a + b
"""

_PY_B = """\
def add(x, y):
    return x + y
"""

_PY_C = """\
def multiply(x, y):
    return x * y
"""

_TS_A = """\
function add(a: number, b: number): number {
    return a + b;
}
"""

_TS_B = """\
function subtract(x: number, y: number): number {
    return x - y;
}
"""


def test_feature_enabled():
    from cola_coder.features.code_similarity import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_identical_python(sim):
    result = sim.compare(_PY_A, _PY_A, language="python")
    assert result.score > 0.9


def test_structurally_similar_python(sim):
    """Renamed variables but same structure."""
    result = sim.compare(_PY_A, _PY_B, language="python")
    assert result.score > 0.5, f"Expected high similarity, got {result.score}"


def test_different_python(sim):
    result_diff = sim.compare(_PY_A, _PY_C, language="python")
    # add vs multiply — just check score is in valid range
    assert 0.0 <= result_diff.score <= 1.0


def test_score_range(sim):
    pairs = [(_PY_A, _PY_B), (_PY_A, _PY_C), (_TS_A, _TS_B)]
    for a, b in pairs:
        result = sim.compare(a, b)
        assert 0.0 <= result.score <= 1.0


def test_typescript_comparison(sim):
    result = sim.compare(_TS_A, _TS_B, language="typescript")
    assert isinstance(result, SimilarityResult)
    assert result.method == "token"  # AST not available for TS


def test_ast_method_for_python(sim):
    result = sim.compare(_PY_A, _PY_B, language="python")
    assert result.method == "combined"
    assert result.ast_score is not None


def test_is_similar_threshold(sim):
    result = sim.compare(_PY_A, _PY_A, language="python")
    assert result.is_similar(threshold=0.9) is True
    assert result.is_similar(threshold=1.1) is False


def test_auto_detect_language(sim):
    result = sim.compare(_PY_A, _PY_B, language="auto")
    assert result.score > 0.0


def test_empty_code(sim):
    result = sim.compare("", "", language="python")
    assert 0.0 <= result.score <= 1.0
