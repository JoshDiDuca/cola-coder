"""Tests for the enhanced complexity_scorer.py (feature 23)."""

import pytest

from cola_coder.features.complexity_scorer import ComplexityScorer


@pytest.fixture
def scorer():
    return ComplexityScorer()


def test_feature_enabled():
    from cola_coder.features.complexity_scorer import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_cognitive_complexity_nested_python(scorer):
    code = """
def complex_func(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
"""
    metrics = scorer.score(code, language="python")
    # Deeply nested code should have higher cognitive complexity
    assert metrics.cognitive_complexity > metrics.cyclomatic


def test_cognitive_complexity_flat_python(scorer):
    code = """
def flat_func(a, b, c):
    if a:
        return 1
    if b:
        return 2
    if c:
        return 3
    return 0
"""
    metrics = scorer.score(code, language="python")
    # Flat branches have lower cognitive complexity
    assert metrics.cognitive_complexity > 0


def test_function_lengths_python(scorer):
    code = """
def short_fn():
    return 1

def long_fn():
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    return x + y + z + a + b
"""
    metrics = scorer.score(code, language="python")
    assert metrics.max_function_length >= 7
    assert metrics.avg_function_length > 0


def test_nesting_per_function_python(scorer):
    code = """
def nested():
    if True:
        for i in range(10):
            pass
"""
    metrics = scorer.score(code, language="python")
    assert len(metrics.nesting_depth_per_function) >= 1


def test_typescript_cognitive_complexity(scorer):
    code = """
function process(x) {
    if (x > 0) {
        for (let i = 0; i < x; i++) {
            if (i % 2 === 0) {
                console.log(i);
            }
        }
    }
}
"""
    metrics = scorer.score(code, language="typescript")
    assert metrics.cognitive_complexity > 0


def test_summary_includes_new_fields(scorer):
    code = "def f(x):\n    return x\n"
    metrics = scorer.score(code, language="python")
    s = metrics.summary()
    assert "cognitive" in s
    assert "avg_fn_len" in s
    assert "max_fn_len" in s


def test_empty_code(scorer):
    metrics = scorer.score("", language="python")
    assert metrics.line_count == 0
    assert metrics.cognitive_complexity == 0


def test_difficulty_bucket_range(scorer):
    codes = [
        ("def f(): pass\n", "python"),
        ("const x = 1;\n", "typescript"),
    ]
    for code, lang in codes:
        m = scorer.score(code, language=lang)
        assert 1 <= m.difficulty_bucket <= 5


def test_filter_by_complexity_still_works(scorer):
    samples = [
        "def f(): pass",
        "def f(x):\n    if x:\n        return 1\n    return 0",
    ]
    results = scorer.filter_by_complexity(samples, min_bucket=1, max_bucket=5)
    assert len(results) == 2
