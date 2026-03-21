"""Tests for readability_scorer.py (feature 50)."""

from __future__ import annotations

import textwrap


from cola_coder.features.readability_scorer import (
    FEATURE_ENABLED,
    ReadabilityReport,
    ReadabilityScorer,
    _count_line_types,
    _detect_indentation_style,
    _score_comment_density,
    _score_naming,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Line-based helpers
# ---------------------------------------------------------------------------


def test_count_line_types():
    code = textwrap.dedent("""\
        # This is a comment
        def foo():
            return 1
        # Another comment
    """)
    comment, code_lines = _count_line_types(code.splitlines())
    assert comment == 2
    assert code_lines == 2


def test_detect_indentation_spaces():
    lines = ["def foo():", "    return 1"]
    assert _detect_indentation_style(lines) == "spaces"


def test_detect_indentation_tabs():
    lines = ["def foo():", "\treturn 1"]
    assert _detect_indentation_style(lines) == "tabs"


def test_detect_indentation_mixed():
    lines = ["def foo():", "    return 1", "\tpass"]
    assert _detect_indentation_style(lines) == "mixed"


def test_score_comment_density_ideal():
    # At ideal density (0.15), score should be near 1.0
    score = _score_comment_density(comment_lines=15, code_lines=85, ideal=0.15)
    assert score > 0.9


def test_score_comment_density_no_comments():
    score = _score_comment_density(comment_lines=0, code_lines=100, ideal=0.15)
    # No comments at all should score below perfect (1.0)
    assert score < 1.0


def test_score_naming_all_descriptive():
    names = ["calculate_total", "user_name", "process_items", "result"]
    score = _score_naming(names, min_length=3)
    assert score == 1.0


def test_score_naming_cryptic_names():
    names = ["a", "b", "c", "calculate_total"]  # 3 of 4 are cryptic
    score = _score_naming(names, min_length=3)
    assert score < 1.0


def test_score_naming_allowed_short():
    # i, j, x, y etc. are convention-allowed short names
    names = ["i", "j", "x", "calculate_sum"]
    score = _score_naming(names, min_length=3)
    assert score == 1.0  # all allowed


# ---------------------------------------------------------------------------
# Scorer integration
# ---------------------------------------------------------------------------


_CLEAN_CODE = textwrap.dedent("""\
    def calculate_total(items, discount):
        \"\"\"Calculate the total price with discount applied.\"\"\"
        # Apply the discount to each item
        total = 0.0
        for item in items:
            total += item * (1 - discount)
        return total
""")

_COMPLEX_CODE = textwrap.dedent("""\
    def process(a, b, c, d, e):
        if a:
            for i in range(b):
                if i % 2 == 0:
                    while c > 0:
                        if d and e:
                            c -= 1
                        elif not d:
                            break
        return a
""")


def test_clean_code_high_score():
    scorer = ReadabilityScorer()
    report = scorer.score(_CLEAN_CODE)
    assert report.parse_error is None
    assert report.overall_score > 0.5


def test_complex_code_lower_complexity_score():
    scorer = ReadabilityScorer(max_cyclomatic=5)
    report = scorer.score(_COMPLEX_CODE)
    assert report.complexity_score < 1.0


def test_function_with_docstring():
    scorer = ReadabilityScorer()
    report = scorer.score(_CLEAN_CODE)
    fn = report.function_reports[0]
    assert fn.has_docstring


def test_function_without_docstring():
    code = "def foo(x):\n    return x + 1\n"
    scorer = ReadabilityScorer()
    report = scorer.score(code)
    assert not report.function_reports[0].has_docstring


def test_long_function_penalised():
    body = "\n".join(f"    x_{i} = {i}" for i in range(60))
    code = f"def long_fn():\n{body}\n    return x_0\n"
    scorer = ReadabilityScorer(max_function_lines=30)
    report = scorer.score(code)
    assert report.function_length_score < 1.0


def test_mixed_indent_penalised():
    code = "def foo():\n    x = 1\n\treturn x\n"
    scorer = ReadabilityScorer()
    report = scorer.score(code)
    assert report.indentation_score < 0.5


def test_syntax_error_handled_gracefully():
    scorer = ReadabilityScorer()
    report = scorer.score("def foo(: pass")
    assert report.parse_error is not None
    # Overall score still computed (partially)
    assert 0.0 <= report.overall_score <= 1.0


def test_summary_format():
    scorer = ReadabilityScorer()
    report = scorer.score(_CLEAN_CODE)
    s = report.summary()
    assert "overall=" in s
    assert "naming=" in s


def test_grade_output():
    report = ReadabilityReport(overall_score=0.9)
    assert report.grade() == "A"
    report.overall_score = 0.72
    assert report.grade() == "B"
    report.overall_score = 0.3
    assert report.grade() == "F"


def test_empty_code():
    scorer = ReadabilityScorer()
    report = scorer.score("")
    assert report.total_lines == 0
    assert 0.0 <= report.overall_score <= 1.0
