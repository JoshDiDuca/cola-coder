"""Tests for repetition_detector.py."""

import pytest

from cola_coder.features.repetition_detector import RepetitionDetector, RepetitionReport


@pytest.fixture
def detector():
    return RepetitionDetector()


def test_feature_enabled():
    from cola_coder.features.repetition_detector import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_empty_code(detector):
    report = detector.detect("")
    assert report.score == 0.0
    assert report.total_lines == 0
    assert isinstance(report, RepetitionReport)


def test_unique_code_low_score(detector):
    code = """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

class Calculator:
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
"""
    report = detector.detect(code)
    assert report.score < 0.3, f"Expected low score for unique code, got {report.score}"


def test_fully_repeated_code_high_score(detector):
    line = "print('hello world')\n"
    code = line * 30
    report = detector.detect(code)
    assert report.score > 0.5, f"Expected high score for repeated code, got {report.score}"
    assert report.duplicate_line_ratio > 0.5


def test_repeated_lines_detected(detector):
    code = "x = 1\nx = 1\nx = 1\ny = 2\nz = 3\n"
    report = detector.detect(code)
    assert "x = 1" in report.repeated_lines


def test_repeated_blocks_detected(detector):
    block = "result = compute_value(x)\noutput = transform(result)\nlog_data(output)\n"
    code = block * 5
    report = detector.detect(code)
    assert report.max_block_repeat > 1
    assert len(report.repeated_blocks) > 0


def test_score_range(detector):
    codes = [
        "def f(): pass",
        "x = 1\nx = 1\nx = 1\n" * 20,
        "a = 1\nb = 2\nc = 3\nd = 4\n",
    ]
    for code in codes:
        report = detector.detect(code)
        assert 0.0 <= report.score <= 1.0, f"Score out of range: {report.score}"


def test_summary_returns_string(detector):
    report = detector.detect("x = 1\ny = 2\n")
    s = report.summary()
    assert isinstance(s, str)
    assert "RepetitionScore" in s


def test_trigram_repetition_contributes(detector):
    # Repeated trigram pattern
    code = " ".join(["foo bar baz"] * 20)
    report = detector.detect(code)
    assert report.score > 0.0


def test_single_line_no_crash(detector):
    report = detector.detect("def hello(): pass")
    assert report.score >= 0.0
    assert report.total_lines >= 0
