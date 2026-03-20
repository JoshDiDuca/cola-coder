"""Tests for the FineWeb-Edu style quality classifier.

Tests the heuristic scorer, classifier fallback behavior, and filter plugin.
"""

import pytest

from cola_coder.data.filters.quality_classifier import (
    CodeQualityClassifier,
    HeuristicQualityScorer,
    QualityAnnotator,
    QualityClassifierFilter,
)
from cola_coder.data.pipeline import DataRecord


# ---------------------------------------------------------------------------
# Sample code fixtures
# ---------------------------------------------------------------------------

EXCELLENT_PYTHON = '''\
"""Calculator module with basic arithmetic operations.

Provides a Calculator class that tracks operation history,
useful for auditing and undo functionality.
"""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


class Calculator:
    """A simple calculator that tracks operation history.

    Usage:
        calc = Calculator()
        result = calc.add(2, 3)  # 5
        print(calc.history)      # [('+', 2, 3, 5)]
    """

    def __init__(self) -> None:
        self.history: list[tuple[str, Number, Number, Number]] = []

    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers and record in history."""
        result = a + b
        self.history.append(("+", a, b, result))
        return result

    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a and record in history."""
        result = a - b
        self.history.append(("-", a, b, result))
        return result

    def multiply(self, a: Number, b: Number) -> Number:
        """Multiply two numbers and record in history."""
        result = a * b
        self.history.append(("*", a, b, result))
        return result

    def clear_history(self) -> None:
        """Clear the operation history."""
        self.history.clear()
'''

GOOD_TYPESCRIPT = '''\
import { useState, useCallback } from 'react';

interface TodoItem {
    id: string;
    text: string;
    completed: boolean;
}

/**
 * Custom hook for managing a todo list.
 *
 * @returns Todo list state and mutation functions
 */
export function useTodos() {
    const [todos, setTodos] = useState<TodoItem[]>([]);

    const addTodo = useCallback((text: string) => {
        setTodos(prev => [
            ...prev,
            { id: crypto.randomUUID(), text, completed: false },
        ]);
    }, []);

    const toggleTodo = useCallback((id: string) => {
        setTodos(prev =>
            prev.map(todo =>
                todo.id === id ? { ...todo, completed: !todo.completed } : todo
            )
        );
    }, []);

    return { todos, addTodo, toggleTodo };
}
'''

POOR_PYTHON = '''\
x=1
y=2
z=x+y
a=z*2
b=a-1
print(b)
if b>5:
    print("big")
else:
    print("small")
for i in range(10):
    for j in range(10):
        for k in range(10):
            print(i,j,k)
'''

MINIFIED_JS = (
    'function a(b,c){return b+c}function d(e,f){return e*f}'
    'var g=a(1,2);var h=d(3,4);console.log(g,h);'
    'function x(y){for(var i=0;i<y;i++){console.log(i)}}'
    'var z=x(10);var w=a(g,h);console.log(w,z);'
)

EMPTY_CODE = ""

TRIVIAL_CODE = "x = 1\n"

AVERAGE_PYTHON = '''\
import os
import sys

def process_files(directory):
    files = os.listdir(directory)
    results = []
    for f in files:
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            with open(path) as fh:
                content = fh.read()
                results.append(len(content))
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = process_files(sys.argv[1])
        print(f"Processed {len(data)} files")
    else:
        print("Usage: script.py <directory>")
'''


# ---------------------------------------------------------------------------
# HeuristicQualityScorer tests
# ---------------------------------------------------------------------------

class TestHeuristicQualityScorer:
    """Test the heuristic quality scorer with various code samples."""

    def setup_method(self):
        self.scorer = HeuristicQualityScorer()

    def test_excellent_python_scores_high(self):
        score = self.scorer.score(EXCELLENT_PYTHON, "python")
        assert score >= 0.6, f"Excellent Python code scored too low: {score:.3f}"

    def test_good_typescript_scores_high(self):
        score = self.scorer.score(GOOD_TYPESCRIPT, "typescript")
        assert score >= 0.5, f"Good TypeScript code scored too low: {score:.3f}"

    def test_poor_python_scores_low(self):
        score = self.scorer.score(POOR_PYTHON, "python")
        assert score < 0.55, f"Poor Python code scored too high: {score:.3f}"

    def test_minified_js_scores_low(self):
        score = self.scorer.score(MINIFIED_JS, "javascript")
        assert score < 0.4, f"Minified JS scored too high: {score:.3f}"

    def test_empty_code_scores_zero(self):
        score = self.scorer.score(EMPTY_CODE)
        assert score == 0.0, f"Empty code should score 0.0, got {score:.3f}"

    def test_trivial_code_scores_very_low(self):
        score = self.scorer.score(TRIVIAL_CODE)
        assert score < 0.2, f"Trivial code scored too high: {score:.3f}"

    def test_average_python_scores_middle(self):
        score = self.scorer.score(AVERAGE_PYTHON, "python")
        assert 0.3 <= score <= 0.8, f"Average Python should be mid-range, got {score:.3f}"

    def test_scores_in_valid_range(self):
        """All scores should be between 0.0 and 1.0."""
        samples = [
            EXCELLENT_PYTHON, GOOD_TYPESCRIPT, POOR_PYTHON,
            MINIFIED_JS, EMPTY_CODE, TRIVIAL_CODE, AVERAGE_PYTHON,
        ]
        for code in samples:
            score = self.scorer.score(code)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score:.3f}"

    def test_excellent_beats_poor(self):
        """Well-structured code should score higher than poorly-structured."""
        excellent = self.scorer.score(EXCELLENT_PYTHON, "python")
        poor = self.scorer.score(POOR_PYTHON, "python")
        assert excellent > poor, (
            f"Excellent ({excellent:.3f}) should beat poor ({poor:.3f})"
        )

    def test_excellent_beats_minified(self):
        """Readable code should score higher than minified."""
        excellent = self.scorer.score(EXCELLENT_PYTHON, "python")
        minified = self.scorer.score(MINIFIED_JS, "javascript")
        assert excellent > minified, (
            f"Excellent ({excellent:.3f}) should beat minified ({minified:.3f})"
        )

    def test_whitespace_only_scores_zero(self):
        score = self.scorer.score("   \n\n  \t\n")
        assert score == 0.0

    def test_language_agnostic_default(self):
        """Scorer should work without specifying a language."""
        score = self.scorer.score(EXCELLENT_PYTHON)
        assert score > 0.0, "Should produce a valid score without language hint"


# ---------------------------------------------------------------------------
# CodeQualityClassifier tests
# ---------------------------------------------------------------------------

class TestCodeQualityClassifier:
    """Test the neural classifier with fallback behavior."""

    def test_fallback_to_heuristic_when_no_model(self):
        """Classifier without a model path should fall back to heuristics."""
        classifier = CodeQualityClassifier(model_path=None)
        score = classifier.score(EXCELLENT_PYTHON, "python")
        assert score > 0.0, "Should produce a valid score via heuristic fallback"

    def test_fallback_to_heuristic_when_model_missing(self):
        """Classifier with invalid model path should fall back to heuristics."""
        classifier = CodeQualityClassifier(model_path="/nonexistent/model")
        score = classifier.score(EXCELLENT_PYTHON, "python")
        assert score > 0.0, "Should fall back to heuristic when model is missing"

    def test_batch_scoring_fallback(self):
        """Batch scoring should work via heuristic fallback."""
        classifier = CodeQualityClassifier(model_path=None)
        codes = [EXCELLENT_PYTHON, POOR_PYTHON, MINIFIED_JS]
        scores = classifier.score_batch(codes, "python")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = CodeQualityClassifier.is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# QualityAnnotator tests
# ---------------------------------------------------------------------------

class TestQualityAnnotator:
    """Test the LLM annotator (without actual API calls)."""

    def test_no_api_key_raises(self):
        """Should raise RuntimeError without an API key."""
        annotator = QualityAnnotator(api_key=None)
        with pytest.raises(RuntimeError, match="No API key"):
            annotator.annotate_batch(["print('hello')"])

    def test_prompt_formatting(self):
        """Prompt should include the code and language."""
        annotator = QualityAnnotator()
        prompt = annotator.format_prompt("print('hello')", "python")
        assert "print('hello')" in prompt
        assert "python" in prompt
        assert "1 =" in prompt  # rubric
        assert "5 =" in prompt

    def test_prompt_truncation(self):
        """Very long code should be truncated in the prompt."""
        annotator = QualityAnnotator()
        long_code = "x = 1\n" * 2000  # ~12k chars
        prompt = annotator.format_prompt(long_code, "python")
        # Prompt should use truncated code (4000 chars max)
        assert len(prompt) < 5000


# ---------------------------------------------------------------------------
# QualityClassifierFilter tests
# ---------------------------------------------------------------------------

class TestQualityClassifierFilter:
    """Test the pipeline filter plugin."""

    def test_filter_name(self):
        filt = QualityClassifierFilter(mode="heuristic")
        assert "quality_classifier" in filt.name()
        assert "heuristic" in filt.name()

    def test_heuristic_mode_keeps_good_code(self):
        filt = QualityClassifierFilter(threshold=0.4, mode="heuristic")
        record = DataRecord(content=EXCELLENT_PYTHON, metadata={"language": "python"})
        keep, reason = filt.check(record)
        assert keep, f"Good code should pass filter, got rejected: {reason}"

    def test_heuristic_mode_rejects_empty(self):
        filt = QualityClassifierFilter(threshold=0.4, mode="heuristic")
        record = DataRecord(content="", metadata={})
        keep, reason = filt.check(record)
        assert not keep, "Empty code should be rejected"

    def test_heuristic_mode_rejects_trivial(self):
        filt = QualityClassifierFilter(threshold=0.4, mode="heuristic")
        record = DataRecord(content="x = 1\n", metadata={})
        keep, reason = filt.check(record)
        assert not keep, "Trivial code should be rejected"

    def test_threshold_configuration(self):
        """Higher threshold should reject more code."""
        record = DataRecord(content=AVERAGE_PYTHON, metadata={"language": "python"})

        low_filter = QualityClassifierFilter(threshold=0.1, mode="heuristic")
        high_filter = QualityClassifierFilter(threshold=0.9, mode="heuristic")

        low_keep, _ = low_filter.check(record)
        high_keep, _ = high_filter.check(record)

        assert low_keep, "Very low threshold should keep average code"
        assert not high_keep, "Very high threshold should reject average code"

    def test_classifier_mode_falls_back_to_heuristic(self):
        """Classifier mode without a trained model should fall back to heuristic."""
        filt = QualityClassifierFilter(
            threshold=0.4, mode="classifier", model_path=None
        )
        record = DataRecord(content=EXCELLENT_PYTHON, metadata={"language": "python"})
        keep, reason = filt.check(record)
        # Should work (via fallback) rather than crash
        assert keep, f"Should fall back to heuristic and keep good code: {reason}"

    def test_setup_from_config(self):
        """Filter should accept config via setup()."""
        filt = QualityClassifierFilter()
        filt.setup({"threshold": 0.7, "mode": "heuristic"})
        assert filt._threshold == 0.7
        assert filt._mode == "heuristic"

    def test_quality_score_stored_in_metadata(self):
        """Filter should store the quality score in record metadata."""
        filt = QualityClassifierFilter(threshold=0.1, mode="heuristic")
        record = DataRecord(content=EXCELLENT_PYTHON, metadata={"language": "python"})
        filt.check(record)
        assert "quality_score" in record.metadata
        assert 0.0 <= record.metadata["quality_score"] <= 1.0

    def test_filter_with_minified_code(self):
        """Minified code should be rejected at default threshold."""
        filt = QualityClassifierFilter(threshold=0.4, mode="heuristic")
        record = DataRecord(content=MINIFIED_JS, metadata={"language": "javascript"})
        keep, reason = filt.check(record)
        assert not keep, f"Minified JS should be rejected, but was kept"
