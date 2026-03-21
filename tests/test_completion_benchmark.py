"""Tests for CompletionBenchmark (evaluation/completion_benchmark.py).

All tests run without GPU or model inference.
"""

from __future__ import annotations

import pytest

from cola_coder.evaluation.completion_benchmark import (
    PROBLEMS,
    BenchmarkReport,
    CompletionBenchmark,
    CompletionProblem,
    get_problems_by_category,
    get_problems_by_difficulty,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bench() -> CompletionBenchmark:
    return CompletionBenchmark()


def perfect_generator(prompt: str) -> str:
    """Returns the canonical correct completion for every easy problem."""
    completions: dict[str, str] = {
        "complete_add": "return a + b\n",
        "complete_len_check": "return len(lst) == 0\n",
        "complete_absolute": "return -x\n",
        "complete_string_reverse": "return s[::-1]\n",
        "complete_max_of_two": "return a\n    return b\n",
        "complete_list_sum": "total += n\n    return total\n",
        "complete_factorial_base": "return 1\n    return n * factorial(n - 1)\n",
        "complete_contains": "return item in lst\n",
        "complete_greet": 'return f"Hello {name}"\n',
        "complete_square": "return x ** 2\n",
    }
    for key, completion in completions.items():
        if key in prompt or ("add" in prompt and "int" in prompt):
            return completion
    return "pass\n"


def always_wrong_generator(prompt: str) -> str:
    return "pass\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProblems:
    def test_problems_count(self):
        assert len(PROBLEMS) >= 30

    def test_all_have_required_fields(self):
        for p in PROBLEMS:
            assert p.task_id, f"task_id empty for {p}"
            assert p.prefix, f"prefix empty for {p.task_id}"
            assert p.required_patterns, f"required_patterns empty for {p.task_id}"
            assert p.difficulty in ("easy", "medium", "hard"), p.task_id

    def test_unique_task_ids(self):
        ids = [p.task_id for p in PROBLEMS]
        assert len(ids) == len(set(ids)), "Duplicate task_ids found"

    def test_get_by_difficulty(self):
        easy = get_problems_by_difficulty("easy")
        assert len(easy) >= 5
        assert all(p.difficulty == "easy" for p in easy)

    def test_get_by_category(self):
        math_probs = get_problems_by_category("math")
        assert len(math_probs) >= 2
        assert all(p.category == "math" for p in math_probs)


class TestScoreSingle:
    def test_correct_completion_passes(self, bench):
        problem = CompletionProblem(
            task_id="test_add",
            prefix="def add(a, b):\n    ",
            required_patterns=[r"return\s+a\s*\+\s*b"],
        )
        result = bench.score_single(problem, "return a + b\n")
        assert result.passed is True
        assert result.missed_patterns == []

    def test_wrong_completion_fails(self, bench):
        problem = CompletionProblem(
            task_id="test_add",
            prefix="def add(a, b):\n    ",
            required_patterns=[r"return\s+a\s*\+\s*b"],
        )
        result = bench.score_single(problem, "pass\n")
        assert result.passed is False
        assert len(result.missed_patterns) == 1

    def test_forbidden_pattern_fails(self, bench):
        problem = CompletionProblem(
            task_id="test_safe",
            prefix="def safe():\n    ",
            required_patterns=[r"return"],
            forbidden_patterns=[r"\beval\b"],
        )
        result = bench.score_single(problem, "return eval('1+1')\n")
        assert result.passed is False
        assert len(result.forbidden_found) == 1

    def test_score_by_id(self, bench):
        result = bench.score("complete_add", "return a + b\n")
        assert result.task_id == "complete_add"

    def test_unknown_task_id_raises(self, bench):
        with pytest.raises(KeyError, match="Unknown task_id"):
            bench.score("nonexistent_task", "pass\n")


class TestRun:
    def test_run_returns_report(self, bench):
        report = bench.run(always_wrong_generator)
        assert isinstance(report, BenchmarkReport)
        assert report.total == len(PROBLEMS)
        assert report.passed + report.failed == report.total

    def test_run_all_fail(self, bench):
        # Note: a few problems (abstract method, protocol) have "pass" as a valid answer.
        # "pass\n" also matches fibonacci because memo[n] appears in the prefix.
        # So we just assert the majority fail.
        report = bench.run(always_wrong_generator)
        assert report.passed < report.total
        assert report.pass_rate < 0.5

    def test_by_difficulty_populated(self, bench):
        report = bench.run(always_wrong_generator)
        assert "easy" in report.by_difficulty
        assert "hard" in report.by_difficulty

    def test_by_category_populated(self, bench):
        report = bench.run(always_wrong_generator)
        assert "math" in report.by_category

    def test_run_error_in_generator(self, bench):
        def exploding(prompt):
            raise RuntimeError("oops")

        report = bench.run(exploding)
        assert report.total == len(PROBLEMS)
        # Error completions "<ERROR: ...>" should not match most patterns
        # (a tiny few problems like abstract_method accept "pass" which won't match)
        assert report.pass_rate < 0.2


class TestMarkdown:
    def test_to_markdown_contains_header(self, bench):
        report = bench.run(always_wrong_generator)
        md = bench.to_markdown(report)
        assert "# Code Completion Benchmark" in md

    def test_to_markdown_has_pass_rate(self, bench):
        report = bench.run(always_wrong_generator)
        md = bench.to_markdown(report)
        assert "Pass rate" in md or "pass_rate" in md.lower() or "0.0%" in md

    def test_to_markdown_has_per_problem_table(self, bench):
        report = bench.run(always_wrong_generator)
        md = bench.to_markdown(report)
        assert "complete_add" in md


class TestCustomProblems:
    def test_custom_problem_list(self):
        custom = [
            CompletionProblem(
                task_id="custom_1",
                prefix="x = ",
                required_patterns=[r"42"],
            )
        ]
        bench = CompletionBenchmark(problems=custom)
        report = bench.run(lambda p: "42\n")
        assert report.passed == 1

    def test_latency_tracked(self, bench):
        report = bench.run(lambda p: "pass\n")
        assert report.total_latency_ms >= 0
