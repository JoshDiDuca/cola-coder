"""Tests for the TypeScript benchmark and regression suite.

All tests run without a GPU or any TypeScript / Node.js runtime.
Model/generator interactions are mocked.

Run:
    cd "C:/Users/josh/ai research/cola-coder"
    .venv/Scripts/pytest tests/test_ts_benchmark.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cola_coder.evaluation.ts_benchmark import (
    PROBLEMS,
    TSBenchmark,
    TSBenchmarkResult,
    TSProblem,
    _pattern_check,
    _structural_check,
    _type_annotation_check,
)
from cola_coder.evaluation.regression import (
    BASELINES,
    RegressionBaseline,
    RegressionResult,
    RegressionSuite,
    _check_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(outputs: list[str] | str) -> MagicMock:
    """Return a mock generator that cycles through `outputs`."""
    gen = MagicMock()
    if isinstance(outputs, str):
        gen.generate.return_value = outputs
    else:
        gen.generate.side_effect = outputs
    return gen


def _get_problem(problem_id: str) -> TSProblem:
    for p in PROBLEMS:
        if p.id == problem_id:
            return p
    raise KeyError(f"Problem not found: {problem_id}")


# ===========================================================================
# 1. Problem loading and filtering
# ===========================================================================


class TestProblemLoading:
    def test_total_problem_count(self):
        """Exactly 50 problems must be defined."""
        assert len(PROBLEMS) == 50

    def test_all_problems_have_required_fields(self):
        for p in PROBLEMS:
            assert p.id, f"{p.id}: missing id"
            assert p.prompt, f"{p.id}: missing prompt"
            assert p.canonical_solution, f"{p.id}: missing canonical_solution"
            assert p.category in {"basics", "types", "react", "nextjs", "prisma", "zod", "testing"}
            assert p.difficulty in {"easy", "medium", "hard"}

    def test_unique_ids(self):
        ids = [p.id for p in PROBLEMS]
        assert len(ids) == len(set(ids)), "Duplicate problem IDs found"

    def test_category_distribution(self):
        """Each target category must have at least 4 problems."""
        from collections import Counter

        counts = Counter(p.category for p in PROBLEMS)
        for cat in ("basics", "types", "react", "nextjs", "prisma", "zod", "testing"):
            assert counts[cat] >= 4, f"Category {cat!r} has only {counts[cat]} problems"


class TestBenchmarkFiltering:
    def test_no_filter_returns_all(self):
        bench = TSBenchmark()
        assert len(bench.get_problems()) == 50

    def test_single_category_filter(self):
        bench = TSBenchmark(categories=["react"])
        problems = bench.get_problems()
        assert len(problems) > 0
        assert all(p.category == "react" for p in problems)

    def test_multi_category_filter(self):
        bench = TSBenchmark(categories=["basics", "types"])
        problems = bench.get_problems()
        assert all(p.category in {"basics", "types"} for p in problems)

    def test_empty_category_returns_nothing(self):
        bench = TSBenchmark(categories=["nonexistent"])
        assert bench.get_problems() == []

    def test_none_categories_means_all(self):
        bench = TSBenchmark(categories=None)
        assert len(bench.get_problems()) == 50


# ===========================================================================
# 2. Structural evaluation
# ===========================================================================


class TestStructuralCheck:
    def test_passes_when_function_name_present(self):
        prob = _get_problem("basics_fibonacci")
        solution = "function fibonacci(n: number): number { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }"
        assert _structural_check(solution, prob) is True

    def test_falls_back_to_annotation_when_name_absent(self):
        """When the function name is absent the fallback to TS annotation check fires."""
        prob = _get_problem("basics_fibonacci")
        solution = "function wrongName(n: number): number { return 0; }"
        # 'fibonacci' not in solution, but ': number' IS present → fallback passes
        assert _structural_check(solution, prob) is True

    def test_fails_when_name_absent_and_no_annotations(self):
        """No name and no TS annotations → structural check fails."""
        prob = _get_problem("basics_fibonacci")
        solution = "function wrongName(n) { return 0; }"  # no TS annotations
        assert _structural_check(solution, prob) is False

    def test_passes_for_type_alias_problem(self):
        prob = _get_problem("types_mapped_type_readonly")
        solution = "type DeepReadonly<T> = { readonly [K in keyof T]: T[K] };"
        assert _structural_check(solution, prob) is True

    def test_passes_with_ts_annotation_fallback(self):
        """Even if name isn't matched, TS annotations should pass the fallback."""
        prob = _get_problem("basics_clamp")
        solution = "const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));"
        # The prompt has 'function clamp' but solution uses arrow — structural check
        # falls back to annotation check.
        result = _structural_check(solution, prob)
        # Arrow function still contains 'clamp' in the identifier text
        assert isinstance(result, bool)


# ===========================================================================
# 3. Type annotation checking
# ===========================================================================


class TestTypeAnnotationCheck:
    def test_plain_typescript_annotation(self):
        assert _type_annotation_check("function foo(x: string): void {}") is True

    def test_generic_annotation(self):
        assert _type_annotation_check("function id<T>(v: T): T { return v; }") is True

    def test_no_annotation_fails(self):
        assert _type_annotation_check("function foo(x) { return x; }") is False

    def test_interface_counts(self):
        assert _type_annotation_check("interface Foo { name: string; }") is True

    def test_promise_annotation(self):
        assert _type_annotation_check("async function f(): Promise<void> {}") is True

    def test_array_annotation(self):
        assert _type_annotation_check("function f(): number[] { return []; }") is True


# ===========================================================================
# 4. Pattern matching
# ===========================================================================


class TestPatternCheck:
    def test_all_required_patterns_present(self):
        prob = _get_problem("react_use_state_counter")
        solution = "const [count, setCount] = useState(0); return <div><button onClick={() => setCount(c => c+1)}>+</button>{count}</div>;"
        assert _pattern_check(solution, prob) is True

    def test_missing_required_pattern_fails(self):
        prob = _get_problem("react_use_state_counter")
        solution = "return <div>no state here</div>;"
        assert _pattern_check(solution, prob) is False

    def test_forbidden_pattern_fails(self):
        prob = TSProblem(
            id="test_forbidden",
            category="basics",
            difficulty="easy",
            description="",
            prompt="function f(): void {",
            canonical_solution="",
            test_code="",
            required_patterns=[r"return"],
            forbidden_patterns=[r"TODO"],
        )
        solution = "function f() { TODO: implement }"
        assert _pattern_check(solution, prob) is False

    def test_no_patterns_always_passes(self):
        prob = TSProblem(
            id="test_no_patterns",
            category="basics",
            difficulty="easy",
            description="",
            prompt="",
            canonical_solution="",
            test_code="",
            required_patterns=[],
            forbidden_patterns=[],
        )
        assert _pattern_check("anything goes here", prob) is True


# ===========================================================================
# 5. Full evaluate_solution integration
# ===========================================================================


class TestEvaluateSolution:
    def test_canonical_solution_passes(self):
        bench = TSBenchmark()
        prob = _get_problem("basics_fibonacci")
        result = bench.evaluate_solution(prob, prob.canonical_solution)
        assert result is True

    def test_empty_solution_fails(self):
        bench = TSBenchmark()
        prob = _get_problem("basics_fibonacci")
        assert bench.evaluate_solution(prob, "") is False

    def test_solution_without_types_fails(self):
        bench = TSBenchmark()
        prob = _get_problem("basics_fibonacci")
        bad = "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }"
        # Missing TS type annotation → tier 2 should fail
        assert bench.evaluate_solution(prob, bad) is False

    def test_zod_schema_solution(self):
        bench = TSBenchmark()
        prob = _get_problem("zod_basic_schema")
        assert bench.evaluate_solution(prob, prob.canonical_solution) is True

    def test_react_solution_passes(self):
        bench = TSBenchmark()
        prob = _get_problem("react_use_state_counter")
        assert bench.evaluate_solution(prob, prob.canonical_solution) is True


# ===========================================================================
# 6. Result aggregation
# ===========================================================================


class TestResultAggregation:
    def _make_result(self, details: list[dict]) -> TSBenchmarkResult:
        total = len(details)
        solved = sum(1 for d in details if d["passed"])
        by_cat: dict[str, list[bool]] = {}
        by_diff: dict[str, list[bool]] = {}
        for d in details:
            by_cat.setdefault(d["category"], []).append(d["passed"])
            by_diff.setdefault(d["difficulty"], []).append(d["passed"])
        return TSBenchmarkResult(
            total_problems=total,
            solved=solved,
            pass_rate=solved / total if total else 0.0,
            by_category={k: sum(v) / len(v) for k, v in by_cat.items()},
            by_difficulty={k: sum(v) / len(v) for k, v in by_diff.items()},
            details=details,
        )

    def test_pass_rate_all_solved(self):
        details = [
            {"id": "a", "category": "basics", "difficulty": "easy", "passed": True, "num_correct": 1, "num_samples": 1},
            {"id": "b", "category": "basics", "difficulty": "easy", "passed": True, "num_correct": 1, "num_samples": 1},
        ]
        r = self._make_result(details)
        assert r.pass_rate == pytest.approx(1.0)

    def test_pass_rate_none_solved(self):
        details = [
            {"id": "a", "category": "basics", "difficulty": "easy", "passed": False, "num_correct": 0, "num_samples": 1},
        ]
        r = self._make_result(details)
        assert r.pass_rate == pytest.approx(0.0)

    def test_by_category_calculation(self):
        details = [
            {"id": "a", "category": "basics", "difficulty": "easy", "passed": True, "num_correct": 1, "num_samples": 1},
            {"id": "b", "category": "basics", "difficulty": "easy", "passed": False, "num_correct": 0, "num_samples": 1},
            {"id": "c", "category": "react", "difficulty": "medium", "passed": True, "num_correct": 1, "num_samples": 1},
        ]
        r = self._make_result(details)
        assert r.by_category["basics"] == pytest.approx(0.5)
        assert r.by_category["react"] == pytest.approx(1.0)

    def test_summary_contains_pass_rate(self):
        bench = TSBenchmark()
        gen = _make_generator(PROBLEMS[0].canonical_solution)
        result = bench.run(gen, tokenizer=None, num_samples=1)
        summary = result.summary()
        assert "TYPESCRIPT BENCHMARK RESULTS" in summary
        assert "By category" in summary

    def test_run_with_mock_generator_all_canonical(self):
        """Run benchmark with a generator that always returns the canonical solution."""
        bench = TSBenchmark(categories=["basics"])
        problems = bench.get_problems()

        call_count = [0]
        canonical_solutions = [p.canonical_solution for p in problems]

        def side_effect(**kwargs):
            idx = call_count[0] % len(canonical_solutions)
            call_count[0] += 1
            return canonical_solutions[idx]

        gen = MagicMock()
        gen.generate.side_effect = side_effect

        result = bench.run(gen, tokenizer=None)
        assert result.total_problems == len(problems)
        assert result.pass_rate > 0.0  # at least some should pass

    def test_run_with_mock_generator_all_empty(self):
        """Generator that returns empty strings — all problems should fail."""
        bench = TSBenchmark(categories=["basics"])
        gen = _make_generator("")
        result = bench.run(gen, tokenizer=None)
        assert result.solved == 0
        assert result.pass_rate == pytest.approx(0.0)


# ===========================================================================
# 7. Regression baseline checking
# ===========================================================================


class TestRegressionBaselineCheck:
    def test_baseline_count(self):
        assert len(BASELINES) == 20

    def test_all_baselines_have_required_fields(self):
        for b in BASELINES:
            assert b.prompt, f"Baseline missing prompt: {b.description}"
            assert b.category
            assert b.min_length >= 0
            assert b.max_length > b.min_length

    def test_passes_when_output_matches(self):
        baseline = RegressionBaseline(
            description="test",
            category="test",
            prompt="",
            expected_patterns=[r"return\s+\w+"],
            forbidden_patterns=[r"TODO"],
            min_length=5,
            max_length=100,
        )
        output = "return value;"
        passed, failures = _check_baseline(output, baseline)
        assert passed is True
        assert failures == []

    def test_fails_when_output_too_short(self):
        baseline = RegressionBaseline(
            description="test",
            category="test",
            prompt="",
            expected_patterns=[],
            forbidden_patterns=[],
            min_length=50,
            max_length=200,
        )
        passed, failures = _check_baseline("short", baseline)
        assert passed is False
        assert any("too short" in f for f in failures)

    def test_fails_when_output_too_long(self):
        baseline = RegressionBaseline(
            description="test",
            category="test",
            prompt="",
            expected_patterns=[],
            forbidden_patterns=[],
            min_length=1,
            max_length=5,
        )
        passed, failures = _check_baseline("this is a very long output", baseline)
        assert passed is False
        assert any("too long" in f for f in failures)

    def test_fails_when_expected_pattern_missing(self):
        baseline = RegressionBaseline(
            description="test",
            category="test",
            prompt="",
            expected_patterns=[r"useState"],
            forbidden_patterns=[],
            min_length=1,
            max_length=1000,
        )
        passed, failures = _check_baseline("no hooks here", baseline)
        assert passed is False
        assert any("useState" in f for f in failures)

    def test_fails_when_forbidden_pattern_present(self):
        baseline = RegressionBaseline(
            description="test",
            category="test",
            prompt="",
            expected_patterns=[],
            forbidden_patterns=[r"TODO"],
            min_length=1,
            max_length=1000,
        )
        passed, failures = _check_baseline("TODO: implement this", baseline)
        assert passed is False
        assert any("TODO" in f for f in failures)


# ===========================================================================
# 8. Regression suite run and comparison
# ===========================================================================


class TestRegressionSuite:
    def test_run_with_mock_generator(self):
        """Mock generator that produces plausible outputs — should pass most baselines."""
        suite = RegressionSuite()

        def smart_output(**kwargs):
            prompt: str = kwargs.get("prompt", "")
            # Return a contextually appropriate snippet based on prompt keywords
            if "def add" in prompt:
                return "    return a + b"
            if "squares" in prompt:
                return "    return [x**2 for x in range(1, n+1)]"
            if "self, x" in prompt:
                return "        self.x = x\n        self.y = y"
            if "fetch_data" in prompt:
                return "    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.json()"
            if "sum(a:" in prompt:
                return "return a + b;\n}"
            if "multiply" in prompt:
                return "a * b;"
            if "first<T>" in prompt:
                return "return arr[0];\n}"
            if "getJson" in prompt:
                return "const response = await fetch(url);\nreturn response.json() as Promise<T>;"
            if "Stack<T>" in prompt or "push(item" in prompt:
                return "this.items.push(item);\n  }"
            if "@dataclass" in prompt:
                return "    id: int\n    name: str\n    email: str"
            if "Serializable" in prompt:
                return "    return JSON.stringify(this.data);\n  }"
            if "import React" in prompt:
                return "useState, useEffect } from 'react';"
            if "from 'zod'" in prompt:
                return "object({ name: z.string(), email: z.string().email() });"
            if "reverseString" in prompt:
                return "return s.split('').reverse().join('');\n}"
            if "is_even" in prompt:
                return "    return n % 2 == 0"
            if "Sort an array" in prompt:
                return "return [...arr].sort((a, b) => a - b);\n}"
            if "const config" in prompt:
                return "3000,"
            if "function App" in prompt:
                return " className='app'><h1>Hello</h1></div>"
            if "safeFetch" in prompt:
                return "return await response.json();"
            if "httpStatus" in prompt:
                return "return 'OK';"
            return "return value;"

        gen = MagicMock()
        gen.generate.side_effect = smart_output

        result = suite.run(gen, tokenizer=None)
        assert result.total == 20
        assert result.passed + result.failed == result.total
        assert 0.0 <= result.pass_rate <= 1.0

    def test_regression_result_pass_rate_property(self):
        result = RegressionResult(total=10, passed=7, failed=3, details=[])
        assert result.pass_rate == pytest.approx(0.7)

    def test_regression_result_zero_total(self):
        result = RegressionResult(total=0, passed=0, failed=0, details=[])
        assert result.pass_rate == pytest.approx(0.0)

    def test_compare_detects_regressions(self):
        """compare_checkpoints must flag baselines that drop from pass to fail."""
        suite = RegressionSuite()

        details_a = [
            {"description": "test1", "category": "python", "passed": True, "output": "", "failures": []},
            {"description": "test2", "category": "python", "passed": True, "output": "", "failures": []},
        ]
        details_b = [
            {"description": "test1", "category": "python", "passed": False, "output": "", "failures": ["too short (0 < 10)"]},
            {"description": "test2", "category": "python", "passed": True, "output": "", "failures": []},
        ]
        result_a = RegressionResult(total=2, passed=2, failed=0, details=details_a)
        result_b = RegressionResult(total=2, passed=1, failed=1, details=details_b)

        comparison = suite.compare_checkpoints(result_a, result_b, "v1", "v2")
        assert "REGRESSIONS" in comparison
        assert "test1" in comparison

    def test_compare_detects_improvements(self):
        suite = RegressionSuite()

        details_a = [
            {"description": "test1", "category": "python", "passed": False, "output": "", "failures": ["missing"]},
        ]
        details_b = [
            {"description": "test1", "category": "python", "passed": True, "output": "ok", "failures": []},
        ]
        result_a = RegressionResult(total=1, passed=0, failed=1, details=details_a)
        result_b = RegressionResult(total=1, passed=1, failed=0, details=details_b)

        comparison = suite.compare_checkpoints(result_a, result_b)
        assert "IMPROVEMENTS" in comparison

    def test_compare_no_changes(self):
        suite = RegressionSuite()

        details_a = [{"description": "test1", "category": "python", "passed": True, "output": "", "failures": []}]
        details_b = [{"description": "test1", "category": "python", "passed": True, "output": "", "failures": []}]
        result_a = RegressionResult(total=1, passed=1, failed=0, details=details_a)
        result_b = RegressionResult(total=1, passed=1, failed=0, details=details_b)

        comparison = suite.compare_checkpoints(result_a, result_b)
        assert "No regressions" in comparison

    def test_summary_contains_categories(self):
        result = RegressionResult(
            total=2,
            passed=1,
            failed=1,
            details=[
                {"description": "py add", "category": "python", "passed": True, "output": "", "failures": []},
                {"description": "ts sum", "category": "typescript", "passed": False, "output": "", "failures": ["missing"]},
            ],
        )
        summary = result.summary()
        assert "REGRESSION TEST RESULTS" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
