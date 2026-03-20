"""Tests for TypeScript type check reward functions.

Tests both with and without tsc available:
- If tsc IS available: test actual type checking scores
- If tsc NOT available: test graceful fallback
- Always: test combined reward, syntax, style, completeness checks
"""

import pytest

from cola_coder.reasoning.rewards.type_check import TypeCheckReward
from cola_coder.reasoning.rewards.batch_type_check import BatchTypeChecker
from cola_coder.reasoning.rewards.combined import (
    CombinedReward,
    _check_syntax,
    _check_style,
    _check_completeness,
)

# --- Test data ---

PERFECT_TS = """\
interface User {
    id: number;
    name: string;
    email: string;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

const users: User[] = [
    { id: 1, name: "Alice", email: "alice@example.com" },
];

const messages: string[] = users.map(greet);
"""

MINOR_ERRORS_TS = """\
interface Config {
    host: string;
    port: number;
}

function createServer(config: Config): string {
    const badPort: string = config.port;
    return `${config.host}:${config.port}`;
}
"""

MODERATE_ERRORS_TS = """\
function process(data) {
    const result = data.map(item => item.value);
    const sum: string = result.reduce((a, b) => a + b, 0);
    const flag: boolean = "hello";
    return sum;
}
"""

SYNTAX_ERROR_TS = """\
function broken( {
    const x = ;
    if (true {
        console.log("oops")
    }
"""

EMPTY_TS = ""

SIMPLE_VALID_TS = "const x: number = 42;\n"


# =============================================================================
# TypeCheckReward tests
# =============================================================================

class TestTypeCheckReward:
    """Tests for TypeCheckReward (requires tsc)."""

    def test_is_available(self):
        """is_available() should return a boolean without crashing."""
        result = TypeCheckReward.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_perfect_code_scores_high(self):
        """Perfect TypeScript code should score 1.0."""
        reward = TypeCheckReward()
        score = reward.score(PERFECT_TS)
        assert score == 1.0

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_simple_valid_code(self):
        """Simple valid code should score 1.0."""
        reward = TypeCheckReward()
        score = reward.score(SIMPLE_VALID_TS)
        assert score == 1.0

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_minor_errors_score_medium(self):
        """Code with 1-2 type errors should score 0.7."""
        reward = TypeCheckReward()
        score = reward.score(MINOR_ERRORS_TS)
        assert score < 1.0
        assert score >= 0.0  # Not a syntax error

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_syntax_error_scores_negative(self):
        """Syntax errors should score -0.5."""
        reward = TypeCheckReward()
        score = reward.score(SYNTAX_ERROR_TS)
        assert score <= 0.0

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_detailed_score_returns_dict(self):
        """detailed_score() should return a dict with expected keys."""
        reward = TypeCheckReward()
        result = reward.detailed_score(PERFECT_TS)
        assert isinstance(result, dict)
        assert "score" in result
        assert "num_errors" in result
        assert "errors" in result
        assert "error_codes" in result
        assert "has_syntax_errors" in result

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_detailed_score_error_info(self):
        """detailed_score() on erroneous code should return error details."""
        reward = TypeCheckReward()
        result = reward.detailed_score(MINOR_ERRORS_TS)
        assert result["num_errors"] > 0
        assert len(result["error_codes"]) > 0
        # Error codes should be like "TS2322"
        for code in result["error_codes"]:
            assert code.startswith("TS")

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_caching(self):
        """Scoring the same code twice should use cache (fast second call)."""
        import time
        reward = TypeCheckReward()

        # First call (uncached)
        start = time.perf_counter()
        score1 = reward.score(PERFECT_TS)
        t1 = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        score2 = reward.score(PERFECT_TS)
        t2 = time.perf_counter() - start

        assert score1 == score2
        # Cache hit should be much faster (at least 10x)
        # But don't make timing assertions too strict
        assert t2 < t1 or t2 < 0.001  # Either faster or both very fast

    @pytest.mark.skipif(
        not TypeCheckReward.is_available(),
        reason="tsc not installed",
    )
    def test_timeout_handling(self):
        """Very short timeout should not hang (may return None/0.0)."""
        reward = TypeCheckReward(timeout=1)
        # This should complete within the timeout or return 0.0
        score = reward.score(PERFECT_TS)
        assert isinstance(score, float)


# =============================================================================
# BatchTypeChecker tests
# =============================================================================

class TestBatchTypeChecker:
    """Tests for BatchTypeChecker."""

    @pytest.mark.skipif(
        not BatchTypeChecker.is_available(),
        reason="tsc not installed",
    )
    def test_batch_scoring(self):
        """Batch scoring should return a score per file."""
        checker = BatchTypeChecker()
        codes = [PERFECT_TS, MINOR_ERRORS_TS, SYNTAX_ERROR_TS]
        scores = checker.score_batch(codes)

        assert len(scores) == 3
        assert scores[0] == 1.0  # Perfect code
        assert scores[1] < 1.0   # Has type errors
        assert scores[2] <= 0.0  # Syntax error

    @pytest.mark.skipif(
        not BatchTypeChecker.is_available(),
        reason="tsc not installed",
    )
    def test_batch_detailed(self):
        """detailed_batch() should return per-file diagnostics."""
        checker = BatchTypeChecker()
        codes = [PERFECT_TS, MINOR_ERRORS_TS]
        results = checker.detailed_batch(codes)

        assert len(results) == 2
        assert results[0]["num_errors"] == 0
        assert results[1]["num_errors"] > 0

    @pytest.mark.skipif(
        not BatchTypeChecker.is_available(),
        reason="tsc not installed",
    )
    def test_empty_batch(self):
        """Empty batch should return empty list."""
        checker = BatchTypeChecker()
        results = checker.score_batch([])
        assert results == []

    @pytest.mark.skipif(
        not BatchTypeChecker.is_available(),
        reason="tsc not installed",
    )
    def test_single_item_batch(self):
        """Single-item batch should work."""
        checker = BatchTypeChecker()
        scores = checker.score_batch([PERFECT_TS])
        assert len(scores) == 1
        assert scores[0] == 1.0

    def test_batch_graceful_without_tsc(self):
        """If tsc not available, batch should return zero scores."""
        checker = BatchTypeChecker()
        if checker._tsc_path is None:
            results = checker.detailed_batch([PERFECT_TS])
            assert len(results) == 1
            assert results[0]["tsc_failed"] is True


# =============================================================================
# Syntax / Style / Completeness checks (always run, no tsc needed)
# =============================================================================

class TestSyntaxCheck:
    """Tests for _check_syntax (bracket/brace matching)."""

    def test_balanced_code(self):
        score = _check_syntax("function f() { return { x: 1 }; }")
        assert score == 1.0

    def test_unbalanced_braces(self):
        score = _check_syntax("function f() { return { x: 1 }")
        assert score < 1.0

    def test_empty_code(self):
        score = _check_syntax("")
        assert score == 0.0

    def test_string_brackets_ignored(self):
        """Brackets inside strings should not count."""
        score = _check_syntax('const s = "{ not a real brace }";')
        assert score == 1.0

    def test_complex_nesting(self):
        code = "if (a) { if (b) { return [1, 2]; } }"
        score = _check_syntax(code)
        assert score == 1.0


class TestStyleCheck:
    """Tests for _check_style (naming conventions)."""

    def test_camel_case_good(self):
        score = _check_style("const myVariable = 42;")
        assert score >= 0.8

    def test_snake_case_penalty(self):
        code = "const my_variable = 42;\nlet another_one = 'hi';"
        score = _check_style(code)
        assert score < 1.0

    def test_empty_code(self):
        score = _check_style("")
        assert score == 0.0

    def test_long_lines_penalty(self):
        code = "const x = " + "a" * 130 + ";"
        score = _check_style(code)
        assert score < 1.0


class TestCompletenessCheck:
    """Tests for _check_completeness (truncation detection)."""

    def test_complete_code(self):
        score = _check_completeness("function f(): number { return 42; }")
        assert score >= 0.8

    def test_truncated_code(self):
        score = _check_completeness("function f() { return")
        assert score < 1.0

    def test_unbalanced_braces(self):
        score = _check_completeness("function f() {\n  if (true) {\n")
        assert score < 1.0

    def test_empty_code(self):
        score = _check_completeness("")
        assert score == 0.0

    def test_very_short_code(self):
        score = _check_completeness("x = 1")
        assert score < 0.5


# =============================================================================
# CombinedReward tests
# =============================================================================

class TestCombinedReward:
    """Tests for CombinedReward (works with or without tsc)."""

    def test_combined_returns_float(self):
        reward = CombinedReward()
        score = reward.score(PERFECT_TS)
        assert isinstance(score, float)

    def test_perfect_code_scores_high(self):
        reward = CombinedReward()
        score = reward.score(PERFECT_TS)
        # Even without tsc, syntax/style/completeness should be high
        assert score > 0.5

    def test_syntax_error_scores_low(self):
        reward = CombinedReward()
        score = reward.score(SYNTAX_ERROR_TS)
        assert score < 0.5

    def test_empty_code_scores_low(self):
        reward = CombinedReward()
        score = reward.score(EMPTY_TS)
        # Empty code: tsc may give it 1.0 (no errors in empty file),
        # but syntax/style/completeness are 0.0, so combined is low-ish
        assert score < 0.7

    def test_detailed_score_has_all_fields(self):
        reward = CombinedReward()
        result = reward.detailed_score(PERFECT_TS)
        assert "combined_score" in result
        assert "type_score" in result
        assert "syntax_score" in result
        assert "style_score" in result
        assert "completeness_score" in result
        assert "weights" in result

    def test_weights_sum_to_one(self):
        reward = CombinedReward()
        result = reward.detailed_score(PERFECT_TS)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_score_batch(self):
        reward = CombinedReward()
        scores = reward.score_batch([PERFECT_TS, SYNTAX_ERROR_TS, EMPTY_TS])
        assert len(scores) == 3
        assert scores[0] > scores[1]  # Perfect > syntax error
        assert scores[0] > scores[2]  # Perfect > empty

    def test_has_type_checker_property(self):
        reward = CombinedReward()
        assert isinstance(reward.has_type_checker, bool)
