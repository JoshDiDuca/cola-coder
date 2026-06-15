"""Tests for the EducationalValueScorer (FineWeb-Edu cheap static proxy).

Verifies the contract: high score for well-documented/tested/well-named code, low
score for obfuscated/comment-free/repetitive blobs, ranges, determinism, graceful
empty handling, and reuse of shared utilities (no reinvented hashing/lang detection).
"""

from __future__ import annotations

import inspect

import pytest

from cola_coder.data.scorers.educational_value import EducationalValueScorer
from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult


# --- Fixtures: representative code samples ---

GOOD_PYTHON = '''
"""Compute Fibonacci numbers iteratively.

This module demonstrates an efficient O(n) Fibonacci implementation with a
worked example and a unit test.
"""


def fibonacci(position: int) -> int:
    """Return the Fibonacci number at the given position.

    Args:
        position: Zero-based index into the Fibonacci sequence.

    Returns:
        The Fibonacci number at that position.
    """
    previous, current = 0, 1
    for _ in range(position):
        previous, current = current, previous + current
    return previous


def test_fibonacci() -> None:
    """Verify the first few Fibonacci numbers."""
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55


if __name__ == "__main__":
    # Example usage: print the 10th Fibonacci number.
    print(fibonacci(10))
'''


GOOD_TYPESCRIPT = '''
/**
 * Compute the factorial of a non-negative integer.
 *
 * This module shows a clear recursive factorial with an inline example test.
 */
export function factorial(value: number): number {
  // Base case: factorial of 0 is 1.
  if (value <= 1) {
    return 1;
  }
  // Recursive case: n * factorial(n - 1).
  return value * factorial(value - 1);
}

describe("factorial", () => {
  it("computes small factorials correctly", () => {
    expect(factorial(0)).toBe(1);
    expect(factorial(5)).toBe(120);
  });
});
'''


OBFUSCATED_BLOB = (
    "var _0x1f=function(a,b){return a+b};"
    "var c=_0x1f(1,2);var d=_0x1f(c,3);var e=_0x1f(d,4);"
    "var f=_0x1f(e,5);var g=_0x1f(f,6);console.log(g);"
)


REPETITIVE_BLOB = "\n".join(["x = x + 1"] * 80)


@pytest.fixture()
def scorer() -> EducationalValueScorer:
    return EducationalValueScorer()


# --- Protocol conformance ---

def test_conforms_to_scorer_protocol(scorer: EducationalValueScorer) -> None:
    assert isinstance(scorer, ScorerProtocol)
    assert scorer.name == "educational_value"
    assert EducationalValueScorer.is_available() is True


# --- Core behaviour: high vs low ---

def test_well_documented_python_scores_high(scorer: EducationalValueScorer) -> None:
    result = scorer.score(GOOD_PYTHON, {"language": "python"})
    assert result.score > 0.6, result.details


def test_well_documented_typescript_scores_high(scorer: EducationalValueScorer) -> None:
    result = scorer.score(GOOD_TYPESCRIPT, {"language": "typescript"})
    assert result.score > 0.6, result.details


def test_obfuscated_blob_scores_low(scorer: EducationalValueScorer) -> None:
    result = scorer.score(OBFUSCATED_BLOB, {"language": "javascript"})
    assert result.score < 0.4, result.details


def test_repetitive_blob_scores_low(scorer: EducationalValueScorer) -> None:
    result = scorer.score(REPETITIVE_BLOB, {"language": "python"})
    assert result.score < 0.4, result.details


def test_good_outscores_bad(scorer: EducationalValueScorer) -> None:
    good = scorer.score(GOOD_PYTHON, {"language": "python"}).score
    bad = scorer.score(OBFUSCATED_BLOB, {"language": "javascript"}).score
    assert good > bad


# --- Range, empty handling, language sanity ---

@pytest.mark.parametrize(
    "code,meta",
    [
        (GOOD_PYTHON, {"language": "python"}),
        (GOOD_TYPESCRIPT, {"language": "typescript"}),
        (OBFUSCATED_BLOB, {"language": "javascript"}),
        (REPETITIVE_BLOB, None),
        ("", None),
        ("   \n\t  \n", None),
        (GOOD_PYTHON, None),  # no language hint -> agnostic path
    ],
)
def test_score_always_in_range(
    scorer: EducationalValueScorer,
    code: str,
    meta: dict[str, object] | None,
) -> None:
    result = scorer.score(code, meta)
    assert isinstance(result, ScorerResult)
    assert 0.0 <= result.score <= 1.0


def test_empty_input_low_no_crash(scorer: EducationalValueScorer) -> None:
    for blank in ("", "   ", "\n\n\t"):
        result = scorer.score(blank, None)
        assert result.score < 0.4
        assert 0.0 <= result.score <= 1.0


def test_agnostic_path_does_not_penalise_python(scorer: EducationalValueScorer) -> None:
    # With no language hint, the good Python sample should still score well.
    result = scorer.score(GOOD_PYTHON, None)
    assert result.score > 0.6, result.details


# --- Determinism ---

def test_deterministic(scorer: EducationalValueScorer) -> None:
    a = scorer.score(GOOD_PYTHON, {"language": "python"}).score
    b = scorer.score(GOOD_PYTHON, {"language": "python"}).score
    assert a == b

    # Fresh instance must agree too.
    c = EducationalValueScorer().score(GOOD_PYTHON, {"language": "python"}).score
    assert a == c


# --- Batch parity ---

def test_score_batch_matches_score(scorer: EducationalValueScorer) -> None:
    items: list[tuple[str, dict[str, object] | None]] = [
        (GOOD_PYTHON, {"language": "python"}),
        (OBFUSCATED_BLOB, {"language": "javascript"}),
        ("", None),
    ]
    batch = scorer.score_batch(items)
    assert len(batch) == len(items)
    for (code, meta), batched in zip(items, batch):
        assert batched.score == scorer.score(code, meta).score


# --- Details exposed for transparency ---

def test_details_contain_signal_breakdown(scorer: EducationalValueScorer) -> None:
    result = scorer.score(GOOD_PYTHON, {"language": "python"})
    for key in (
        "comment_docstring",
        "example_or_test",
        "naming_quality",
        "structural_completeness",
        "non_degenerate",
    ):
        assert key in result.details


# --- Registry wiring ---

def test_registered_in_registry() -> None:
    from cola_coder.data.scorers.registry import _instantiate_scorer
    from cola_coder.data.scorers.sandbox import SandboxedRunner

    scorer = _instantiate_scorer("educational_value", {}, SandboxedRunner(), None)
    assert scorer is not None
    assert scorer.name == "educational_value"


# --- DRY: shared utilities are reused, not reinvented ---

def test_reuses_shared_utilities_no_reinvention() -> None:
    import cola_coder.data.scorers.educational_value as mod

    src = inspect.getsource(mod)
    # No reinvented MD5 hashing.
    assert "hashlib" not in src
    assert ".md5(" not in src
    # Reuses shared language detection + repetition + ScoreMapper.
    assert "language_detect" in src
    assert "compute_repetition_metrics" in src
    assert "ScoreMapper" in src
