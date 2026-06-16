"""Graded Gopher-repetition scorer (DATA-072).

`RepetitionScorer` turns the existing hard-reject repetition metrics into a
continuous 0–1 quality signal (down-weight, not just drop). These tests pin the
score ordering (clean > repetitive), the threshold-boundary behavior, the
short-doc n-gram guard, and the ScorerProtocol contract.
"""

from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult
from cola_coder.data.scorers.repetition_scorer import RepetitionScorer

# Clean, non-repetitive code (distinct lines, varied tokens).
_CLEAN = (
    "export function add(a: number, b: number): number {\n"
    "  const sum = a + b\n"
    "  return sum\n"
    "}\n"
    "export function multiply(x: number, y: number): number {\n"
    "  const product = x * y\n"
    "  return product\n"
    "}\n"
    "const greeting = 'hello world'\n"
    "console.log(greeting.toUpperCase())\n"
)

# Degenerate: the same line repeated many times (high duplicate-line fraction).
_REPETITIVE = "console.log('spam')\n" * 40


class TestRepetitionScorerBasics:
    def test_is_a_scorer(self) -> None:
        assert isinstance(RepetitionScorer(), ScorerProtocol)

    def test_is_available(self) -> None:
        assert RepetitionScorer.is_available() is True

    def test_name(self) -> None:
        assert RepetitionScorer().name == "repetition"

    def test_returns_scorer_result(self) -> None:
        result = RepetitionScorer().score(_CLEAN)
        assert isinstance(result, ScorerResult)
        assert result.scorer_name == "repetition"
        assert 0.0 <= result.score <= 1.0


class TestScoreOrdering:
    def test_clean_scores_higher_than_repetitive(self) -> None:
        scorer = RepetitionScorer()
        clean = scorer.score(_CLEAN).score
        repetitive = scorer.score(_REPETITIVE).score
        assert clean > repetitive

    def test_clean_scores_high(self) -> None:
        # Non-repetitive code should land comfortably in the upper range.
        assert RepetitionScorer().score(_CLEAN).score >= 0.6

    def test_degenerate_repetition_scores_zero(self) -> None:
        # 40x the same line blows past the dup_line_frac threshold (ratio >= 1).
        assert RepetitionScorer().score(_REPETITIVE).score == 0.0

    def test_details_name_dominant_metric_on_repetition(self) -> None:
        details = RepetitionScorer().score(_REPETITIVE).details
        assert details["dominant_metric"] != "none"
        assert isinstance(details["max_ratio"], float)
        assert details["max_ratio"] >= 1.0


class TestEdgeCases:
    def test_empty_is_clean(self) -> None:
        # No text → no repetition evidence → max score (other scorers handle empties).
        result = RepetitionScorer().score("")
        assert result.score == 1.0
        assert result.details["reason"] == "empty"

    def test_whitespace_only_is_clean(self) -> None:
        assert RepetitionScorer().score("   \n\t  \n").score == 1.0

    def test_short_doc_skips_ngram_metrics(self) -> None:
        # A short snippet (< min_words) must not be n-gram-penalised.
        short = "const x = compute(a, b, c)\n"
        details = RepetitionScorer().score(short).details
        assert details["ngrams_evaluated"] is False

    def test_long_doc_evaluates_ngrams(self) -> None:
        details = RepetitionScorer().score(_REPETITIVE).details
        assert details["ngrams_evaluated"] is True

    def test_score_batch_matches_score(self) -> None:
        scorer = RepetitionScorer()
        items: list[tuple[str, dict[str, object] | None]] = [(_CLEAN, None), (_REPETITIVE, None)]
        batch = scorer.score_batch(items)
        assert [r.score for r in batch] == [scorer.score(_CLEAN).score, scorer.score(_REPETITIVE).score]


class TestRegistryWiring:
    def test_instantiates_via_registry(self) -> None:
        from cola_coder.data.scorers.registry import _instantiate_scorer
        from cola_coder.data.scorers.sandbox import SandboxedRunner
        from cola_coder.data.scorers.security import SecurityConfig

        runner = SandboxedRunner.from_config(SecurityConfig.from_dict({}))
        scorer = _instantiate_scorer("repetition", {"min_words": 30}, runner, None)
        assert scorer is not None
        assert scorer.name == "repetition"
        assert scorer.is_available() is True
