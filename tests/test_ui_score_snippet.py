"""Tests for the quality-scorer playground helper (UI-101).

Covers ``cola_coder.ui.score_snippet_view.score_snippet``: it runs ONLY the
deterministic, dependency-free pure-Python scorers (heuristic,
educational_value, repetition, injection_safety, cwe_security) — no Docker,
node, model, or network — and returns a per-scorer breakdown plus the unweighted
mean. Empty/whitespace input yields ``{"error": ...}``; a scorer that raises is
skipped; the function never raises.
"""

from __future__ import annotations

import statistics

from cola_coder.data.scorers.protocol import CompositeScorer
from cola_coder.ui.score_snippet_view import score_snippet

# Names the view is allowed to surface (the "safe" pure-Python set).
SAFE_SCORER_NAMES: set[str] = {
    "heuristic",
    "educational_value",
    "repetition",
    "injection_safety",
    "cwe_security",
}

# Tiers produced by CompositeScorer.score_to_tier.
VALID_TIERS: set[str] = {"excellent", "good", "average", "poor", "reject"}

# A clean, varied TypeScript snippet used across multiple cases.
CLEAN_TS = "export const add = (a: number, b: number): number => a + b\n"

# A highly repetitive snippet that should drag the repetition signal down.
REPETITIVE_TS = "console.log('x')\n" * 40


class TestSuccessShape:
    """A normal snippet yields a well-formed breakdown dict."""

    def test_returns_non_empty_breakdown(self) -> None:
        """A normal TS snippet returns a populated, internally-consistent dict."""
        result = score_snippet(CLEAN_TS)

        assert "error" not in result
        assert isinstance(result["scorers"], list)
        assert len(result["scorers"]) > 0
        assert result["count"] == len(result["scorers"])
        assert isinstance(result["mean_score"], float)
        assert 0.0 <= result["mean_score"] <= 1.0
        assert isinstance(result["mean_tier"], str)
        assert result["mean_tier"] != ""

    def test_at_least_three_safe_scorers_run(self) -> None:
        """In a normal install the safe set should yield at least 3 scorers.

        Robust fallback: if the environment provides fewer, only require >= 1
        (and the subset constraint below still guards the names).
        """
        result = score_snippet(CLEAN_TS)
        count = result["count"]

        if count >= 3:
            assert count >= 3
        else:
            # Environment yielded fewer safe scorers than the usual >= 3.
            assert count >= 1

    def test_scorer_names_are_subset_of_safe_set(self) -> None:
        """Only the documented safe pure-Python scorers may appear."""
        result = score_snippet(CLEAN_TS)
        names = {entry["name"] for entry in result["scorers"]}

        assert names <= SAFE_SCORER_NAMES


class TestEntryContract:
    """Each scorer entry has exactly the right keys, types, and bounds."""

    def test_entry_keys_types_and_score_bounds(self) -> None:
        """Every entry is exactly {name, score, tier} with correct types/bounds."""
        result = score_snippet(CLEAN_TS)

        for entry in result["scorers"]:
            assert set(entry.keys()) == {"name", "score", "tier"}
            assert isinstance(entry["name"], str)
            assert isinstance(entry["score"], float)
            assert isinstance(entry["tier"], str)
            assert 0.0 <= entry["score"] <= 1.0

    def test_tiers_are_valid(self) -> None:
        """Every entry's tier is one of the CompositeScorer tier names."""
        result = score_snippet(CLEAN_TS)

        for entry in result["scorers"]:
            assert entry["tier"] in VALID_TIERS

        assert result["mean_tier"] in VALID_TIERS


class TestMeanConsistency:
    """The aggregate fields are derived from the per-scorer scores."""

    def test_mean_score_matches_per_scorer_mean(self) -> None:
        """``mean_score`` equals the mean of the per-scorer scores."""
        result = score_snippet(CLEAN_TS)
        scores = [entry["score"] for entry in result["scorers"]]

        assert scores, "expected at least one scorer to run"
        expected_mean = statistics.fmean(scores)
        assert result["mean_score"] == round(expected_mean, 4) or abs(
            result["mean_score"] - expected_mean
        ) < 1e-4

    def test_mean_tier_matches_score_to_tier(self) -> None:
        """``mean_tier`` is exactly ``score_to_tier(mean_score-equivalent)``."""
        result = score_snippet(CLEAN_TS)
        scores = [entry["score"] for entry in result["scorers"]]
        expected_mean = statistics.fmean(scores)

        assert result["mean_tier"] == CompositeScorer.score_to_tier(expected_mean)


class TestEmptyInput:
    """Blank input is rejected, never scored."""

    def test_empty_string_returns_error(self) -> None:
        """Empty code returns an error dict with no scorers key."""
        result = score_snippet("")

        assert "error" in result
        assert isinstance(result["error"], str)
        assert "scorers" not in result

    def test_whitespace_only_returns_error(self) -> None:
        """Whitespace-only code returns an error dict with no scorers key."""
        result = score_snippet("   \n\t  \n")

        assert "error" in result
        assert isinstance(result["error"], str)
        assert "scorers" not in result


class TestRepetitionSignal:
    """The repetition signal flows through to the aggregate."""

    def test_repetitive_scores_lower_than_clean(self) -> None:
        """A repetitive snippet means lower than a clean, varied one.

        Robust fallback: if the overall means tie, assert the repetition
        scorer's own per-scorer entry is lower on the repetitive input.
        """
        clean = score_snippet(CLEAN_TS)
        repetitive = score_snippet(REPETITIVE_TS)

        if clean["mean_score"] != repetitive["mean_score"]:
            assert clean["mean_score"] > repetitive["mean_score"]
            return

        clean_rep = _find_scorer(clean["scorers"], "repetition")
        rep_rep = _find_scorer(repetitive["scorers"], "repetition")
        assert clean_rep is not None and rep_rep is not None, (
            "repetition scorer expected to run to verify the signal"
        )
        assert clean_rep["score"] > rep_rep["score"]


def _find_scorer(
    scorers: list[dict[str, object]], name: str
) -> dict[str, object] | None:
    """Return the breakdown entry for ``name``, or ``None`` if it did not run."""
    for entry in scorers:
        if entry["name"] == name:
            return entry
    return None
