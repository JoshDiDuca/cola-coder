"""Tests for StarsScorer."""

from __future__ import annotations

import pytest

from cola_coder.data.scorers.protocol import ScorerProtocol
from cola_coder.data.scorers.stars_scorer import StarsScorer


class TestStarsScorer:
    def test_implements_protocol(self) -> None:
        scorer = StarsScorer()
        assert isinstance(scorer, ScorerProtocol)

    def test_name_is_stars(self) -> None:
        assert StarsScorer.name == "stars"

    def test_always_available(self) -> None:
        assert StarsScorer.is_available() is True

    # -- Star normalization tests ------------------------------------------

    def test_zero_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 0})
        assert result.score == pytest.approx(0.1)

    def test_negative_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": -5})
        assert result.score == pytest.approx(0.1)

    def test_one_star(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 1})
        assert result.score == pytest.approx(0.1, abs=0.05)

    def test_ten_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 10})
        assert result.score == pytest.approx(0.3, abs=0.01)

    def test_hundred_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 100})
        assert result.score == pytest.approx(0.5, abs=0.01)

    def test_thousand_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 1000})
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_ten_thousand_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 10000})
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_hundred_thousand_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 100000})
        assert result.score == pytest.approx(1.0)

    def test_monotonically_increasing(self) -> None:
        """More stars should always produce equal or higher score."""
        scorer = StarsScorer()
        star_counts = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
        scores = [
            scorer.score("code", {"repo_stars": s}).score
            for s in star_counts
        ]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Score for {star_counts[i]} stars ({scores[i]}) < "
                f"score for {star_counts[i-1]} stars ({scores[i-1]})"
            )

    # -- Missing/invalid metadata tests ------------------------------------

    def test_no_metadata_returns_default(self) -> None:
        scorer = StarsScorer(default_score=0.3)
        result = scorer.score("code", None)
        assert result.score == 0.3
        assert result.details["source"] == "default"

    def test_metadata_without_stars_returns_default(self) -> None:
        scorer = StarsScorer(default_score=0.4)
        result = scorer.score("code", {"language": "typescript"})
        assert result.score == 0.4

    def test_invalid_stars_value_returns_default(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": "not_a_number"})
        assert result.score == 0.3
        assert result.details.get("parse_error") is True

    def test_custom_default_score(self) -> None:
        scorer = StarsScorer(default_score=0.5)
        result = scorer.score("code", None)
        assert result.score == 0.5

    def test_default_score_clamped(self) -> None:
        scorer = StarsScorer(default_score=1.5)
        result = scorer.score("code", None)
        assert result.score == 1.0

    # -- Batch scoring tests -----------------------------------------------

    def test_batch_scoring(self) -> None:
        scorer = StarsScorer()
        items = [
            ("code1", {"repo_stars": 100}),
            ("code2", {"repo_stars": 1000}),
            ("code3", None),
        ]
        results = scorer.score_batch(items)
        assert len(results) == 3
        assert results[0].score == pytest.approx(0.5, abs=0.01)
        assert results[1].score == pytest.approx(0.8, abs=0.01)
        assert results[2].score == pytest.approx(0.3)  # default

    def test_details_include_star_count(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 42})
        assert result.details["stars"] == 42
        assert result.details["source"] == "metadata"

    def test_empty_batch(self) -> None:
        scorer = StarsScorer()
        results = scorer.score_batch([])
        assert results == []

    # -- Intermediate star values ------------------------------------------

    def test_fifty_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 50})
        assert 0.3 < result.score < 0.5  # Between 10-star and 100-star

    def test_five_hundred_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 500})
        assert 0.5 < result.score < 0.8  # Between 100-star and 1000-star

    def test_five_thousand_stars(self) -> None:
        scorer = StarsScorer()
        result = scorer.score("code", {"repo_stars": 5000})
        assert 0.8 < result.score < 1.0  # Between 1000-star and 10000-star
