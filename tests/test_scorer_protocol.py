"""Tests for the scorer protocol, ScorerResult, and CompositeScorer."""

from __future__ import annotations

import pytest

from cola_coder.data.scorers.protocol import (
    CompositeResult,
    CompositeScorer,
    ScorerProtocol,
    ScorerResult,
)


# -- Mock scorer for testing ---------------------------------------------------


class MockScorer:
    """A test scorer that returns a fixed score."""

    name: str

    def __init__(self, name: str, fixed_score: float) -> None:
        self.name = name
        self._score = fixed_score

    def score(self, code: str, metadata: dict | None = None) -> ScorerResult:
        return ScorerResult(score=self._score, scorer_name=self.name)

    def score_batch(self, items: list[tuple[str, dict | None]]) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True


# -- ScorerResult tests --------------------------------------------------------


class TestScorerResult:
    def test_construction(self) -> None:
        r = ScorerResult(score=0.85, scorer_name="test")
        assert r.score == 0.85
        assert r.scorer_name == "test"
        assert r.details == {}

    def test_with_details(self) -> None:
        r = ScorerResult(score=0.5, scorer_name="tsc", details={"errors": 3})
        assert r.details["errors"] == 3

    def test_score_clamping_not_enforced(self) -> None:
        """ScorerResult doesn't enforce 0-1 range (scorers do)."""
        r = ScorerResult(score=1.5, scorer_name="test")
        assert r.score == 1.5


# -- Protocol conformance -----------------------------------------------------


class TestProtocolConformance:
    def test_mock_scorer_is_protocol(self) -> None:
        scorer = MockScorer("test", 0.5)
        assert isinstance(scorer, ScorerProtocol)

    def test_missing_method_fails_protocol(self) -> None:
        class BadScorer:
            name = "bad"
            def score(self, code, metadata=None):
                pass
            # Missing score_batch and is_available
        assert not isinstance(BadScorer(), ScorerProtocol)


# -- CompositeScorer tests ----------------------------------------------------


class TestCompositeScorer:
    def test_single_scorer(self) -> None:
        scorer = MockScorer("a", 0.8)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.overall == pytest.approx(0.8)
        assert "a" in result.per_scorer

    def test_two_scorers_weighted(self) -> None:
        s1 = MockScorer("a", 1.0)
        s2 = MockScorer("b", 0.0)
        composite = CompositeScorer([(s1, 0.5), (s2, 0.5)])
        result = composite.score("code")
        assert result.overall == pytest.approx(0.5)

    def test_weights_normalized(self) -> None:
        s1 = MockScorer("a", 1.0)
        s2 = MockScorer("b", 0.0)
        # Weights 3:1, should normalize to 0.75:0.25
        composite = CompositeScorer([(s1, 3.0), (s2, 1.0)])
        result = composite.score("code")
        assert result.overall == pytest.approx(0.75)

    def test_tier_mapping_excellent(self) -> None:
        scorer = MockScorer("a", 0.9)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.weight == 2.0  # excellent tier

    def test_tier_mapping_good(self) -> None:
        scorer = MockScorer("a", 0.7)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.weight == 1.5  # good tier

    def test_tier_mapping_average(self) -> None:
        scorer = MockScorer("a", 0.5)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.weight == 1.0  # average tier

    def test_tier_mapping_poor(self) -> None:
        scorer = MockScorer("a", 0.3)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.weight == 0.3  # poor tier

    def test_tier_mapping_reject(self) -> None:
        scorer = MockScorer("a", 0.1)
        composite = CompositeScorer([(scorer, 1.0)])
        result = composite.score("code")
        assert result.weight == 0.0  # reject tier

    def test_custom_tier_weights(self) -> None:
        scorer = MockScorer("a", 0.9)
        custom = {"excellent": 3.0, "good": 2.0, "average": 1.0, "poor": 0.5, "reject": 0.1}
        composite = CompositeScorer([(scorer, 1.0)], tier_weights=custom)
        result = composite.score("code")
        assert result.weight == 3.0

    def test_batch_scoring(self) -> None:
        s1 = MockScorer("a", 0.8)
        s2 = MockScorer("b", 0.6)
        composite = CompositeScorer([(s1, 0.5), (s2, 0.5)])
        items = [("code1", None), ("code2", None), ("code3", None)]
        results = composite.score_batch(items)
        assert len(results) == 3
        for r in results:
            assert r.overall == pytest.approx(0.7)

    def test_empty_scorers(self) -> None:
        composite = CompositeScorer([])
        result = composite.score("code")
        assert result.overall == 0.0

    def test_score_to_tier_static(self) -> None:
        assert CompositeScorer.score_to_tier(0.9) == "excellent"
        assert CompositeScorer.score_to_tier(0.7) == "good"
        assert CompositeScorer.score_to_tier(0.5) == "average"
        assert CompositeScorer.score_to_tier(0.3) == "poor"
        assert CompositeScorer.score_to_tier(0.1) == "reject"


# -- CompositeResult tests ----------------------------------------------------


class TestCompositeResult:
    def test_construction(self) -> None:
        r = CompositeResult(overall=0.75, per_scorer={}, weight=1.5)
        assert r.overall == 0.75
        assert r.weight == 1.5


# -- DATA-031: score_batch must equal score(), even with name collisions -------


class TestScoreBatchEqualsScore:
    def test_batch_matches_single_for_each_item(self) -> None:
        s1 = MockScorer("a", 0.9)
        s2 = MockScorer("b", 0.3)
        composite = CompositeScorer([(s1, 0.5), (s2, 0.5)])
        items = [("code1", None), ("code2", {"language": "python"})]
        batch = composite.score_batch(items)
        singles = [composite.score(c, m) for c, m in items]
        for b, s in zip(batch, singles):
            assert b.overall == pytest.approx(s.overall)
            assert b.weight == s.weight

    def test_colliding_scorer_names_dont_corrupt_batch(self) -> None:
        # Two scorers sharing a name: keying batch results by name (the old bug)
        # made both read the second's score. With position keying, each scorer's
        # distinct score is used and score_batch matches score().
        s1 = MockScorer("dup", 0.0)
        s2 = MockScorer("dup", 1.0)
        composite = CompositeScorer([(s1, 0.5), (s2, 0.5)])
        single = composite.score("x")
        batch = composite.score_batch([("x", None)])
        assert single.overall == pytest.approx(0.5)  # 0.5*0 + 0.5*1
        assert batch[0].overall == pytest.approx(single.overall)
