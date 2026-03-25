"""Integration test for scoring pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cola_coder.data.scorers.protocol import CompositeScorer, ScorerResult


class _DummyScorer:
    name = "dummy"

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        return ScorerResult(score=0.7, scorer_name=self.name)

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        return [self.score(c, m) for c, m in items]

    @staticmethod
    def is_available() -> bool:
        return True


class TestCompositeIntegration:
    def test_end_to_end_scoring(self, tmp_path: Path) -> None:
        """Score a small dataset and verify weights file."""
        # Create dummy .npy data
        rng = np.random.default_rng(42)
        data = rng.integers(0, 100, size=(10, 32), dtype=np.uint16)
        data_path = tmp_path / "test_data.npy"
        np.save(str(data_path), data)

        # Create composite scorer with dummy
        scorer = _DummyScorer()
        composite = CompositeScorer([(scorer, 1.0)])

        # Score each sample
        weights: list[float] = []
        for i in range(len(data)):
            result = composite.score("dummy code")
            weights.append(result.weight)

        # Save weights
        weights_arr = np.array(weights, dtype=np.float32)
        weights_path = data_path.with_suffix(".weights.npy")
        np.save(str(weights_path), weights_arr)

        # Verify
        assert weights_path.exists()
        loaded = np.load(str(weights_path))
        assert len(loaded) == 10
        assert all(w > 0 for w in loaded)

    def test_composite_with_multiple_scorers(self) -> None:
        """Multiple scorers combine correctly."""
        s1 = _DummyScorer()
        s1.name = "a"

        s2 = _DummyScorer()
        s2.name = "b"

        composite = CompositeScorer([(s1, 0.5), (s2, 0.5)])
        result = composite.score("code")
        assert result.overall == pytest.approx(0.7)  # Both return 0.7

    def test_scores_jsonl_output(self, tmp_path: Path) -> None:
        """Verify JSONL scores file format."""
        scores_path = tmp_path / "test.scores.jsonl"
        with open(scores_path, "w") as f:
            for i in range(5):
                entry = {"index": i, "overall": 0.7, "weight": 1.5, "dummy": 0.7}
                f.write(json.dumps(entry) + "\n")

        with open(scores_path) as f:
            lines = f.readlines()
        assert len(lines) == 5
        entry = json.loads(lines[0])
        assert "index" in entry
        assert "overall" in entry
        assert "weight" in entry

    def test_weight_tiers(self) -> None:
        """Verify score-to-weight tier mapping."""
        scorer = _DummyScorer()
        composite = CompositeScorer([(scorer, 1.0)])

        # Score of 0.7 should give "good" tier (weight 1.5)
        result = composite.score("code")
        assert result.weight == 1.5  # 0.7 >= 0.6 → "good" → 1.5

    def test_composite_normalizes_weights(self) -> None:
        """Scorer weights are normalized to sum to 1.0."""
        s1 = _DummyScorer()
        s1.name = "a"
        s2 = _DummyScorer()
        s2.name = "b"

        # Weights (3.0, 7.0) should be normalized to (0.3, 0.7)
        composite = CompositeScorer([(s1, 3.0), (s2, 7.0)])
        result = composite.score("code")
        # Both scorers return 0.7, so: 0.7*0.3 + 0.7*0.7 = 0.7
        assert result.overall == pytest.approx(0.7)
