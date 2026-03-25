"""Smoke tests for scoring menu integration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


class TestScoringMenuSmoke:
    """Verify scoring-related menu options don't crash."""

    def test_list_available_scorers(self, tmp_path: Path) -> None:
        """list_available_scorers returns a list without crashing."""
        from cola_coder.data.scorers.registry import list_available_scorers
        result = list_available_scorers(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(result, list)

    def test_build_composite_scorer_empty_config(self, tmp_path: Path) -> None:
        """build_composite_scorer handles missing config gracefully."""
        from cola_coder.data.scorers.registry import build_composite_scorer
        scorer = build_composite_scorer(str(tmp_path / "nonexistent.yaml"))
        # Should return a CompositeScorer (possibly with 0 scorers)
        from cola_coder.data.scorers.protocol import CompositeScorer
        assert isinstance(scorer, CompositeScorer)

    def test_score_data_script_help(self) -> None:
        """score_data.py --help should not crash."""
        project_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["python", "scripts/score_data.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=str(project_root),
        )
        assert result.returncode == 0
        assert "score" in result.stdout.lower()

    def test_train_judge_script_help(self) -> None:
        """train_judge_classifier.py --help should not crash."""
        project_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["python", "scripts/train_judge_classifier.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=str(project_root),
        )
        assert result.returncode == 0

    def test_heuristic_scorer_available(self) -> None:
        """HeuristicScorer.is_available() returns True."""
        from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer
        assert HeuristicScorer.is_available() is True

    def test_heuristic_scorer_implements_protocol(self) -> None:
        """HeuristicScorer follows the ScorerProtocol interface."""
        from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer
        scorer = HeuristicScorer()
        assert hasattr(scorer, "name")
        assert hasattr(scorer, "score")
        assert hasattr(scorer, "score_batch")
        assert hasattr(scorer, "is_available")
        assert scorer.name == "heuristic"

    def test_heuristic_scorer_scores(self) -> None:
        """HeuristicScorer can score a code sample."""
        from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer
        scorer = HeuristicScorer()
        result = scorer.score("function add(a: number, b: number): number { return a + b; }")
        assert 0.0 <= result.score <= 1.0
        assert result.scorer_name == "heuristic"
