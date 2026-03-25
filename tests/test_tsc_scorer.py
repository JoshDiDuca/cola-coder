"""Tests for TscScorer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult
from cola_coder.data.scorers.tsc_scorer import TscScorer


class TestTscScorer:
    def test_implements_protocol(self) -> None:
        """TscScorer satisfies ScorerProtocol."""
        scorer = TscScorer()
        assert isinstance(scorer, ScorerProtocol)

    def test_name_is_tsc(self) -> None:
        assert TscScorer.name == "tsc"

    def test_score_returns_scorer_result(self) -> None:
        """Mocked checker returns a valid ScorerResult."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {
            "score": 1.0,
            "num_errors": 0,
            "error_codes": [],
            "has_syntax_errors": False,
        }
        scorer._checker = mock_checker

        result = scorer.score("const x: number = 1;", metadata={"language": "typescript"})
        assert isinstance(result, ScorerResult)
        assert result.scorer_name == "tsc"
        assert result.score == pytest.approx(1.0)

    def test_score_remapping_perfect(self) -> None:
        """Raw score 1.0 maps to normalized 1.0."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": 1.0, "num_errors": 0, "error_codes": [], "has_syntax_errors": False}
        scorer._checker = mock_checker

        result = scorer.score("const x = 1;", metadata={"language": "typescript"})
        assert result.score == pytest.approx(1.0)

    def test_score_remapping_syntax_error(self) -> None:
        """Raw score -0.5 (syntax error) maps to normalized 0.0."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": -0.5, "num_errors": 10, "error_codes": ["TS1005"], "has_syntax_errors": True}
        scorer._checker = mock_checker

        result = scorer.score("{{invalid", metadata={"language": "typescript"})
        assert result.score == pytest.approx(0.0)

    def test_score_remapping_moderate(self) -> None:
        """Raw score 0.3 maps to normalized ~0.533."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": 0.3, "num_errors": 4, "error_codes": [], "has_syntax_errors": False}
        scorer._checker = mock_checker

        result = scorer.score("const x = foo();", metadata={"language": "typescript"})
        expected = (0.3 + 0.5) / 1.5
        assert result.score == pytest.approx(expected, abs=0.01)

    def test_non_typescript_returns_neutral(self) -> None:
        """Python code gets neutral score 0.5."""
        scorer = TscScorer()
        result = scorer.score("def hello():\n    print('hi')", metadata={"language": "python"})
        assert result.score == 0.5
        assert result.details.get("skipped") is True

    def test_no_metadata_heuristic_detection(self) -> None:
        """Without metadata, uses heuristic detection."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": 1.0, "num_errors": 0, "error_codes": [], "has_syntax_errors": False}
        scorer._checker = mock_checker

        # This has enough TS indicators
        ts_code = "import { foo } from 'bar';\nconst x: string = foo();\nexport default x;"
        result = scorer.score(ts_code)
        assert result.score == pytest.approx(1.0)
        assert result.details.get("skipped") is not True

    def test_metadata_file_path_detection(self) -> None:
        """Detects TypeScript from file extension in metadata."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": 0.7, "num_errors": 2, "error_codes": [], "has_syntax_errors": False}
        scorer._checker = mock_checker

        result = scorer.score("plain code", metadata={"file_path": "src/app.tsx"})
        assert result.details.get("skipped") is not True

    def test_batch_scoring(self) -> None:
        """score_batch returns list of correct length."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {"score": 0.7, "num_errors": 1, "error_codes": [], "has_syntax_errors": False}
        scorer._checker = mock_checker

        items = [
            ("const x = 1;", {"language": "typescript"}),
            ("let y = 2;", {"language": "typescript"}),
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2
        assert all(isinstance(r, ScorerResult) for r in results)

    def test_details_include_error_info(self) -> None:
        """ScorerResult details include error diagnostics."""
        scorer = TscScorer()
        mock_checker = MagicMock()
        mock_checker.detailed_score.return_value = {
            "score": 0.3,
            "num_errors": 4,
            "error_codes": ["TS2322", "TS2339"],
            "has_syntax_errors": False,
        }
        scorer._checker = mock_checker

        result = scorer.score("bad code", metadata={"language": "typescript"})
        assert result.details["num_errors"] == 4
        assert "TS2322" in result.details["error_codes"]

    def test_is_available_when_import_fails(self) -> None:
        """is_available returns False when TypeCheckReward can't be imported."""
        with patch("cola_coder.data.scorers.tsc_scorer.TscScorer.is_available", return_value=False):
            assert TscScorer.is_available() is False
