"""Tests for EslintScorer."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from cola_coder.data.scorers.eslint_scorer import EslintScorer
from cola_coder.data.scorers.language_detect import detect_extension
from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult


def _mock_eslint_output(files: list[dict]) -> str:
    """Create mock eslint JSON output."""
    return json.dumps(files)


class TestEslintScorer:
    def test_implements_protocol(self) -> None:
        scorer = EslintScorer()
        assert isinstance(scorer, ScorerProtocol)

    def test_name_is_eslint(self) -> None:
        assert EslintScorer.name == "eslint"

    def test_perfect_score_no_issues(self) -> None:
        """0 errors + 0 warnings = score 1.0."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=_mock_eslint_output([{"filePath": "file.ts", "errorCount": 0, "warningCount": 0}]),
        )
        scorer._runner = mock_runner

        result = scorer.score("const x = 1;", metadata={"language": "typescript"})
        assert result.score == 1.0
        assert result.details["error_count"] == 0

    def test_few_warnings_high_score(self) -> None:
        """1 warning = score 0.9."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout=_mock_eslint_output([{"filePath": "f.ts", "errorCount": 0, "warningCount": 1}]),
        )
        scorer._runner = mock_runner

        result = scorer.score("let x;", metadata={"language": "typescript"})
        assert result.score == 0.9

    def test_many_errors_low_score(self) -> None:
        """15 errors = score 0.3."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout=_mock_eslint_output([{"filePath": "f.ts", "errorCount": 15, "warningCount": 0}]),
        )
        scorer._runner = mock_runner

        result = scorer.score("bad code", metadata={"language": "typescript"})
        assert result.score == 0.3

    def test_extreme_errors_minimum_score(self) -> None:
        """25+ errors = score 0.1."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout=_mock_eslint_output([{"filePath": "f.ts", "errorCount": 25, "warningCount": 5}]),
        )
        scorer._runner = mock_runner

        result = scorer.score("terrible", metadata={"language": "typescript"})
        assert result.score == 0.1

    def test_non_js_ts_returns_neutral(self) -> None:
        """Python code gets neutral score 0.5."""
        scorer = EslintScorer()
        result = scorer.score("def hello():\n    pass", metadata={"language": "python"})
        assert result.score == 0.5
        assert result.details.get("skipped") is True

    def test_eslint_failure_returns_neutral(self) -> None:
        """Failed eslint run returns neutral score."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=-2, stdout="", stderr="eslint not found",
        )
        scorer._runner = mock_runner

        result = scorer.score("const x = 1;", metadata={"language": "typescript"})
        assert result.score == 0.5
        assert result.details.get("skipped") is True

    def test_batch_scoring(self) -> None:
        """score_batch handles multiple items."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        # Return results for both files
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=_mock_eslint_output([
                {"filePath": "whatever", "errorCount": 0, "warningCount": 0},
                {"filePath": "whatever2", "errorCount": 3, "warningCount": 0},
            ]),
        )
        scorer._runner = mock_runner

        items = [
            ("const x = 1;", {"language": "typescript"}),
            ("let y = 2;", {"language": "typescript"}),
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2

    def test_batch_mixed_languages(self) -> None:
        """Batch with mixed languages: TS scored, Python skipped."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=_mock_eslint_output([{"filePath": "x", "errorCount": 0, "warningCount": 0}]),
        )
        scorer._runner = mock_runner

        items = [
            ("const x = 1;", {"language": "typescript"}),
            ("def foo(): pass", {"language": "python"}),
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2
        assert results[1].details.get("skipped") is True

    def test_invalid_json_returns_neutral(self) -> None:
        """Invalid JSON from eslint returns neutral score."""
        scorer = EslintScorer()
        mock_runner = MagicMock()
        mock_runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json at all",
        )
        scorer._runner = mock_runner

        result = scorer.score("const x = 1;", metadata={"language": "typescript"})
        assert result.score == 0.5

    def test_issues_to_score_boundaries(self) -> None:
        """Test score mapping at exact boundaries."""
        assert EslintScorer._issues_to_score(0) == 1.0
        assert EslintScorer._issues_to_score(2) == 0.9
        assert EslintScorer._issues_to_score(5) == 0.7
        assert EslintScorer._issues_to_score(10) == 0.5
        assert EslintScorer._issues_to_score(20) == 0.3
        assert EslintScorer._issues_to_score(21) == 0.1
        assert EslintScorer._issues_to_score(100) == 0.1

    def test_detect_extension_from_metadata(self) -> None:
        """Extension detected from file_path metadata."""
        assert detect_extension({"file_path": "app.tsx"}) == ".tsx"
        assert detect_extension({"file_path": "index.js"}) == ".js"
        assert detect_extension({"language": "typescript"}) == ".ts"
        assert detect_extension(None) == ".ts"
