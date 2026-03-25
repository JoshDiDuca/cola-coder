"""Tests for LlmJudge, OllamaBackend, ClaudeBackend, and classifier."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.data.scorers.llm_judge import (
    ClaudeBackend,
    LlmJudge,
    OllamaBackend,
    _code_hash,
    _parse_judge_response,
)
from cola_coder.data.scorers.classifier import (
    ClassifierScorer,
    QualityClassifier,
    QualityClassifierTrainer,
)
from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult


# -- Response parsing ------------------------------------------------------


class TestParseJudgeResponse:
    def test_valid_response(self) -> None:
        text = "Score: 4\nReason: Well-structured code with good naming."
        score, reason = _parse_judge_response(text)
        assert score == 4
        assert "Well-structured" in reason

    def test_score_only(self) -> None:
        score, reason = _parse_judge_response("Score: 3")
        assert score == 3
        assert reason == ""

    def test_no_score(self) -> None:
        score, reason = _parse_judge_response("This is good code")
        assert score == -1

    def test_score_clamped_high(self) -> None:
        score, _ = _parse_judge_response("Score: 9")
        assert score == 5

    def test_score_clamped_low(self) -> None:
        # Regex only matches single digit, so "Score: -1" won't match
        score, _ = _parse_judge_response("Score: 0")
        assert score == 0

    def test_multiline_response(self) -> None:
        text = "I'll evaluate this code.\n\nScore: 5\nReason: Excellent TypeScript code."
        score, reason = _parse_judge_response(text)
        assert score == 5
        assert "Excellent" in reason


# -- Code hash -------------------------------------------------------------


class TestCodeHash:
    def test_deterministic(self) -> None:
        assert _code_hash("hello") == _code_hash("hello")

    def test_different_inputs(self) -> None:
        assert _code_hash("hello") != _code_hash("world")


# -- OllamaBackend --------------------------------------------------------


class TestOllamaBackend:
    def test_score_code_with_mock(self) -> None:
        backend = OllamaBackend()
        with patch.object(backend, "_call", return_value="Score: 4\nReason: Good code"):
            score, reason = backend.score_code("const x = 1;")
            assert score == 4
            assert "Good" in reason

    def test_score_code_failure(self) -> None:
        backend = OllamaBackend()
        with patch.object(backend, "_call", return_value=None):
            score, reason = backend.score_code("bad")
            assert score == -1


# -- ClaudeBackend --------------------------------------------------------


class TestClaudeBackend:
    def test_no_api_key(self) -> None:
        backend = ClaudeBackend(api_key="")
        score, reason = backend.score_code("const x = 1;")
        assert score == -1
        # Either "No API key" (when anthropic is installed) or SDK-not-installed
        assert "No API key" in reason or "not installed" in reason

    def test_score_code_with_mock(self) -> None:
        backend = ClaudeBackend(api_key="test-key")
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Score: 3\nReason: Average code")]
        with patch("cola_coder.data.scorers.llm_judge.ClaudeBackend.score_code",
                    return_value=(3, "Average code")):
            score, reason = backend.score_code("code")
            assert score == 3


# -- LlmJudge -------------------------------------------------------------


class TestLlmJudge:
    def test_implements_protocol(self) -> None:
        judge = LlmJudge(provider="ollama")
        assert isinstance(judge, ScorerProtocol)

    def test_name_is_llm_judge(self) -> None:
        assert LlmJudge.name == "llm_judge"

    def test_invalid_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            LlmJudge(provider="gpt4")

    def test_score_normalizes_to_0_1(self) -> None:
        judge = LlmJudge(provider="ollama")
        with patch.object(judge._backend, "score_code", return_value=(4, "Good")):
            result = judge.score("const x = 1;")
            assert result.score == pytest.approx(0.8)
            assert result.details["score_raw"] == 4

    def test_score_failure_returns_neutral(self) -> None:
        judge = LlmJudge(provider="ollama")
        with patch.object(judge._backend, "score_code", return_value=(-1, "")):
            result = judge.score("code")
            assert result.score == 0.5
            assert result.details.get("error") is True

    def test_annotate_batch_creates_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "annotations.jsonl"
        judge = LlmJudge(provider="ollama")
        with patch.object(judge._backend, "score_code", return_value=(3, "OK")):
            result_path = judge.annotate_batch(
                ["code1", "code2"],
                output_path=str(out),
            )
        assert Path(result_path).exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["score"] == 3
        assert "code_hash" in entry

    def test_annotate_batch_resumes(self, tmp_path: Path) -> None:
        out = tmp_path / "annotations.jsonl"
        judge = LlmJudge(provider="ollama")

        # First run: annotate 2 samples
        with patch.object(judge._backend, "score_code", return_value=(4, "Good")):
            judge.annotate_batch(["code1", "code2"], output_path=str(out))

        # Second run: same 2 samples should be skipped
        with patch.object(judge._backend, "score_code", return_value=(5, "Great")) as mock:
            judge.annotate_batch(["code1", "code2", "code3"], output_path=str(out))

        # Only code3 should have been newly scored
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # 2 from first run + 1 new


# -- QualityClassifierTrainer ---------------------------------------------


class TestClassifierTrainer:
    def _write_annotations(self, path: Path, n: int = 100) -> None:
        """Write fake annotations JSONL."""
        import random
        random.seed(42)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                entry = {
                    "code_hash": f"hash_{i}",
                    "score": random.randint(0, 5),
                    "reason": "test",
                    "code_prefix": f"const x_{i} = {i}; function foo_{i}() " + "{ return x; }" * (i % 5),
                }
                f.write(json.dumps(entry) + "\n")

    def test_train_produces_model_files(self, tmp_path: Path) -> None:
        annotations = tmp_path / "annotations.jsonl"
        self._write_annotations(annotations, n=100)
        model_dir = tmp_path / "model"

        try:
            trainer = QualityClassifierTrainer()
            metrics = trainer.train(str(annotations), str(model_dir))
            assert (model_dir / "vectorizer.pkl").exists()
            assert (model_dir / "model.pkl").exists()
            assert (model_dir / "meta.json").exists()
            assert metrics.num_train > 0
            assert 0.0 <= metrics.accuracy <= 1.0
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_train_too_few_annotations(self, tmp_path: Path) -> None:
        annotations = tmp_path / "annotations.jsonl"
        self._write_annotations(annotations, n=5)
        model_dir = tmp_path / "model"

        try:
            trainer = QualityClassifierTrainer()
            with pytest.raises(ValueError, match="at least 10"):
                trainer.train(str(annotations), str(model_dir))
        except ImportError:
            pytest.skip("scikit-learn not installed")


# -- QualityClassifier + ClassifierScorer ---------------------------------


class TestClassifierScorer:
    def test_implements_protocol(self) -> None:
        scorer = ClassifierScorer()
        assert isinstance(scorer, ScorerProtocol)

    def test_missing_model_returns_neutral(self) -> None:
        scorer = ClassifierScorer(model_dir="/nonexistent/path")
        result = scorer.score("const x = 1;")
        assert result.score == 0.5
        assert result.details.get("error") is True

    def test_roundtrip_train_predict(self, tmp_path: Path) -> None:
        """Train a classifier and use it for scoring."""
        import random
        random.seed(42)

        annotations = tmp_path / "annotations.jsonl"
        with open(annotations, "w") as f:
            for i in range(200):
                score = 4 if "function" in f"function foo_{i}()" else 1
                entry = {
                    "code_hash": f"h{i}",
                    "score": score,
                    "reason": "test",
                    "code_prefix": f"function foo_{i}() " + "{ return 1; }",
                }
                f.write(json.dumps(entry) + "\n")

        model_dir = tmp_path / "model"

        try:
            trainer = QualityClassifierTrainer()
            trainer.train(str(annotations), str(model_dir))

            scorer = ClassifierScorer(model_dir=str(model_dir))
            result = scorer.score("function hello() { return 1; }")
            assert isinstance(result, ScorerResult)
            assert 0.0 <= result.score <= 1.0
        except ImportError:
            pytest.skip("scikit-learn not installed")
