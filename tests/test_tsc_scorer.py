"""Tests for TscScorer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.data.scorers.protocol import ScorerProtocol, ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.tsc_scorer import TscScorer


def _make_runner(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    """Create a mock SandboxedRunner that returns fixed output."""
    runner = MagicMock(spec=SandboxedRunner)
    runner.run.return_value = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )
    return runner


class TestTscScorer:
    def test_implements_protocol(self) -> None:
        """TscScorer satisfies ScorerProtocol."""
        scorer = TscScorer()
        assert isinstance(scorer, ScorerProtocol)

    def test_name_is_tsc(self) -> None:
        assert TscScorer.name == "tsc"

    def test_score_returns_scorer_result(self) -> None:
        """No errors returns perfect ScorerResult."""
        runner = _make_runner(stdout="", stderr="")
        scorer = TscScorer(runner=runner)

        result = scorer.score("const x: number = 1;", metadata={"language": "typescript"})
        assert isinstance(result, ScorerResult)
        assert result.scorer_name == "tsc"
        assert result.score == pytest.approx(1.0)

    def test_score_perfect_no_errors(self) -> None:
        """Zero errors maps to score 1.0."""
        runner = _make_runner()
        scorer = TscScorer(runner=runner)

        result = scorer.score("const x = 1;", metadata={"language": "typescript"})
        assert result.score == pytest.approx(1.0)

    def test_score_one_error(self) -> None:
        """One error maps to score 0.8."""
        tsc_output = "check.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.\n"
        runner = _make_runner(stdout=tsc_output, returncode=1)
        scorer = TscScorer(runner=runner)

        result = scorer.score("const x: number = 'hi';", metadata={"language": "typescript"})
        assert result.score == pytest.approx(0.8)
        assert result.details["num_errors"] == 1

    def test_score_syntax_error_penalty(self) -> None:
        """Syntax errors (TS1XXX) get capped at 0.3."""
        tsc_output = "check.ts(1,1): error TS1005: ';' expected.\n"
        runner = _make_runner(stdout=tsc_output, returncode=1)
        scorer = TscScorer(runner=runner)

        result = scorer.score("{{invalid", metadata={"language": "typescript"})
        assert result.score <= 0.3
        assert result.details["has_syntax_errors"] is True

    def test_score_many_errors(self) -> None:
        """11+ errors maps to 0.1."""
        errors = "\n".join(
            f"check.ts({i},1): error TS2322: msg{i}" for i in range(12)
        )
        runner = _make_runner(stdout=errors, returncode=1)
        scorer = TscScorer(runner=runner)

        result = scorer.score("bad code", metadata={"language": "typescript"})
        assert result.score == pytest.approx(0.1)

    def test_non_typescript_returns_neutral(self) -> None:
        """Python code gets neutral score 0.5."""
        scorer = TscScorer()
        result = scorer.score("def hello():\n    print('hi')", metadata={"language": "python"})
        assert result.score == 0.5
        assert result.details.get("skipped") is True

    def test_no_metadata_heuristic_detection(self) -> None:
        """Without metadata, uses heuristic detection."""
        runner = _make_runner()
        scorer = TscScorer(runner=runner)

        # This has enough TS indicators (: string, : number = 2 matches)
        ts_code = "const x: string = 'foo';\nconst y: number = 1;"
        result = scorer.score(ts_code)
        assert result.score == pytest.approx(1.0)
        assert result.details.get("skipped") is not True

    def test_metadata_file_path_detection(self) -> None:
        """Detects TypeScript from file extension in metadata."""
        runner = _make_runner()
        scorer = TscScorer(runner=runner)

        result = scorer.score("plain code", metadata={"file_path": "src/app.tsx"})
        assert result.details.get("skipped") is not True

    def test_batch_scoring(self) -> None:
        """score_batch returns list of correct length."""
        runner = _make_runner()
        scorer = TscScorer(runner=runner)

        items: list[tuple[str, dict[str, object] | None]] = [
            ("const x: number = 1;", {"language": "typescript"}),
            ("const y: string = 'hello';", {"language": "typescript"}),
        ]
        results = scorer.score_batch(items)
        assert len(results) == 2
        assert all(isinstance(r, ScorerResult) for r in results)

    def test_details_include_error_info(self) -> None:
        """ScorerResult details include error diagnostics."""
        tsc_output = (
            "check.ts(1,7): error TS2322: Type 'string' is not assignable.\n"
            "check.ts(2,3): error TS2339: Property 'foo' does not exist.\n"
        )
        runner = _make_runner(stdout=tsc_output, returncode=1)
        scorer = TscScorer(runner=runner)

        result = scorer.score("bad code", metadata={"language": "typescript"})
        assert result.details["num_errors"] == 2

    def test_is_available_checks_shutil_which(self) -> None:
        """is_available checks for tsc via shutil.which."""
        with patch("shutil.which", return_value=None):
            assert TscScorer.is_available() is False
        with patch("shutil.which", return_value="/usr/local/bin/tsc"):
            assert TscScorer.is_available() is True

    def test_cache_hit_avoids_runner_call(self) -> None:
        """Second call with same code hits cache, no runner call."""
        runner = _make_runner()
        scorer = TscScorer(runner=runner)

        code = "const x: number = 1;"
        meta: dict[str, object] = {"language": "typescript"}
        scorer.score(code, metadata=meta)
        scorer.score(code, metadata=meta)

        # Runner should only be called once (cache hit on second call)
        assert runner.run.call_count == 1


class TestTscScorerSandboxEnforcement:
    """Verify TscScorer uses SandboxedRunner, not direct subprocess."""

    def test_score_calls_runner(self) -> None:
        """TscScorer MUST call runner.run(), not subprocess.run."""
        runner = MagicMock(spec=SandboxedRunner)
        runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        scorer = TscScorer(runner=runner)
        scorer.score("const x: number = 1;", metadata={"language": "typescript"})
        runner.run.assert_called_once()

    def test_does_not_import_type_check_reward(self) -> None:
        """TscScorer should not import TypeCheckReward directly."""
        import cola_coder.data.scorers.tsc_scorer as mod

        source = Path(mod.__file__).read_text()
        assert "TypeCheckReward" not in source
        # TscRunner import IS allowed (it's the secure sandboxed path)
        # type_check.py import is NOT allowed (it bypasses sandbox)
        assert "from cola_coder.reasoning.rewards.type_check" not in source

    def test_writes_hardened_tsconfig(self, tmp_path: Path) -> None:
        """TscScorer writes tsconfig.json with plugins=[]."""
        import json

        written_files: list[str] = []

        def capture_run(cmd: list[str], cwd: str | Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
            # Read what was written to the temp dir
            cwd_path = Path(cwd)
            tsconfig_path = cwd_path / "tsconfig.json"
            if tsconfig_path.exists():
                tsconfig = json.loads(tsconfig_path.read_text())
                assert tsconfig["compilerOptions"]["plugins"] == []
                assert tsconfig["compilerOptions"]["types"] == []
                assert tsconfig["compilerOptions"]["typeRoots"] == []
                written_files.append("tsconfig.json")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        runner = MagicMock(spec=SandboxedRunner)
        runner.run.side_effect = capture_run
        scorer = TscScorer(runner=runner)
        scorer.score("const x: number = 1;", metadata={"language": "typescript"})
        assert "tsconfig.json" in written_files

    def test_batch_uses_runner(self) -> None:
        """score_batch MUST call runner.run()."""
        runner = MagicMock(spec=SandboxedRunner)
        runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        scorer = TscScorer(runner=runner)
        items: list[tuple[str, dict[str, object] | None]] = [
            ("const x: number = 1;", {"language": "typescript"}),
            ("const y: string = 'hello';", {"language": "typescript"}),
        ]
        scorer.score_batch(items)
        runner.run.assert_called_once()  # Single invocation for batch
