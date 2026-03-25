"""Security enforcement tests for the scoring pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cola_coder.data.scorers.protocol import CompositeScorer, ScorerResult
from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.tsc_scorer import TscScorer
from cola_coder.data.scorers.eslint_scorer import EslintScorer


class TestTscScorerEnforcement:
    """Verify TscScorer uses SandboxedRunner exclusively."""

    def test_score_calls_runner_run(self) -> None:
        """TscScorer MUST call runner.run(), not subprocess.run."""
        runner = MagicMock(spec=SandboxedRunner)
        runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        scorer = TscScorer(runner=runner)
        scorer.score("const x: number = 1;", metadata={"language": "typescript"})
        runner.run.assert_called_once()

    def test_batch_calls_runner_once(self) -> None:
        """score_batch runs tsc once through runner for multiple files."""
        runner = MagicMock(spec=SandboxedRunner)
        runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        scorer = TscScorer(runner=runner)
        items: list[tuple[str, dict[str, object] | None]] = [
            ("const a: number = 1;", {"language": "typescript"}),
            ("const b: string = 'hi';", {"language": "typescript"}),
        ]
        scorer.score_batch(items)
        runner.run.assert_called_once()

    def test_no_type_check_reward_import(self) -> None:
        """TscScorer source must not reference TypeCheckReward."""
        import cola_coder.data.scorers.tsc_scorer as mod
        source = Path(mod.__file__).read_text()
        assert "TypeCheckReward" not in source
        assert "from cola_coder.reasoning" not in source


class TestEslintScorerEnforcement:
    """Verify EslintScorer continues to use SandboxedRunner."""

    def test_score_calls_runner(self) -> None:
        runner = MagicMock(spec=SandboxedRunner)
        runner.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=json.dumps([{"filePath": "f.ts", "errorCount": 0, "warningCount": 0}]),
            stderr="",
        )
        scorer = EslintScorer(runner=runner)
        scorer.score("const x = 1;", metadata={"language": "typescript"})
        runner.run.assert_called()


class TestTsconfigHardening:
    """Verify tsconfig.json security measures."""

    def test_plugins_always_empty(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        assert config["compilerOptions"]["plugins"] == []

    def test_types_empty(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        assert config["compilerOptions"]["types"] == []

    def test_type_roots_empty(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        assert config["compilerOptions"]["typeRoots"] == []

    def test_no_paths_or_base_url(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        opts = config["compilerOptions"]
        assert "paths" not in opts
        assert "baseUrl" not in opts

    def test_no_emit(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        assert config["compilerOptions"]["noEmit"] is True

    def test_explicit_include(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig(include_files=["check.ts"])
        assert config["include"] == ["check.ts"]

    def test_excludes_node_modules(self) -> None:
        from cola_coder.data.scorers.tsconfig_factory import create_hardened_tsconfig
        config = create_hardened_tsconfig()
        assert "node_modules" in config["exclude"]

    def test_write_to_directory(self, tmp_path: Path) -> None:
        from cola_coder.data.scorers.tsconfig_factory import write_hardened_tsconfig
        path = write_hardened_tsconfig(tmp_path, include_files=["test.ts"])
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["compilerOptions"]["plugins"] == []


class TestDockerSecurityFlags:
    """Verify Docker container security flags."""

    def test_pids_limit(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode
        config = SecurityConfig(mode=SecurityMode.DOCKER)
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner.from_config(config)
            # Simulate a Docker run to check flags
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                runner.run(["echo", "test"], cwd="/tmp")
                call_args = mock_run.call_args[0][0]
                assert "--pids-limit" in call_args

    def test_cap_drop_all(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode
        config = SecurityConfig(mode=SecurityMode.DOCKER)
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner.from_config(config)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                runner.run(["echo", "test"], cwd="/tmp")
                call_args = mock_run.call_args[0][0]
                assert "--cap-drop" in call_args

    def test_no_new_privileges(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode
        config = SecurityConfig(mode=SecurityMode.DOCKER)
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner.from_config(config)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                runner.run(["echo", "test"], cwd="/tmp")
                call_args = mock_run.call_args[0][0]
                assert any("no-new-privileges" in str(a) for a in call_args)

    def test_runs_as_nobody(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode
        config = SecurityConfig(mode=SecurityMode.DOCKER)
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner.from_config(config)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
                runner.run(["echo", "test"], cwd="/tmp")
                call_args = mock_run.call_args[0][0]
                assert "--user" in call_args


class TestSecurityConfigLoading:
    """Verify SecurityConfig loads from YAML correctly."""

    def test_default_mode_is_native(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig
        config = SecurityConfig.from_dict({})
        assert config.mode.value == "native"

    def test_docker_mode_from_config(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig
        config = SecurityConfig.from_dict({"security": {"mode": "docker"}})
        assert config.mode.value == "docker"

    def test_backward_compat_sandbox_key(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig
        config = SecurityConfig.from_dict({"sandbox": {"use_docker": True}})
        assert config.mode.value == "docker"

    def test_credential_scan_mode(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig
        config = SecurityConfig.from_dict({"security": {"credential_scan": {"mode": "reject"}}})
        assert config.credential_scan_mode == "reject"

    def test_require_docker(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityError
        config = SecurityConfig.from_dict({"security": {"require_docker": True}})
        assert config.require_docker is True


class TestLlmJudgeCredentialScanning:
    """Verify LLM Judge scans credentials before API calls."""

    def test_score_strips_credentials(self) -> None:
        from cola_coder.data.scorers.llm_judge import LlmJudge
        from cola_coder.data.scorers.credential_scanner import CredentialScanner

        scanner = CredentialScanner(mode="strip")
        judge = LlmJudge(provider="ollama", credential_scanner=scanner)

        with patch.object(judge._backend, "score_code", return_value=(3, "OK")) as mock:
            result = judge.score('const key = "AKIAIOSFODNN7EXAMPLE1";')
            # Verify the code sent to backend has been redacted
            called_code = mock.call_args[0][0]
            assert "AKIA" not in called_code
            assert "[REDACTED]" in called_code

    def test_score_rejects_credentials(self) -> None:
        from cola_coder.data.scorers.llm_judge import LlmJudge
        from cola_coder.data.scorers.credential_scanner import CredentialScanner

        scanner = CredentialScanner(mode="reject")
        judge = LlmJudge(provider="ollama", credential_scanner=scanner)

        result = judge.score('const key = "AKIAIOSFODNN7EXAMPLE1";')
        assert result.score == 0.5
        assert result.details.get("skipped") is True
        assert result.details.get("reason") == "credential_detected"

    def test_score_passes_clean_code(self) -> None:
        from cola_coder.data.scorers.llm_judge import LlmJudge
        from cola_coder.data.scorers.credential_scanner import CredentialScanner

        scanner = CredentialScanner(mode="reject")
        judge = LlmJudge(provider="ollama", credential_scanner=scanner)

        with patch.object(judge._backend, "score_code", return_value=(4, "Good")):
            result = judge.score("const x: number = 42;")
            assert result.score == pytest.approx(0.8)  # 4/5 = 0.8

    def test_annotate_batch_skips_credential_samples(self, tmp_path: Path) -> None:
        from cola_coder.data.scorers.llm_judge import LlmJudge
        from cola_coder.data.scorers.credential_scanner import CredentialScanner

        out = tmp_path / "annotations.jsonl"
        scanner = CredentialScanner(mode="reject")
        judge = LlmJudge(provider="ollama", credential_scanner=scanner)

        with patch.object(judge._backend, "score_code", return_value=(3, "OK")):
            judge.annotate_batch(
                ['const key = "AKIAIOSFODNN7EXAMPLE1";', "const x = 42;"],
                output_path=str(out),
            )

        # Only clean code should be annotated
        lines = out.read_text().strip().split("\n") if out.exists() else []
        assert len(lines) == 1  # Only the clean sample
