"""Tests for SandboxedRunner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cola_coder.data.scorers.sandbox import SandboxedRunner


class TestSandboxedRunner:
    def test_native_echo(self, tmp_path: Path) -> None:
        """Basic native command execution."""
        runner = SandboxedRunner(use_docker=False, timeout=5)
        # Use a platform-safe command
        result = runner.run(["python", "-c", "print('hello')"], cwd=tmp_path)
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_timeout_returns_negative_rc(self, tmp_path: Path) -> None:
        """Timeout produces returncode -1."""
        runner = SandboxedRunner(use_docker=False, timeout=1)
        result = runner.run(["python", "-c", "import time; time.sleep(10)"], cwd=tmp_path)
        assert result.returncode == -1
        assert "Timeout" in result.stderr

    def test_command_not_found(self, tmp_path: Path) -> None:
        """Missing command produces returncode -2."""
        runner = SandboxedRunner(use_docker=False, timeout=5)
        result = runner.run(["nonexistent_command_12345"], cwd=tmp_path)
        assert result.returncode == -2
        assert "not found" in result.stderr

    def test_docker_available_check(self) -> None:
        """_docker_available returns bool without crashing."""
        result = SandboxedRunner._docker_available()
        assert isinstance(result, bool)

    def test_docker_mode_disabled_when_unavailable(self) -> None:
        """If Docker is not available, falls back to native."""
        with patch.object(SandboxedRunner, "_docker_available", return_value=False):
            runner = SandboxedRunner(use_docker=True)
            assert runner.use_docker is False

    def test_cwd_isolation(self, tmp_path: Path) -> None:
        """Command runs in the specified working directory."""
        runner = SandboxedRunner(use_docker=False, timeout=5)
        result = runner.run(
            ["python", "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
        )
        assert result.returncode == 0
        # The cwd should be the tmp_path (normalized)
        assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


class TestSandboxedRunnerSecurity:
    """Security hardening tests."""

    def test_from_config_factory(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode

        config = SecurityConfig(mode=SecurityMode.NATIVE, timeout=15)
        runner = SandboxedRunner.from_config(config)
        assert runner.timeout == 15

    def test_from_config_docker_mode(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode

        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            config = SecurityConfig(mode=SecurityMode.DOCKER)
            runner = SandboxedRunner.from_config(config)
            assert runner.use_docker is True

    def test_verify_or_fail_raises_when_docker_required(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode, SecurityError

        config = SecurityConfig(mode=SecurityMode.DOCKER, require_docker=True)
        with patch.object(SandboxedRunner, "_docker_available", return_value=False):
            runner = SandboxedRunner.from_config(config)
            with pytest.raises(SecurityError, match="Docker is required"):
                runner.verify_or_fail()

    def test_verify_or_fail_passes_when_docker_available(self) -> None:
        from cola_coder.data.scorers.security import SecurityConfig, SecurityMode

        config = SecurityConfig(mode=SecurityMode.DOCKER, require_docker=True)
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner.from_config(config)
            runner.verify_or_fail()  # Should not raise

    def test_cleanup_stale_temps(self, tmp_path: Path) -> None:
        """cleanup_stale_temps removes orphaned dirs."""
        import shutil
        import tempfile

        # Create a fake stale temp dir
        stale = Path(tempfile.gettempdir()) / "cola_test_stale_12345"
        stale.mkdir(exist_ok=True)
        (stale / "file.ts").write_text("test")
        try:
            cleaned = SandboxedRunner.cleanup_stale_temps(prefix="cola_test_stale_")
            assert cleaned >= 1
            assert not stale.exists()
        finally:
            if stale.exists():
                shutil.rmtree(stale)

    def test_run_counter_increments(self, tmp_path: Path) -> None:
        """Run counter tracks native executions."""
        runner = SandboxedRunner(use_docker=False, timeout=5)
        assert runner.get_run_summary()["total_runs"] == 0

        runner.run(["python", "-c", "print(1)"], cwd=tmp_path)
        runner.run(["python", "-c", "print(2)"], cwd=tmp_path)

        summary = runner.get_run_summary()
        assert summary["total_runs"] == 2
        assert summary["native_runs"] == 2
        assert summary["docker_runs"] == 0

    def test_error_counter_on_timeout(self, tmp_path: Path) -> None:
        """Error counter increments on timeout (returncode -1)."""
        runner = SandboxedRunner(use_docker=False, timeout=1)
        runner.run(["python", "-c", "import time; time.sleep(10)"], cwd=tmp_path)

        summary = runner.get_run_summary()
        assert summary["errors"] == 1
        assert summary["total_runs"] == 1

    def test_error_counter_on_missing_command(self, tmp_path: Path) -> None:
        """Error counter increments on command not found (returncode -2)."""
        runner = SandboxedRunner(use_docker=False, timeout=5)
        runner.run(["nonexistent_xyz_12345"], cwd=tmp_path)

        summary = runner.get_run_summary()
        assert summary["errors"] == 1

    def test_log_status_native(self) -> None:
        """log_status returns correct info for native mode."""
        runner = SandboxedRunner(use_docker=False, timeout=10)
        status = runner.log_status()
        assert status["mode"] == "native"
        assert status["docker_connected"] is False

    def test_log_status_docker_connected(self) -> None:
        """log_status shows docker connected when active."""
        with patch.object(SandboxedRunner, "_docker_available", return_value=True):
            runner = SandboxedRunner(use_docker=True, timeout=10)
            status = runner.log_status()
            assert status["mode"] == "docker"
            assert status["docker_connected"] is True
            assert status["docker_image"] == "node:20-alpine"

    def test_log_status_docker_requested_unavailable(self) -> None:
        """log_status shows warning when Docker requested but unavailable."""
        with patch.object(SandboxedRunner, "_docker_available", return_value=False):
            runner = SandboxedRunner(use_docker=True, timeout=10)
            status = runner.log_status()
            assert status["mode"] == "native"
            assert status["docker_requested"] is True
            assert status["docker_connected"] is False

    def test_audit_integration(self, tmp_path: Path) -> None:
        """run() logs to audit logger when configured."""
        import json

        from cola_coder.data.scorers.audit import ScoringAuditLogger
        from cola_coder.data.scorers.security import SecurityConfig

        log_path = tmp_path / "audit.jsonl"
        logger = ScoringAuditLogger(log_path)
        config = SecurityConfig()
        runner = SandboxedRunner.from_config(config, audit_logger=logger)

        runner.run(
            ["python", "-c", "print('hi')"],
            cwd=tmp_path, label="test_scorer", file_hash="abc123",
        )

        assert log_path.exists()
        data = json.loads(log_path.read_text().strip())
        assert data["scorer"] == "test_scorer"
        assert data["file_hash"] == "abc123"
