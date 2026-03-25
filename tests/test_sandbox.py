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
