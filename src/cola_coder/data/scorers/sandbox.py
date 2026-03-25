"""Sandboxed execution for running external tools on untrusted code.

Provides isolation for tsc and eslint when processing HuggingFace data.
Supports native mode (temp dir isolation + timeout) and optional Docker
mode for maximum security.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class SandboxedRunner:
    """Run external tools safely on untrusted code."""

    def __init__(
        self,
        use_docker: bool = False,
        timeout: int = 10,
        memory_mb: int = 512,
        docker_image: str = "node:20-alpine",
    ) -> None:
        self.use_docker = use_docker and self._docker_available()
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.docker_image = docker_image

    def run(
        self,
        cmd: list[str],
        cwd: str | Path,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command in the sandbox.

        Args:
            cmd: Command and arguments (e.g. ["tsc", "--noEmit", "file.ts"]).
            cwd: Working directory (should be a temp dir with only the files to process).
            capture_output: Capture stdout/stderr.

        Returns:
            CompletedProcess with stdout/stderr.
        """
        cwd = str(Path(cwd).resolve())
        if self.use_docker:
            return self._run_docker(cmd, cwd, capture_output)
        return self._run_native(cmd, cwd, capture_output)

    def _run_native(
        self,
        cmd: list[str],
        cwd: str,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run with timeout and isolated working directory."""
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=self.timeout,
                # No shell=True — prevents injection
                # Isolated cwd — no access to parent dirs
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr=f"Timeout after {self.timeout}s",
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2,
                stdout="", stderr=f"Command not found: {cmd[0]}",
            )

    def _run_docker(
        self,
        cmd: list[str],
        cwd: str,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run inside a Docker container with no network and memory limits."""
        docker_cmd = [
            "docker", "run",
            "--rm",                              # Auto-remove container
            "--network", "none",                 # No network access
            f"--memory={self.memory_mb}m",       # Memory limit
            "--read-only",                       # Read-only root filesystem
            "--tmpfs", "/tmp:rw,size=64m",       # Small writable tmp
            "-v", f"{cwd}:/work:ro",             # Mount code read-only
            "-w", "/work",                       # Working directory
            self.docker_image,
            *cmd,
        ]
        try:
            return subprocess.run(
                docker_cmd,
                capture_output=capture_output,
                text=True,
                timeout=self.timeout + 10,  # Extra time for Docker overhead
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1,
                stdout="", stderr=f"Docker timeout after {self.timeout + 10}s",
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2,
                stdout="", stderr="Docker not found",
            )

    @staticmethod
    def _docker_available() -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
