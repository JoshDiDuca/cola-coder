"""Sandboxed execution for running external tools on untrusted code.

Provides isolation for tsc and eslint when processing HuggingFace data.
Supports native mode (temp dir isolation + timeout) and optional Docker
mode for maximum security.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _kill_process_tree(name: str) -> None:
    """Kill a process and its children on Windows."""
    if sys.platform != "win32":
        return
    try:
        # Use taskkill /T to kill the process tree
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", name],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


class SandboxedRunner:
    """Run external tools safely on untrusted code."""

    def __init__(
        self,
        use_docker: bool = False,
        timeout: int = 10,
        memory_mb: int = 512,
        docker_image: str = "node:20-alpine",
    ) -> None:
        self._docker_requested = use_docker
        self.use_docker = use_docker and self._docker_available()
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.docker_image = docker_image

        # Execution counters for reporting
        self._docker_runs: int = 0
        self._native_runs: int = 0
        self._total_runs: int = 0
        self._errors: int = 0

    @classmethod
    def from_config(
        cls,
        config: "SecurityConfig",
        audit_logger: "ScoringAuditLogger | None" = None,
    ) -> SandboxedRunner:
        """Construct from SecurityConfig with optional audit logger."""
        from cola_coder.data.scorers.security import SecurityMode

        instance = cls(
            use_docker=(config.mode == SecurityMode.DOCKER),
            timeout=config.timeout,
            memory_mb=config.memory_mb,
            docker_image=config.docker_image,
        )
        instance._config = config
        instance._audit_logger = audit_logger
        return instance

    def log_status(self) -> dict[str, object]:
        """Return sandbox status info for display. Also logs via Python logging."""
        mode = "docker" if self.use_docker else "native"
        docker_available = self._docker_available()

        status: dict[str, object] = {
            "mode": mode,
            "docker_requested": self._docker_requested,
            "docker_available": docker_available,
            "docker_connected": self.use_docker,
            "timeout": self.timeout,
            "memory_mb": self.memory_mb,
            "docker_image": self.docker_image if self.use_docker else None,
        }

        if self.use_docker:
            logger.info(
                "Sandbox: Docker mode ACTIVE — image=%s, network=none, "
                "memory=%dMB, timeout=%ds",
                self.docker_image, self.memory_mb, self.timeout,
            )
        elif self._docker_requested:
            logger.warning(
                "Sandbox: Docker was REQUESTED but is NOT AVAILABLE — "
                "falling back to native isolation",
            )
        else:
            logger.info(
                "Sandbox: native mode — temp dir isolation, timeout=%ds "
                "(set security.mode=docker for container isolation)",
                self.timeout,
            )

        return status

    def get_run_summary(self) -> dict[str, int]:
        """Return execution statistics for end-of-run reporting."""
        return {
            "total_runs": self._total_runs,
            "docker_runs": self._docker_runs,
            "native_runs": self._native_runs,
            "errors": self._errors,
        }

    def verify_or_fail(self) -> None:
        """Raises SecurityError if Docker is required but unavailable."""
        if hasattr(self, '_config') and self._config.require_docker and not self._docker_available():
            from cola_coder.data.scorers.security import SecurityError

            raise SecurityError(
                "Docker is required (security.require_docker=true) but not available. "
                "Install/start Docker Desktop or set require_docker=false."
            )

    def run(
        self,
        cmd: list[str],
        cwd: str | Path,
        capture_output: bool = True,
        label: str = "",
        file_hash: str = "",
    ) -> subprocess.CompletedProcess[str]:
        """Run a command in the sandbox.

        Args:
            cmd: Command and arguments (e.g. ["tsc", "--noEmit", "file.ts"]).
            cwd: Working directory (should be a temp dir with only the files to process).
            capture_output: Capture stdout/stderr.
            label: Scorer name for audit logging.
            file_hash: File content hash for audit logging.

        Returns:
            CompletedProcess with stdout/stderr.
        """
        import time

        start = time.perf_counter()
        result = self._do_run(cmd, cwd, capture_output)
        duration_ms = (time.perf_counter() - start) * 1000

        if hasattr(self, '_audit_logger') and self._audit_logger:
            from cola_coder.data.scorers.audit import AuditEntry

            self._audit_logger.log(AuditEntry(
                scorer=label,
                file_hash=file_hash,
                security_mode=self._config.mode.value if hasattr(self, '_config') else "native",
                command=cmd[:5],  # Truncate long command lists
                exit_code=result.returncode,
                duration_ms=round(duration_ms, 1),
            ))

        return result

    def _do_run(
        self,
        cmd: list[str],
        cwd: str | Path,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Internal run dispatcher."""
        cwd = str(Path(cwd).resolve())
        self._total_runs += 1

        if self.use_docker:
            self._docker_runs += 1
            result = self._run_docker(cmd, cwd, capture_output)
        else:
            self._native_runs += 1
            result = self._run_native(cmd, cwd, capture_output)

        if result.returncode < 0:
            self._errors += 1

        return result

    def _run_native(
        self,
        cmd: list[str],
        cwd: str,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run with timeout and isolated working directory."""
        try:
            kwargs: dict[str, Any] = {
                "cwd": cwd,
                "capture_output": capture_output,
                "text": True,
                "timeout": self.timeout,
                # No shell=True -- prevents injection
                # Isolated cwd -- no access to parent dirs
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

            return subprocess.run(cmd, **kwargs)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                _kill_process_tree(cmd[0] if cmd else "")
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
        # Get docker config if available
        docker_cfg = None
        if hasattr(self, '_config'):
            docker_cfg = self._config.docker

        docker_cmd = [
            "docker", "run",
            "--rm",                              # Auto-remove container
            "--network", "none",                 # No network access
            f"--memory={self.memory_mb}m",       # Memory limit
            "--read-only",                       # Read-only root filesystem
            "--tmpfs", "/tmp:rw,size=64m",       # Small writable tmp
            "--pids-limit", str(docker_cfg.pids_limit if docker_cfg else 64),  # Prevent fork bombs
            "--cap-drop", "ALL",                 # Drop all capabilities
            "--security-opt", "no-new-privileges",  # Prevent privilege escalation
            "--user", "65534:65534",             # Run as nobody
            "-v", f"{cwd}:/work:ro",             # Mount code read-only
            "-w", "/work",                       # Working directory
            self.docker_image,
            *cmd,
        ]
        try:
            kwargs: dict[str, Any] = {
                "capture_output": capture_output,
                "text": True,
                "timeout": self.timeout + 10,  # Extra time for Docker overhead
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

            return subprocess.run(docker_cmd, **kwargs)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                _kill_process_tree("docker")
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

    @staticmethod
    def cleanup_stale_temps(prefix: str = "cola_") -> int:
        """Remove stale temp directories from crashed scoring runs."""
        import glob

        tmpdir = tempfile.gettempdir()
        stale = glob.glob(os.path.join(tmpdir, f"{prefix}*"))
        cleaned = 0
        for path in stale:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    cleaned += 1
            except OSError:
                pass
        return cleaned
