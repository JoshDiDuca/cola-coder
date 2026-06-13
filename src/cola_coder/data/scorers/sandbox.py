"""Sandboxed execution for running external tools on untrusted code.

Provides isolation for tsc and eslint when processing HuggingFace data.
Supports native mode (temp dir isolation + timeout) and optional Docker
mode for maximum security.

SEC-013 — FAIL CLOSED:
When Docker isolation is *requested* (``use_docker=True`` / security.mode=docker)
but Docker is unavailable (not installed / daemon down), the runner does NOT
silently fall back to running untrusted code on the host. It refuses to execute
and returns a sentinel result (returncode ``RC_SANDBOX_UNAVAILABLE``) that
callers treat as "not executed". The ONLY way to host-exec when Docker was
requested-but-missing is to opt in explicitly via ``allow_native_fallback=True``
(off by default). Plain ``native`` mode (the default) is itself the explicit,
documented opt-in to host execution with temp-dir isolation + timeout — it is
intended for trusted/already-vetted code.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cola_coder.data.scorers.audit import ScoringAuditLogger
    from cola_coder.data.scorers.security import SecurityConfig

logger = logging.getLogger(__name__)

# Sentinel return code: Docker isolation was requested but unavailable, and
# host-exec fallback was not explicitly allowed. The command was NOT executed.
# Callers treat this like any other negative returncode (error / skip / score-0).
RC_SANDBOX_UNAVAILABLE = -3


def _kill_proc_tree(pid: int) -> None:
    """Kill a process AND its descendants, scoped to a single PID tree.

    NEVER kill by image name: `taskkill /IM <name>` (Windows) / killall (POSIX)
    terminate EVERY process sharing the name — e.g. all `node` processes on the
    box, including the VS Code extension host or unrelated work. Scoping to the
    pid's tree kills only the runaway sandboxed process and what it spawned.

    Windows: `taskkill /F /T /PID` walks the child tree. POSIX: the child is
    started in its own session (start_new_session=True) so its pid is the
    process-group id, and killpg reaps the whole group.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _finish_proc(
    proc: "subprocess.Popen[str]",
    cmd: list[str],
    timeout_s: float,
    timeout_msg: str,
) -> subprocess.CompletedProcess[str]:
    """Wait for a Popen process, killing its whole tree by PID on timeout.

    Returns a CompletedProcess. On timeout the return code is -1 and stdout is
    discarded (partial output from a runaway process is not trustworthy).
    """
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(
            args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        _kill_proc_tree(proc.pid)
        try:
            proc.communicate(timeout=5)  # reap so we don't leak a zombie
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        return subprocess.CompletedProcess(
            args=cmd, returncode=-1, stdout="", stderr=timeout_msg,
        )


class SandboxedRunner:
    """Run external tools safely on untrusted code."""

    def __init__(
        self,
        use_docker: bool = False,
        timeout: int = 10,
        memory_mb: int = 512,
        docker_image: str = "node:20-alpine",
        allow_native_fallback: bool = False,
    ) -> None:
        self._docker_requested = use_docker
        self.use_docker = use_docker and self._docker_available()
        # SEC-013: when Docker was requested but is unavailable, host-exec is
        # refused UNLESS the caller explicitly opts in here. Off by default so
        # the failure mode is "don't run untrusted code", never "run it on host".
        self.allow_native_fallback = allow_native_fallback
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
            # SEC-013: explicit, off-by-default opt-in for host-exec fallback
            # when Docker was requested but is unavailable. Read defensively so
            # this works whether or not SecurityConfig defines the field.
            allow_native_fallback=getattr(config, "allow_native_fallback", False),
        )
        instance._config = config
        instance._audit_logger = audit_logger
        return instance

    def log_status(self) -> dict[str, object]:
        """Return sandbox status info for display. Also logs via Python logging."""
        mode = "docker" if self.use_docker else "native"
        docker_available = self._docker_available()

        # SEC-013: Docker requested + unavailable + no explicit fallback opt-in
        # means we will refuse to execute (fail closed), NOT run on the host.
        fail_closed = (
            self._docker_requested
            and not self.use_docker
            and not self.allow_native_fallback
        )

        status: dict[str, object] = {
            "mode": mode,
            "docker_requested": self._docker_requested,
            "docker_available": docker_available,
            "docker_connected": self.use_docker,
            "allow_native_fallback": self.allow_native_fallback,
            "fail_closed": fail_closed,
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
        elif fail_closed:
            logger.warning(
                "Sandbox: Docker isolation REQUESTED but NOT AVAILABLE — "
                "FAILING CLOSED: untrusted code will NOT be executed. Install/"
                "start Docker, set security.mode=native to accept host "
                "execution, or set allow_native_fallback=true to opt in.",
            )
        elif self._docker_requested:
            logger.warning(
                "Sandbox: Docker was REQUESTED but is NOT AVAILABLE — "
                "allow_native_fallback is set, falling back to native isolation "
                "(host execution of untrusted code)",
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
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command in the sandbox.

        Args:
            cmd: Command and arguments (e.g. ["tsc", "--noEmit", "file.ts"]).
            cwd: Working directory (should be a temp dir with only the files to process).
            capture_output: Capture stdout/stderr.
            label: Scorer name for audit logging.
            file_hash: File content hash for audit logging.
            env: Restricted environment for native mode (Docker mode already
                isolates the environment inside the container).
            timeout: Per-call timeout override in seconds (defaults to the
                runner's configured timeout).

        Returns:
            CompletedProcess with stdout/stderr.
        """
        import time

        start = time.perf_counter()
        result = self._do_run(cmd, cwd, capture_output, env=env, timeout=timeout)
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
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Internal run dispatcher."""
        cwd = str(Path(cwd).resolve())
        self._total_runs += 1
        effective_timeout = timeout if timeout is not None else self.timeout

        if self.use_docker:
            self._docker_runs += 1
            result = self._run_docker(cmd, cwd, capture_output, timeout=effective_timeout)
        elif self._docker_requested and not self.allow_native_fallback:
            # SEC-013 FAIL CLOSED: Docker isolation was requested but Docker is
            # unavailable. Running untrusted code directly on the host would be a
            # silent fail-OPEN, so we refuse to execute and report a sentinel.
            self._errors += 1
            logger.error(
                "Sandbox: Docker isolation REQUESTED but UNAVAILABLE — refusing "
                "to run %r on the host (fail closed). Install/start Docker, set "
                "security.mode=native to accept host execution, or pass "
                "allow_native_fallback=True to opt in explicitly.",
                cmd[:3],
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=RC_SANDBOX_UNAVAILABLE,
                stdout="",
                stderr=(
                    "Sandbox unavailable: Docker isolation was required but "
                    "Docker is not available; command not executed (fail closed)."
                ),
            )
        else:
            self._native_runs += 1
            result = self._run_native(
                cmd, cwd, capture_output, env=env, timeout=effective_timeout,
            )

        if result.returncode < 0:
            self._errors += 1

        return result

    def _run_native(
        self,
        cmd: list[str],
        cwd: str,
        capture_output: bool,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run with timeout and isolated working directory."""
        effective_timeout = timeout if timeout is not None else self.timeout
        # Popen (not subprocess.run) so we hold the child's PID: on timeout we
        # must kill its WHOLE tree by PID (untrusted code may spawn children),
        # never by image name. No shell=True; isolated cwd; restricted env.
        popen_kwargs: dict[str, Any] = {"cwd": cwd, "text": True}
        if capture_output:
            popen_kwargs["stdout"] = subprocess.PIPE
            popen_kwargs["stderr"] = subprocess.PIPE
        if env is not None:
            popen_kwargs["env"] = env
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        else:
            # Own session → child pid is its process-group id, so a timeout can
            # killpg the entire tree, not just the direct child.
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2,
                stdout="", stderr=f"Command not found: {cmd[0]}",
            )

        return _finish_proc(
            proc, cmd, effective_timeout, f"Timeout after {effective_timeout}s",
        )

    def _run_docker(
        self,
        cmd: list[str],
        cwd: str,
        capture_output: bool,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run inside a Docker container with no network and memory limits."""
        effective_timeout = timeout if timeout is not None else self.timeout
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
        popen_kwargs: dict[str, Any] = {"text": True}
        if capture_output:
            popen_kwargs["stdout"] = subprocess.PIPE
            popen_kwargs["stderr"] = subprocess.PIPE
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(docker_cmd, **popen_kwargs)
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2, stdout="", stderr="Docker not found",
            )

        # The container is constrained by --rm/--network none/--memory/timeout;
        # killing the `docker run` client tree by PID (not /IM docker, which
        # would kill every docker client) is the host-side stop on timeout.
        return _finish_proc(
            proc, cmd, effective_timeout + 10,  # extra time for Docker overhead
            f"Docker timeout after {effective_timeout + 10}s",
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
