"""Docker sandbox for running code in isolated containers.

Makes Docker OPTIONAL — if not installed, callers should fall back to
subprocess mode with appropriate warnings.

Security defaults (SEC-012 — bulletproof against untrusted code):
    - Non-root user (--user 65534:65534 / nobody) — never runs as root
    - Read-only root filesystem (--read-only) with exactly ONE small writable
      tmpfs at /tmp (--tmpfs /tmp:rw,...,size=64m) for the workdir
    - No network access (--network=none)
    - All Linux capabilities dropped (--cap-drop=ALL)
    - No privilege escalation (--security-opt no-new-privileges)
      NEVER --privileged, NEVER seccomp=unconfined
    - Fork-bomb protection (--pids-limit)
    - Memory limit AND --memory-swap equal to it (swap disabled)
    - CPU limit (--cpus)
    - File-descriptor / process ulimits (--ulimit nofile, --ulimit nproc)
    - No host namespaces shared (never --pid=host / --ipc=host / --net=host)
    - Clean environment — host env/secrets are NOT forwarded into the container
    - Read-only code mount
    - Captured stdout/stderr bounded to a max byte size (output-bomb defence)
    - Timeout enforcement + an outer wall-clock watchdog
    - Container force-removed on EVERY exit path (timeout, error, interrupt,
      normal completion) so no untrusted code outlives ``run``
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Docker images for common language runtimes
DEFAULT_IMAGES = {
    "node": "node:20-slim",
    "python": "python:3.11-slim",
    "go": "golang:1.22-alpine",
    "rust": "rust:1.77-slim",
}

# Unprivileged uid:gid present in virtually every Linux image (the "nobody"
# user). Running untrusted code as this user — combined with --cap-drop=ALL and
# no-new-privileges — means a container-escape would land as an unprivileged
# account with no capabilities rather than as root.
NOBODY_UID_GID = "65534:65534"


class DockerSandbox:
    """Run code in isolated Docker containers.

    Usage:
        sandbox = DockerSandbox()
        if sandbox.is_available():
            exit_code, stdout, stderr = sandbox.run(
                repo_path=Path("./my-repo"),
                command="npm test",
                image="node:20-slim",
            )

    All resource limits are constructor-configurable with safe defaults. The
    sandbox is written to be compatible with Docker Desktop on Windows: only
    flags broadly supported by that engine are used (``--storage-opt size`` is
    deliberately NOT used because it is unsupported on the overlay2/Desktop
    storage driver — the writable tmpfs is size-capped instead).
    """

    def __init__(
        self,
        memory_limit: str = "2g",
        cpu_limit: float = 2.0,
        pid_limit: int = 64,
        network: bool = False,
        timeout: int = 300,
        *,
        user: str = NOBODY_UID_GID,
        read_only: bool = True,
        tmpfs_size: str = "64m",
        tmpfs_path: str = "/tmp",
        nofile_limit: int = 256,
        nproc_limit: int = 256,
        max_output_bytes: int = 1_000_000,
        watchdog_grace: int = 10,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pid_limit = pid_limit
        self.network = network
        self.timeout = timeout
        # SEC-012 hardening knobs (configurable, safe defaults):
        self.user = user
        self.read_only = read_only
        self.tmpfs_size = tmpfs_size
        self.tmpfs_path = tmpfs_path
        self.nofile_limit = nofile_limit
        self.nproc_limit = nproc_limit
        self.max_output_bytes = max_output_bytes
        # Extra seconds the outer wall-clock watchdog waits beyond the
        # subprocess timeout before force-removing the container itself.
        self.watchdog_grace = watchdog_grace
        self._docker_path: str | None = None

    @staticmethod
    def is_available() -> bool:
        """Check if Docker is installed and the daemon is running."""
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _build_run_argv(self, container_name: str, image: str, command: str,
                        env: dict[str, str] | None, repo_path: Path) -> list[str]:
        """Assemble the hardened ``docker run`` argv.

        Every entry here is a deliberate security control — see the module
        docstring. Keep them in sync with the SEC-012 assertions in
        tests/test_curation.py.
        """
        mount_path = str(repo_path).replace("\\", "/")

        cmd = [
            "docker", "run",
            "--rm",
            f"--name={container_name}",
            # (1) non-root — drop from root to nobody.
            "--user", self.user,
            # (4) capabilities + no privilege escalation. NEVER --privileged
            #     and NEVER --security-opt seccomp=unconfined.
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            # (5) fork-bomb cap.
            f"--pids-limit={self.pid_limit}",
            # (6) memory cap with swap disabled (swap == memory means no swap).
            f"--memory={self.memory_limit}",
            f"--memory-swap={self.memory_limit}",
            # (7) CPU cap.
            f"--cpus={self.cpu_limit}",
            # (8) file-descriptor and process ulimits.
            f"--ulimit=nofile={self.nofile_limit}:{self.nofile_limit}",
            f"--ulimit=nproc={self.nproc_limit}:{self.nproc_limit}",
        ]

        # (2) read-only rootfs + exactly one small writable tmpfs for the
        #     workdir/tmp. The code-copy step in run_with_install copies INTO
        #     this tmpfs (/tmp/workdir), so install/test still work.
        if self.read_only:
            cmd.append("--read-only")
            cmd.append(f"--tmpfs={self.tmpfs_path}:rw,size={self.tmpfs_size},mode=1777")

        # (3) / (9) network off — and because we never pass --net=host/--pid=host
        #     /--ipc=host, no host namespaces are shared.
        if not self.network:
            cmd.append("--network=none")

        # Mount repo read-only at /code; work out of it.
        cmd.extend(["-v", f"{mount_path}:/code:ro", "-w", "/code"])

        # (10) clean environment: only EXPLICIT vars the caller asked for are
        #      forwarded. The host's process environment (which may hold tokens,
        #      HF_TOKEN, AWS creds, etc.) is never passed through — Docker does
        #      not inherit the parent env into the container by default, and we
        #      pass nothing implicitly.
        if env:
            for key, val in env.items():
                cmd.extend(["-e", f"{key}={val}"])

        # Image and command.
        cmd.append(image)
        cmd.extend(["sh", "-c", command])
        return cmd

    def _truncate_output(self, text: str) -> str:
        """Bound captured output to ``max_output_bytes`` (output-bomb defence).

        A malicious test can emit gigabytes to stdout/stderr to exhaust host
        memory while we buffer it. ``subprocess.run`` already buffers fully, so
        we cap the *returned* size and append a clear truncation marker so the
        caller can tell the output was cut.
        """
        if self.max_output_bytes is None or self.max_output_bytes <= 0:
            return text
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return text
        clipped = encoded[: self.max_output_bytes].decode("utf-8", errors="replace")
        return (
            clipped
            + f"\n...[output truncated: exceeded {self.max_output_bytes} bytes]"
        )

    def run(
        self,
        repo_path: Path,
        command: str,
        image: str = "node:20-slim",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run a command in a hardened Docker container.

        Args:
            repo_path: Path to the repo to mount (read-only).
            command: Shell command to run inside the container.
            image: Docker image to use.
            timeout: Override default timeout (seconds).
            env: Extra environment variables to pass (ONLY these — the host
                environment is never forwarded).

        Returns:
            Tuple of (exit_code, stdout, stderr). stdout/stderr are bounded to
            ``max_output_bytes`` and carry a truncation marker if clipped.

        Raises:
            RuntimeError: If Docker is not available.
        """
        if not self.is_available():
            raise RuntimeError(
                "Docker is not available. Install Docker or use subprocess mode."
            )

        effective_timeout = timeout or self.timeout
        repo_path = repo_path.resolve()

        # Unique container name so we can force-kill it on ANY exit path.
        # Killing the `docker run` client process (what subprocess.run does on
        # timeout) does NOT stop the container — the daemon keeps it (and the
        # untrusted code inside) running. We must explicitly `docker rm -f` it
        # by name. Generated ONCE here and reused for the run command and every
        # cleanup path below.
        container_name = f"cola-curation-{uuid.uuid4().hex}"

        cmd = self._build_run_argv(container_name, image, command, env, repo_path)

        logger.info("Docker run: %s", " ".join(cmd))

        # (12) Outer wall-clock watchdog (belt-and-braces): independent of the
        # subprocess timeout, a background timer force-removes the container if
        # the whole call somehow overruns. subprocess.run's timeout only kills
        # the `docker run` CLIENT — if that timeout machinery is ever defeated
        # (a wedged client, a swallowed TimeoutExpired), this still tears the
        # container down so untrusted code cannot outlive the call.
        watchdog = threading.Timer(
            effective_timeout + self.watchdog_grace,
            self._force_remove_container,
            args=(container_name,),
        )
        watchdog.daemon = True
        watchdog.start()

        # Defense-in-depth (SEC-002): no container/`docker run` child may
        # outlive this method, regardless of how it exits.
        #   - timeout      → return the timeout sentinel (the `finally` cleans up)
        #   - KeyboardInterrupt → clean up, then re-raise so Ctrl-C still works
        #   - any other exception → propagate, but clean up first via `finally`
        #   - normal completion → `finally` is a harmless no-op if `--rm` fired
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return (
                result.returncode,
                self._truncate_output(result.stdout),
                self._truncate_output(result.stderr),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Docker command timed out after %ds: %s", effective_timeout, command
            )
            # Force-kill happens in `finally`; the timeout above only kills the
            # `docker run` client, not the container the daemon is running.
            # Without cleanup, untrusted code keeps executing past the timeout.
            return -1, "", f"Timeout after {effective_timeout}s"
        except KeyboardInterrupt:
            # A Ctrl-C mid-run must still tear down the container before the
            # interrupt unwinds the stack — otherwise the daemon keeps running
            # untrusted code after the Python process is gone.
            logger.warning("Docker run interrupted; cleaning up container %s", container_name)
            raise
        finally:
            # The watchdog is no longer needed once we are tearing down here.
            watchdog.cancel()
            # Runs on every path: timeout, KeyboardInterrupt, unexpected
            # exception, AND normal completion (in case `--rm` did not fire).
            # `_force_remove_container` is best-effort and idempotent — if the
            # container already exited it is a harmless no-op.
            self._force_remove_container(container_name)

    @staticmethod
    def _force_remove_container(name: str) -> None:
        """Force-remove a (possibly still-running) container by name.

        ``docker rm -f`` both kills and removes the container, so it stops
        untrusted code that survived a client-side timeout. Best-effort: if
        the container already exited (``--rm`` cleaned it up) the command is a
        harmless no-op, and any failure is logged rather than raised so the
        caller's result/exception still propagates.
        """
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.error("Failed to force-remove container %s: %s", name, exc)

    def run_with_install(
        self,
        repo_path: Path,
        install_cmd: str,
        test_cmd: str,
        image: str = "node:20-slim",
        install_timeout: int = 120,
        test_timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """Run install + test in a single container (writable copy).

        Since the code mount is read-only AND the root filesystem is read-only,
        this copies code into the writable tmpfs (under ``tmpfs_path``, e.g.
        ``/tmp/workdir``) first, runs install, then runs tests — all in one
        container invocation. The tmpfs is the only writable location, so the
        copy target lives inside it.

        Args:
            repo_path: Path to repo on host.
            install_cmd: Dependency install command (e.g., "npm install").
            test_cmd: Test command (e.g., "npm test").
            image: Docker image.
            install_timeout: Max seconds for install step.
            test_timeout: Max seconds for test step (uses self.timeout if None).

        Returns:
            Tuple of (exit_code, stdout, stderr) for the test step.
        """
        effective_test_timeout = test_timeout or self.timeout
        total_timeout = install_timeout + effective_test_timeout + 30  # buffer

        # Copy into the writable tmpfs (read-only rootfs makes everything else
        # unwritable). HOME is also pointed at the tmpfs so package managers
        # that need a writable cache/HOME (npm, pip) work under --user nobody.
        workdir = f"{self.tmpfs_path}/workdir"
        combined = (
            f"cp -r /code {workdir} && cd {workdir} && export HOME={self.tmpfs_path} && "
            f"timeout {install_timeout} {install_cmd} 2>&1 && "
            f"timeout {effective_test_timeout} {test_cmd} 2>&1"
        )

        return self.run(
            repo_path=repo_path,
            command=combined,
            image=image,
            timeout=total_timeout,
        )

    @staticmethod
    def image_for_language(language: str) -> str:
        """Get the default Docker image for a language."""
        return DEFAULT_IMAGES.get(language, "ubuntu:22.04")
