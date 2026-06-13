"""Docker sandbox for running code in isolated containers.

Makes Docker OPTIONAL — if not installed, callers should fall back to
subprocess mode with appropriate warnings.

Security defaults:
    - No network access (--network none)
    - Memory limit (2GB)
    - CPU limit (2 cores)
    - PID limit (64)
    - All Linux capabilities dropped (--cap-drop=ALL)
    - No privilege escalation (--security-opt no-new-privileges)
    - Read-only code mount
    - Timeout enforcement
    - Container force-removed on EVERY exit path (timeout, error, interrupt,
      normal completion) so no untrusted code outlives ``run``
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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
    """

    def __init__(
        self,
        memory_limit: str = "2g",
        cpu_limit: float = 2.0,
        pid_limit: int = 64,
        network: bool = False,
        timeout: int = 300,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pid_limit = pid_limit
        self.network = network
        self.timeout = timeout
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

    def run(
        self,
        repo_path: Path,
        command: str,
        image: str = "node:20-slim",
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run a command in a Docker container.

        Args:
            repo_path: Path to the repo to mount (read-only).
            command: Shell command to run inside the container.
            image: Docker image to use.
            timeout: Override default timeout (seconds).
            env: Extra environment variables to pass.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            RuntimeError: If Docker is not available.
            subprocess.TimeoutExpired: If the command exceeds timeout.
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

        # Build docker run command.
        # The container user stays root because npm/pip installs need a
        # writable HOME and cache across arbitrary images — but with ALL
        # capabilities dropped, no-new-privileges, resource limits, and
        # (by default) no network, root-in-container has little to abuse.
        cmd = [
            "docker", "run",
            "--rm",
            f"--name={container_name}",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            f"--pids-limit={self.pid_limit}",
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
        ]

        if not self.network:
            cmd.append("--network=none")

        # Mount repo read-only
        # Convert Windows paths to Docker-compatible format
        mount_path = str(repo_path).replace("\\", "/")
        cmd.extend(["-v", f"{mount_path}:/code:ro", "-w", "/code"])

        # Add environment variables
        if env:
            for key, val in env.items():
                cmd.extend(["-e", f"{key}={val}"])

        # Image and command
        cmd.append(image)
        cmd.extend(["sh", "-c", command])

        logger.info("Docker run: %s", " ".join(cmd))

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
            return result.returncode, result.stdout, result.stderr
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

        Since the code mount is read-only, this copies code to /tmp/code first,
        runs install, then runs tests. All in one container invocation.

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

        # Combined command: copy, install, test
        combined = (
            f"cp -r /code /tmp/workdir && cd /tmp/workdir && "
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
