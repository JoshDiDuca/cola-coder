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
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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

        # Build docker run command.
        # The container user stays root because npm/pip installs need a
        # writable HOME and cache across arbitrary images — but with ALL
        # capabilities dropped, no-new-privileges, resource limits, and
        # (by default) no network, root-in-container has little to abuse.
        cmd = [
            "docker", "run",
            "--rm",
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
            # Try to kill any lingering container
            return -1, "", f"Timeout after {effective_timeout}s"

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
