"""Security configuration and enforcement for the scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SecurityMode(str, Enum):
    """Scoring pipeline security level."""
    OFF = "off"         # Trust all data, no isolation
    NATIVE = "native"   # Temp dir isolation + timeout + CREATE_NO_WINDOW on Windows
    DOCKER = "docker"   # Full container isolation


class SecurityError(Exception):
    """Raised when a security requirement is not met."""
    pass


@dataclass
class DockerConfig:
    """Docker-specific security settings."""
    pids_limit: int = 64
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    network: str = "none"
    read_only: bool = True
    tmpfs_size_mb: int = 64


@dataclass
class SecurityConfig:
    """Complete security configuration for the scoring pipeline."""
    mode: SecurityMode = SecurityMode.NATIVE
    require_docker: bool = False
    timeout: int = 10
    memory_mb: int = 512
    docker_image: str = "node:20-alpine"
    audit_log_path: str = "logs/scoring_audit.jsonl"
    credential_scan_mode: str = "strip"  # off | warn | strip | reject
    docker: DockerConfig = field(default_factory=DockerConfig)

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> SecurityConfig:
        """Load from a scoring config dict (the 'security' or 'sandbox' section)."""
        # Support new 'security' key
        security = cfg.get("security", {})

        # Backward compat: fall back to old 'sandbox' key
        if not security:
            sandbox = cfg.get("sandbox", {})
            if sandbox:
                use_docker = sandbox.get("use_docker", False)
                return SecurityConfig(
                    mode=SecurityMode.DOCKER if use_docker else SecurityMode.NATIVE,
                    timeout=sandbox.get("timeout", 10),
                    memory_mb=sandbox.get("memory_mb", 512),
                )

        # Parse mode
        mode_str = security.get("mode", "native")
        try:
            mode = SecurityMode(mode_str)
        except ValueError:
            mode = SecurityMode.NATIVE

        # Parse docker subsection
        docker_cfg = security.get("docker", {})
        docker = DockerConfig(
            pids_limit=docker_cfg.get("pids_limit", 64),
            cap_drop=docker_cfg.get("cap_drop", ["ALL"]),
            network=docker_cfg.get("network", "none"),
            read_only=docker_cfg.get("read_only", True),
            tmpfs_size_mb=docker_cfg.get("tmpfs_size_mb", 64),
        )

        # Parse credential scan
        cred_cfg = security.get("credential_scan", {})
        cred_mode = cred_cfg.get("mode", "strip") if isinstance(cred_cfg, dict) else "strip"

        return SecurityConfig(
            mode=mode,
            require_docker=security.get("require_docker", False),
            timeout=security.get("timeout", 10),
            memory_mb=security.get("memory_mb", 512),
            docker_image=security.get("docker_image", "node:20-alpine"),
            audit_log_path=security.get("audit_log", "logs/scoring_audit.jsonl"),
            credential_scan_mode=cred_mode,
            docker=docker,
        )
