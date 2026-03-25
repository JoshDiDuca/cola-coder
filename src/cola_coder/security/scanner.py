"""Malware scanner protocol and composite scanner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ThreatFinding:
    """A single malware/threat detection."""
    name: str              # e.g. "CryptoMiner", "ReverseShell"
    severity: str          # "high", "medium", "low"
    file_path: str         # Path to the file containing the threat
    scanner: str           # Which scanner found it (e.g. "yara", "defender")
    details: str = ""      # Additional details


@dataclass
class MalwareScanResult:
    """Result of scanning files for malware."""
    is_clean: bool = True
    threats: list[ThreatFinding] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration_ms: float = 0.0

    def merge(self, other: MalwareScanResult) -> MalwareScanResult:
        """Merge two scan results."""
        return MalwareScanResult(
            is_clean=self.is_clean and other.is_clean,
            threats=self.threats + other.threats,
            files_scanned=max(self.files_scanned, other.files_scanned),
            scan_duration_ms=self.scan_duration_ms + other.scan_duration_ms,
        )


@runtime_checkable
class MalwareScannerProtocol(Protocol):
    """Interface for malware scanners."""
    name: str

    def scan_file(self, path: Path) -> MalwareScanResult: ...
    def scan_directory(self, path: Path) -> MalwareScanResult: ...
    def is_available(self) -> bool: ...


class CompositeMalwareScanner:
    """Run multiple scanners and combine results."""

    def __init__(
        self,
        scanners: list[MalwareScannerProtocol] | None = None,
        audit_logger: object | None = None,
    ) -> None:
        self._scanners = [s for s in (scanners or []) if s.is_available()]
        self._audit_logger = audit_logger

    @classmethod
    def from_config(
        cls,
        config: dict | None = None,
        audit_logger: object | None = None,
    ) -> CompositeMalwareScanner:
        """Build scanner from config dict. Auto-detects available scanners."""
        config = config or {}
        scanners_cfg = config.get("scanners", {})
        scanners: list[MalwareScannerProtocol] = []

        if scanners_cfg.get("defender", True):
            try:
                from cola_coder.security.defender_scanner import DefenderScanner
                scanners.append(DefenderScanner())
            except ImportError:
                pass

        if scanners_cfg.get("yara", True):
            try:
                from cola_coder.security.yara_scanner import YaraScanner
                rules_dir = config.get("yara_rules_dir")
                scanners.append(YaraScanner(rules_dir=rules_dir))
            except ImportError:
                pass

        if scanners_cfg.get("clamav", False):
            try:
                from cola_coder.security.clamav_scanner import ClamAvScanner
                scanners.append(ClamAvScanner())
            except ImportError:
                pass

        return cls(scanners, audit_logger=audit_logger)

    def scan_file(self, path: Path) -> MalwareScanResult:
        result = MalwareScanResult(files_scanned=1)
        for scanner in self._scanners:
            try:
                sub = scanner.scan_file(path)
                result = result.merge(sub)
            except Exception:
                pass
        if not result.is_clean:
            for t in result.threats:
                logger.warning(
                    "MALWARE DETECTED [%s/%s]: %s in %s",
                    t.scanner, t.severity, t.name, t.file_path,
                )
            if self._audit_logger:
                for t in result.threats:
                    self._audit_logger.log_security_event(
                        f"malware_detected:{t.scanner}:{t.name}",
                        scorer=t.scanner,
                        file_hash=t.file_path,
                    )
        return result

    def scan_directory(self, path: Path) -> MalwareScanResult:
        result = MalwareScanResult()
        for scanner in self._scanners:
            try:
                sub = scanner.scan_directory(path)
                result = result.merge(sub)
            except Exception:
                pass
        if not result.is_clean:
            for t in result.threats:
                logger.warning(
                    "MALWARE DETECTED [%s/%s]: %s in %s",
                    t.scanner, t.severity, t.name, t.file_path,
                )
            if self._audit_logger:
                for t in result.threats:
                    self._audit_logger.log_security_event(
                        f"malware_detected:{t.scanner}:{t.name}",
                        scorer=t.scanner,
                        file_hash=t.file_path,
                    )
        return result

    @property
    def available_scanners(self) -> list[str]:
        return [s.name for s in self._scanners]
