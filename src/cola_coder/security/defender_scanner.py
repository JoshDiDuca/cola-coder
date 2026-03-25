"""Windows Defender scanner via MpCmdRun.exe CLI."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from cola_coder.security.scanner import MalwareScanResult, ThreatFinding

logger = logging.getLogger(__name__)


class DefenderScanner:
    """Scan files using Windows Defender (MpCmdRun.exe)."""

    name: str = "defender"

    # Standard locations for MpCmdRun.exe
    _MPCMDRUN_PATHS = [
        r"C:\Program Files\Windows Defender\MpCmdRun.exe",
        r"C:\ProgramData\Microsoft\Windows Defender\Platform",
    ]

    def __init__(self, timeout: int = 300) -> None:
        self._timeout = timeout
        self._exe_path = self._find_mpcmdrun()

    def scan_file(self, path: Path) -> MalwareScanResult:
        """Scan a single file."""
        return self._scan(Path(path))

    def scan_directory(self, path: Path) -> MalwareScanResult:
        """Scan an entire directory."""
        return self._scan(Path(path))

    def _scan(self, target: Path) -> MalwareScanResult:
        """Run MpCmdRun.exe -Scan -ScanType 3 on target."""
        start = time.perf_counter()

        if self._exe_path is None:
            return MalwareScanResult(is_clean=True)

        try:
            result = subprocess.run(
                [
                    self._exe_path,
                    "-Scan",
                    "-ScanType", "3",         # Custom scan
                    "-File", str(target),
                    "-DisableRemediation",     # Don't auto-delete, just report
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            duration = (time.perf_counter() - start) * 1000

            # Exit codes: 0 = clean, 2 = threat found
            if result.returncode == 2:
                threats = self._parse_threats(result.stdout, str(target))
                for t in threats:
                    logger.warning(
                        "Defender threat [%s/%s]: %s in %s",
                        t.name, t.severity, t.details, t.file_path,
                    )
                return MalwareScanResult(
                    is_clean=False,
                    threats=threats,
                    files_scanned=1,
                    scan_duration_ms=duration,
                )

            return MalwareScanResult(
                is_clean=True,
                files_scanned=1,
                scan_duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            return MalwareScanResult(
                is_clean=True,  # Assume clean on timeout
                scan_duration_ms=(time.perf_counter() - start) * 1000,
            )
        except (FileNotFoundError, OSError):
            return MalwareScanResult(is_clean=True)

    def _parse_threats(self, output: str, target: str) -> list[ThreatFinding]:
        """Parse MpCmdRun output for threat details."""
        threats: list[ThreatFinding] = []
        # MpCmdRun outputs threat info in lines like:
        # Threat  : TrojanDropper:Win32/...
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("Threat") and ":" in line:
                threat_name = line.split(":", 1)[1].strip()
                threats.append(ThreatFinding(
                    name=threat_name,
                    severity="high",
                    file_path=target,
                    scanner=self.name,
                    details=f"Windows Defender: {threat_name}",
                ))
        if not threats and output.strip():
            # Generic threat if we can't parse specifics
            threats.append(ThreatFinding(
                name="MalwareDetected",
                severity="high",
                file_path=target,
                scanner=self.name,
                details="Windows Defender detected a threat",
            ))
        return threats

    def is_available(self) -> bool:
        """Check if Windows Defender CLI is available."""
        return sys.platform == "win32" and self._exe_path is not None

    @classmethod
    def _find_mpcmdrun(cls) -> str | None:
        """Find MpCmdRun.exe on the system."""
        if sys.platform != "win32":
            return None

        # Check standard path first
        standard = Path(cls._MPCMDRUN_PATHS[0])
        if standard.exists():
            return str(standard)

        # Check platform-versioned path
        platform_dir = Path(cls._MPCMDRUN_PATHS[1])
        if platform_dir.exists():
            # Find latest version directory
            versions = sorted(platform_dir.iterdir(), reverse=True)
            for v in versions:
                exe = v / "MpCmdRun.exe"
                if exe.exists():
                    return str(exe)

        return None
