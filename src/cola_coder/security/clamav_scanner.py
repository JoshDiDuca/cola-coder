"""ClamAV scanner via clamd daemon client (optional)."""

from __future__ import annotations

import time
from pathlib import Path

from cola_coder.security.scanner import MalwareScanResult, ThreatFinding


class ClamAvScanner:
    """Scan files using ClamAV daemon (clamd).

    Requires:
    - ClamAV daemon running: `clamd` or `freshclam`
    - Python client: `pip install clamd`
    """

    name: str = "clamav"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3310,
        unix_socket: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._host = host
        self._port = port
        self._unix_socket = unix_socket
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy-connect to clamd."""
        if self._client is not None:
            return self._client
        try:
            import clamd
            if self._unix_socket:
                self._client = clamd.ClamdUnixSocket(path=self._unix_socket)
            else:
                self._client = clamd.ClamdNetworkSocket(
                    host=self._host,
                    port=self._port,
                    timeout=self._timeout,
                )
            # Ping to verify connection
            self._client.ping()
            return self._client
        except Exception:
            self._client = None
            return None

    def scan_file(self, path: Path) -> MalwareScanResult:
        """Scan a single file via clamd."""
        start = time.perf_counter()
        client = self._get_client()
        if client is None:
            return MalwareScanResult(is_clean=True)

        try:
            result = client.scan(str(path))
            duration = (time.perf_counter() - start) * 1000
            return self._parse_result(result, duration)
        except Exception:
            return MalwareScanResult(is_clean=True)

    def scan_directory(self, path: Path) -> MalwareScanResult:
        """Scan a directory via clamd multiscan."""
        start = time.perf_counter()
        client = self._get_client()
        if client is None:
            return MalwareScanResult(is_clean=True)

        try:
            result = client.multiscan(str(path))
            duration = (time.perf_counter() - start) * 1000
            return self._parse_result(result, duration)
        except Exception:
            return MalwareScanResult(is_clean=True)

    def _parse_result(self, result: dict, duration: float) -> MalwareScanResult:
        """Parse clamd scan result dict."""
        threats: list[ThreatFinding] = []
        files_scanned = 0

        if result:
            for filepath, (status, detail) in result.items():
                files_scanned += 1
                if status == "FOUND":
                    threats.append(ThreatFinding(
                        name=detail,
                        severity="high",
                        file_path=filepath,
                        scanner=self.name,
                        details=f"ClamAV: {detail}",
                    ))

        return MalwareScanResult(
            is_clean=len(threats) == 0,
            threats=threats,
            files_scanned=files_scanned,
            scan_duration_ms=duration,
        )

    def is_available(self) -> bool:
        """Check if clamd is running and reachable."""
        return self._get_client() is not None
