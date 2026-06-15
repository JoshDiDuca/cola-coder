"""Security / malware-scan endpoint helper for the local UI.

Mirrors the CLI ``data_menu`` "Scan Data for Malware" action: it builds a
:class:`~cola_coder.security.scanner.CompositeMalwareScanner` (auto-detecting the
available backends — Microsoft Defender, YARA, ClamAV) and runs it over a path.

This view is deliberately *bounded and synchronous*: it scans at most
``max_files`` files (default 500) so a UI request can never wedge on a giant
tree. There is no background-job launcher here — a summary is enough for v1.

Robust to missing scanner dependencies and bad paths: returns an
``{"error": ...}`` dict, never raises.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILES = 500


def _collect_files(root: Path, max_files: int) -> list[Path]:
    """Return up to ``max_files`` regular files under ``root`` (or ``root`` itself)."""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            files.append(candidate)
            if len(files) >= max_files:
                break
    return files


def scan_summary(path: str, max_files: int = _DEFAULT_MAX_FILES) -> dict:
    """Run the composite malware scanner over ``path`` and summarize the result.

    The scan is bounded to the first ``max_files`` files (default 500) found
    under ``path`` to keep the request responsive; ``path`` may also point at a
    single file. Returns a payload matching ``schemas.MalwareScanResult`` on
    success, or ``{"error": ...}`` on a bad path / missing scanner backends.
    Never raises.
    """
    target = Path(path)
    if not target.exists():
        return {"error": f"path not found: {path}"}
    if max_files <= 0:
        return {"error": f"max_files must be positive, got {max_files}"}

    try:
        from cola_coder.security.scanner import CompositeMalwareScanner
    except ImportError as exc:
        return {"error": f"security scanner unavailable: {exc}"}

    scanner = CompositeMalwareScanner.from_config()
    if not scanner.available_scanners:
        return {
            "error": (
                "no malware scanner available — install yara-python, or enable "
                "Microsoft Defender / ClamAV"
            )
        }

    files = _collect_files(target, max_files)
    if not files:
        return {"error": f"no files to scan under: {path}"}

    start = time.perf_counter()
    try:
        result = scanner.scan_file(files[0])
        for file_path in files[1:]:
            result = result.merge(scanner.scan_file(file_path))
    except Exception as exc:  # defensive: scanner backends run external tools
        logger.warning("Malware scan failed on %s: %s", path, exc)
        return {"error": f"scan failed: {exc}"}
    duration_ms = (time.perf_counter() - start) * 1000.0

    threats = [
        {
            "file_path": t.file_path,
            "name": t.name,
            "severity": t.severity,
            "scanner": t.scanner,
            "details": t.details or None,
        }
        for t in result.threats
    ]
    return {
        "path": str(target),
        "files_scanned": len(files),
        "is_clean": result.is_clean and not result.had_errors,
        "threats": threats,
        "duration_ms": duration_ms,
    }
