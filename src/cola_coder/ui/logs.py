"""Log/artifact browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of training/job log files so the dashboard can
"see everything" without the user shelling in. Covers top-level ``*.log`` /
``*.err`` files plus any files inside a ``ui_jobs/`` job-log directory.

All functions are robust to missing or malformed inputs and never raise — they
return empty results or an ``{"error": ...}`` dict instead. ``tail_log`` is a
strict reader: it never opens for writing and never executes anything.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _split_lines(text: str) -> list[str]:
    """Split on both carriage returns and newlines (tqdm uses \\r).

    Mirrors ``status._split_lines`` so tqdm progress bars in ``.err`` files
    render as separate lines rather than one giant carriage-return blob.
    """
    return [ln for ln in re.split(r"[\r\n]+", text) if ln != ""]


def list_logs(root: str = ".") -> list[dict]:
    """List candidate log files under ``root``.

    Includes top-level ``*.log`` and ``*.err`` files, plus any files inside a
    ``ui_jobs/`` (job logs) directory if present. Newest-first by mtime.

    Each entry: {"name": str, "path": str, "size_bytes": int, "mtime": float}.

    Never raises. Returns ``[]`` on a missing ``root``.
    """
    if not os.path.isdir(root):
        return []

    results: list[dict] = []

    # Top-level *.log and *.err files.
    try:
        top_entries = os.listdir(root)
    except OSError:
        top_entries = []

    for name in top_entries:
        if not (name.endswith(".log") or name.endswith(".err")):
            continue
        path = os.path.join(root, name)
        entry = _stat_entry(name, path)
        if entry is not None:
            results.append(entry)

    # Any files inside a ui_jobs/ job-log directory.
    jobs_dir = os.path.join(root, "ui_jobs")
    if os.path.isdir(jobs_dir):
        for dirpath, _dirnames, filenames in os.walk(jobs_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                entry = _stat_entry(name, path)
                if entry is not None:
                    results.append(entry)

    results.sort(key=lambda entry: entry["mtime"], reverse=True)
    return results


def _stat_entry(name: str, path: str) -> dict | None:
    """Build a listing entry for ``path`` or ``None`` if it cannot be stat'd."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {
        "name": name,
        "path": path,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def tail_log(path: str, lines: int = 200) -> dict:
    """Return the last ``lines`` lines of a UTF-8 (errors="replace") text file.

    Returns {"path": str, "lines": list[str], "size_bytes": int,
    "truncated": bool} where ``truncated`` is ``True`` if the file had more
    lines than returned.

    tqdm progress bars use carriage returns (\\r) not newlines, so the text is
    split on both \\r and \\n (mirroring ``status._split_lines``) — that way
    ``.err`` progress lines render as separate lines.

    On any failure returns {"error": str}. Never raises. Read-only: the file is
    never opened for writing and nothing is executed.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    try:
        stat = os.stat(path)
        size_bytes = stat.st_size
    except OSError as exc:
        return {"error": str(exc)}

    try:
        # MB-scale logs: reading the whole file then slicing is acceptable.
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    all_lines = _split_lines(text)
    n = max(0, lines)
    truncated = len(all_lines) > n
    tail = all_lines[-n:] if n > 0 else []

    return {
        "path": path,
        "lines": tail,
        "size_bytes": size_bytes,
        "truncated": truncated,
    }
