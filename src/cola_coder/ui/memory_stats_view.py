"""Project-memory-inspector endpoint helper for the local UI.

Read-only mirror of the CLI "Project Memory > stats / view" actions
(:mod:`cola_coder.features.menus.tools_menu`). Reads the REAL memory store —
:class:`cola_coder.memory.manager.MemoryManager`, which persists project
knowledge as markdown files under ``<project_root>/.cola/memory/`` (one file
per category: project, patterns, errors, decisions, domain_knowledge,
session_log). Each ``## `` heading inside a file is one logical entry; entries
carry an embedded ``_Added: YYYY-MM-DD HH:MM_`` timestamp.

This view performs NO mutations (no compaction, no writes). It maps the store's
real fields onto the UI model:

- ``type``    -> the memory file's category key (e.g. ``"errors"``); the store
  groups entries by file, not by a per-entry type field.
- ``pinned``  -> always ``0``: the markdown store has no notion of pinned
  entries, so this is reported as a constant.
- ``created_at`` -> the ``_Added:`` timestamp parsed from the entry body, or
  ``""`` when absent.
- ``size_bytes`` -> total bytes of all existing memory files on disk.

Robust to an uninitialised or empty store: returns a valid empty-stats result
(zeros / empty lists), never an ``{"error": ...}`` and never raises.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cola_coder.memory.manager import MemoryManager, _iter_sections

logger = logging.getLogger(__name__)

_ADDED_RE = re.compile(r"_Added:\s*(.+?)_")
_PREVIEW_LEN = 120
_RECENT_SAMPLE_MAX = 10


def _extract_added(body: str) -> str:
    """Return the entry's ``_Added:`` timestamp, or ``""`` when absent."""
    match = _ADDED_RE.search(body)
    return match.group(1).strip() if match else ""


def _make_preview(title: str, body: str) -> str:
    """Build a single-line, ~120-char preview from a section title + body."""
    # Strip the timestamp marker so it does not dominate the preview.
    cleaned = _ADDED_RE.sub("", body)
    combined = f"{title} — {cleaned}".strip(" —")
    single_line = " ".join(combined.split())
    if len(single_line) > _PREVIEW_LEN:
        return single_line[: _PREVIEW_LEN - 1].rstrip() + "…"
    return single_line


def memory_stats(project_root: str | None = None) -> dict:
    """Summarise the project memory store for the UI inspector.

    ``project_root`` defaults to the current working directory; the store is
    resolved to ``<project_root>/.cola/memory/`` by :class:`MemoryConfig`.
    Returns an empty-but-valid :class:`~cola_coder.ui.schemas.MemoryStats`
    shape when the store is missing or empty.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()

    empty: dict[str, object] = {
        "total_entries": 0,
        "pinned": 0,
        "types": [],
        "size_bytes": 0,
        "oldest_at": None,
        "newest_at": None,
        "recent_sample": [],
    }

    try:
        manager = MemoryManager(root)
    except Exception:  # defensive: config construction should not raise
        logger.exception("Failed to construct MemoryManager for %s", root)
        return empty

    if not manager.is_initialized:
        return empty

    total_entries = 0
    present_types: list[str] = []
    timestamps: list[str] = []
    # (created_at, MemoryEntry-dict) so we can sort newest-first by timestamp.
    collected: list[tuple[str, dict[str, str]]] = []
    size_bytes = 0

    for type_key, filename in manager.config.files.items():
        file_path = manager.memory_path / filename
        if not file_path.exists():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            size_bytes += file_path.stat().st_size
        except OSError:
            logger.warning("Could not read memory file %s", file_path)
            continue

        file_entry_count = 0
        for title, body in _iter_sections(content):
            file_entry_count += 1
            created_at = _extract_added(body)
            if created_at:
                timestamps.append(created_at)
            entry_id = f"{type_key}:{title}"
            collected.append(
                (
                    created_at,
                    {
                        "id": entry_id[:200],
                        "type": type_key,
                        "created_at": created_at,
                        "content_preview": _make_preview(title, body),
                    },
                )
            )

        if file_entry_count > 0:
            present_types.append(type_key)
            total_entries += file_entry_count

    # ISO-style ("YYYY-MM-DD HH:MM") timestamps sort correctly as strings.
    sorted_ts = sorted(t for t in timestamps if t)
    oldest_at = sorted_ts[0] if sorted_ts else None
    newest_at = sorted_ts[-1] if sorted_ts else None

    # Most-recent first; entries with no timestamp sort last (empty string).
    collected.sort(key=lambda item: item[0], reverse=True)
    recent_sample = [entry for _, entry in collected[:_RECENT_SAMPLE_MAX]]

    return {
        "total_entries": total_entries,
        "pinned": 0,  # markdown store has no pinned concept — constant.
        "types": present_types,
        "size_bytes": size_bytes,
        "oldest_at": oldest_at,
        "newest_at": newest_at,
        "recent_sample": recent_sample,
    }
