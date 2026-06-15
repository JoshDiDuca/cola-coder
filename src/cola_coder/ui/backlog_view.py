"""Backlog viewer helpers for the local UI/dashboard.

Lightweight, read-only parser for the autonomous-improvement-loop backlog at the
project root (``ai_backlog.md``), so the UI can render the loop's tracked work as
a structured, filterable table (IDs, category, severity, status, date,
description) instead of forcing the user to read raw markdown.

Backlog item format (one bullet per item, description may span indented lines)::

    - **UI-046..047** [ui, medium] `done` (2026-06-15) — <description first line...>
      <continued description on indented following lines...>

Items may carry an ID *range* (``UI-046..047``), varied categories
(``ui``/``bug``/``ops``/``data``/``model``/``eval``/``typing``/``post-training`` ...),
varied severities, and statuses (``open``/``in-progress``/``done``/``dropped``).
The ``[cat, sev]`` bracket is split on the FIRST comma — the category is the head,
the severity is the tail (which itself may contain commas/qualifiers, kept verbatim).

This module only *reads* the markdown — it never mutates it. All functions are
robust to a missing or malformed file and never raise; they return an
``{"error": ...}`` dict (or an empty backlog when the file is simply absent).

Returned shape (mirrors ``schemas.BacklogView`` / ``schemas.BacklogItem``)::

    {"items": [ {"id": str, "category": str, "severity": str, "status": str,
                 "date": str | None, "description": str} ],
     "count": int,
     "open_count": int,
     "done_count": int}
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Backlog markdown file, relative to the project root.
_BACKLOG_FILENAME = "ai_backlog.md"

# Max length (chars) for a single-line description before truncation.
_MAX_DESCRIPTION = 200

# Recognised status values; anything else collapses to ``"unknown"``.
_KNOWN_STATUSES: frozenset[str] = frozenset(
    {"open", "in-progress", "done", "dropped"}
)

# An item bullet header. Anatomy of a match:
#   - **<id>**            → group "id"   (the bold token, e.g. ``UI-046..047``)
#   [<cat>, <sev>]        → group "meta" (split on the first comma downstream)
#   `<status>`            → group "status"
#   (YYYY-MM-DD)          → group "date" (optional)
#   — <description>       → group "desc" (optional first line; em-dash OR hyphen)
_ITEM_RE = re.compile(
    r"^-\s+\*\*(?P<id>[^*]+?)\*\*"          # - **ID**
    r"\s*\[(?P<meta>[^\]]*)\]"              # [cat, sev]
    r"\s*`(?P<status>[^`]*)`"               # `status`
    r"(?:\s*\((?P<date>\d{4}-\d{2}-\d{2})\))?"  # optional (YYYY-MM-DD)
    r"(?:\s*[—-]\s*(?P<desc>.*))?$"         # optional — description
)

# Any other ``- `` list bullet (used to detect where an item's continuation ends).
_BULLET_RE = re.compile(r"^-\s+")

# A YYYY-MM-DD anywhere (fallback when the date isn't immediately after status).
_DATE_RE = re.compile(r"\((?P<date>\d{4}-\d{2}-\d{2})\)")

# Collapse any run of whitespace (incl. newlines) to a single space.
_WS_RE = re.compile(r"\s+")


def backlog(root: str = ".") -> dict:
    """Parse ``ai_backlog.md`` (under ``root``) into a structured backlog view.

    Returns the ``BacklogView`` dict — ``items`` plus ``count``/``open_count``/
    ``done_count`` rollups. If the file is missing, returns an empty backlog
    (not an error). On any genuine failure returns ``{"error": "..."}``. Never
    raises.
    """
    try:
        path = Path(root) / _BACKLOG_FILENAME
        if not path.is_file():
            return _empty()

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("could not read backlog %s: %s", path, exc)
            return {"error": str(exc)}

        items = _parse(raw)
        open_count = sum(1 for item in items if item["status"] == "open")
        done_count = sum(1 for item in items if item["status"] == "done")

        return {
            "items": items,
            "count": len(items),
            "open_count": open_count,
            "done_count": done_count,
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        logger.exception("failed to parse backlog")
        return {"error": str(exc)}


# ── Internals ───────────────────────────────────────────────────────────────


def _empty() -> dict:
    """Return an empty backlog view (used when the file is absent)."""
    return {"items": [], "count": 0, "open_count": 0, "done_count": 0}


def _parse(raw: str) -> list[dict]:
    """Parse all item bullets from the markdown, gathering multi-line descriptions.

    Walks line by line: each line matching ``_ITEM_RE`` opens a new item whose
    description starts at the bullet's first line and absorbs subsequent indented
    continuation lines (until the next ``- `` bullet or a blank line). Non-item
    lines (headers, separators, the preamble) are skipped.
    """
    lines = raw.splitlines()
    items: list[dict] = []

    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        match = _ITEM_RE.match(line)
        if match is None:
            index += 1
            continue

        # Gather continuation lines: indented, non-blank, not a new bullet.
        desc_parts: list[str] = []
        first_desc = match.group("desc")
        if first_desc:
            desc_parts.append(first_desc)

        cursor = index + 1
        while cursor < total:
            follow = lines[cursor]
            if follow.strip() == "":
                break
            if not follow.startswith((" ", "\t")):
                break
            if _BULLET_RE.match(follow.strip()):
                break
            desc_parts.append(follow.strip())
            cursor += 1

        items.append(_build_item(match, desc_parts))
        index = cursor

    return items


def _build_item(match: re.Match[str], desc_parts: list[str]) -> dict:
    """Assemble a single ``BacklogItem`` dict from a header match + description lines."""
    category, severity = _split_meta(match.group("meta"))
    status = _normalize_status(match.group("status"))
    date = match.group("date")
    description = _collapse(" ".join(desc_parts))

    # Fallback: a date that lived inside the description rather than after status.
    if date is None and description:
        date_match = _DATE_RE.search(description)
        if date_match:
            date = date_match.group("date")

    return {
        "id": match.group("id").strip(),
        "category": category,
        "severity": severity,
        "status": status,
        "date": date,
        "description": description,
    }


def _split_meta(meta: str) -> tuple[str, str]:
    """Split a ``[cat, sev]`` bracket body into ``(category, severity)``.

    Splits on the FIRST comma only — the severity tail may itself contain commas
    or qualifiers (e.g. ``medium-potential``, ``research/WORKTREE``) and is kept
    verbatim. A bracket with no comma is treated as category-only.
    """
    head, _, tail = meta.partition(",")
    category = head.strip()
    severity = tail.strip()
    return category, severity


def _normalize_status(status: str) -> str:
    """Lowercase + trim a backticked status; collapse unknowns to ``"unknown"``."""
    cleaned = status.strip().lower()
    return cleaned if cleaned in _KNOWN_STATUSES else "unknown"


def _collapse(text: str) -> str:
    """Collapse whitespace to single spaces and truncate to ``_MAX_DESCRIPTION`` chars."""
    collapsed = _WS_RE.sub(" ", text).strip()
    if len(collapsed) > _MAX_DESCRIPTION:
        return collapsed[: _MAX_DESCRIPTION - 1].rstrip() + "…"
    return collapsed
