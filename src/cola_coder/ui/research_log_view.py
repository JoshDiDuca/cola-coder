"""Research-log viewer helper for the local cola-coder UI.

Pure library module (no Rich, no CLI) that parses ``docs/research-log.md`` — the
living log the autonomous improvement loop appends to each cycle — into a list of
structured entries the web UI can render.

The log format (newest first) is a sequence of entries delimited by a header::

    ## YYYY-MM-DD — <Title> (rotate: <area>)

    **Sources (...):** ...one or more http(s) URLs...

    **Summary:** ...

    **Original idea ...:** ...

Older entries use minor variants (``Sources:`` / ``Findings:`` instead of the
bold ``**Sources (...):**`` / ``**Summary:**`` markers, and
``**ORIGINAL cross-technique idea ...**`` for the original-idea marker), so the
parser matches each field loosely and never assumes one fixed layout. Entries are
split on the date header (not the ``---`` rule) because some entries omit the
trailing separator.

Per entry the parser extracts: ``date``, ``title`` (with the ``(rotate: ...)``
suffix stripped), ``area`` (the ``rotate:`` value or ``None``), ``source_count``
(distinct http(s) URLs in the entry body), ``has_original_idea`` (whether a bold
"original idea" marker appears) and a one-line ``summary`` (first ~240 chars of
the Summary/Findings section, or the first body paragraph as a fallback).

Best-effort and never raises: on any failure it returns ``{"error": ...}``; a
missing log file is NOT an error (an empty entry list is returned).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Relative location of the research log under the project root.
_RESEARCH_LOG_REL = ("docs", "research-log.md")

# Max characters of the summary section kept (collapsed to one line).
_SUMMARY_MAX_CHARS = 240

# Entry header: "## 2026-06-15 — <title>" (em dash or hyphen between date/title).
_ENTRY_HEADER = re.compile(
    r"^##\s+(?P<date>\d{4}-\d{2}-\d{2})\s*[—–-]\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

# "(rotate: <area>)" — the trailing rotation tag in a title.
_ROTATE = re.compile(r"\(\s*rotate:\s*(?P<area>[^)]+?)\s*\)", re.IGNORECASE)

# Any http(s) URL — counted for source_count.
_URL = re.compile(r"https?://\S+")

# The "Summary" / "Findings" section marker (bold or plain, line-leading).
_SUMMARY_MARKER = re.compile(
    r"^\s*\*{0,2}(?:Summary|Findings)\s*:?\*{0,2}\s*", re.IGNORECASE
)

# A bold "Original idea" / "Original cross-technique idea" marker anywhere.
_ORIGINAL_IDEA = re.compile(r"\*\*\s*original[^*]*idea", re.IGNORECASE)


def _strip_url_trailing_punct(url: str) -> str:
    """Drop trailing markdown/sentence punctuation glued onto a bare URL."""
    return url.rstrip(".,);]")


def _count_sources(body: str) -> int:
    """Count distinct http(s) URLs appearing in an entry body."""
    seen: set[str] = set()
    for match in _URL.finditer(body):
        seen.add(_strip_url_trailing_punct(match.group(0)))
    return len(seen)


def _extract_area(title: str) -> tuple[str, str | None]:
    """Return ``(clean_title, area)`` splitting off any ``(rotate: ...)`` tag."""
    match = _ROTATE.search(title)
    if match is None:
        return title.strip(), None
    area = match.group("area").strip()
    clean = _ROTATE.sub("", title).strip()
    # Tidy up any trailing connective left after removing the tag (e.g. "+ ").
    clean = re.sub(r"[\s+]+$", "", clean).strip()
    return clean, (area or None)


def _collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (incl. newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_summary(body: str) -> str:
    """Return a one-line summary: the Summary/Findings section, else first para.

    The Summary section runs from its marker to the next blank line; if no marker
    is present (some older entries), the first non-empty body paragraph is used.
    The result is collapsed to one line and clipped to ``_SUMMARY_MAX_CHARS``.
    """
    lines = body.splitlines()

    summary_lines: list[str] = []
    capturing = False
    for line in lines:
        if not capturing:
            marker = _SUMMARY_MARKER.match(line)
            if marker is not None:
                capturing = True
                remainder = line[marker.end():].strip()
                if remainder:
                    summary_lines.append(remainder)
            continue
        if line.strip() == "":
            break
        summary_lines.append(line.strip())

    if not summary_lines:
        summary_lines = _first_paragraph(lines)

    collapsed = _collapse_whitespace(" ".join(summary_lines))
    if len(collapsed) > _SUMMARY_MAX_CHARS:
        collapsed = collapsed[:_SUMMARY_MAX_CHARS].rstrip() + "…"
    return collapsed


def _first_paragraph(lines: list[str]) -> list[str]:
    """First non-empty run of body lines, skipping a leading Sources block."""
    paragraph: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped == "":
                continue
            # Skip an opening Sources block / bullet list — not a summary.
            lowered = stripped.lower()
            if lowered.startswith(("sources", "**sources", "- ", "* ")):
                continue
            started = True
            paragraph.append(stripped)
        else:
            if stripped == "":
                break
            paragraph.append(stripped)
    return paragraph


def _parse_entry(date: str, raw_title: str, body: str) -> dict:
    """Build one structured entry record from a header + body slice."""
    title, area = _extract_area(raw_title)
    return {
        "date": date,
        "title": title,
        "area": area,
        "source_count": _count_sources(body),
        "has_original_idea": _ORIGINAL_IDEA.search(body) is not None,
        "summary": _extract_summary(body),
    }


def _parse_entries(text: str) -> list[dict]:
    """Split the log on date headers and parse each entry (newest first)."""
    matches = list(_ENTRY_HEADER.finditer(text))
    entries: list[dict] = []
    for idx, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        entries.append(_parse_entry(match.group("date"), match.group("title"), body))
    return entries


def research_log(root: str = ".") -> dict:
    """Parse ``docs/research-log.md`` into structured entries.

    Returns::

        {"entries": [ {"date", "title", "area", "source_count",
                       "has_original_idea", "summary"} ],  # newest first
         "count": int}

    Entries preserve the file's order (the log is maintained newest-first). A
    missing log file yields an empty list (not an error). On any unexpected
    failure returns ``{"error": "..."}``. Never raises.
    """
    try:
        log_path = Path(root).joinpath(*_RESEARCH_LOG_REL)
        if not log_path.is_file():
            return {"entries": [], "count": 0}

        text = log_path.read_text(encoding="utf-8", errors="replace")
        entries = _parse_entries(text)
        return {"entries": entries, "count": len(entries)}
    except Exception as exc:  # noqa: BLE001 — contract: never raise, return error dict
        logger.warning("research_log parse failed: %s", exc)
        return {"error": str(exc)}
