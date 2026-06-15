"""Docs browser helpers for the local UI/dashboard.

Lightweight, read-only enumeration and reading of the educational markdown guides
under ``docs/`` (numbered guides ``01-*.md``..``06-*.md`` plus other top-level
``*.md`` files and the ``deep-dives/`` subdirectory, recursed one level). The UI
renders a "Docs Browser" view so the user can read the guides without leaving the
dashboard.

All functions are robust to missing or malformed inputs and never raise — they
return empty results or an ``{"error": ...}`` dict instead. ``doc_content`` is a
strict reader: it never opens for writing, never executes anything, and is
**path-guarded** — the resolved real path must live inside ``docs/`` or the read
is rejected (traversal like ``../`` cannot escape the docs tree).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory (relative to ``root``) holding the documentation.
_DOCS_DIRNAME = "docs"

# Read one level into this subdirectory of ``docs/`` (the deep-dive essays).
_DEEP_DIVES_DIRNAME = "deep-dives"

# Only markdown files are surfaced.
_MD_SUFFIX = ".md"

# Cap returned content so a huge file (e.g. research-log.md ~160KB) can't blow up
# the response. Bytes are counted on the decoded text length.
_MAX_CONTENT_CHARS = 200_000


def docs_list(root: str = ".") -> dict:
    """Enumerate markdown docs under ``<root>/docs`` (recursing into ``deep-dives/``).

    Returns::

        {"docs": [ {"name": str, "path": str, "rel": str, "title": str,
                    "size_bytes": int} ],
         "count": int}

    ``path`` is repo-relative with forward slashes (e.g. ``docs/03_training.md``),
    ``rel`` is the display path relative to ``docs/`` (e.g. ``03_training.md`` or
    ``deep-dives/mixture-of-experts.md``), and ``title`` is derived from the first
    ``# `` heading (falling back to the filename). Sorted by ``rel`` so the
    numbered guides come first, then the ``deep-dives/`` entries.

    On any failure returns ``{"error": "..."}``. Never raises.
    """
    try:
        docs_dir = Path(root) / _DOCS_DIRNAME
        if not docs_dir.is_dir():
            return {"docs": [], "count": 0}

        entries: list[dict] = []

        # Top-level *.md files.
        entries.extend(_collect_dir(docs_dir, docs_dir, recurse_name=_DEEP_DIVES_DIRNAME))

        entries.sort(key=lambda entry: entry["rel"])
        return {"docs": entries, "count": len(entries)}
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        logger.warning("docs_list failed: %s", exc)
        return {"error": str(exc)}


def doc_content(path: str, root: str = ".") -> dict:
    """Return the UTF-8 (errors="replace") text of a markdown doc under ``docs/``.

    ``path`` is repo-relative (as produced by :func:`docs_list`, e.g.
    ``docs/03_training.md``). The path is **guarded**: the resolved real path must
    be a file inside the ``docs/`` directory. Traversal (``../``), absolute paths
    pointing elsewhere, and non-``.md`` files are rejected.

    Returns ``{"path": str, "content": str, "truncated": bool}`` on success, where
    ``truncated`` is ``True`` if the file was longer than the content cap.

    On any failure (out-of-tree, missing, not markdown) returns
    ``{"error": str}``. Never raises. Read-only: the file is never opened for
    writing and nothing is executed.
    """
    try:
        docs_dir = (Path(root) / _DOCS_DIRNAME).resolve()
        target = (Path(root) / path).resolve()

        # Path guard: the resolved target must live inside docs/.
        if not _is_within(target, docs_dir):
            return {"error": f"path is outside docs/: {path}"}

        if target.suffix.lower() != _MD_SUFFIX:
            return {"error": f"not a markdown doc: {path}"}

        if not target.is_file():
            return {"error": f"doc not found: {path}"}

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": str(exc)}

        truncated = len(text) > _MAX_CONTENT_CHARS
        content = text[:_MAX_CONTENT_CHARS] if truncated else text

        return {
            "path": _rel_to_root(target, Path(root)),
            "content": content,
            "truncated": truncated,
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        logger.warning("doc_content failed for %r: %s", path, exc)
        return {"error": str(exc)}


# ── Internals ────────────────────────────────────────────────────────────────


def _collect_dir(directory: Path, docs_dir: Path, recurse_name: str | None) -> list[dict]:
    """Collect ``*.md`` entries directly under ``directory``.

    If ``recurse_name`` is given and a subdirectory of that name exists, also
    collect its ``*.md`` files (one level deep). ``docs_dir`` is the root used to
    compute the display ``rel`` path.
    """
    entries: list[dict] = []
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return entries

    for child in children:
        if child.is_file() and child.suffix.lower() == _MD_SUFFIX:
            entry = _build_entry(child, docs_dir)
            if entry is not None:
                entries.append(entry)
        elif child.is_dir() and recurse_name is not None and child.name == recurse_name:
            # Recurse exactly one level into the named subdirectory.
            entries.extend(_collect_dir(child, docs_dir, recurse_name=None))

    return entries


def _build_entry(file_path: Path, docs_dir: Path) -> dict | None:
    """Build a listing entry for ``file_path`` or ``None`` if it cannot be stat'd."""
    try:
        size_bytes = file_path.stat().st_size
    except OSError:
        return None

    rel = file_path.relative_to(docs_dir).as_posix()
    repo_rel = f"{_DOCS_DIRNAME}/{rel}"
    title = _extract_title(file_path, fallback=file_path.name)

    return {
        "name": file_path.name,
        "path": repo_rel,
        "rel": rel,
        "title": title,
        "size_bytes": size_bytes,
    }


def _extract_title(file_path: Path, fallback: str) -> str:
    """Return the text of the first ``# `` heading, or ``fallback`` if none found.

    Reads only the first chunk of the file so a large doc isn't fully loaded just
    to find its title.
    """
    try:
        with file_path.open(encoding="utf-8", errors="replace") as handle:
            for _ in range(200):
                line = handle.readline()
                if line == "":
                    break
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip() or fallback
    except OSError:
        return fallback
    return fallback


def _is_within(target: Path, parent: Path) -> bool:
    """Return True if ``target`` is ``parent`` or lives inside it (both resolved)."""
    if target == parent:
        return True
    return parent in target.parents


def _rel_to_root(target: Path, root: Path) -> str:
    """Return ``target`` as a forward-slash path relative to ``root`` when possible."""
    try:
        return target.relative_to(root.resolve()).as_posix()
    except ValueError:
        return target.as_posix()
