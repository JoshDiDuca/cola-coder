"""Project-memory VIEW + WRITE endpoint helpers for the local UI (UI-095).

Turns the read-only memory inspector (:mod:`cola_coder.ui.memory_stats_view`)
into a full workbench by wrapping the mutating/searching side of
:class:`cola_coder.memory.manager.MemoryManager`:

- ``memory_export``  — full markdown content per theme file (for a read view).
- ``memory_add``     — append one entry (pattern/error/decision/domain/session).
- ``memory_search``  — TF-IDF retrieval over the store (CPU only, no model).
- ``memory_compact`` — drop duplicate entries; report what was removed.

ALL operations are MAIN-SAFE: pure markdown file I/O + CPU TF-IDF. Nothing here
imports torch, loads a checkpoint, or touches the GPU / the live trainer. Every
function is defensive — it returns ``{"error": str}`` instead of raising.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cola_coder.memory.manager import MemoryManager, _iter_sections

from .memory_stats_view import memory_stats

logger = logging.getLogger(__name__)

# Per-file content cap for transport (the store is small, but bound it anyway).
_MAX_FILE_CHARS = 20000
# Per-chunk content cap for a search hit.
_MAX_CHUNK_CHARS = 1000
_SEARCH_MAX_CHUNKS = 20

# UI ``kind`` -> the MemoryManager add method it maps to. ``domain`` maps to
# ``add_domain_knowledge`` (topic, content); ``session`` to ``log_session``
# (summary, domain). The two fields are (primary, secondary) in every case.
_ADD_KINDS = ("pattern", "error", "decision", "domain", "session")


def _manager(project_root: str | None) -> MemoryManager:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return MemoryManager(root)


def memory_export(project_root: str | None = None) -> dict:
    """Return the full markdown content of every existing memory file.

    ``{"initialized": bool, "files": [{type, name, content, truncated,
    entry_count}]}``. An uninitialised store yields ``initialized=False`` with an
    empty file list. Never raises.
    """
    try:
        manager = _manager(project_root)
    except Exception:  # defensive — config construction should not raise
        logger.exception("Failed to construct MemoryManager")
        return {"initialized": False, "files": []}

    if not manager.is_initialized:
        return {"initialized": False, "files": []}

    files: list[dict[str, object]] = []
    for type_key, filename in manager.config.files.items():
        file_path = manager.memory_path / filename
        if not file_path.exists():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read memory file %s", file_path)
            continue
        entry_count = sum(1 for _ in _iter_sections(content))
        files.append(
            {
                "type": type_key,
                "name": filename,
                "content": content[:_MAX_FILE_CHARS],
                "truncated": len(content) > _MAX_FILE_CHARS,
                "entry_count": entry_count,
            }
        )
    return {"initialized": True, "files": files}


def memory_add(
    project_root: str | None,
    kind: str,
    primary: str,
    secondary: str = "",
) -> dict:
    """Append one entry to the store, then return the refreshed memory stats.

    ``kind`` selects the theme: pattern/error/decision/domain/session. ``primary``
    is the main text (pattern / error / decision / topic / summary); ``secondary``
    is the optional second field (example / fix / rationale / content / domain).
    The store is auto-initialised on first write so the UI "just works". Returns
    ``{"error": str}`` for an unknown kind or empty primary. Never raises.
    """
    if kind not in _ADD_KINDS:
        return {"error": f"unknown kind: {kind!r} (expected one of {_ADD_KINDS})"}
    primary = primary.strip()
    secondary = secondary.strip()
    if not primary:
        return {"error": "primary content is required"}

    try:
        manager = _manager(project_root)
        if not manager.is_initialized:
            manager.init_project()
        if kind == "pattern":
            manager.add_pattern(primary, secondary)
        elif kind == "error":
            manager.add_error(primary, secondary)
        elif kind == "decision":
            manager.add_decision(primary, secondary)
        elif kind == "domain":
            manager.add_domain_knowledge(primary, secondary)
        elif kind == "session":
            manager.log_session(primary, secondary)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    return memory_stats(project_root)


def memory_search(
    project_root: str | None,
    query: str,
    max_chunks: int = 5,
) -> dict:
    """TF-IDF search the memory store (CPU only). No model/GPU is loaded.

    ``{"query": str, "results": [{content, source_file, section,
    relevance_score}]}``. Empty query returns ``{"error": ...}``; an empty or
    uninitialised store returns an empty result list. Never raises.
    """
    query = query.strip()
    if not query:
        return {"error": "query is required"}
    bounded = max(1, min(max_chunks, _SEARCH_MAX_CHUNKS))

    try:
        manager = _manager(project_root)
        chunks = manager.retrieve(query, max_chunks=bounded)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    results = [
        {
            "content": chunk.content[:_MAX_CHUNK_CHARS],
            "source_file": chunk.source_file,
            "section": chunk.section,
            "relevance_score": round(float(chunk.relevance_score), 4),
        }
        for chunk in chunks
    ]
    return {"query": query, "results": results}


def memory_compact(project_root: str | None = None) -> dict:
    """Drop duplicate entries and report what was removed per file.

    ``{"removed_total": int, "removed": [{name, removed}]}``. An uninitialised
    store returns ``{"error": ...}``. Never raises.
    """
    try:
        manager = _manager(project_root)
        if not manager.is_initialized:
            return {"error": "memory not initialized — add an entry first"}
        removed = manager.compact()
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    removed_list = [{"name": name, "removed": count} for name, count in removed.items()]
    return {"removed_total": sum(removed.values()), "removed": removed_list}
