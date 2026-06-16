"""Checkpoint notes/tags endpoint helpers for the local UI (UI-100).

Lets the user annotate checkpoints with a short label + free-text note (e.g.
"best so far", "before reasoning warmup") from the web UI. Notes are persisted
to a sidecar ``<root>/.cola/checkpoint_notes.json`` keyed by the checkpoint's
path — DELIBERATELY OUTSIDE the ``checkpoints/`` tree, so this NEVER writes into
a checkpoint directory and cannot interfere with the live trainer (which owns
and writes ``checkpoints/<run>/step_*``). Pure JSON file I/O — no model, no GPU.

All functions are defensive: they return ``{"error": str}`` instead of raising,
and a missing/malformed sidecar yields an empty-but-valid notes view.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Sidecar location — under .cola/, never under checkpoints/.
_NOTES_REL = (".cola", "checkpoint_notes.json")
_MAX_LABEL = 80
_MAX_NOTE = 2000


def _notes_path(root: str) -> Path:
    return Path(root).joinpath(*_NOTES_REL)


def _load(root: str) -> dict[str, dict[str, str]]:
    """Load the raw {key: {label, note, updated_at}} map, tolerating absence."""
    path = _notes_path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    # Keep only well-formed entries (defensive against hand edits).
    clean: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            clean[str(key)] = {
                "label": str(value.get("label", "")),
                "note": str(value.get("note", "")),
                "updated_at": str(value.get("updated_at", "")),
            }
    return clean


def _view(raw: dict[str, dict[str, str]]) -> dict:
    """Shape the raw map into the ``CheckpointNotes`` response (sorted by key)."""
    notes = [
        {
            "key": key,
            "label": entry.get("label", ""),
            "note": entry.get("note", ""),
            "updated_at": entry.get("updated_at", ""),
        }
        for key, entry in sorted(raw.items())
    ]
    return {"notes": notes}


def _atomic_write(path: Path, raw: dict[str, dict[str, str]]) -> None:
    """Atomically write the notes map (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(raw, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def checkpoint_notes(root: str = ".") -> dict:
    """Return all checkpoint notes (``{"notes": [...]}``). Never raises."""
    try:
        return _view(_load(root))
    except ValueError as exc:
        return {"error": str(exc)}


def set_checkpoint_note(
    root: str,
    key: str,
    label: str = "",
    note: str = "",
    now: str | None = None,
) -> dict:
    """Upsert a note for checkpoint ``key`` (its path); returns the refreshed view.

    ``key`` is required; ``label``/``note`` are both optional but at least one
    must be non-empty (an all-empty note is a delete — use that explicitly).
    ``now`` overrides the ``updated_at`` stamp (for deterministic tests). Returns
    ``{"error": str}`` on bad input/IO. Never raises.
    """
    key = key.strip()
    label = label.strip()[:_MAX_LABEL]
    note = note.strip()[:_MAX_NOTE]
    if not key:
        return {"error": "key is required"}
    if not label and not note:
        return {"error": "provide a label or a note (empty clears via delete)"}
    stamp = now if now is not None else datetime.datetime.now().isoformat(timespec="seconds")

    try:
        raw = _load(root)
        raw[key] = {"label": label, "note": note, "updated_at": stamp}
        _atomic_write(_notes_path(root), raw)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}
    return _view(raw)


def delete_checkpoint_note(root: str, key: str) -> dict:
    """Remove the note for ``key``; returns the refreshed view. Never raises."""
    key = key.strip()
    if not key:
        return {"error": "key is required"}
    try:
        raw = _load(root)
        if key not in raw:
            return {"error": f"no note for key: {key}"}
        del raw[key]
        _atomic_write(_notes_path(root), raw)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}
    return _view(raw)
