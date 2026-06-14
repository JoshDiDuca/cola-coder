"""Instruction/SFT and reasoning-problem dataset browsing helpers for the UI.

Lightweight, read-only inspection of the *text* JSONL datasets the project
produces — the instruction-tuning pairs and the reasoning coding-problem sets —
which the tokenized ``.npy`` browser in ``datasets.py`` does NOT cover. All
functions are robust to missing or malformed inputs and never raise on bad data:
they return empty results or an ``{"error": ...}`` dict instead.

The relevant JSONL shapes (discovered from the producing scripts):

- ``scripts/generate_instructions.py`` -> ``data/processed/instructions*.jsonl``
  in ChatML form ``{"messages": [{"role": "user", "content": ...},
  {"role": "assistant", "content": ...}]}`` (consumed by ``SFTDataset`` /
  ``scripts/train_sft.py``). Older/template rows may instead carry flat
  ``{"instruction", "output", ...}`` keys.
- reasoning problems (``ProblemSet.add_from_jsonl`` in
  ``cola_coder/evaluation/problem_loader.py``) -> rows with the required keys
  ``task_id``, ``prompt``, ``test_code``, ``entry_point`` plus optional
  ``difficulty``/``category``/``language``/``canonical_solution``; these
  conventionally live under ``data/reasoning/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Cap individual string field values in previews so the response stays light.
_MAX_FIELD_CHARS = 2000

# Directories that conventionally hold instruction/SFT/reasoning JSONL datasets.
_SCAN_DIRS = ("data/sft", "data/reasoning", "data/processed", "data")

# Required keys that identify a reasoning coding-problem row.
_REASONING_KEYS = {"task_id", "prompt", "test_code", "entry_point"}


def list_sft_files(root: str = ".") -> list[dict]:
    """Discover instruction/SFT/reasoning JSONL datasets under ``root``.

    Scans the conventional directories (``data/sft``, ``data/reasoning``,
    ``data/processed``, and the top of ``data``) for ``*.jsonl`` files. Each
    entry is a dict with keys::

        {"name": str, "path": str, "kind": str, "num_records": int,
         "size_bytes": int, "mtime": float}

    ``kind`` is one of ``"instructions"``, ``"reasoning_problems"``, ``"sft"``,
    ``"jsonl"``, inferred from the filename, its directory, and a sniff of the
    first row. ``num_records`` is the count of non-empty lines (cheap streaming —
    lines are not JSON-parsed). Results are newest-first by ``mtime``. Unreadable
    files are skipped. Missing directories are ignored. Returns ``[]`` when
    nothing is found. Never raises.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    seen: set[str] = set()
    results: list[dict] = []

    for found in _candidate_files(root_path):
        try:
            resolved = str(found.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        try:
            stat = found.stat()
        except OSError:
            continue

        try:
            rel = os.path.relpath(str(found), str(root_path))
        except (OSError, ValueError):
            rel = str(found)

        results.append(
            {
                "name": found.name,
                "path": rel,
                "kind": _classify(found),
                "num_records": _count_lines(found),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    results.sort(key=lambda entry: entry["mtime"], reverse=True)
    return results


def preview_sft(path: str, n: int = 10) -> dict:
    """Parse the first ``n`` rows of a JSONL instruction/reasoning dataset.

    Returns::

        {"path": str, "records": list[dict], "fields": list[str],
         "count": int, "truncated": bool}

    ``records`` holds up to ``n`` parsed rows; ``fields`` is the union of keys
    across those rows (sorted); ``count`` is the total non-empty line count;
    ``truncated`` is ``True`` when ``count > n``. Malformed JSON lines are
    skipped (they never crash the preview and never count toward ``n``). Any huge
    string field value is capped at ~2000 chars so the response stays light. On
    any failure (missing file, unreadable) returns ``{"error": str}``. Never
    raises.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {"error": f"path not found: {path}"}

    records: list[dict] = []
    fields: set[str] = set()
    count = 0

    try:
        with file_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if len(records) >= n:
                    continue
                try:
                    parsed = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(parsed, dict):
                    continue
                capped = _cap_record(parsed)
                records.append(capped)
                fields.update(capped.keys())
    except OSError as exc:
        return {"error": str(exc)}

    return {
        "path": path,
        "records": records,
        "fields": sorted(fields),
        "count": count,
        "truncated": count > n,
    }


# ── Internals ───────────────────────────────────────────────────────────────


def _candidate_files(root_path: Path) -> list[Path]:
    """Collect ``*.jsonl`` files from the conventional instruction/SFT dirs."""
    candidates: list[Path] = []
    for dirname in _SCAN_DIRS:
        scan_dir = root_path / dirname
        if not scan_dir.is_dir():
            continue
        try:
            for found in scan_dir.rglob("*.jsonl"):
                if found.is_file():
                    candidates.append(found)
        except OSError:
            continue
    return candidates


def _classify(path: Path) -> str:
    """Best-effort kind from filename + parent dir, falling back to a row sniff."""
    name = path.name.lower()
    parent = path.parent.name.lower()

    if "instruction" in name:
        return "instructions"
    if "reasoning" in name or "problem" in name or parent == "reasoning":
        return "reasoning_problems"
    if parent == "sft" or "sft" in name:
        return "sft"

    sniffed = _sniff_kind(path)
    return sniffed if sniffed else "jsonl"


def _sniff_kind(path: Path) -> str | None:
    """Inspect the first parseable row to tell reasoning problems from SFT."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    return None
                if not isinstance(row, dict):
                    return None
                keys = set(row.keys())
                if _REASONING_KEYS <= keys:
                    return "reasoning_problems"
                if "messages" in keys or {"instruction", "output"} <= keys:
                    return "instructions"
                return None
    except OSError:
        return None
    return None


def _count_lines(path: Path) -> int:
    """Cheap count of non-empty lines (no JSON parsing). 0 on any error."""
    count = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return 0
    return count


def _cap_record(record: dict) -> dict:
    """Return a shallow copy with long string values truncated to the cap."""
    capped: dict = {}
    for key, value in record.items():
        if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
            capped[key] = value[:_MAX_FIELD_CHARS]
        else:
            capped[key] = value
    return capped
