"""Storage inspection helpers for the local cola-coder UI/dashboard.

Lightweight, read-only view of where data, checkpoints, and the tokenizer
actually live (per ``configs/storage.yaml``) plus their on-disk footprint.
All functions are robust to missing or malformed inputs and never raise — they
return an ``{"error": ...}`` dict instead.

The real keys consumed here mirror ``StorageConfig`` / ``get_storage_config``
and ``DatasetResolver``: the YAML ``storage:`` block carries ``data_dir``,
``checkpoints_dir`` (plural), ``tokenizer_path``, ``cache_dir``, ``hf_cache_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Hard cap on the number of files any single directory walk will stat before
# bailing out — keeps a huge data dir from hanging the request.
_DEFAULT_WALK_CAP = 20000


def _dir_size(path: str, cap: int = _DEFAULT_WALK_CAP) -> int | None:
    """Best-effort recursive byte size of ``path``.

    Sums file sizes via ``os.scandir`` (walking subdirectories iteratively) but
    bails out after ``cap`` files have been counted, returning what it has so
    far. Per-entry ``stat`` errors are swallowed. Returns ``None`` if ``path``
    does not exist or is not a directory.
    """
    p = Path(path)
    if not p.is_dir():
        return None

    total = 0
    counted = 0
    stack: list[str] = [str(p)]
    while stack:
        if counted >= cap:
            break
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if counted >= cap:
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            counted += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _resolve(root: str, value: str | None) -> str | None:
    """Resolve ``value`` relative to ``root`` when it is a relative path.

    Returns ``None`` when ``value`` is missing/blank. Absolute paths are kept
    as-is (only normalized to string form).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(root) / candidate)


def _entry(name: str, path: str | None, *, is_dir: bool, cap: int) -> dict:
    """Build one ``entries`` record with existence + best-effort size."""
    if path is None:
        return {"name": name, "path": None, "exists": False, "size_bytes": None}
    p = Path(path)
    exists = p.exists()
    if not exists:
        size_bytes: int | None = None
    elif p.is_dir():
        size_bytes = _dir_size(path, cap)
    else:
        try:
            size_bytes = p.stat().st_size
        except OSError:
            size_bytes = None
    return {"name": name, "path": path, "exists": exists, "size_bytes": size_bytes}


def read_storage(root: str = ".", cap: int = _DEFAULT_WALK_CAP) -> dict:
    """Summarize storage configuration + on-disk footprint. Returns:
      {"path": str,                         # resolved configs/storage.yaml
       "raw": dict,                         # full safe_loaded yaml
       "tokenizer_path": str | None,
       "data_dir": str | None,
       "checkpoint_dir": str | None,
       "entries": [ {"name": str, "path": str, "exists": bool, "size_bytes": int | None} ]}

    ``entries`` covers the key resolved locations (tokenizer, data dir,
    checkpoints dir, data/processed) with a best-effort recursive byte size
    (capped at ``cap`` files so a huge data dir can never hang the request;
    size ``None`` if unresolved/missing). On any failure (missing
    storage.yaml / bad YAML) return ``{"error": "..."}``. Never raises.
    """
    storage_yaml = Path(root) / "configs" / "storage.yaml"
    if not storage_yaml.is_file():
        return {"error": f"storage.yaml not found: {storage_yaml}"}

    try:
        with open(storage_yaml, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        return {"error": str(exc)}

    if not isinstance(raw, dict):
        return {"error": f"invalid storage.yaml (not a mapping): {storage_yaml}"}

    storage = raw.get("storage")
    if not isinstance(storage, dict):
        return {"error": f"invalid storage.yaml (missing 'storage' mapping): {storage_yaml}"}

    # Real key names mirror StorageConfig / DatasetResolver. Note the YAML uses
    # the plural "checkpoints_dir"; the contract surfaces it as "checkpoint_dir".
    tokenizer_path = _resolve(root, storage.get("tokenizer_path"))
    data_dir = _resolve(root, storage.get("data_dir"))
    checkpoint_dir = _resolve(root, storage.get("checkpoints_dir"))

    processed_dir = str(Path(data_dir) / "processed") if data_dir is not None else None

    entries = [
        _entry("tokenizer", tokenizer_path, is_dir=False, cap=cap),
        _entry("data_dir", data_dir, is_dir=True, cap=cap),
        _entry("checkpoints_dir", checkpoint_dir, is_dir=True, cap=cap),
        _entry("data_processed", processed_dir, is_dir=True, cap=cap),
    ]

    return {
        "path": str(storage_yaml),
        "raw": raw,
        "tokenizer_path": tokenizer_path,
        "data_dir": data_dir,
        "checkpoint_dir": checkpoint_dir,
        "entries": entries,
    }
