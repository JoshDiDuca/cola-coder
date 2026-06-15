"""Config browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of the project's YAML configs (``configs/*.yaml``).
All functions are robust to missing or malformed inputs and never raise — they return
empty results or an ``{"error": ...}`` dict instead.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml


def list_configs(configs_dir: str = "configs") -> list[dict]:
    """Recursively scan ``configs_dir`` for ``*.yaml`` / ``*.yml`` files.

    Each entry is a dict with keys: name (filename), path, rel (path relative to
    ``configs_dir``), size_bytes, mtime.

    Missing ``configs_dir`` yields ``[]``. Results are sorted by ``rel``.
    """
    if not os.path.isdir(configs_dir):
        return []

    results: list[dict] = []
    for dirpath, _dirnames, filenames in os.walk(configs_dir):
        for filename in filenames:
            if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                continue

            path = os.path.join(dirpath, filename)

            try:
                stat = os.stat(path)
                size_bytes = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                continue

            rel = os.path.relpath(path, configs_dir)

            results.append(
                {
                    "name": filename,
                    "path": path,
                    "rel": rel,
                    "size_bytes": size_bytes,
                    "mtime": mtime,
                }
            )

    results.sort(key=lambda entry: entry["rel"])
    return results


def read_config(path: str, max_chars: int = 40000) -> dict:
    """Read a YAML config file.

    Returns {"path", "content" (raw text, truncated to ``max_chars``),
    "parsed" (yaml.safe_load result: dict|list|None), "truncated" (bool)}.

    ``parsed`` is ``None`` when the (full) text does not parse as YAML. On a
    missing/unreadable path returns {"error": str}. Never raises.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        return {"error": str(exc)}

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        parsed = None

    truncated = len(raw) > max_chars
    content = raw[:max_chars]

    return {
        "path": path,
        "content": content,
        "parsed": parsed,
        "truncated": truncated,
    }


def write_config(path: str, content: str, configs_dir: str = "configs") -> dict:
    """Validate and atomically write YAML ``content`` to ``path``.

    Safety: ``path`` MUST resolve to a ``.yaml``/``.yml`` file inside ``configs_dir``
    (path-traversal is rejected), and ``content`` MUST parse as YAML — an invalid edit
    is refused BEFORE touching disk, so a config can never be corrupted from the UI.
    The write is atomic (temp file + ``os.replace``). Editing a file on disk does NOT
    affect an already-running trainer (it read its config at launch).

    Returns {"ok": True, "path", "bytes_written"} or {"error": str}. Never raises.
    """
    try:
        base = Path(configs_dir).resolve()
        target = Path(path).resolve()
    except OSError as exc:
        return {"error": str(exc)}

    if base != target and base not in target.parents:
        return {"error": f"refused: {path} is outside the configs directory"}
    if target.suffix not in (".yaml", ".yml"):
        return {"error": f"refused: {target.name} is not a .yaml/.yml file"}

    try:
        yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return {"error": f"invalid YAML, not written: {exc}"}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp, target)
    except OSError as exc:
        return {"error": str(exc)}

    return {"ok": True, "path": str(target), "bytes_written": len(content.encode("utf-8"))}
