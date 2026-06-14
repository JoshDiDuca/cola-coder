"""Feature-toggle write helpers for the local UI/dashboard.

A single, SAFE write-path: flip one existing flag in ``configs/features.yaml``
without disturbing the file's human-curated category comments or key ordering.

The on-disk file is a flat ``{key: bool}`` mapping under a top-level
``features:`` key, with category comments interleaved between groups. A naive
``yaml.safe_load`` + ``yaml.safe_dump`` round-trip would erase those comments
and reorder keys, so we do a *line-level* edit instead: regex-replace only the
single matching ``  <key>: <bool>`` line, preserving indentation, any trailing
inline comment, every other line verbatim, and the original key order.

As with the read-side helpers, this function NEVER raises — on any failure it
returns an ``{"error": ...}`` dict.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml


def set_feature(key: str, enabled: bool, path: str = "configs/features.yaml") -> dict:
    """Toggle one existing feature flag via a line-level edit (preserves comments + order). Returns:
      {"ok": True, "key": str, "enabled": bool, "path": str}
    The key MUST already exist under the `features:` mapping — if it does not, return
    {"error": "unknown feature: <key>"} (never CREATE a new key — that would make a phantom flag).
    Write atomically (temp file + os.replace). On any failure (missing file, key not found, write error)
    return {"error": "..."}. Never raise.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": str(exc)}

    # Read-only validation: the key must already exist under `features:`.
    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return {"error": f"invalid YAML: {exc}"}

    if not isinstance(parsed, dict):
        return {"error": "top-level YAML is not a mapping"}

    section = parsed.get("features")
    if not isinstance(section, dict):
        return {"error": "'features' section is not a mapping"}

    if key not in section:
        return {"error": f"unknown feature: {key}"}

    # Line-level edit: match exactly the `  <key>: <bool>` line, preserving the
    # leading indentation and any trailing inline comment in `rest`.
    line_pattern = re.compile(
        r"^(?P<indent>\s+)"
        + re.escape(key)
        + r"\s*:\s*(?:true|false|True|False)(?P<rest>\s*(?:#.*)?)$",
    )

    new_bool = "true" if enabled else "false"

    new_lines: list[str] = []
    replaced = False
    for line in raw_text.splitlines(keepends=True):
        # Separate the trailing newline so the regex can anchor on the content.
        stripped = line.rstrip("\n").rstrip("\r")
        newline = line[len(stripped):]
        match = line_pattern.match(stripped)
        if match and not replaced:
            new_content = f"{match.group('indent')}{key}: {new_bool}{match.group('rest')}"
            new_lines.append(new_content + newline)
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # Key exists in the parse but no scalar bool line matched (e.g. nested
        # mapping or unexpected formatting). Refuse rather than guess.
        return {"error": f"could not locate scalar line for feature: {key}"}

    new_text = "".join(new_lines)

    # Atomic write: temp file in the same directory, then os.replace.
    target = Path(path)
    directory = str(target.parent) if str(target.parent) else "."
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        if tmp_path is not None and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {"error": str(exc)}

    # Verify the on-disk result parses and the key now equals `enabled`.
    verify = _verify(path, key, enabled)
    if "error" in verify:
        return verify

    return {"ok": True, "key": key, "enabled": enabled, "path": path}


def _verify(path: str, key: str, enabled: bool) -> dict:
    """Re-read the file and confirm ``features[key]`` is now ``enabled``."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": f"verify read failed: {exc}"}

    try:
        parsed: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {"error": f"verify parse failed: {exc}"}

    if not isinstance(parsed, dict):
        return {"error": "verify failed: top-level YAML is not a mapping"}

    section = parsed.get("features")
    if not isinstance(section, dict) or key not in section:
        return {"error": f"verify failed: feature missing after write: {key}"}

    if bool(section[key]) != enabled:
        return {"error": f"verify failed: feature {key} did not change"}

    return {"ok": True}
