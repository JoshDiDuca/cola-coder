"""Feature-toggle browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of the project's feature-toggle system
(``configs/features.yaml``, ~175 toggles). All functions are robust to missing
or malformed inputs and never raise — they return an ``{"error": ...}`` dict
instead.

The on-disk file is a flat ``{key: bool}`` mapping under a top-level
``features:`` key (categories live only in YAML comments, so a parsed file has
no machine-readable grouping). This module is also defensive about nested
``{key: {enabled: bool, category: str, ...}}`` shapes in case a future schema
adds them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CATEGORY = "All"


def _coerce_feature(value: Any) -> tuple[bool, Any]:
    """Return ``(enabled, value)`` for a single toggle.

    - Plain bool → ``(bool, bool)`` (value equals enabled).
    - Nested dict → enabled from its ``enabled``/``on`` key (truthy), value is
      the raw dict.
    - Anything else (number, str, list, None) → ``(truthiness, raw value)``.
    """
    if isinstance(value, bool):
        return value, value

    if isinstance(value, dict):
        if "enabled" in value:
            enabled = bool(value["enabled"])
        elif "on" in value:
            enabled = bool(value["on"])
        else:
            enabled = bool(value)
        return enabled, value

    return bool(value), value


def _category_of(value: Any) -> str:
    """Extract a category for a toggle, defaulting to ``"All"``.

    Only nested-dict toggles can carry a ``category`` field; flat bools cannot,
    so they fall into the default group.
    """
    if isinstance(value, dict):
        category = value.get("category")
        if isinstance(category, str) and category:
            return category
    return _DEFAULT_CATEGORY


def list_features(path: str = "configs/features.yaml") -> dict:
    """Parse features.yaml into a grouped, UI-friendly view.

    Returns:
      {"path": str,
       "total": int, "enabled": int,
       "groups": [ {"category": str,
                    "features": [ {"key": str, "enabled": bool, "value": <any>} ] } ] }
    Group by category if the YAML/category metadata provides one; otherwise put
    everything in a single {"category": "All", ...} group. ``enabled`` is the
    boolean truthiness of the toggle; ``value`` is the raw value when it's not a
    plain bool (e.g. a nested dict/number), else equal to enabled.
    On any failure (missing file, bad YAML) return {"error": "..."}. Never raise.
    """
    if not os.path.isfile(path):
        return {"error": f"path not found: {path}"}

    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": str(exc)}

    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return {"error": f"invalid YAML: {exc}"}

    if parsed is None:
        return {"error": "empty or null YAML"}
    if not isinstance(parsed, dict):
        return {"error": "top-level YAML is not a mapping"}

    # Toggles live under a "features:" key; fall back to the whole document if
    # that key is absent (defensive against a flat top-level file).
    section = parsed.get("features", parsed)
    if not isinstance(section, dict):
        return {"error": "'features' section is not a mapping"}

    total = 0
    enabled_count = 0
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []

    for key, raw_value in section.items():
        enabled, value = _coerce_feature(raw_value)
        category = _category_of(raw_value)

        total += 1
        if enabled:
            enabled_count += 1

        if category not in grouped:
            grouped[category] = []
            order.append(category)
        grouped[category].append(
            {"key": str(key), "enabled": enabled, "value": value}
        )

    groups = [
        {"category": category, "features": grouped[category]} for category in order
    ]

    return {
        "path": path,
        "total": total,
        "enabled": enabled_count,
        "groups": groups,
    }
