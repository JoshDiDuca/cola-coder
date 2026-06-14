"""Data-source config browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of the multi-source data-collection config
(``configs/data_sources.yaml``). The on-disk file mixes code/text/math at
Qwen2.5-Coder ratios (70% / 20% / 10%). All functions are robust to missing or
malformed inputs and never raise — they return an ``{"error": ...}`` dict
instead.

The real schema is a ``sources:`` mapping keyed by source name, where each entry
carries ``dataset`` / ``weight`` / ``enabled`` and optional ``languages`` /
``min_length`` / ``max_length``::

    sources:
      code:
        dataset: "bigcode/starcoderdata"
        weight: 0.7
        languages: [typescript, javascript, python]
      text: {dataset: "...", weight: 0.2}
      math: {dataset: "...", weight: 0.1}

The source-name key (code/text/math) doubles as its category, so it is surfaced
as ``kind``. This module is also defensive about a ``sources`` *list* shape (in
case a future schema lists named entries) and about weights given as percentages
rather than fractions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _coerce_weight(value: Any) -> float | None:
    """Return a float weight or ``None`` when not numeric/parseable."""
    if isinstance(value, bool):  # bool is an int subclass — reject it explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_languages(value: Any) -> list[str]:
    """Normalize a languages field into a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _summarize_source(name: str, raw: Any) -> dict:
    """Build a UI-friendly summary dict for one source entry.

    ``raw`` is the per-source mapping. ``name`` is the source key (which also
    serves as ``kind`` when no explicit ``kind``/``type`` field is present).
    """
    entry: dict[str, Any] = raw if isinstance(raw, dict) else {}

    # A name may be carried inside the entry (list shape) or supplied by the key.
    display_name = entry.get("name", name)

    kind = entry.get("kind") or entry.get("type") or entry.get("source") or name

    return {
        "name": str(display_name),
        "weight": _coerce_weight(entry.get("weight")),
        "dataset": entry.get("dataset") if isinstance(entry.get("dataset"), str) else None,
        "languages": _coerce_languages(entry.get("languages")),
        "kind": str(kind) if kind is not None else None,
    }


def _iter_sources(sources: Any) -> list[dict]:
    """Normalize either a name-keyed dict or a list of entries into summaries."""
    summaries: list[dict] = []

    if isinstance(sources, dict):
        for name, raw in sources.items():
            summaries.append(_summarize_source(str(name), raw))
    elif isinstance(sources, list):
        for index, raw in enumerate(sources):
            # Prefer an explicit name/kind/source field; fall back to the index.
            name: Any = None
            if isinstance(raw, dict):
                name = raw.get("name") or raw.get("kind") or raw.get("source")
            summaries.append(_summarize_source(str(name) if name else str(index), raw))

    return summaries


def _format_weight(weight: float | None) -> str | None:
    """Render a weight as a percentage string (``0.7`` -> ``70%``)."""
    if weight is None:
        return None
    # Treat values > 1 as already-percentages, else fractions.
    pct = weight if weight > 1 else weight * 100
    if pct == int(pct):
        return f"{int(pct)}%"
    return f"{pct:g}%"


def read_data_sources(path: str = "configs/data_sources.yaml") -> dict:
    """Parse data_sources.yaml into a UI-friendly summary. Returns:
      {"path": str,
       "sources": [ {"name": str, "weight": float | None, "dataset": str | None,
                     "languages": list[str], "kind": str | None} ],
       "total_weight": float | None,    # sum of source weights if present
       "parsed": dict,                  # the full safe_loaded yaml (for a raw view)
       "summary": str}                  # one-line, e.g. "3 sources, code 70% / text 20% / math 10%"
    Pull fields defensively with .get using the REAL key names you find. ``kind`` is the
    source category (e.g. code/text/math or the HF/github/local source type) if present.
    On any failure (missing file / bad YAML / non-mapping) return {"error": "..."}. Never raise.
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

    sources_raw = parsed.get("sources")
    if not isinstance(sources_raw, (dict, list)):
        return {"error": "'sources' section is missing or not a mapping/list"}

    sources = _iter_sources(sources_raw)

    weights = [s["weight"] for s in sources if s["weight"] is not None]
    total_weight = float(sum(weights)) if weights else None

    # One-line summary: "3 sources, code 70% / text 20% / math 10%".
    parts: list[str] = []
    for s in sources:
        formatted = _format_weight(s["weight"])
        parts.append(f"{s['name']} {formatted}" if formatted else s["name"])
    count = len(sources)
    noun = "source" if count == 1 else "sources"
    summary = f"{count} {noun}"
    if parts:
        summary = f"{summary}, " + " / ".join(parts)

    return {
        "path": path,
        "sources": sources,
        "total_weight": total_weight,
        "parsed": parsed,
        "summary": summary,
    }
