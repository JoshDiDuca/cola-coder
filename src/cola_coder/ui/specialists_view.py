"""Router specialist-registry endpoint helper for the local UI.

Read-only mirror of the CLI "Router & Specialists" registry
(``configs/specialists.yaml``): the domain → checkpoint map the confidence
router uses to dispatch a request to a per-domain specialist model (project
Vision: a 125M router + 50M domain specialists). This surfaces the registry in
the UI; it never edits or loads a model. Never raises — a missing/empty/malformed
file yields a valid "no specialists" result.

Registry shape (per the file's own documented example)::

    specialists:
      react:
        checkpoint: checkpoints/react/latest
        config: configs/tiny.yaml
        keywords: [react, jsx, component]
        confidence_threshold: 0.6
        description: "React specialist"
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SPECIALISTS_REL = Path("configs") / "specialists.yaml"


def _coerce_entry(domain: str, raw: dict) -> dict:
    """Coerce one registry entry into the SpecialistEntry shape (defensive)."""
    keywords_raw = raw.get("keywords", [])
    keywords = [str(k) for k in keywords_raw] if isinstance(keywords_raw, list) else []
    threshold = raw.get("confidence_threshold")
    return {
        "domain": domain,
        "checkpoint": str(raw.get("checkpoint", "")),
        "config": str(raw["config"]) if raw.get("config") is not None else None,
        "keywords": keywords,
        "confidence_threshold": float(threshold) if isinstance(threshold, (int, float)) else None,
        "description": str(raw["description"]) if raw.get("description") is not None else None,
    }


def specialists_view(root: str = ".") -> dict:
    """Read ``configs/specialists.yaml`` and return a ``SpecialistsView`` dict.

    Returns ``{"error": str}`` only on an unreadable/malformed file; a missing file
    or empty ``specialists: {}`` is a valid empty registry (``exists`` reflects the
    file's presence).
    """
    path = Path(root) / _SPECIALISTS_REL
    if not path.is_file():
        return {"path": str(path), "exists": False, "count": 0, "specialists": []}

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"error": f"could not read {path}: {exc}"}

    registry = parsed.get("specialists") if isinstance(parsed, dict) else None
    entries: list[dict] = []
    if isinstance(registry, dict):
        for domain, raw in registry.items():
            if isinstance(raw, dict):
                entries.append(_coerce_entry(str(domain), raw))
    entries.sort(key=lambda e: e["domain"])

    return {"path": str(path), "exists": True, "count": len(entries), "specialists": entries}
