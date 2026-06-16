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
import os
import tempfile
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


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Atomically write ``data`` as YAML (temp file + ``os.replace``), like configs.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(data, handle, sort_keys=True, default_flow_style=False)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_registry(path: Path) -> dict:
    """Return the full parsed YAML doc (a dict), tolerating a missing/empty file.

    Preserves any sibling top-level keys so a write only touches ``specialists``.
    Returns ``{"specialists": {}}`` for a missing file and raises ``ValueError`` for
    a malformed one (so a write never silently drops existing entries).
    """
    if not path.is_file():
        return {"specialists": {}}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if parsed is None:
        return {"specialists": {}}
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    if not isinstance(parsed.get("specialists"), dict):
        parsed["specialists"] = {}
    return parsed


def save_specialist(
    root: str,
    domain: str,
    checkpoint: str,
    keywords: list[str],
    config: str | None = None,
    confidence_threshold: float | None = None,
    description: str | None = None,
) -> dict:
    """Upsert one specialist entry into ``configs/specialists.yaml`` (add or update).

    Validates the inputs and atomically rewrites the file, preserving every other
    domain and any sibling top-level keys. Editing this registry on disk does NOT
    affect a running trainer (it loaded its config at launch) and never loads a model.

    Returns the refreshed ``specialists_view`` dict on success, or ``{"error": str}``
    on a validation/IO failure (never raises).
    """
    domain = domain.strip()
    if not domain:
        return {"error": "domain is required"}
    if "/" in domain or "\\" in domain:
        return {"error": "domain must not contain path separators"}
    if not checkpoint.strip():
        return {"error": "checkpoint is required"}
    if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
        return {"error": f"confidence_threshold must be in [0, 1], got {confidence_threshold}"}

    entry: dict[str, object] = {
        "checkpoint": checkpoint.strip(),
        "keywords": [str(k).strip() for k in keywords if str(k).strip()],
    }
    if config is not None and config.strip():
        entry["config"] = config.strip()
    if confidence_threshold is not None:
        entry["confidence_threshold"] = confidence_threshold
    if description is not None and description.strip():
        entry["description"] = description.strip()

    path = Path(root) / _SPECIALISTS_REL
    try:
        doc = _load_registry(path)
        doc["specialists"][domain] = entry
        _atomic_write_yaml(path, doc)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}

    return specialists_view(root)


def remove_specialist(root: str, domain: str) -> dict:
    """Remove one specialist entry from ``configs/specialists.yaml`` by domain.

    Returns the refreshed ``specialists_view`` dict on success, or ``{"error": str}``
    when the file/domain is missing or the write fails (never raises).
    """
    domain = domain.strip()
    if not domain:
        return {"error": "domain is required"}

    path = Path(root) / _SPECIALISTS_REL
    if not path.is_file():
        return {"error": f"no registry to edit: {path} does not exist"}

    try:
        doc = _load_registry(path)
        if domain not in doc["specialists"]:
            return {"error": f"domain not in registry: {domain}"}
        del doc["specialists"][domain]
        _atomic_write_yaml(path, doc)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}

    return specialists_view(root)
