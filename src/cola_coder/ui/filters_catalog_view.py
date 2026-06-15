"""Data-quality filter catalog helpers for the local UI/dashboard.

Lightweight, read-only inventory of the modular data-quality filter plugins, so
the UI can render a "Data Filters Catalog" view (which filter plugins exist,
their category, one-line purpose, and whether they run by default). The canonical
source of truth is the plugin registry in ``cola_coder.data.registry``: each
filter is a ``FilterPlugin`` subclass decorated with ``@register_filter("<name>")``,
stored in ``_FILTER_REGISTRY`` and enumerated via ``list_filters()`` /
``get_filter()``. Importing ``cola_coder.data.filters`` fires all the decorators.

This module only *enumerates* the registered plugins — it never instantiates or
runs them. All functions are robust to missing or malformed inputs and never
raise; they return an ``{"error": ...}`` dict instead.

Returned shape (mirrors ``schemas.FiltersCatalog`` / ``schemas.FilterInfo``)::

    {"filters": [ {"name": str, "category": str, "purpose": str,
                   "module": str, "default_enabled": bool} ],
     "count": int,
     "categories": list[str]}   # distinct categories, sorted
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Best-effort category per registered filter name. Filters not listed here fall
# back to ``_DEFAULT_CATEGORY``. Keyed on the registry name (the @register_filter
# argument), which is stable and config-facing.
_CATEGORY_BY_NAME: dict[str, str] = {
    "content": "quality",
    "quality": "quality",
    "quality_classifier": "quality",
    "length": "quality",
    "syntax": "format",
    "deduplication": "dedup",
    "decontamination": "dedup",
    "license": "safety",
    "pii": "safety",
    "injection": "safety",
    "repetition": "quality",
}

_DEFAULT_CATEGORY = "quality"

# Filters that run in the standard conservative (``--filter``) pipeline by
# default. The remaining registered filters are opt-in (strict mode, explicit
# YAML config, or safety pre-screens enabled per data source).
_DEFAULT_ENABLED_NAMES: frozenset[str] = frozenset(
    {"content", "quality", "syntax", "length"}
)


def filters_catalog() -> dict:
    """Enumerate the registered data-quality filter plugins (read-only).

    Imports ``cola_coder.data.filters`` to fire the ``@register_filter``
    decorators, then reads the registry via ``list_filters()`` / ``get_filter()``.
    Never instantiates or runs a filter. Returns the ``FiltersCatalog`` dict, or
    ``{"error": "..."}`` on genuine failure. Never raises.
    """
    try:
        # Fire the @register_filter decorators (idempotent import).
        import cola_coder.data.filters  # noqa: F401
        from cola_coder.data.registry import get_filter, list_filters

        names = list_filters()

        filters: list[dict] = []
        categories: set[str] = set()
        for name in names:
            try:
                cls = get_filter(name)
            except KeyError:
                # Race against registry mutation — skip this entry, don't fail.
                logger.warning("filter %r vanished from registry during scan", name)
                continue

            category = _CATEGORY_BY_NAME.get(name, _DEFAULT_CATEGORY)
            categories.add(category)
            filters.append(
                {
                    "name": name,
                    "category": category,
                    "purpose": _purpose_from_docstring(cls),
                    "module": getattr(cls, "__module__", "") or "",
                    "default_enabled": name in _DEFAULT_ENABLED_NAMES,
                }
            )

        filters.sort(key=lambda entry: (entry["category"], entry["name"]))

        return {
            "filters": filters,
            "count": len(filters),
            "categories": sorted(categories),
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        logger.exception("failed to enumerate filter catalog")
        return {"error": str(exc)}


# ── Internals ───────────────────────────────────────────────────────────────


def _purpose_from_docstring(cls: type) -> str:
    """Return the first non-empty line of ``cls``'s docstring (``""`` if none)."""
    doc = cls.__doc__
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
