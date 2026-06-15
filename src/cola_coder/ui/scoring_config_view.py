"""Data-quality scoring config helpers for the local UI/dashboard.

Read-only inventory of the data-quality *scorers* — complements the Data Filters
Catalog (UI-046). For each scorer it merges two sources of truth:

* ``configs/scoring.yaml`` (``scoring.scorers.<name>.{enabled, weight}``) — what
  the user has turned on and how each contributes to the composite score.
* the scorer registry (``cola_coder.data.scorers.registry.list_available_scorers``)
  — whether the scorer can actually *run* on this machine (its deps / external
  tools are present), via each scorer's ``is_available()``.

The two are joined by scorer name. The registry helper already returns the merged
``{name, enabled, weight, available}`` shape (it reads the same YAML), so we reuse
it directly and enrich each entry with a one-line ``purpose`` resolved from the
scorer class docstring. ``curriculum`` reflects ``curriculum.strategy`` when
``curriculum.enabled`` is set in ``scoring.yaml`` (else ``None``).

This module never instantiates a scorer to *score* anything — the registry only
constructs scorers to probe availability. All functions are robust to a missing
or malformed ``scoring.yaml`` and never raise; on genuine failure they return an
``{"error": ...}`` dict (a missing config yields an empty scorer list, not a
crash).

Returned shape (mirrors ``schemas.ScoringConfig`` / ``schemas.ScorerConfigEntry``)::

    {"path": str, "count": int, "enabled_count": int,
     "curriculum": str | None,
     "scorers": [ {"name": str, "enabled": bool, "weight": float,
                   "available": bool, "purpose": str} ]}
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SCORING_CONFIG_PATH = "configs/scoring.yaml"


def _resolve_purpose(name: str) -> str:
    """Return the first docstring line of the scorer class named ``name`` ("" if unresolved)."""
    try:
        if name == "tsc":
            from cola_coder.data.scorers.tsc_scorer import TscScorer as cls
        elif name == "eslint":
            from cola_coder.data.scorers.eslint_scorer import EslintScorer as cls
        elif name == "stars":
            from cola_coder.data.scorers.stars_scorer import StarsScorer as cls
        elif name == "heuristic":
            from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer as cls
        elif name == "injection_safety":
            from cola_coder.data.scorers.injection_scorer import InjectionScorer as cls
        elif name == "educational_value":
            from cola_coder.data.scorers.educational_value import EducationalValueScorer as cls
        elif name == "classifier":
            from cola_coder.data.scorers.classifier import ClassifierScorer as cls
        elif name == "llm_judge":
            from cola_coder.data.scorers.llm_judge import LlmJudge as cls
        else:
            return ""
    except ImportError:
        logger.debug("scorer %r class not importable; purpose left blank", name)
        return ""
    return _purpose_from_docstring(cls)


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


def _curriculum_mode(config_path: str | Path = _SCORING_CONFIG_PATH) -> str | None:
    """Return the curriculum strategy if ``scoring.yaml`` enables one, else ``None``."""
    from cola_coder.data.scorers.registry import load_scoring_config

    cfg = load_scoring_config(config_path)
    curriculum = cfg.get("curriculum")
    if not isinstance(curriculum, dict) or not curriculum.get("enabled", False):
        return None
    strategy = curriculum.get("strategy")
    return strategy if isinstance(strategy, str) and strategy else None


def scoring_config(config_path: str | Path = _SCORING_CONFIG_PATH) -> dict:
    """Merge ``scoring.yaml`` scorer config with registry availability (read-only).

    Reuses ``registry.list_available_scorers()`` for the merged
    ``{name, enabled, weight, available}`` shape, enriches each entry with a
    one-line ``purpose`` from the scorer class docstring, and adds the curriculum
    mode. Returns the ``ScoringConfig`` dict, or ``{"error": "..."}`` on genuine
    failure. A missing ``scoring.yaml`` yields an empty scorer list (no crash).
    Never raises.
    """
    try:
        from cola_coder.data.scorers.registry import list_available_scorers

        raw = list_available_scorers(config_path)

        scorers: list[dict] = []
        enabled_count = 0
        for entry in raw:
            name = str(entry.get("name", ""))
            enabled = bool(entry.get("enabled", False))
            weight = float(entry.get("weight", 0.0) or 0.0)
            available = bool(entry.get("available", False))
            if enabled:
                enabled_count += 1
            scorers.append(
                {
                    "name": name,
                    "enabled": enabled,
                    "weight": weight,
                    "available": available,
                    "purpose": _resolve_purpose(name),
                }
            )

        scorers.sort(key=lambda e: (not e["enabled"], e["name"]))

        return {
            "path": str(config_path),
            "scorers": scorers,
            "count": len(scorers),
            "enabled_count": enabled_count,
            "curriculum": _curriculum_mode(config_path),
        }
    except Exception as exc:  # noqa: BLE001 — contract: never raise.
        logger.exception("failed to assemble scoring config view")
        return {"error": str(exc)}
