"""Scorer registry — instantiates scorers from YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cola_coder.data.scorers.protocol import CompositeScorer, ScorerProtocol
from cola_coder.data.scorers.sandbox import SandboxedRunner


def load_scoring_config(
    config_path: str | Path = "configs/scoring.yaml",
) -> dict[str, Any]:
    """Load scoring configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return raw
    except (yaml.YAMLError, OSError):
        return {}


def build_composite_scorer(
    config_path: str | Path = "configs/scoring.yaml",
    scorer_names: list[str] | None = None,
) -> CompositeScorer:
    """Build a CompositeScorer from config, optionally filtering to specific scorers.

    Args:
        config_path: Path to scoring.yaml.
        scorer_names: If given, only include these scorers (e.g. ["tsc", "eslint"]).

    Returns:
        CompositeScorer with all enabled and available scorers.
    """
    cfg = load_scoring_config(config_path)
    scoring_cfg = cfg.get("scoring", {})
    scorers_cfg = scoring_cfg.get("scorers", {})
    sandbox_cfg = scoring_cfg.get("sandbox", {})
    tier_weights = scoring_cfg.get("tier_weights")

    # Build sandbox runner for tool-based scorers
    runner = SandboxedRunner(
        use_docker=sandbox_cfg.get("use_docker", False),
        timeout=sandbox_cfg.get("timeout", 10),
        memory_mb=sandbox_cfg.get("memory_mb", 512),
    )

    scorers: list[tuple[ScorerProtocol, float]] = []

    for name, scfg in scorers_cfg.items():
        if not isinstance(scfg, dict):
            continue
        if not scfg.get("enabled", False):
            continue
        if scorer_names is not None and name not in scorer_names:
            continue

        weight = float(scfg.get("weight", 0.0))
        if weight <= 0:
            continue

        scorer = _instantiate_scorer(name, scfg, runner)
        if scorer is not None and scorer.is_available():
            scorers.append((scorer, weight))

    return CompositeScorer(scorers, tier_weights=tier_weights)


def _instantiate_scorer(
    name: str,
    cfg: dict[str, Any],
    runner: SandboxedRunner,
) -> ScorerProtocol | None:
    """Instantiate a scorer by name. Returns None if import fails."""
    try:
        if name == "tsc":
            from cola_coder.data.scorers.tsc_scorer import TscScorer
            return TscScorer(
                strict=cfg.get("strict", True),
                timeout=cfg.get("timeout", 10),
                runner=runner,
            )
        elif name == "eslint":
            from cola_coder.data.scorers.eslint_scorer import EslintScorer
            return EslintScorer(
                timeout=cfg.get("timeout", 15),
                runner=runner,
            )
        elif name == "stars":
            from cola_coder.data.scorers.stars_scorer import StarsScorer
            return StarsScorer(
                default_score=cfg.get("default_score", 0.3),
            )
        elif name == "heuristic":
            from cola_coder.data.scorers.heuristic_scorer import HeuristicScorer
            return HeuristicScorer()
        elif name == "classifier":
            from cola_coder.data.scorers.classifier import ClassifierScorer
            model_dir = cfg.get("model_dir", "models/quality_classifier")
            return ClassifierScorer(model_dir=model_dir)
    except (ImportError, Exception):
        pass
    return None


def list_available_scorers(
    config_path: str | Path = "configs/scoring.yaml",
) -> list[dict[str, object]]:
    """List all configured scorers with their availability status."""
    cfg = load_scoring_config(config_path)
    scorers_cfg = cfg.get("scoring", {}).get("scorers", {})
    sandbox_cfg = cfg.get("scoring", {}).get("sandbox", {})

    runner = SandboxedRunner(
        use_docker=sandbox_cfg.get("use_docker", False),
        timeout=sandbox_cfg.get("timeout", 10),
    )

    results: list[dict[str, object]] = []
    for name, scfg in scorers_cfg.items():
        if not isinstance(scfg, dict):
            continue
        scorer = _instantiate_scorer(name, scfg, runner)
        results.append({
            "name": name,
            "enabled": scfg.get("enabled", False),
            "weight": scfg.get("weight", 0.0),
            "available": scorer is not None and scorer.is_available() if scorer else False,
        })
    return results
