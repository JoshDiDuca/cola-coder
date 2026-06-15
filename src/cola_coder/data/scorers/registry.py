"""Scorer registry — instantiates scorers from YAML config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from cola_coder.data.scorers.audit import ScoringAuditLogger
from cola_coder.data.scorers.credential_scanner import CredentialScanner
from cola_coder.data.scorers.protocol import CompositeScorer, ScorerProtocol
from cola_coder.data.scorers.sandbox import SandboxedRunner
from cola_coder.data.scorers.security import SecurityConfig

logger = logging.getLogger(__name__)


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
    tier_weights = scoring_cfg.get("tier_weights")

    # Build security config
    security_cfg = SecurityConfig.from_dict(scoring_cfg)

    # Build audit logger
    audit_logger = ScoringAuditLogger(security_cfg.audit_log_path)

    # Build sandbox runner with security config
    runner = SandboxedRunner.from_config(security_cfg, audit_logger=audit_logger)

    # Log sandbox status (visible in CLI output)
    runner.log_status()

    # Verify Docker if required
    runner.verify_or_fail()

    # Clean stale temp dirs
    SandboxedRunner.cleanup_stale_temps()

    # Build credential scanner
    scanner = CredentialScanner(mode=security_cfg.credential_scan_mode)

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

        scorer = _instantiate_scorer(name, scfg, runner, scanner)
        if scorer is not None and scorer.is_available():
            scorers.append((scorer, weight))

    return CompositeScorer(scorers, tier_weights=tier_weights)


def _instantiate_scorer(
    name: str,
    cfg: dict[str, Any],
    runner: SandboxedRunner,
    scanner: CredentialScanner | None = None,
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
        elif name == "injection_safety":
            from cola_coder.data.scorers.injection_scorer import InjectionScorer
            return InjectionScorer()
        elif name == "cwe_security":
            from cola_coder.data.scorers.cwe_security import CweSecurityScorer
            return CweSecurityScorer()
        elif name == "educational_value":
            from cola_coder.data.scorers.educational_value import EducationalValueScorer
            return EducationalValueScorer()
        elif name == "classifier":
            from cola_coder.data.scorers.classifier import ClassifierScorer
            model_dir = cfg.get("model_dir", "models/quality_classifier")
            return ClassifierScorer(model_dir=model_dir)
        elif name == "llm_judge":
            from cola_coder.data.scorers.llm_judge import LlmJudge
            return LlmJudge(
                provider=cfg.get("provider", "ollama"),
                model=cfg.get("model", "codellama"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url", "http://localhost:11434"),
                timeout=cfg.get("timeout", 30),
                credential_scanner=scanner,
            )
    except (ImportError, Exception):
        pass
    return None


def list_available_scorers(
    config_path: str | Path = "configs/scoring.yaml",
) -> list[dict[str, object]]:
    """List all configured scorers with their availability status."""
    cfg = load_scoring_config(config_path)
    scoring_cfg = cfg.get("scoring", {})
    scorers_cfg = scoring_cfg.get("scorers", {})

    # Build security config for runner construction
    security_cfg = SecurityConfig.from_dict(scoring_cfg)
    runner = SandboxedRunner.from_config(security_cfg)
    scanner = CredentialScanner(mode=security_cfg.credential_scan_mode)

    results: list[dict[str, object]] = []
    for name, scfg in scorers_cfg.items():
        if not isinstance(scfg, dict):
            continue
        scorer = _instantiate_scorer(name, scfg, runner, scanner)
        results.append({
            "name": name,
            "enabled": scfg.get("enabled", False),
            "weight": scfg.get("weight", 0.0),
            "available": scorer is not None and scorer.is_available() if scorer else False,
        })
    return results
