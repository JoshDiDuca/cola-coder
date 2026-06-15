"""VRAM-estimate endpoint helper for the local UI.

Mirrors the CLI ``scripts/vram_estimate.py`` (``cola-vram.ps1``): given a config
name (e.g. ``"small.yaml"``) it loads that YAML and reuses the SHARED estimator
:func:`cola_coder.features.vram_estimator.estimate_vram` — the exact same math the
CLI prints — rather than reinventing the memory formulas.

The estimator returns a GB-based breakdown; this view flattens it into a list of
named MB components plus a training total and a fixed VRAM budget so the panel can
render a fits / over-budget badge. Pure computation: the result does NOT depend on
any detected GPU (the estimator's optional GPU probe result is ignored here — the
budget is a fixed assumption). Robust to a missing/malformed config or absent
torch: returns an ``{"error": ...}`` dict, never raises.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configs live under ``configs/`` relative to the project root.
_CONFIGS_DIR = "configs"

# Assumed VRAM budget in MB (RTX 4080 Super class — the project's primary GPU).
# Fixed so the estimate is reproducible and GPU-independent.
_DEFAULT_BUDGET_MB = 16384.0

_GB_TO_MB = 1024.0


def _resolve_config_path(config: str) -> Path | None:
    """Resolve a config name to a path under ``configs/``.

    Accepts a bare filename (``small.yaml``), a name without extension
    (``small``), or a path already containing ``configs`` (``configs/small.yaml``).
    Returns the existing path, or None if it cannot be found. Never raises.
    """
    candidate = Path(config)
    search: list[Path] = []

    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.append(Path(_CONFIGS_DIR) / candidate)
        search.append(candidate)
        if candidate.suffix == "":
            search.append(Path(_CONFIGS_DIR) / f"{candidate.name}.yaml")
            search.append(Path(_CONFIGS_DIR) / f"{candidate.name}.yml")

    for path in search:
        if path.is_file():
            return path
    return None


def vram_estimate(config: str, budget_mb: float = _DEFAULT_BUDGET_MB) -> dict:
    """Estimate VRAM usage for ``config`` and return a flat, UI-ready dict.

    Args:
        config: Config name/path (e.g. ``"small.yaml"`` or ``"4080_max"``).
        budget_mb: Assumed VRAM budget in MB the estimate is checked against.

    Returns:
        A dict matching :class:`cola_coder.ui.schemas.VramEstimate`, or
        ``{"error": ...}`` on any failure.
    """
    resolved = _resolve_config_path(config)
    if resolved is None:
        return {"error": f"config not found: {config}"}

    try:
        from cola_coder.model.config import Config
    except ImportError as exc:  # pragma: no cover - torch/yaml stack absent
        logger.warning("config module unavailable: %s", exc)
        return {"error": f"config module unavailable: {exc}"}

    try:
        cfg = Config.from_yaml(resolved)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("failed to load config %s: %s", resolved, exc)
        return {"error": f"failed to load config: {exc}"}

    try:
        from cola_coder.features.vram_estimator import estimate_vram
    except ImportError as exc:  # pragma: no cover - estimator deps absent
        logger.warning("vram estimator unavailable: %s", exc)
        return {"error": f"vram estimator unavailable: {exc}"}

    try:
        est = estimate_vram(model_config=cfg.model, training_config=cfg.training)
    except (ValueError, AttributeError) as exc:
        logger.warning("vram estimation failed for %s: %s", resolved, exc)
        return {"error": f"vram estimation failed: {exc}"}

    # Flatten the GB breakdown into named MB components (GPU-independent math).
    components: list[dict] = [
        {"name": "model weights", "mb": est.model_weights_gb * _GB_TO_MB},
        {"name": "optimizer state", "mb": est.optimizer_state_gb * _GB_TO_MB},
        {"name": "gradients", "mb": est.gradients_gb * _GB_TO_MB},
        {"name": "activations", "mb": est.activations_gb * _GB_TO_MB},
        {"name": "kv-cache (inference)", "mb": est.kv_cache_gb * _GB_TO_MB},
    ]

    total_mb = est.total_training_gb * _GB_TO_MB

    return {
        "config": resolved.name,
        "params_millions": cfg.model.total_params / 1e6,
        "precision": cfg.training.precision,
        "batch_size": cfg.training.batch_size,
        "seq_len": cfg.model.max_seq_len,
        "components": components,
        "total_mb": total_mb,
        "budget_mb": budget_mb,
        "fits": total_mb <= budget_mb,
    }
