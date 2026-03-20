"""
Gradient Norm Monitor — Feature 13

Monitors gradient norms during training to detect vanishing/exploding gradients.
Tracks per-layer gradient norms and overall model gradient norm.
Alerts on abnormal gradient patterns.

Self-contained — no modifications to existing files required.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether the gradient norm monitor feature is enabled."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Norm category helpers
# ---------------------------------------------------------------------------

#: Threshold boundaries used for classification
_VANISHING_THRESHOLD = 1e-7
_WEAK_THRESHOLD = 1e-4
_HEALTHY_MAX = 10.0
_ELEVATED_MAX = 100.0


def _classify_norm(norm: float) -> str:
    """Return the category string for a given gradient norm value."""
    if norm < _VANISHING_THRESHOLD:
        return "vanishing"
    if norm < _WEAK_THRESHOLD:
        return "weak"
    if norm <= _HEALTHY_MAX:
        return "healthy"
    if norm <= _ELEVATED_MAX:
        return "elevated"
    return "exploding"


# ---------------------------------------------------------------------------
# LayerGradStats dataclass (matches plan design)
# ---------------------------------------------------------------------------

@dataclass
class LayerGradStats:
    name: str
    norm: float
    num_params: int
    has_grad: bool
    is_frozen: bool

    @property
    def norm_per_param(self) -> float:
        """Gradient norm normalised by sqrt(num_params) for fair layer comparison."""
        return self.norm / math.sqrt(self.num_params) if self.num_params > 0 else 0.0

    @property
    def category(self) -> str:
        if not self.has_grad or self.is_frozen:
            return "frozen"
        return _classify_norm(self.norm)

    @property
    def color(self) -> str:
        return {
            "frozen": "dim",
            "vanishing": "bold red",
            "weak": "yellow",
            "healthy": "green",
            "elevated": "yellow",
            "exploding": "bold red",
        }[self.category]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GradientNormMonitor:
    """
    Monitor gradient norms during training.

    Usage::

        monitor = GradientNormMonitor()

        # After loss.backward(), before optimizer.step():
        monitor.update(model, step=global_step)

        # Periodically inspect health:
        health = monitor.check_health()
        print(health)
    """

    def __init__(
        self,
        exploding_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
    ) -> None:
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold

        # history: list of {"step": int, "norms": {param_name: float}}
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_grad_norms(self, model: nn.Module) -> dict[str, float]:
        """
        Compute the L2 gradient norm for every named parameter that has a
        gradient.  Parameters without gradients (or that are frozen) are
        omitted.

        Returns a dict mapping parameter name → float norm.
        """
        norms: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norms[name] = param.grad.detach().norm(2).item()
        return norms

    def update(self, model: nn.Module, step: int) -> dict[str, float]:
        """
        Record gradient norms at *step*.  Call after ``loss.backward()``
        and before ``optimizer.step()`` / ``clip_grad_norm_()``.

        Returns the norms dict that was recorded.
        """
        norms = self.compute_grad_norms(model)
        self._history.append({"step": step, "norms": norms})
        return norms

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_health(self) -> dict:
        """
        Return a dict describing the current gradient health.

        Keys
        ----
        status : str
            "healthy" | "vanishing" | "exploding" | "mixed" | "no_data"
        vanishing_params : list[str]
            Parameter names whose latest norm is below the vanishing threshold.
        exploding_params : list[str]
            Parameter names whose latest norm is above the exploding threshold.
        total_norm : float | None
            Global L2 norm computed from the most recent recorded step.
        step : int | None
            The step number of the most recent recording.
        """
        if not self._history:
            return {
                "status": "no_data",
                "vanishing_params": [],
                "exploding_params": [],
                "total_norm": None,
                "step": None,
            }

        latest = self._history[-1]
        norms = latest["norms"]

        vanishing_params = [
            name for name, norm in norms.items() if norm < self.vanishing_threshold
        ]
        exploding_params = [
            name for name, norm in norms.items() if norm > self.exploding_threshold
        ]

        total_norm = (
            math.sqrt(sum(n ** 2 for n in norms.values())) if norms else 0.0
        )

        if exploding_params and vanishing_params:
            status = "mixed"
        elif exploding_params:
            status = "exploding"
        elif vanishing_params:
            status = "vanishing"
        else:
            status = "healthy"

        return {
            "status": status,
            "vanishing_params": vanishing_params,
            "exploding_params": exploding_params,
            "total_norm": total_norm,
            "step": latest["step"],
        }

    def detect_vanishing(self, threshold: float = 1e-7) -> bool | list[str]:
        """
        Check for vanishing gradients across all recorded steps.

        Returns
        -------
        False
            If no vanishing gradients were detected in the most recent step.
        list[str]
            Names of parameters whose latest norm is below *threshold* if any
            were found.
        """
        if not self._history:
            return False
        norms = self._history[-1]["norms"]
        bad = [name for name, norm in norms.items() if norm < threshold]
        return bad if bad else False

    def detect_exploding(self, threshold: float = 100.0) -> bool | list[str]:
        """
        Check for exploding gradients across all recorded steps.

        Returns
        -------
        False
            If no exploding gradients were detected in the most recent step.
        list[str]
            Names of parameters whose latest norm is above *threshold* if any
            were found.
        """
        if not self._history:
            return False
        norms = self._history[-1]["norms"]
        bad = [name for name, norm in norms.items() if norm > threshold]
        return bad if bad else False

    # ------------------------------------------------------------------
    # History / summary
    # ------------------------------------------------------------------

    def get_history(self, param_name: Optional[str] = None) -> list[dict]:
        """
        Return the full recording history.

        Parameters
        ----------
        param_name
            If given, each entry is filtered to ``{"step": ..., "norm": float}``
            for that specific parameter only.  Steps where the parameter had no
            gradient are omitted.

        Returns
        -------
        list[dict]
            If *param_name* is None: list of ``{"step": int, "norms": dict}``.
            If *param_name* is set:  list of ``{"step": int, "norm": float}``.
        """
        if param_name is None:
            return list(self._history)

        filtered = []
        for entry in self._history:
            if param_name in entry["norms"]:
                filtered.append({"step": entry["step"], "norm": entry["norms"][param_name]})
        return filtered

    def summary(self) -> dict:
        """
        Return a summary dict over all recorded steps.

        Keys
        ----
        total_norm : float
            Global L2 norm from the most recent step.
        mean_norm : float
            Mean per-parameter norm across the most recent step.
        max_norm : float
            Maximum per-parameter norm in the most recent step.
        min_norm : float
            Minimum per-parameter norm in the most recent step.
        max_norm_param : str | None
            Name of the parameter with the highest norm.
        num_params : int
            Number of parameters that had gradients in the most recent step.
        num_steps : int
            Total number of steps recorded.
        """
        if not self._history:
            return {
                "total_norm": 0.0,
                "mean_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "max_norm_param": None,
                "num_params": 0,
                "num_steps": 0,
            }

        latest_norms = self._history[-1]["norms"]
        values = list(latest_norms.values())

        if not values:
            return {
                "total_norm": 0.0,
                "mean_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "max_norm_param": None,
                "num_params": 0,
                "num_steps": len(self._history),
            }

        total_norm = math.sqrt(sum(v ** 2 for v in values))
        max_norm_param = max(latest_norms, key=latest_norms.__getitem__)

        return {
            "total_norm": total_norm,
            "mean_norm": sum(values) / len(values),
            "max_norm": max(values),
            "min_norm": min(values),
            "max_norm_param": max_norm_param,
            "num_params": len(values),
            "num_steps": len(self._history),
        }

    # ------------------------------------------------------------------
    # Per-layer aggregation (useful for display / logging)
    # ------------------------------------------------------------------

    def get_layer_stats(self, model: nn.Module) -> list[LayerGradStats]:
        """
        Compute ``LayerGradStats`` for every parameter in *model* (using
        current ``.grad`` tensors).  Groups by the first 3 name components
        to produce logical layer-level stats.

        Returns a list sorted by layer name.
        """
        # Collect per-parameter stats first
        param_stats: list[LayerGradStats] = []
        for name, param in model.named_parameters():
            has_grad = param.grad is not None
            norm = param.grad.detach().norm(2).item() if has_grad else 0.0
            param_stats.append(
                LayerGradStats(
                    name=name,
                    norm=norm,
                    num_params=param.numel(),
                    has_grad=has_grad,
                    is_frozen=not param.requires_grad,
                )
            )

        # Aggregate to logical layers
        layer_sums: dict[str, dict] = defaultdict(
            lambda: {"sq_sum": 0.0, "n_params": 0, "any_grad": False, "is_frozen": True}
        )
        for stat in param_stats:
            parts = stat.name.split(".")
            key = ".".join(parts[:3]) if len(parts) >= 3 else stat.name
            layer_sums[key]["sq_sum"] += stat.norm ** 2
            layer_sums[key]["n_params"] += stat.num_params
            if stat.has_grad:
                layer_sums[key]["any_grad"] = True
            if not stat.is_frozen:
                layer_sums[key]["is_frozen"] = False

        results = []
        for layer_name, agg in layer_sums.items():
            results.append(
                LayerGradStats(
                    name=layer_name,
                    norm=math.sqrt(agg["sq_sum"]),
                    num_params=agg["n_params"],
                    has_grad=agg["any_grad"],
                    is_frozen=agg["is_frozen"],
                )
            )

        return sorted(results, key=lambda s: s.name)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"GradientNormMonitor("
            f"steps_recorded={len(self._history)}, "
            f"vanishing_threshold={self.vanishing_threshold}, "
            f"exploding_threshold={self.exploding_threshold})"
        )
