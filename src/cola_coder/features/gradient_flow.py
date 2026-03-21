"""Gradient Flow Visualizer — Feature 99

Track gradient magnitudes through the network, detect vanishing/exploding
gradients per layer, and generate diagnostic data for visualization.

Key capabilities
----------------
- Collect per-layer gradient L2 norms after each backward pass.
- Classify each layer as vanishing / healthy / elevated / exploding.
- Produce a time-series of gradient norms per layer for plotting.
- Generate a "gradient flow report" summarising the health of each layer.

Works with or without a live PyTorch model — the data structures and
analysis are pure Python.  The optional :func:`register_hooks` helper
attaches backward hooks to a ``torch.nn.Module`` if torch is available.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if gradient flow tracking is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class GradientHealth(str, Enum):
    VANISHING = "vanishing"
    WEAK = "weak"
    HEALTHY = "healthy"
    ELEVATED = "elevated"
    EXPLODING = "exploding"


_VANISHING_THRESHOLD = 1e-7
_WEAK_THRESHOLD = 1e-4
_ELEVATED_THRESHOLD = 10.0
_EXPLODING_THRESHOLD = 1000.0


def classify_gradient(norm: float) -> GradientHealth:
    """Classify a gradient L2 norm into a health category."""
    if math.isnan(norm) or math.isinf(norm):
        return GradientHealth.EXPLODING
    if norm < _VANISHING_THRESHOLD:
        return GradientHealth.VANISHING
    if norm < _WEAK_THRESHOLD:
        return GradientHealth.WEAK
    if norm <= _ELEVATED_THRESHOLD:
        return GradientHealth.HEALTHY
    if norm <= _EXPLODING_THRESHOLD:
        return GradientHealth.ELEVATED
    return GradientHealth.EXPLODING


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LayerGradRecord:
    """Gradient norm history for a single layer."""

    name: str
    norms: list[float] = field(default_factory=list)

    @property
    def latest(self) -> Optional[float]:
        return self.norms[-1] if self.norms else None

    @property
    def mean(self) -> Optional[float]:
        return statistics.mean(self.norms) if self.norms else None

    @property
    def health(self) -> Optional[GradientHealth]:
        if self.latest is None:
            return None
        return classify_gradient(self.latest)

    @property
    def trend(self) -> Optional[str]:
        """Simple trend: 'increasing', 'decreasing', or 'stable'."""
        if len(self.norms) < 2:
            return None
        delta = self.norms[-1] - self.norms[0]
        span = max(abs(self.norms[-1]), abs(self.norms[0]), 1e-15)
        rel = delta / span
        if rel > 0.2:
            return "increasing"
        if rel < -0.2:
            return "decreasing"
        return "stable"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "latest": self.latest,
            "mean": self.mean,
            "health": self.health.value if self.health else None,
            "trend": self.trend,
            "n_steps": len(self.norms),
        }


@dataclass
class GradientFlowReport:
    """Snapshot of gradient health across all layers at one point in time."""

    step: int
    layers: list[LayerGradRecord]
    global_norm: Optional[float] = None

    @property
    def n_vanishing(self) -> int:
        return sum(1 for rec in self.layers if rec.health == GradientHealth.VANISHING)

    @property
    def n_exploding(self) -> int:
        return sum(
            1
            for rec in self.layers
            if rec.health in (GradientHealth.EXPLODING, GradientHealth.ELEVATED)
        )

    @property
    def is_healthy(self) -> bool:
        return self.n_vanishing == 0 and all(
            rec.health not in (GradientHealth.EXPLODING,) for rec in self.layers
        )

    def layer(self, name: str) -> Optional[LayerGradRecord]:
        for rec in self.layers:
            if rec.name == name:
                return rec
        return None

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "global_norm": self.global_norm,
            "n_layers": len(self.layers),
            "n_vanishing": self.n_vanishing,
            "n_exploding": self.n_exploding,
            "is_healthy": self.is_healthy,
            "layers": [rec.as_dict() for rec in self.layers],
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class GradientFlowTracker:
    """Track per-layer gradient norms over multiple training steps."""

    def __init__(self) -> None:
        self._records: dict[str, LayerGradRecord] = {}
        self._global_norms: list[float] = []
        self._steps: list[int] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        layer_norms: dict[str, float],
        global_norm: Optional[float] = None,
    ) -> None:
        """Record gradient norms for one training step.

        Parameters
        ----------
        step: Global step number.
        layer_norms: Mapping of layer-name → gradient L2 norm.
        global_norm: Overall model gradient norm (optional).
        """
        self._steps.append(step)
        if global_norm is not None:
            self._global_norms.append(global_norm)
        for name, norm in layer_norms.items():
            if name not in self._records:
                self._records[name] = LayerGradRecord(name=name)
            self._records[name].norms.append(norm)

    def clear(self) -> None:
        """Reset all recorded history."""
        self._records.clear()
        self._global_norms.clear()
        self._steps.clear()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def report(self, step: Optional[int] = None) -> GradientFlowReport:
        """Generate a :class:`GradientFlowReport` from current history."""
        s = step if step is not None else (self._steps[-1] if self._steps else 0)
        global_norm = self._global_norms[-1] if self._global_norms else None
        return GradientFlowReport(
            step=s,
            layers=list(self._records.values()),
            global_norm=global_norm,
        )

    def vanishing_layers(self) -> list[str]:
        """Return names of layers currently classified as vanishing."""
        return [
            name
            for name, rec in self._records.items()
            if rec.health == GradientHealth.VANISHING
        ]

    def exploding_layers(self) -> list[str]:
        """Return names of layers currently classified as exploding."""
        return [
            name
            for name, rec in self._records.items()
            if rec.health == GradientHealth.EXPLODING
        ]

    def gradient_history(self, layer_name: str) -> list[float]:
        """Return the full norm history for a named layer."""
        rec = self._records.get(layer_name)
        return list(rec.norms) if rec else []

    def diagnostic_data(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for plotting / logging."""
        return {
            "steps": list(self._steps),
            "global_norms": list(self._global_norms),
            "layers": {
                name: {"norms": list(rec.norms), "health": rec.health.value if rec.health else None}
                for name, rec in self._records.items()
            },
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    @property
    def layer_names(self) -> list[str]:
        return list(self._records.keys())


# ---------------------------------------------------------------------------
# Optional torch hook integration
# ---------------------------------------------------------------------------


def register_hooks(model: Any, tracker: GradientFlowTracker) -> list[Any]:
    """Register backward hooks on a ``torch.nn.Module``.

    Returns a list of hook handles that can be removed via ``handle.remove()``.
    Returns an empty list if torch is not available.
    """
    try:
        import torch.nn as nn
    except ImportError:
        return []

    if not isinstance(model, nn.Module):
        return []

    handles: list[Any] = []

    def _make_hook(name: str) -> Callable:
        def hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            for go in grad_output:
                if go is not None:
                    norm = go.detach().norm().item()
                    # Record inline (step number not tracked here — caller manages)
                    if name not in tracker._records:
                        tracker._records[name] = LayerGradRecord(name=name)
                    tracker._records[name].norms.append(float(norm))

        return hook

    for name, module in model.named_modules():
        if name:  # skip root module
            handle = module.register_full_backward_hook(_make_hook(name))
            handles.append(handle)

    return handles
