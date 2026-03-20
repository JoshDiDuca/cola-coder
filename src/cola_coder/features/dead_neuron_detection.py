"""
Feature: Dead Neuron Detection

Detect neurons that never activate (output is always 0 or near-0) across
forward passes. Tracks per-neuron activation statistics and reports dead
neuron percentage per layer.

Designed for diagnostic use after training, not during training.
"""

from __future__ import annotations

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402


@dataclass
class ActivationStats:
    """Per-layer activation statistics."""

    layer_name: str
    n_neurons: int
    mean_activation: float
    max_activation: float
    activation_rate: float   # Fraction of tokens where |activation| > threshold
    dead_neurons: int        # Count with max_activation < threshold
    weak_neurons: int        # Count with activation_rate < 0.01 but not fully dead

    @property
    def dead_pct(self) -> float:
        return self.dead_neurons / self.n_neurons * 100 if self.n_neurons > 0 else 0.0

    @property
    def weak_pct(self) -> float:
        return self.weak_neurons / self.n_neurons * 100 if self.n_neurons > 0 else 0.0

    @property
    def health_color(self) -> str:
        if self.dead_pct > 20:
            return "red"
        if self.dead_pct > 5 or self.weak_pct > 20:
            return "yellow"
        return "green"


class DeadNeuronDetector:
    """
    Detects dead neurons in a PyTorch model by attaching forward hooks and
    accumulating activation statistics across multiple forward passes.

    Usage:
        detector = DeadNeuronDetector()
        detector.register_hooks(model)

        for batch in data_loader:
            model(batch)  # hooks fire automatically

        dead = detector.get_dead_neurons()
        pcts = detector.get_dead_percentage()
        print(detector.summary())

        detector.remove_hooks()
    """

    def __init__(self) -> None:
        # layer_name -> list of per-batch max absolute activations: shape (n_neurons,)
        self._max_acts: dict[str, torch.Tensor] = {}
        # layer_name -> running sum of absolute activations (for mean)
        self._sum_acts: dict[str, torch.Tensor] = {}
        # layer_name -> running count of tokens seen
        self._token_counts: dict[str, int] = defaultdict(int)
        # layer_name -> running count of tokens where |act| > threshold (used with threshold=1e-6)
        self._active_counts: dict[str, torch.Tensor] = {}
        # layer_name -> n_neurons (last seen)
        self._n_neurons: dict[str, int] = {}

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._threshold: float = 1e-6  # default threshold used during accumulation

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, model: nn.Module) -> None:
        """
        Attach forward hooks to all trackable layers in the model.

        Hooks fire automatically on every forward pass. Layers targeted:
        - nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh (activation modules)
        - nn.Linear layers (output activations)

        All layer types are included so the detector works with arbitrary
        architectures, not just Cola-Coder's SwiGLU transformer.
        """
        self.remove_hooks()  # clean up any previous hooks

        target_types = (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Linear)

        for name, module in model.named_modules():
            if name == "":
                continue  # skip the model root
            if isinstance(module, target_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Return a forward hook closure for the given layer name."""

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self.update(layer_name, output)

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Statistics accumulation
    # ------------------------------------------------------------------

    def update(self, layer_name: str, output: torch.Tensor) -> None:
        """
        Called automatically by the forward hook. Updates running activation
        statistics for the given layer.

        Can also be called manually if you want to feed activations directly.

        Args:
            layer_name: Identifier for the layer.
            output: The layer's output tensor. Accepts shapes:
                    - (batch, n_neurons)
                    - (batch, seq_len, n_neurons)   <- transformer hidden states
                    - (batch, channels, h, w)        <- conv feature maps
                    Any shape with last dim = n_neurons (or channels folded in).
        """
        if isinstance(output, tuple):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            return

        # Move to CPU immediately to avoid accumulating VRAM
        acts = output.detach().cpu().float()

        # Flatten all dims except the last one (treated as neuron index)
        # shape -> (total_tokens, n_neurons)
        n_neurons = acts.shape[-1]
        flat = acts.reshape(-1, n_neurons)  # (N, n_neurons)
        abs_flat = flat.abs()

        n_tokens = flat.shape[0]

        # Initialize accumulators on first call
        if layer_name not in self._max_acts:
            self._max_acts[layer_name] = torch.zeros(n_neurons)
            self._sum_acts[layer_name] = torch.zeros(n_neurons)
            self._active_counts[layer_name] = torch.zeros(n_neurons)
            self._n_neurons[layer_name] = n_neurons

        # Update running max per neuron
        batch_max = abs_flat.max(dim=0).values  # (n_neurons,)
        self._max_acts[layer_name] = torch.maximum(self._max_acts[layer_name], batch_max)

        # Update running sum and token count
        self._sum_acts[layer_name] += abs_flat.sum(dim=0)
        self._token_counts[layer_name] += n_tokens

        # Count tokens where |act| > threshold
        active = (abs_flat > self._threshold).float().sum(dim=0)
        self._active_counts[layer_name] += active

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_dead_neurons(self, threshold: float = 1e-6) -> dict[str, int]:
        """
        Return the number of dead neurons per layer.

        A neuron is considered dead if its maximum absolute activation across
        all observed tokens is below `threshold`.

        Args:
            threshold: Activation magnitude below which a neuron is "dead".

        Returns:
            dict mapping layer_name -> count of dead neurons.
        """
        result: dict[str, int] = {}
        for name, max_acts in self._max_acts.items():
            dead = int((max_acts < threshold).sum().item())
            result[name] = dead
        return result

    def get_dead_percentage(self, threshold: float = 1e-6) -> dict[str, float]:
        """
        Return the percentage of dead neurons per layer.

        Args:
            threshold: Same as in `get_dead_neurons`.

        Returns:
            dict mapping layer_name -> percentage (0-100) of dead neurons.
        """
        dead_counts = self.get_dead_neurons(threshold=threshold)
        result: dict[str, float] = {}
        for name, dead in dead_counts.items():
            n = self._n_neurons.get(name, 0)
            result[name] = dead / n * 100.0 if n > 0 else 0.0
        return result

    def _compute_stats(self, threshold: float = 1e-6) -> list[ActivationStats]:
        """Compute full ActivationStats for every tracked layer."""
        stats: list[ActivationStats] = []

        for name in self._max_acts:
            max_acts = self._max_acts[name]       # (n_neurons,)
            sum_acts = self._sum_acts[name]        # (n_neurons,)
            active_counts = self._active_counts[name]  # (n_neurons,)
            n_tokens = self._token_counts[name]
            n_neurons = self._n_neurons[name]

            if n_tokens == 0:
                continue

            mean_per_neuron = sum_acts / n_tokens          # (n_neurons,)
            act_rate_per_neuron = active_counts / n_tokens  # (n_neurons,)

            dead_mask = max_acts < threshold
            weak_mask = (~dead_mask) & (act_rate_per_neuron < 0.01)

            stats.append(ActivationStats(
                layer_name=name,
                n_neurons=n_neurons,
                mean_activation=float(mean_per_neuron.mean().item()),
                max_activation=float(max_acts.max().item()),
                activation_rate=float(act_rate_per_neuron.mean().item()),
                dead_neurons=int(dead_mask.sum().item()),
                weak_neurons=int(weak_mask.sum().item()),
            ))

        return sorted(stats, key=lambda s: s.dead_pct, reverse=True)

    def summary(self, threshold: float = 1e-6) -> dict:
        """
        Return an overall summary dict with per-layer and aggregate statistics.

        Returns:
            {
                "layers": [
                    {
                        "layer_name": str,
                        "n_neurons": int,
                        "dead_neurons": int,
                        "dead_pct": float,
                        "weak_neurons": int,
                        "weak_pct": float,
                        "mean_activation": float,
                        "max_activation": float,
                        "activation_rate": float,
                    },
                    ...
                ],
                "total_neurons": int,
                "total_dead": int,
                "total_dead_pct": float,
                "total_weak": int,
                "layers_tracked": int,
            }
        """
        stats = self._compute_stats(threshold=threshold)

        layer_dicts = [
            {
                "layer_name": s.layer_name,
                "n_neurons": s.n_neurons,
                "dead_neurons": s.dead_neurons,
                "dead_pct": round(s.dead_pct, 4),
                "weak_neurons": s.weak_neurons,
                "weak_pct": round(s.weak_pct, 4),
                "mean_activation": round(s.mean_activation, 6),
                "max_activation": round(s.max_activation, 6),
                "activation_rate": round(s.activation_rate, 6),
            }
            for s in stats
        ]

        total_neurons = sum(s.n_neurons for s in stats)
        total_dead = sum(s.dead_neurons for s in stats)
        total_weak = sum(s.weak_neurons for s in stats)
        total_dead_pct = total_dead / total_neurons * 100.0 if total_neurons > 0 else 0.0

        return {
            "layers": layer_dicts,
            "total_neurons": total_neurons,
            "total_dead": total_dead,
            "total_dead_pct": round(total_dead_pct, 4),
            "total_weak": total_weak,
            "layers_tracked": len(stats),
        }

    def total_dead_percentage(self, threshold: float = 1e-6) -> float:
        """
        Return a single number: percentage of all tracked neurons that are dead.

        Returns:
            Float in [0, 100].
        """
        dead_counts = self.get_dead_neurons(threshold=threshold)
        total_dead = sum(dead_counts.values())
        total_neurons = sum(self._n_neurons.values())
        if total_neurons == 0:
            return 0.0
        return total_dead / total_neurons * 100.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated statistics (keeps hooks intact)."""
        self._max_acts.clear()
        self._sum_acts.clear()
        self._token_counts.clear()
        self._active_counts.clear()
        self._n_neurons.clear()

    def __repr__(self) -> str:
        n_layers = len(self._n_neurons)
        n_hooks = len(self._hooks)
        total = sum(self._n_neurons.values())
        return (
            f"DeadNeuronDetector("
            f"hooks={n_hooks}, layers_seen={n_layers}, total_neurons={total})"
        )
