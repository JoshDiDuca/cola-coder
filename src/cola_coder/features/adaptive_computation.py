"""Adaptive Computation: dynamically adjust compute per token based on difficulty.

Implements early exit from transformer layers when the model is confident enough.
Harder tokens use more layers; easy tokens exit early, saving compute.

For a TS dev: like short-circuit evaluation — if you already know the answer with
high confidence, don't bother evaluating the rest of the expression (layers).

Inspired by the "Patience is a Virtue" and PonderNet lines of work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive computation / early exit.

    confidence_threshold: exit early when confidence exceeds this value (0–1).
    min_layers: always run at least this many transformer layers.
    max_layers: cap layers used; None means use all layers in the base model.
    ponder_lambda: weight for the ponder (layer-count) regularization loss.
    """

    confidence_threshold: float = 0.9
    min_layers: int = 2
    max_layers: Optional[int] = None
    ponder_lambda: float = 0.01


# ---------------------------------------------------------------------------
# Early Exit Classifier
# ---------------------------------------------------------------------------


class EarlyExitClassifier(nn.Module):
    """Small MLP on top of hidden states that predicts a confidence score 0–1.

    Applied after each transformer layer to decide whether to exit early.
    Kept deliberately lightweight so it doesn't dominate compute.

    Input:  hidden  — (batch, seq_len, dim)  or  (batch, dim)
    Output: confidence tensor — (batch, seq_len)  or  (batch,)  — values in [0, 1]
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 4, 16)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return per-position confidence scores.

        Args:
            hidden: (..., dim) — any leading batch/seq dimensions are preserved.

        Returns:
            Tensor with the last dimension squeezed away — (...,) values in [0, 1].
        """
        return self.net(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# Adaptive Transformer wrapper
# ---------------------------------------------------------------------------


class AdaptiveTransformer(nn.Module):
    """Wraps a transformer model to support layer-wise early exit.

    The base model must expose:
      - base_model.layers  (nn.ModuleList of transformer blocks)
      - base_model.embed(token_ids) -> hidden  (embedding lookup + positional)
      - base_model.head(hidden) -> logits  (LM head)

    If the wrapped model does not follow this interface you can subclass and
    override _embed / _run_layer / _lm_head.
    """

    def __init__(self, base_model: nn.Module, config: AdaptiveConfig) -> None:
        super().__init__()
        self.base = base_model
        self.config = config

        # Determine number of layers
        layers = getattr(base_model, "layers", None)
        self.num_layers: int = len(layers) if layers is not None else 0

        max_layers = config.max_layers if config.max_layers is not None else self.num_layers
        self.effective_max_layers = min(max_layers, self.num_layers)

        # One early-exit classifier per layer (lazily sized; needs dim at runtime)
        self._classifiers: Optional[nn.ModuleList] = None

        # Stats tracking (not differentiable — plain Python accumulators)
        self._stat_total_exits: list[int] = [0] * max(self.num_layers, 1)
        self._stat_layers_used: list[int] = []

    # ------------------------------------------------------------------
    # Internal helpers — override these for non-standard base models
    # ------------------------------------------------------------------

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_fn = getattr(self.base, "embed", None)
        if embed_fn is not None:
            return embed_fn(token_ids)
        # Fallback: try common attribute names
        for name in ("embedding", "tok_embeddings", "wte"):
            mod = getattr(self.base, name, None)
            if mod is not None:
                return mod(token_ids)
        raise AttributeError(
            "base_model has no embed() method or known embedding attribute."
        )

    def _run_layer(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        layers = getattr(self.base, "layers", None)
        if layers is None:
            raise AttributeError("base_model.layers not found.")
        return layers[layer_idx](hidden)

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        head_fn = getattr(self.base, "head", None)
        if head_fn is not None:
            return head_fn(hidden)
        for name in ("lm_head", "output"):
            mod = getattr(self.base, name, None)
            if mod is not None:
                return mod(hidden)
        raise AttributeError(
            "base_model has no head() method or known LM-head attribute."
        )

    def _get_or_build_classifiers(self, dim: int) -> nn.ModuleList:
        if self._classifiers is None or len(self._classifiers) != self.effective_max_layers:
            self._classifiers = nn.ModuleList(
                [EarlyExitClassifier(dim) for _ in range(self.effective_max_layers)]
            )
            # Move to same device as base model parameters (best-effort)
            try:
                param = next(self.base.parameters())
                self._classifiers = self._classifiers.to(param.device)
            except StopIteration:
                pass
        return self._classifiers

    # ------------------------------------------------------------------
    # Main adaptive forward
    # ------------------------------------------------------------------

    def forward_adaptive(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Run the transformer with early exit.

        Args:
            token_ids: (batch, seq_len) integer token IDs.

        Returns:
            (logits, layers_used)
              logits      — (batch, seq_len, vocab_size)
              layers_used — number of layers actually executed this call
        """
        hidden = self._embed(token_ids)
        dim = hidden.shape[-1]
        classifiers = self._get_or_build_classifiers(dim)

        cfg = self.config
        layers_used = 0

        for layer_idx in range(self.effective_max_layers):
            hidden = self._run_layer(layer_idx, hidden)
            layers_used += 1

            # Never exit before min_layers
            if layers_used < cfg.min_layers:
                continue

            # Compute per-token confidence, then take the batch-min as
            # the conservative "are we sure?" signal.
            with torch.no_grad():
                conf = classifiers[layer_idx](hidden)  # (batch, seq_len) or (batch,)
                # Scalar: minimum confidence across all positions
                min_conf = conf.min().item()

            if min_conf >= cfg.confidence_threshold:
                # Record which layer triggered the exit
                self._stat_total_exits[layer_idx] += 1
                break

        self._stat_layers_used.append(layers_used)
        logits = self._lm_head(hidden)
        return logits, layers_used

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a dict with average layers used and per-layer exit counts."""
        if not self._stat_layers_used:
            return {
                "avg_layers_used": 0.0,
                "exits_per_layer": list(self._stat_total_exits),
                "total_forward_calls": 0,
            }
        avg = sum(self._stat_layers_used) / len(self._stat_layers_used)
        return {
            "avg_layers_used": avg,
            "exits_per_layer": list(self._stat_total_exits),
            "total_forward_calls": len(self._stat_layers_used),
        }

    def reset_stats(self) -> None:
        self._stat_total_exits = [0] * max(self.num_layers, 1)
        self._stat_layers_used = []


# ---------------------------------------------------------------------------
# Ponder loss
# ---------------------------------------------------------------------------


def compute_ponder_loss(layer_confidences: list[float]) -> torch.Tensor:
    """Penalize using too many layers (PonderNet-style regularisation).

    Encourages the model to exit early by penalising late exits.
    The loss is the expected layer index weighted by (1 - confidence) at each
    layer — intuitively, how long the model kept "pondering" before being sure.

    Args:
        layer_confidences: confidence scores at each layer, in order.
            Values should be in [0, 1].  The list represents a single forward
            pass up to the exit layer.

    Returns:
        Scalar tensor >= 0.  Higher = model stayed in more layers.
    """
    if not layer_confidences:
        return torch.tensor(0.0)

    confs = torch.tensor(layer_confidences, dtype=torch.float32)

    # Probability of *halting* at each step: halt_prob[i] = conf[i] * prod(1-conf[:i])
    # This follows the geometric / PonderNet formulation.
    n = len(confs)
    halt_probs = torch.zeros(n)
    running_not_halted = 1.0
    for i in range(n):
        halt_probs[i] = confs[i] * running_not_halted
        running_not_halted *= (1.0 - confs[i].item())

    # Normalise so probabilities sum to 1 (handle edge case of all-zero confs)
    total = halt_probs.sum()
    if total > 0:
        halt_probs = halt_probs / total

    # Expected step index (1-indexed) — penalise higher expected step
    steps = torch.arange(1, n + 1, dtype=torch.float32)
    ponder_loss = (halt_probs * steps).sum()

    return ponder_loss
