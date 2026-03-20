"""Cascade Routing: try small/fast models first, escalate to larger models on low confidence.

This feature saves compute by handling easy requests with small models and only
escalating to larger (more expensive) models when the small model is not confident.

Routing flow:
  small model → if confidence < threshold → medium model → if confidence < threshold → large model

Uses entropy-based confidence scoring: low entropy over output distribution = high confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether cascade routing is enabled."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CascadeConfig:
    """Configuration for cascade routing.

    Attributes:
        confidence_threshold: Minimum confidence [0, 1] required to accept a
            model's output. If the model's confidence is below this value, the
            next (larger) model in the cascade is tried.
        max_cascade_depth: Maximum number of models to try before accepting
            whatever the last model produced regardless of confidence.
    """
    confidence_threshold: float = 0.7
    max_cascade_depth: int = 3


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """Result from a cascade routing run.

    Attributes:
        output: The generated output text (or any object returned by a model).
        model_index: Index into the model list of the model that produced this
            output (0 = smallest/fastest).
        confidence: Confidence score in [0, 1] assigned to this output.
        cascade_depth: Number of models tried before this result was accepted
            (1 = first model was accepted, 2 = first model was rejected and
            second was accepted, etc.).
    """
    output: Any
    model_index: int
    confidence: float
    cascade_depth: int


# ---------------------------------------------------------------------------
# Default confidence function
# ---------------------------------------------------------------------------

def default_confidence_fn(logits: "torch.Tensor") -> float:  # noqa: F821
    """Entropy-based confidence scoring.

    Converts a logits tensor into a scalar confidence value in [0, 1].

    Low entropy over the softmax distribution → the model is concentrated on
    a small number of tokens → high confidence.
    High entropy → the model is spread across many tokens → low confidence.

    The raw entropy is normalised by the maximum possible entropy
    (log(vocab_size)) so the result is always in [0, 1].

    Args:
        logits: Tensor of shape (batch, vocab_size) or (vocab_size,).
                Only the last token's logits are used when batch dim is present.

    Returns:
        Confidence score in [0, 1].  Values close to 1 mean high confidence.
    """
    import torch
    import torch.nn.functional as F

    # Accept shapes:
    #   (vocab,)           — single-token logits, no batch/seq dims
    #   (batch, vocab)     — batched single-token logits
    #   (batch, seq, vocab) — batched multi-token logits (take last token)
    if logits.dim() == 3:
        # (batch, seq, vocab) — take last token of first batch element
        last_logits = logits[0, -1, :]
    elif logits.dim() == 2:
        # (batch, vocab) — take first batch element
        last_logits = logits[0]
    else:
        # (vocab,)
        last_logits = logits

    probs = F.softmax(last_logits.float(), dim=-1)

    # Shannon entropy: H = -sum(p * log(p))
    # Clamp to avoid log(0)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum().item()

    vocab_size = last_logits.shape[-1]
    max_entropy = math.log(vocab_size) if vocab_size > 1 else 1.0

    # Normalise to [0, 1] and invert so that low entropy = high confidence
    normalised_entropy = entropy / max_entropy
    confidence = 1.0 - normalised_entropy

    # Clamp to guard against floating-point drift
    return float(max(0.0, min(1.0, confidence)))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class CascadeRouter:
    """Routes prompts through a cascade of models ordered small → large.

    The router tries each model in order.  It calls *confidence_fn* on the
    model's raw output to obtain a scalar confidence in [0, 1].  If the
    confidence is at or above ``config.confidence_threshold`` the result is
    returned immediately.  Otherwise the next model is tried.  If all models
    are exhausted the result from the last model is returned regardless.

    Usage::

        router = CascadeRouter(models=[small_model, large_model], config=CascadeConfig())
        result = router.route(prompt, confidence_fn=default_confidence_fn)

    The *models* list is expected to be callables with the signature::

        model(prompt: Any) -> (output: Any, logits: torch.Tensor)

    i.e. each model returns a tuple of ``(output, logits)`` where *logits* is
    a tensor suitable for ``confidence_fn``.
    """

    def __init__(
        self,
        models: List[Callable],
        config: Optional[CascadeConfig] = None,
    ) -> None:
        """Initialise the cascade router.

        Args:
            models: List of model callables ordered small → large.  Each
                callable must return ``(output, logits)`` where *logits* is a
                tensor that can be passed to *confidence_fn*.
            config: Cascade configuration.  Uses ``CascadeConfig()`` defaults
                when not provided.
        """
        if not models:
            raise ValueError("CascadeRouter requires at least one model.")
        self.models = models
        self.config = config if config is not None else CascadeConfig()

        # Statistics: how many times each model tier was the final model used
        self._tier_counts: List[int] = [0] * len(models)
        self._total_routes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        prompt: Any,
        confidence_fn: Callable[["torch.Tensor"], float] = default_confidence_fn,
    ) -> CascadeResult:
        """Route *prompt* through the cascade.

        Tries each model in order.  Returns as soon as a model produces output
        with confidence >= ``config.confidence_threshold``, or after
        ``config.max_cascade_depth`` models have been tried.

        Args:
            prompt: The input to pass to each model callable.
            confidence_fn: A function that takes a logits tensor and returns a
                confidence score in [0, 1].  Defaults to
                :func:`default_confidence_fn`.

        Returns:
            A :class:`CascadeResult` describing which model was used, what it
            produced, and how deep the cascade went.
        """
        max_depth = min(self.config.max_cascade_depth, len(self.models))

        last_result: Optional[CascadeResult] = None

        for depth, (model_idx, model) in enumerate(
            zip(range(len(self.models)), self.models), start=1
        ):
            if depth > max_depth:
                break

            output, logits = model(prompt)
            confidence = confidence_fn(logits)

            last_result = CascadeResult(
                output=output,
                model_index=model_idx,
                confidence=confidence,
                cascade_depth=depth,
            )

            if confidence >= self.config.confidence_threshold:
                # Confident enough — accept this result
                self._tier_counts[model_idx] += 1
                self._total_routes += 1
                return last_result

        # Exhausted models (or max_cascade_depth reached) — return last result
        assert last_result is not None
        self._tier_counts[last_result.model_index] += 1
        self._total_routes += 1
        return last_result

    def get_stats(self) -> dict:
        """Return cascade statistics.

        Returns a dict with:

        - ``total_routes``: total number of prompts routed.
        - ``tier_counts``: list of counts, one per model tier, showing how
          many times that tier was ultimately used.
        - ``tier_usage_pct``: list of percentages (0–100) for each tier.

        Example::

            {
                "total_routes": 100,
                "tier_counts": [72, 21, 7],
                "tier_usage_pct": [72.0, 21.0, 7.0],
            }
        """
        total = self._total_routes
        pct = [
            round(100.0 * count / total, 2) if total > 0 else 0.0
            for count in self._tier_counts
        ]
        return {
            "total_routes": total,
            "tier_counts": list(self._tier_counts),
            "tier_usage_pct": pct,
        }

    def reset_stats(self) -> None:
        """Reset all cascade statistics to zero."""
        self._tier_counts = [0] * len(self.models)
        self._total_routes = 0
