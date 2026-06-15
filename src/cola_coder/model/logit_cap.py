"""Gemma-2-style logit soft-capping.

Soft-capping is a smooth, differentiable clamp that bounds a tensor of logits
to the open interval ``(-cap, +cap)`` without the hard gradient cliff of
``torch.clamp``::

    y = cap * tanh(x / cap)

For ``|x| << cap`` this is the identity (``tanh(z) ~= z`` for small ``z``), so a
large cap is nearly a no-op on the bulk of the distribution while still taming
the rare extreme outliers that drive bf16 logit drift and softmax
over-confidence.

Reference: Gemma 2 (https://arxiv.org/abs/2408.00118) applies this to the
attention logits (pre-softmax) and the final LM-head logits as a
training-stability and calibration lever. cola-coder already ships the
complementary QK-Norm; soft-capping completes the layered logit-magnitude
control stack (see docs/research-log.md 2026-06-15).

This module is intentionally torch-only and side-effect-free so it can be unit
tested in isolation and reused by both the attention and output paths.
"""

import torch


def soft_cap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """Smoothly bound ``logits`` to ``(-cap, +cap)`` via ``cap * tanh(x / cap)``.

    When ``cap`` is falsy or non-positive the input is returned unchanged
    (identity), so this is safe to call unconditionally on a disabled path —
    the result is byte-identical to not calling it at all.

    Args:
        logits: Any float tensor (e.g. pre-softmax attention scores or final
            vocabulary logits). Shape and dtype are preserved.
        cap: The soft-cap magnitude. ``<= 0`` (or ``0.0``) disables capping.

    Returns:
        The soft-capped tensor (same shape/dtype as ``logits``), or ``logits``
        unchanged when ``cap <= 0``.
    """
    if not cap or cap <= 0:
        return logits
    return cap * torch.tanh(logits / cap)
