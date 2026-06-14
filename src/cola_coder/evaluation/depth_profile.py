"""Logit-lens per-token DEPTH / convergence profiler (INFER-031).

The "logit lens" (nostalgebraist 2020) decodes EVERY transformer layer's residual
stream through the model's OWN final norm + tied output head, turning each layer's
hidden state into a next-token distribution. Reading those distributions top-to-bottom
shows how the model's prediction crystallizes with depth.

This module measures, per token position, the EARLIEST layer at which the next-token
prediction has effectively converged to the final layer's answer (the token's
"exit depth"). Aggregated over a few sequences it answers: how many layers does THIS
model actually need per token? Low mean exit depth ⇒ early-exit / layer-skipping is
on the table; a long tail of deep-exiting tokens marks the genuinely hard positions.

MAIN-SAFE: pure analysis over forward passes. No training, no new weights, no
architecture mutation, no checkpoint writes. It reuses the model's existing block
iteration (mirrors Transformer.get_hidden_states / forward), the TIED output head,
and the final norm — it never adds a projection of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@torch.no_grad()
def logit_lens(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Decode every transformer layer's hidden state through the tied output head.

    Runs one forward pass, capturing the residual-stream hidden state AFTER each
    block, then applies the model's OWN ``final_norm`` + tied ``output`` head to each
    layer's hidden state. The last layer's row is therefore identical (up to the
    capture point) to a normal ``model.forward`` — that's the logit-lens anchor.

    Block iteration mirrors ``Transformer.get_hidden_states`` / ``Transformer.forward``
    (rope_freqs from the model, ``start_pos=0``, ``use_cache=False``), so no new
    parameters or forward logic are introduced.

    Args:
        model: A ``Transformer`` (or a stub exposing ``blocks`` / ``tok_emb`` /
            ``final_norm`` / ``output`` / ``rope_freqs``). A single un-batched
            sequence is assumed (batch dim, if present, must be 1).
        input_ids: Token ids, shape ``(seq,)`` or ``(1, seq)``.

    Returns:
        Per-layer logits, shape ``(n_layers, seq, vocab)``, on the model's device.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # (1, seq)
    if input_ids.size(0) != 1:
        raise ValueError(f"logit_lens expects a single sequence; got batch={input_ids.size(0)}")

    device = model.tok_emb.weight.device
    input_ids = input_ids.to(device)

    rope_freqs = getattr(model, "rope_freqs", None)

    # Embedding (no dropout — eval-only analysis; dropout would add noise to the lens).
    h = model.tok_emb(input_ids)  # (1, seq, dim)

    per_layer: list[torch.Tensor] = []
    for block in model.blocks:
        h = block(h, rope_freqs=rope_freqs, start_pos=0, use_cache=False)
        # Decode THIS layer's residual stream with the shared final norm + tied head.
        logits = model.output(model.final_norm(h))  # (1, seq, vocab)
        per_layer.append(logits.squeeze(0))  # (seq, vocab)

    return torch.stack(per_layer, dim=0)  # (n_layers, seq, vocab)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of the softmax over the last dim.

    Computed via ``log_softmax`` for numerical stability (the standard, NaN-safe
    formulation: ``-sum(p * log p)`` where ``log p = log_softmax``).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def convergence_depth(
    per_layer_logits: torch.Tensor,
    *,
    mode: str = "argmax",
    tau: float = 0.0,
) -> torch.Tensor:
    """Per position, the earliest layer index at which the prediction "converged".

    Two notions of convergence:

    - ``mode="argmax"`` (default): the earliest layer whose top-1 token equals the
      FINAL layer's top-1 token AND stays equal for every layer below it. Requiring
      stability (not just a first match) avoids crediting a transient early agreement
      that the model later overturns.
    - ``mode="entropy"``: the earliest layer whose softmax entropy (nats) is ``<= tau``.
      Lower ``tau`` ⇒ a stricter confidence bar ⇒ a deeper (>=) exit.

    The last layer always qualifies, so every position has a defined depth in
    ``[0, n_layers - 1]``.

    Args:
        per_layer_logits: Shape ``(n_layers, seq, vocab)`` (from :func:`logit_lens`).
        mode: ``"argmax"`` or ``"entropy"``.
        tau: Entropy threshold in nats (entropy mode only).

    Returns:
        ``LongTensor`` of shape ``(seq,)`` with each position's exit-depth index.
    """
    n_layers, seq, _ = per_layer_logits.shape
    last = n_layers - 1

    if mode == "argmax":
        preds = per_layer_logits.argmax(dim=-1)  # (n_layers, seq)
        final_pred = preds[last]  # (seq,)
        matches = preds == final_pred.unsqueeze(0)  # (n_layers, seq)
        # Stable convergence: the earliest layer from which ALL deeper layers match.
        # Walk bottom-up so each position latches the shallowest sustained match.
        depth = torch.full((seq,), last, dtype=torch.long, device=per_layer_logits.device)
        ongoing = torch.ones(seq, dtype=torch.bool, device=per_layer_logits.device)
        for layer in range(last, -1, -1):
            ongoing = ongoing & matches[layer]
            depth = torch.where(ongoing, torch.full_like(depth, layer), depth)
        return depth

    if mode == "entropy":
        ent = _entropy(per_layer_logits)  # (n_layers, seq)
        confident = ent <= tau  # (n_layers, seq)
        confident[last] = True  # last layer always qualifies
        # Earliest layer index that is confident, per position.
        first_idx = confident.float().argmax(dim=0)  # (seq,) — argmax finds first True
        return first_idx.to(torch.long)

    raise ValueError(f"Unknown mode '{mode}'. Choose 'argmax' or 'entropy'.")


@dataclass
class DepthReport:
    """Aggregate exit-depth statistics over a set of sequences.

    Attributes:
        mean_exit_depth: Mean exit-depth index across all profiled token positions.
        median_exit_depth: Median exit-depth index.
        n_layers: Number of transformer layers (max possible depth + 1).
        n_tokens: Total token positions aggregated.
        frac_converged_by_depth: For each depth ``d`` in ``[0, n_layers)``, the
            fraction of tokens whose exit depth is ``<= d`` (a cumulative curve;
            the last entry is always 1.0 when there are tokens).
        by_tier: Optional ``{tier_label: {mean/median/n_tokens/...}}`` when
            difficulty tiers are supplied.
    """

    mean_exit_depth: float
    median_exit_depth: float
    n_layers: int
    n_tokens: int
    frac_converged_by_depth: list[float]
    by_tier: dict[str, dict] | None = field(default=None)


def _summarize(depths: list[int], n_layers: int) -> dict:
    """Build the scalar summary for a flat list of per-token exit depths."""
    n_tokens = len(depths)
    if n_tokens == 0:
        return {
            "mean_exit_depth": 0.0,
            "median_exit_depth": 0.0,
            "n_tokens": 0,
            "frac_converged_by_depth": [0.0] * n_layers,
        }
    t = torch.tensor(depths, dtype=torch.float)
    cumulative = [
        float((t <= d).float().mean().item()) for d in range(n_layers)
    ]
    return {
        "mean_exit_depth": float(t.mean().item()),
        "median_exit_depth": float(t.median().item()),
        "n_tokens": n_tokens,
        "frac_converged_by_depth": cumulative,
    }


def profile_depth(
    model,
    sequences,
    *,
    mode: str = "argmax",
    tau: float = 0.0,
    difficulty_tiers: list[str] | None = None,
) -> DepthReport:
    """Profile exit depth over a list of token-id sequences.

    Args:
        model: A ``Transformer`` (or compatible stub) — see :func:`logit_lens`.
        sequences: Iterable of token-id sequences. Each may be a 1-D ``LongTensor``
            or a ``list[int]``.
        mode: ``"argmax"`` or ``"entropy"`` (see :func:`convergence_depth`).
        tau: Entropy threshold (entropy mode only).
        difficulty_tiers: Optional list of tier labels (from
            ``difficulty_profile.TIERS``) aligned 1:1 with ``sequences``. When given,
            the report also carries a per-tier breakdown in ``by_tier``.

    Returns:
        A :class:`DepthReport`. Empty input yields a safe all-zero report.
    """
    sequences = list(sequences)
    n_layers = len(model.blocks)

    if difficulty_tiers is not None and len(difficulty_tiers) != len(sequences):
        raise ValueError(
            f"difficulty_tiers length ({len(difficulty_tiers)}) must match "
            f"sequences length ({len(sequences)})"
        )

    all_depths: list[int] = []
    per_tier: dict[str, list[int]] = {}

    for i, seq in enumerate(sequences):
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.long)
        if seq.numel() == 0:
            continue
        per_layer = logit_lens(model, seq)
        depths = convergence_depth(per_layer, mode=mode, tau=tau).tolist()
        all_depths.extend(depths)
        if difficulty_tiers is not None:
            per_tier.setdefault(difficulty_tiers[i], []).extend(depths)

    summary = _summarize(all_depths, n_layers)

    by_tier: dict[str, dict] | None = None
    if difficulty_tiers is not None:
        by_tier = {
            tier: {**_summarize(depths, n_layers), "n_layers": n_layers}
            for tier, depths in per_tier.items()
        }

    return DepthReport(
        mean_exit_depth=summary["mean_exit_depth"],
        median_exit_depth=summary["median_exit_depth"],
        n_layers=n_layers,
        n_tokens=summary["n_tokens"],
        frac_converged_by_depth=summary["frac_converged_by_depth"],
        by_tier=by_tier,
    )
