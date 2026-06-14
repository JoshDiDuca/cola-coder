"""Spectral-Alignment (SA) training-stability diagnostic (EVAL-035).

Background (arXiv:2510.04202 "Spectral Alignment"): for each linear layer with
weight ``W``, the SA of a forward pass is the cosine between the layer's response
and ``u1(W)`` — the PRINCIPAL LEFT singular vector of ``W`` (the output-space
direction that ``W`` amplifies most). A *healthy* layer has a SIGN-BALANCED SA
distribution: roughly half the token positions align positively with ``u1`` and
half negatively (mean ≈ 0). An impending loss explosion announces itself first as
SIGN-COLLAPSE — the alignments all shift to one sign (all-positive or all-negative)
*before* the loss actually diverges. The "sign-collapse fraction" (share of values
on the majority sign; 0.5 = healthy, → 1.0 = collapsed) is therefore an EARLY
divergence-risk scalar.

This module is a DIAGNOSTIC over a saved checkpoint: it runs forward passes and
inspects weights. It is MAIN-SAFE — no training, no new weights, no architecture
mutation, no checkpoint writes. It REUSES depth_profile's block-iteration pattern
(``model.blocks`` with ``rope_freqs`` / ``start_pos=0`` / ``use_cache=False``) and
the real ``TransformerBlock`` attribute names (``attn_norm`` / ``attention.q_proj`` /
``ffn_norm`` / ``ffn.down_proj``), so it introduces no new forward logic.

The principal left singular vector is found by cheap POWER ITERATION (a handful of
matrix-vector products), never a full SVD — and deliberately NOT Muon's
zeropower/Newton-Schulz orthogonalization (that whitens the whole spectrum for an
optimizer step; here we want only the single dominant direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# Probe-name → (block submodule attr, norm attr that feeds it). The norm is the
# REAL input activation the weight multiplies (pre-norm transformer: q_proj sees
# attn_norm(x); down_proj sees the FFN's gated hidden, so we recompute it).
_PROBE_KEYS = ("q", "fc2")


@torch.no_grad()
def principal_left_singular_vector(W: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Principal LEFT singular vector ``u1`` of a 2-D weight via power iteration.

    Power iteration on ``W W^T`` (implicitly): start from a random ``v`` in the
    input space, repeatedly map ``u = W v`` (output space), ``v = W^T u`` (input
    space), renormalizing each step. ``u`` converges to the left singular vector
    of the largest singular value — the output-space direction ``W`` amplifies
    most. A handful of iterations suffices for a diagnostic (no full SVD).

    This is NOT Muon's Newton-Schulz/zeropower orthogonalization: that whitens the
    entire spectrum to ``U V^T`` for an optimizer step; here we want only the
    single dominant direction, which power iteration gives far more cheaply.

    Args:
        W: A 2-D weight tensor, shape ``(out, in)`` (PyTorch ``nn.Linear.weight``).
        iters: Number of power-iteration steps (default 8).

    Returns:
        Unit-norm left singular vector ``u1``, shape ``(out,)``, on ``W``'s device.
    """
    if W.dim() != 2:
        raise ValueError(f"principal_left_singular_vector expects a 2-D weight; got {W.shape}")
    out_dim, in_dim = W.shape
    Wf = W.detach().float()
    # Deterministic-but-arbitrary start in the INPUT space (seeded by caller).
    v = torch.randn(in_dim, device=Wf.device, dtype=Wf.dtype)
    v = v / v.norm().clamp_min(1e-12)
    u = torch.zeros(out_dim, device=Wf.device, dtype=Wf.dtype)
    for _ in range(max(1, iters)):
        u = Wf @ v  # (out,)
        u = u / u.norm().clamp_min(1e-12)
        v = Wf.t() @ u  # (in,)
        v = v / v.norm().clamp_min(1e-12)
    return u


def _probe_weight(block, key: str) -> torch.Tensor | None:
    """Resolve a probe key to its REAL weight tensor on the block (or None)."""
    if key == "q":
        attn = getattr(block, "attention", None)
        proj = getattr(attn, "q_proj", None) if attn is not None else None
        return proj.weight if proj is not None else None
    if key == "fc2":
        ffn = getattr(block, "ffn", None)
        # Dense SwiGLU second linear is down_proj; MoE blocks have no single fc2.
        proj = getattr(ffn, "down_proj", None) if ffn is not None else None
        return proj.weight if proj is not None else None
    raise ValueError(f"Unknown probe key '{key}'. Choose from {_PROBE_KEYS}.")


def _probe_response(block, key: str, h: torch.Tensor) -> torch.Tensor | None:
    """Response activation ``W·(real input)`` for a probe, shape ``(seq, out)``.

    Mirrors the block's actual pre-norm wiring so the probed input is the genuine
    activation the weight multiplies:
      - ``q``  → ``q_proj(attn_norm(h))``
      - ``fc2``→ ``down_proj(silu(gate_proj(x)) * up_proj(x))`` with ``x = ffn_norm(h)``
    ``h`` is the residual stream entering the block (single sequence, ``(seq, dim)``).
    Returns None when the probe's submodule is absent (e.g. fc2 on a MoE block).
    """
    if key == "q":
        attn = getattr(block, "attention", None)
        norm = getattr(block, "attn_norm", None)
        proj = getattr(attn, "q_proj", None) if attn is not None else None
        if proj is None or norm is None:
            return None
        return proj(norm(h))
    if key == "fc2":
        ffn = getattr(block, "ffn", None)
        norm = getattr(block, "ffn_norm", None)
        if ffn is None or norm is None:
            return None
        gate = getattr(ffn, "gate_proj", None)
        up = getattr(ffn, "up_proj", None)
        down = getattr(ffn, "down_proj", None)
        if gate is None or up is None or down is None:
            return None  # MoE block — no single dense fc2 to probe
        x = norm(h)
        return down(F.silu(gate(x)) * up(x))
    raise ValueError(f"Unknown probe key '{key}'. Choose from {_PROBE_KEYS}.")


@torch.no_grad()
def spectral_alignment(
    model,
    input_ids: torch.Tensor,
    *,
    probes: tuple[str, ...] = ("q",),
    iters: int = 8,
) -> dict[int, torch.Tensor]:
    """Per-layer Spectral Alignment over one sequence's forward pass.

    For each block, captures the residual stream ENTERING the block (reusing the
    depth_profile iteration: ``block(h, rope_freqs=..., start_pos=0,
    use_cache=False)``), computes the probed weight's response ``W·(input)`` per
    token position, and reports the cosine of that response with ``u1(W)`` — the
    SA value for that (token, layer). Cosine with the response (not the raw input)
    keeps the alignment dimensionally valid for non-square projections and is the
    meaningful quantity: how aligned the layer's OUTPUT is with the direction it
    amplifies most.

    When several probes are given, their per-token SA values are concatenated for
    the layer (a layer's stability signal pools its probed weights).

    Args:
        model: A ``Transformer`` (or stub) exposing ``blocks`` / ``tok_emb`` /
            ``rope_freqs`` and the real block attribute names. Single sequence.
        input_ids: Token ids, shape ``(seq,)`` or ``(1, seq)``.
        probes: Which weights to probe per block (subset of ``("q", "fc2")``).
        iters: Power-iteration steps for ``u1``.

    Returns:
        ``{layer_index: sa_values}`` where ``sa_values`` is a 1-D tensor of SA
        cosines (one per token position, per probe present in that layer).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.size(0) != 1:
        raise ValueError(
            f"spectral_alignment expects a single sequence; got batch={input_ids.size(0)}"
        )

    device = model.tok_emb.weight.device
    input_ids = input_ids.to(device)
    rope_freqs = getattr(model, "rope_freqs", None)

    h = model.tok_emb(input_ids)  # (1, seq, dim)

    per_layer: dict[int, torch.Tensor] = {}
    for layer_idx, block in enumerate(model.blocks):
        h_in = h.squeeze(0)  # (seq, dim) — residual stream entering this block
        sa_parts: list[torch.Tensor] = []
        for key in probes:
            W = _probe_weight(block, key)
            response = _probe_response(block, key, h_in)
            if W is None or response is None or W.dim() != 2:
                continue
            u1 = principal_left_singular_vector(W, iters=iters)  # (out,)
            # Cosine of each token's response row with u1 → (seq,) SA values.
            sa = F.cosine_similarity(
                response.float(), u1.unsqueeze(0).to(response.dtype).float(), dim=-1
            )
            sa_parts.append(sa)
        if sa_parts:
            per_layer[layer_idx] = torch.cat(sa_parts)
        # Advance the residual stream through the real block (depth_profile pattern).
        h = block(h, rope_freqs=rope_freqs, start_pos=0, use_cache=False)

    return per_layer


def sign_collapse_stat(sa_values: torch.Tensor) -> float:
    """Fraction of SA values on the MAJORITY sign (0.5 healthy → 1.0 collapsed).

    Zeros are excluded (a zero alignment carries no sign); if every value is zero
    (or the tensor is empty) the result is a neutral ``0.5``. A sign-balanced
    distribution scores ~0.5; an all-positive or all-negative distribution scores
    1.0 — the early-warning extreme.

    Args:
        sa_values: 1-D tensor of SA cosines.

    Returns:
        Majority-sign fraction in ``[0.5, 1.0]``.
    """
    if sa_values.numel() == 0:
        return 0.5
    pos = int((sa_values > 0).sum().item())
    neg = int((sa_values < 0).sum().item())
    total = pos + neg
    if total == 0:
        return 0.5
    return max(pos, neg) / total


@dataclass
class SpectralHealthReport:
    """Aggregate Spectral-Alignment health over a set of sequences.

    Attributes:
        per_layer: One dict per layer: ``{"layer", "sa_mean", "sign_collapse", "n"}``
            (``sa_mean`` is the mean SA cosine; ``sign_collapse`` the majority-sign
            fraction; ``n`` the count of SA values pooled for that layer).
        worst_layer: Index of the layer with the highest sign-collapse (-1 if none).
        worst_collapse: That layer's sign-collapse fraction (0.5 when no data).
        n_layers: Number of transformer layers.
        by_tier: Optional ``{tier_label: {worst_layer, worst_collapse, per_layer}}``
            when difficulty tiers are supplied.
    """

    per_layer: list[dict]
    worst_layer: int
    worst_collapse: float
    n_layers: int
    by_tier: dict[str, dict] | None = field(default=None)


def _summarize_layers(
    layer_values: dict[int, list[torch.Tensor]], n_layers: int
) -> dict:
    """Build per-layer SA mean + sign-collapse and pick the worst layer."""
    per_layer: list[dict] = []
    worst_layer = -1
    worst_collapse = 0.5
    for layer_idx in range(n_layers):
        parts = layer_values.get(layer_idx)
        if not parts:
            continue
        sa = torch.cat(parts)
        collapse = sign_collapse_stat(sa)
        per_layer.append({
            "layer": layer_idx,
            "sa_mean": float(sa.mean().item()),
            "sign_collapse": collapse,
            "n": int(sa.numel()),
        })
        if collapse > worst_collapse or worst_layer == -1:
            worst_collapse = collapse
            worst_layer = layer_idx
    return {
        "per_layer": per_layer,
        "worst_layer": worst_layer,
        "worst_collapse": worst_collapse,
    }


def profile_spectral_health(
    model,
    sequences,
    *,
    probes: tuple[str, ...] = ("q",),
    iters: int = 8,
    by_tier: list[str] | None = None,
) -> SpectralHealthReport:
    """Profile Spectral-Alignment health over a list of token-id sequences.

    Aggregates per-layer SA values across the fed sequences, then computes each
    layer's mean SA and sign-collapse fraction and flags the worst (highest
    sign-collapse) layer — the earliest divergence-risk signal.

    Args:
        model: A ``Transformer`` (or compatible stub).
        sequences: Iterable of token-id sequences (1-D ``LongTensor`` or ``list[int]``).
        probes: Which block weights to probe (subset of ``("q", "fc2")``).
        iters: Power-iteration steps for ``u1``.
        by_tier: Optional tier labels (from ``difficulty_profile.TIERS``) aligned
            1:1 with ``sequences``; adds a per-tier breakdown to the report.

    Returns:
        A :class:`SpectralHealthReport`. Empty input yields a safe neutral report.
    """
    sequences = list(sequences)
    n_layers = len(model.blocks)

    if by_tier is not None and len(by_tier) != len(sequences):
        raise ValueError(
            f"by_tier length ({len(by_tier)}) must match sequences length ({len(sequences)})"
        )

    all_layers: dict[int, list[torch.Tensor]] = {}
    tier_layers: dict[str, dict[int, list[torch.Tensor]]] = {}

    for i, seq in enumerate(sequences):
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.long)
        if seq.numel() < 1:
            continue
        sa_by_layer = spectral_alignment(model, seq, probes=probes, iters=iters)
        for layer_idx, sa in sa_by_layer.items():
            all_layers.setdefault(layer_idx, []).append(sa)
            if by_tier is not None:
                tier = by_tier[i]
                tier_layers.setdefault(tier, {}).setdefault(layer_idx, []).append(sa)

    summary = _summarize_layers(all_layers, n_layers)

    by_tier_out: dict[str, dict] | None = None
    if by_tier is not None:
        by_tier_out = {}
        for tier, layers in tier_layers.items():
            s = _summarize_layers(layers, n_layers)
            by_tier_out[tier] = {
                "worst_layer": s["worst_layer"],
                "worst_collapse": s["worst_collapse"],
                "per_layer": s["per_layer"],
            }

    return SpectralHealthReport(
        per_layer=summary["per_layer"],
        worst_layer=summary["worst_layer"],
        worst_collapse=summary["worst_collapse"],
        n_layers=n_layers,
        by_tier=by_tier_out,
    )
