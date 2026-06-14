"""Tests for the Spectral-Alignment divergence-risk diagnostic (EVAL-035).

Hermetic + fast: power-iteration and sign-collapse math are checked EXACTLY against
synthetic tensors (no model); profile/SA integration runs against a tiny real
Transformer on CPU plus a faithful stub mirroring the REAL block attribute names
(attn_norm / attention.q_proj / ffn_norm / ffn.gate_proj|up_proj|down_proj). No
checkpoint, no GPU.
"""

from __future__ import annotations

import torch

from cola_coder.evaluation.difficulty_profile import TIERS
from cola_coder.evaluation.spectral_health import (
    SpectralHealthReport,
    principal_left_singular_vector,
    profile_spectral_health,
    sign_collapse_stat,
    spectral_alignment,
)
from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer

VOCAB = 64
SEQ = 12


def _tiny_model(seed: int = 0) -> Transformer:
    """A tiny but real Transformer (CPU): dim=32, 4 layers, vocab=64."""
    torch.manual_seed(seed)
    config = ModelConfig(
        vocab_size=VOCAB,
        dim=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )
    model = Transformer(config)
    model.eval()
    return model


# ── power iteration ──────────────────────────────────────────────────────────


def test_power_iteration_recovers_rank1_left_vector():
    """For W = sigma * u u^T (square symmetric rank-1), u1 recovers u (cosine≈1)."""
    torch.manual_seed(1)
    u = torch.randn(16)
    u = u / u.norm()
    W = 5.0 * torch.outer(u, u)  # rank-1, left singular vector is u
    u1 = principal_left_singular_vector(W, iters=30)
    cos = torch.abs(F_cos(u1, u))
    assert cos > 0.999


def test_power_iteration_rank1_rectangular():
    """Rank-1 rectangular W = u v^T → left singular vector is u (cosine≈1)."""
    torch.manual_seed(2)
    u = torch.randn(10)
    u = u / u.norm()
    v = torch.randn(7)
    v = v / v.norm()
    W = 3.0 * torch.outer(u, v)  # shape (10, 7)
    u1 = principal_left_singular_vector(W, iters=30)
    assert u1.shape == (10,)
    assert torch.abs(F_cos(u1, u)) > 0.999


def test_power_iteration_matches_svd_on_random_matrix():
    """u1 from power iteration aligns with the SVD's top left singular vector."""
    torch.manual_seed(3)
    W = torch.randn(20, 12)
    u1 = principal_left_singular_vector(W, iters=50)
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    assert torch.abs(F_cos(u1, U[:, 0])) > 0.99


def test_power_iteration_output_is_unit_norm():
    W = torch.randn(9, 5)
    u1 = principal_left_singular_vector(W, iters=8)
    assert abs(u1.norm().item() - 1.0) < 1e-5


def test_power_iteration_rejects_non_2d():
    try:
        principal_left_singular_vector(torch.randn(4, 4, 4))
        raise AssertionError("expected ValueError on 3-D input")
    except ValueError:
        pass


def test_power_iteration_deterministic_under_seed():
    W = torch.randn(8, 6)
    torch.manual_seed(7)
    a = principal_left_singular_vector(W, iters=10)
    torch.manual_seed(7)
    b = principal_left_singular_vector(W, iters=10)
    assert torch.allclose(a, b)


# ── sign collapse ──────────────────────────────────────────────────────────────


def test_sign_collapse_balanced_is_half():
    sa = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.2, -0.2])
    assert abs(sign_collapse_stat(sa) - 0.5) < 1e-9


def test_sign_collapse_all_positive_is_one():
    sa = torch.tensor([0.1, 0.9, 0.5, 0.3])
    assert sign_collapse_stat(sa) == 1.0


def test_sign_collapse_all_negative_is_one():
    sa = torch.tensor([-0.1, -0.9, -0.5])
    assert sign_collapse_stat(sa) == 1.0


def test_sign_collapse_empty_and_all_zero_are_neutral():
    assert sign_collapse_stat(torch.empty(0)) == 0.5
    assert sign_collapse_stat(torch.zeros(5)) == 0.5


def test_sign_collapse_majority():
    sa = torch.tensor([1.0, 1.0, 1.0, -1.0])  # 3 of 4 positive
    assert abs(sign_collapse_stat(sa) - 0.75) < 1e-9


# ── SA cosine direction (synthetic, exact) ─────────────────────────────────────


def test_sa_aligned_input_scores_near_one():
    """A response parallel to u1 has cosine ~1; orthogonal ~0 — checked directly."""
    torch.manual_seed(5)
    u = torch.randn(16)
    u = u / u.norm()
    # response rows parallel to u → cosine 1; orthogonal vector → cosine ~0.
    aligned = (2.0 * u).unsqueeze(0)
    cos_aligned = F_cos(aligned.squeeze(0), u)
    assert cos_aligned > 0.999
    # Build an orthogonal vector via Gram-Schmidt against u.
    r = torch.randn(16)
    orth = r - (r @ u) * u
    cos_orth = F_cos(orth, u)
    assert abs(cos_orth.item()) < 1e-5


# ── rank-collapsed weight scores higher collapse (early-warning direction) ──────


def test_rank_collapsed_weight_collapses_more_than_random():
    """A near-rank-1 (collapsed) weight pushes SA all one sign vs a random weight.

    Synthetic check of the early-warning DIRECTION: feed positive-mean inputs
    through a rank-1 weight W = u v^T with v aligned to the input mean → every
    response ∝ +u → cosine all +1 (collapse→1.0). A random full-rank weight
    spreads the responses across both signs (collapse near 0.5).
    """
    torch.manual_seed(11)
    dim = 24
    u = torch.randn(dim)
    u = u / u.norm()
    v = torch.randn(dim)
    v = v / v.norm()
    # Inputs: a strong, consistent +v bias (so v·x > 0 for every row → rank-1 W
    # maps them all to +u, collapse→1.0) PLUS large diverse noise (so a random
    # full-rank W scatters the responses across both signs, collapse near 0.5).
    inputs = 3.0 * v.unsqueeze(0) + torch.randn(40, dim)

    W_collapsed = torch.outer(u, v)  # rank-1
    W_random = torch.randn(dim, dim)

    def collapse_for(W: torch.Tensor) -> float:
        u1 = principal_left_singular_vector(W, iters=40)
        resp = inputs @ W.t()  # (n, dim)
        sa = torch.nn.functional.cosine_similarity(resp, u1.unsqueeze(0), dim=-1)
        return sign_collapse_stat(sa)

    assert collapse_for(W_collapsed) > collapse_for(W_random)


# ── integration: real tiny model + stub ────────────────────────────────────────


def test_spectral_alignment_runs_on_tiny_model():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (SEQ,))
    sa = spectral_alignment(model, ids, probes=("q",))
    assert set(sa.keys()) == set(range(model.n_layers))
    for v in sa.values():
        assert v.numel() == SEQ
        assert torch.all(v <= 1.0 + 1e-5) and torch.all(v >= -1.0 - 1e-5)


def test_spectral_alignment_two_probes_pools_values():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (SEQ,))
    sa = spectral_alignment(model, ids, probes=("q", "fc2"))
    # q + fc2 each contribute SEQ values per layer → 2*SEQ.
    for v in sa.values():
        assert v.numel() == 2 * SEQ


def test_spectral_alignment_rejects_batched():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (2, SEQ))
    try:
        spectral_alignment(model, ids)
        raise AssertionError("expected ValueError on batched input")
    except ValueError:
        pass


def test_profile_spectral_health_runs_and_reports():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(3)]
    report = profile_spectral_health(model, seqs, probes=("q",))
    assert isinstance(report, SpectralHealthReport)
    assert report.n_layers == model.n_layers
    assert len(report.per_layer) == model.n_layers
    assert 0 <= report.worst_layer < model.n_layers
    assert 0.5 <= report.worst_collapse <= 1.0


def test_profile_by_tier_keyed_by_tiers():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(3)]
    tiers = ["easy", "medium", "hard"]
    report = profile_spectral_health(model, seqs, probes=("q",), by_tier=tiers)
    assert report.by_tier is not None
    assert set(report.by_tier.keys()) <= set(TIERS)
    assert set(report.by_tier.keys()) == {"easy", "medium", "hard"}
    for stats in report.by_tier.values():
        assert "worst_layer" in stats and "worst_collapse" in stats


def test_profile_by_tier_length_mismatch_raises():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(3)]
    try:
        profile_spectral_health(model, seqs, by_tier=["easy"])
        raise AssertionError("expected ValueError on tier length mismatch")
    except ValueError:
        pass


def test_profile_empty_input_is_safe():
    model = _tiny_model()
    report = profile_spectral_health(model, [], probes=("q",))
    assert report.per_layer == []
    assert report.worst_layer == -1
    assert report.worst_collapse == 0.5
    assert report.n_layers == model.n_layers


def test_profile_skips_empty_sequences():
    model = _tiny_model()
    seqs = [torch.empty(0, dtype=torch.long), torch.randint(0, VOCAB, (SEQ,))]
    report = profile_spectral_health(model, seqs, probes=("q",))
    assert len(report.per_layer) == model.n_layers


def test_profile_deterministic_under_seed():
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(2)]
    torch.manual_seed(0)
    r1 = profile_spectral_health(_tiny_model(0), seqs, probes=("q",))
    torch.manual_seed(0)
    r2 = profile_spectral_health(_tiny_model(0), seqs, probes=("q",))
    assert r1.worst_layer == r2.worst_layer
    assert abs(r1.worst_collapse - r2.worst_collapse) < 1e-9
    assert [r["sign_collapse"] for r in r1.per_layer] == [
        r["sign_collapse"] for r in r2.per_layer
    ]


def F_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine between two 1-D vectors (test helper)."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).squeeze()
