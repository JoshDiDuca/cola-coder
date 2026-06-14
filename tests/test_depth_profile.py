"""Tests for the logit-lens depth / convergence profiler (INFER-031).

Hermetic + fast: builds a TINY real Transformer on CPU (no checkpoint, no GPU) so
the logit_lens anchor test runs against the actual model.forward, plus a faithful
stub (real attribute names) to exercise convergence_depth on synthetic streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cola_coder.evaluation.depth_profile import (
    DepthReport,
    convergence_depth,
    logit_lens,
    profile_depth,
)
from cola_coder.evaluation.difficulty_profile import TIERS
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


class _StubBlock(nn.Module):
    """A no-op-ish block that just returns its input (residual identity)."""

    def forward(self, h, rope_freqs=None, start_pos=0, use_cache=False, mask=None):
        return h


class _StubModel(nn.Module):
    """Stub mirroring the REAL Transformer attribute names used by logit_lens.

    Lets a test inject hand-crafted per-layer logits by controlling tok_emb/output.
    """

    def __init__(self, n_layers: int, vocab: int = VOCAB, dim: int = 8) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.final_norm = nn.Identity()
        self.output = nn.Linear(dim, vocab, bias=False)
        self.blocks = nn.ModuleList([_StubBlock() for _ in range(n_layers)])
        self.rope_freqs = None

    @property
    def n_layers(self) -> int:
        return len(self.blocks)


# ── logit_lens ─────────────────────────────────────────────────────────────


def test_logit_lens_shape():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (SEQ,))
    out = logit_lens(model, ids)
    assert out.shape == (model.n_layers, SEQ, VOCAB)


def test_logit_lens_last_layer_matches_forward():
    """The last layer's lens row must equal the model's real forward logits."""
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (1, SEQ))
    lens = logit_lens(model, ids)
    with torch.no_grad():
        forward_logits = model(ids).squeeze(0)  # (seq, vocab)
    assert torch.allclose(lens[-1], forward_logits, atol=1e-5)


def test_logit_lens_accepts_1d_and_2d():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (SEQ,))
    a = logit_lens(model, ids)
    b = logit_lens(model, ids.unsqueeze(0))
    assert torch.allclose(a, b)


def test_logit_lens_rejects_batch():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (2, SEQ))
    try:
        logit_lens(model, ids)
        assert False, "expected ValueError for batch>1"
    except ValueError:
        pass


# ── convergence_depth (argmax) ───────────────────────────────────────────────


def test_convergence_depth_argmax_in_range():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (SEQ,))
    lens = logit_lens(model, ids)
    depth = convergence_depth(lens, mode="argmax")
    assert depth.shape == (SEQ,)
    assert int(depth.min()) >= 0
    assert int(depth.max()) <= model.n_layers - 1


def test_convergence_depth_argmax_non_converging_is_last():
    """A stream whose argmax differs at every layer must exit at the last layer."""
    n_layers, seq = 5, 3
    logits = torch.full((n_layers, seq, VOCAB), -10.0)
    for layer in range(n_layers):
        # Each layer prefers a DIFFERENT token => never matches the final until itself.
        logits[layer, :, layer] = 10.0
    depth = convergence_depth(logits, mode="argmax")
    assert torch.equal(depth, torch.full((seq,), n_layers - 1, dtype=torch.long))


def test_convergence_depth_constant_logits_exit_at_zero():
    """Degenerate/constant logits across layers converge immediately at layer 0."""
    n_layers, seq = 4, 5
    logits = torch.zeros(n_layers, seq, VOCAB)
    logits[:, :, 7] = 5.0  # same argmax everywhere
    depth = convergence_depth(logits, mode="argmax")
    assert torch.equal(depth, torch.zeros(seq, dtype=torch.long))


# ── convergence_depth (entropy) ──────────────────────────────────────────────


def test_convergence_depth_entropy_lower_tau_deeper():
    """Lower tau (stricter) must yield a deeper-or-equal exit at every position."""
    n_layers, seq = 6, 4
    # Sharpen monotonically with depth so entropy decreases with layer index.
    logits = torch.zeros(n_layers, seq, VOCAB)
    for layer in range(n_layers):
        logits[layer, :, 0] = float(layer)  # bigger peak => lower entropy deeper
    high = convergence_depth(logits, mode="entropy", tau=2.0)
    low = convergence_depth(logits, mode="entropy", tau=0.5)
    assert torch.all(low >= high)


def test_convergence_depth_entropy_constant_converges_zero():
    """A peaked (near-zero-entropy) distribution at every layer converges at 0."""
    n_layers, seq = 4, 3
    logits = torch.full((n_layers, seq, VOCAB), -20.0)
    logits[:, :, 1] = 20.0  # almost one-hot => entropy ~ 0
    depth = convergence_depth(logits, mode="entropy", tau=0.1)
    assert torch.equal(depth, torch.zeros(seq, dtype=torch.long))


def test_convergence_depth_bad_mode_raises():
    logits = torch.zeros(2, 2, VOCAB)
    try:
        convergence_depth(logits, mode="nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── stub-driven anchor for the lens decoding ─────────────────────────────────


def test_logit_lens_with_stub_decodes_each_layer():
    """Stub with identity blocks => every layer's logits equal output(emb)."""
    model = _StubModel(n_layers=3, dim=8)
    ids = torch.randint(0, VOCAB, (SEQ,))
    lens = logit_lens(model, ids)
    assert lens.shape == (3, SEQ, VOCAB)
    # Identity blocks: all layers must be identical.
    assert torch.allclose(lens[0], lens[1])
    assert torch.allclose(lens[1], lens[2])


# ── profile_depth aggregation ────────────────────────────────────────────────


def test_profile_depth_aggregates():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(3)]
    report = profile_depth(model, seqs, mode="argmax")
    assert isinstance(report, DepthReport)
    assert report.n_layers == model.n_layers
    assert report.n_tokens == 3 * SEQ
    assert 0.0 <= report.mean_exit_depth <= model.n_layers - 1
    assert len(report.frac_converged_by_depth) == model.n_layers
    # Cumulative curve is non-decreasing and ends at 1.0.
    fc = report.frac_converged_by_depth
    assert all(fc[i] <= fc[i + 1] + 1e-6 for i in range(len(fc) - 1))
    assert abs(fc[-1] - 1.0) < 1e-6
    assert report.by_tier is None


def test_profile_depth_by_tier():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(4)]
    tiers = ["easy", "hard", "easy", "medium"]
    report = profile_depth(model, seqs, mode="argmax", difficulty_tiers=tiers)
    assert report.by_tier is not None
    assert set(report.by_tier.keys()) == {"easy", "hard", "medium"}
    # Tier labels must be drawn from the canonical difficulty_profile.TIERS.
    assert set(report.by_tier.keys()) <= set(TIERS)
    for stats in report.by_tier.values():
        assert stats["n_layers"] == model.n_layers
        assert "mean_exit_depth" in stats
    assert report.by_tier["easy"]["n_tokens"] == 2 * SEQ


def test_profile_depth_empty_is_safe():
    model = _tiny_model()
    report = profile_depth(model, [], mode="argmax")
    assert report.n_tokens == 0
    assert report.mean_exit_depth == 0.0
    assert report.median_exit_depth == 0.0
    assert report.frac_converged_by_depth == [0.0] * model.n_layers


def test_profile_depth_accepts_list_sequences_and_skips_empty():
    model = _tiny_model()
    seqs = [[1, 2, 3, 4], [], [5, 6, 7]]  # middle is empty
    report = profile_depth(model, seqs, mode="argmax")
    assert report.n_tokens == 4 + 3


def test_profile_depth_tier_length_mismatch_raises():
    model = _tiny_model()
    seqs = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(2)]
    try:
        profile_depth(model, seqs, difficulty_tiers=["easy"])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ── determinism ──────────────────────────────────────────────────────────────


def test_determinism_under_fixed_seed():
    a = _tiny_model(seed=123)
    b = _tiny_model(seed=123)
    ids = torch.arange(SEQ) % VOCAB
    la = logit_lens(a, ids)
    lb = logit_lens(b, ids)
    assert torch.allclose(la, lb, atol=1e-6)
    da = convergence_depth(la, mode="argmax")
    db = convergence_depth(lb, mode="argmax")
    assert torch.equal(da, db)
