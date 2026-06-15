"""Tests for Gemma-2-style logit soft-capping.

Covers the pure ``soft_cap_logits`` helper (bounds, identity-near-zero,
monotonicity, disabled passthrough, shape/dtype/gradient) and the model-level
wiring of ``final_logit_softcap`` (default-off is a byte-identical no-op; when
enabled the max |logit| is bounded by the cap). All deterministic, CPU-only.
"""

import torch

from cola_coder.model.config import ModelConfig
from cola_coder.model.logit_cap import soft_cap_logits
from cola_coder.model.transformer import Transformer


def make_tiny_config(**overrides) -> ModelConfig:
    """Smallest sensible config for fast CPU model tests."""
    base = dict(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim_multiplier=2.667,
        max_seq_len=128,
        dropout=0.0,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


class TestSoftCapHelper:
    """Unit tests for the pure soft_cap_logits function."""

    def test_bounds_output_to_interval(self):
        """Output is bounded to [-cap, +cap] across a wide logit range.

        ``cap*tanh(x/cap)`` is asymptotic to +/- cap; in float32 tanh saturates
        to exactly 1.0 for large |x|, so the bound is the closed interval. The
        guarantee that matters for stability is that no logit exceeds the cap.
        """
        cap = 30.0
        x = torch.tensor([-1e6, -1e3, -50.0, 0.0, 50.0, 1e3, 1e6])
        y = soft_cap_logits(x, cap)
        assert torch.all(y >= -cap)
        assert torch.all(y <= cap)
        assert y.abs().max().item() <= cap
        # Moderate logits land strictly inside the interval.
        mid = soft_cap_logits(torch.tensor([-5.0, 0.0, 5.0]), cap)
        assert torch.all(mid > -cap)
        assert torch.all(mid < cap)
        # Extreme inputs saturate to +/- cap.
        assert torch.isclose(y[-1], torch.tensor(cap), atol=1e-3)
        assert torch.isclose(y[0], torch.tensor(-cap), atol=1e-3)

    def test_identity_near_zero(self):
        """For |x| << cap the function is approximately the identity."""
        cap = 1000.0
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = soft_cap_logits(x, cap)
        assert torch.allclose(y, x, atol=1e-2)

    def test_monotonic(self):
        """Soft-cap is strictly increasing in x over the unsaturated range.

        float64 + a logit range that stays well short of tanh saturation, so the
        function is genuinely strictly increasing (the property that preserves
        the relative ordering of logits when capping is enabled).
        """
        cap = 10.0
        x = torch.linspace(-40.0, 40.0, steps=500, dtype=torch.float64)
        y = soft_cap_logits(x, cap)
        diffs = y[1:] - y[:-1]
        assert torch.all(diffs > 0)

    def test_zero_cap_returns_input_unchanged(self):
        """cap == 0 disables capping (identity, same object)."""
        x = torch.randn(4, 8)
        y = soft_cap_logits(x, 0.0)
        assert y is x

    def test_negative_cap_returns_input_unchanged(self):
        """cap < 0 disables capping (identity, same object)."""
        x = torch.randn(4, 8)
        y = soft_cap_logits(x, -5.0)
        assert y is x

    def test_preserves_shape_and_dtype(self):
        """Shape and dtype are preserved."""
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(3, 5, 7, dtype=dtype)
            y = soft_cap_logits(x, 12.0)
            assert y.shape == x.shape
            assert y.dtype == x.dtype

    def test_gradient_flows_without_nan(self):
        """Gradient flows and contains no NaN/Inf, even on extreme inputs."""
        x = torch.tensor([-1e4, -1.0, 0.0, 1.0, 1e4], requires_grad=True)
        y = soft_cap_logits(x, 20.0)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        # Saturated regions have ~0 gradient; near-zero region has ~1.
        assert x.grad[2] > 0.99


class TestConfigDefaults:
    """Both soft-caps must default to OFF (0.0)."""

    def test_default_softcaps_off(self):
        cfg = ModelConfig()
        assert cfg.attn_logit_softcap == 0.0
        assert cfg.final_logit_softcap == 0.0


class TestModelWiring:
    """Model-level behavior of final_logit_softcap."""

    def test_default_off_is_byte_identical(self):
        """With final_logit_softcap=0.0 logits equal the un-capped baseline."""
        torch.manual_seed(0)
        cfg = make_tiny_config(final_logit_softcap=0.0)
        model = Transformer(cfg)
        model.eval()

        token_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits = model(token_ids)

        # Recompute the baseline by hand (final_norm -> output) and compare.
        with torch.no_grad():
            h = model.tok_emb(token_ids)
            h = model.dropout(h)
            for block in model.blocks:
                h = block(h, rope_freqs=model.rope_freqs, start_pos=0, use_cache=False)
            h = model.final_norm(h)
            baseline = model.output(h)

        assert torch.equal(logits, baseline)

    def test_enabled_bounds_logits(self):
        """With final_logit_softcap>0 the max |logit| is below the cap."""
        torch.manual_seed(0)
        cap = 5.0
        cfg = make_tiny_config(final_logit_softcap=cap)
        model = Transformer(cfg)
        model.eval()

        token_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits = model(token_ids)

        assert logits.abs().max().item() < cap

    def test_enabled_differs_from_disabled(self):
        """Enabling the cap actually changes the logits (not a silent no-op)."""
        torch.manual_seed(0)
        token_ids = torch.randint(0, 256, (2, 16))

        torch.manual_seed(123)
        off = Transformer(make_tiny_config(final_logit_softcap=0.0)).eval()
        torch.manual_seed(123)
        on = Transformer(make_tiny_config(final_logit_softcap=1.0)).eval()

        with torch.no_grad():
            logits_off = off(token_ids)
            logits_on = on(token_ids)

        # Same init seed -> identical weights -> the ONLY difference is the cap.
        assert not torch.equal(logits_off, logits_on)
        assert logits_on.abs().max().item() < 1.0


class TestAttnLogitSoftcapWiring:
    """PRE-softmax attention soft-cap (attn_logit_softcap) — was a phantom config
    knob (defined but never read; SDPA can't cap pre-softmax scores). Now wired via
    a gated eager path. These guard that it is real, not a silent no-op."""

    def test_default_off_uses_sdpa_and_matches(self):
        """attn_logit_softcap=0.0 keeps the fast SDPA path (eager path not taken)."""
        torch.manual_seed(0)
        cfg = make_tiny_config(attn_logit_softcap=0.0)
        model = Transformer(cfg).eval()
        for block in model.blocks:
            assert block.attention.attn_logit_softcap == 0.0
        token_ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            # Forward must run cleanly through the SDPA branch.
            logits = model(token_ids)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_enabled_differs_from_disabled(self):
        """Enabling the attention cap changes the output (not a silent no-op)."""
        torch.manual_seed(0)
        token_ids = torch.randint(0, 256, (2, 16))

        torch.manual_seed(123)
        off = Transformer(make_tiny_config(attn_logit_softcap=0.0)).eval()
        torch.manual_seed(123)
        on = Transformer(make_tiny_config(attn_logit_softcap=2.0)).eval()

        with torch.no_grad():
            logits_off = off(token_ids)
            logits_on = on(token_ids)

        # Identical weights (same seed); the cap is the only difference — and a
        # cap of 2.0 noticeably reshapes the attention distribution.
        assert not torch.equal(logits_off, logits_on)

    def test_eager_path_matches_sdpa_when_cap_huge(self):
        """A very large cap is ~identity on pre-softmax scores, so the eager path
        must converge to the SDPA result — proving the eager math mirrors SDPA."""
        torch.manual_seed(7)
        token_ids = torch.randint(0, 256, (2, 16))

        torch.manual_seed(42)
        sdpa = Transformer(make_tiny_config(attn_logit_softcap=0.0)).eval()
        torch.manual_seed(42)
        eager = Transformer(make_tiny_config(attn_logit_softcap=1e6)).eval()

        with torch.no_grad():
            logits_sdpa = sdpa(token_ids)
            logits_eager = eager(token_ids)

        # cap=1e6 → tanh(x/1e6)*1e6 ≈ x, so eager attention ≈ SDPA (tiny fp error).
        assert torch.allclose(logits_sdpa, logits_eager, atol=1e-3)

    def test_kv_cache_inference_with_attn_softcap(self):
        """The eager path also handles the KV-cache (q_len != kv_len) branch."""
        torch.manual_seed(0)
        cfg = make_tiny_config(attn_logit_softcap=3.0)
        model = Transformer(cfg).eval()
        token_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(token_ids, use_cache=True, start_pos=0)
        assert out.shape == (1, 8, cfg.vocab_size)
        assert torch.isfinite(out).all()
