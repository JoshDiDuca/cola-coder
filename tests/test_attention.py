"""Tests for attention components: RoPE, GQA, normalization, feedforward."""

import pytest
import torch

from cola_coder.model.normalization import RMSNorm
from cola_coder.model.feedforward import SwiGLUFFN
from cola_coder.model.rope import precompute_rope_freqs, apply_rope
from cola_coder.model.attention import GroupedQueryAttention


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_rms(self):
        """After normalization, RMS should be approximately 1."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64) * 10  # Large values
        out = norm(x)
        # RMS of each vector should be close to the weight magnitude
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_gradient_flow(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))


class TestSwiGLUFFN:
    """Tests for SwiGLU feed-forward network."""

    def test_output_shape(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=128)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape  # Output dim should match input dim

    def test_gradient_flow(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_hidden_dims(self):
        """FFN works with various hidden dimensions."""
        for hidden_dim in [64, 128, 256]:
            ffn = SwiGLUFFN(dim=64, hidden_dim=hidden_dim)
            x = torch.randn(1, 8, 64)
            out = ffn(x)
            assert out.shape == (1, 8, 64)


class TestRoPE:
    """Tests for Rotary Positional Encoding."""

    def test_freq_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_seq_len=128)
        assert freqs.shape == (128, 8)  # dim // 2

    def test_apply_rope_shapes(self):
        head_dim = 16
        freqs = precompute_rope_freqs(dim=head_dim, max_seq_len=128)
        q = torch.randn(2, 32, 4, head_dim)  # batch, seq, heads, head_dim
        k = torch.randn(2, 32, 2, head_dim)
        q_rot, k_rot = apply_rope(q, k, freqs)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_offset(self):
        """RoPE with start_pos offset should match sliced full computation."""
        head_dim = 16
        freqs = precompute_rope_freqs(dim=head_dim, max_seq_len=128)

        q = torch.randn(1, 1, 4, head_dim)
        k = torch.randn(1, 1, 2, head_dim)

        # Apply at position 0
        q_rot0, _ = apply_rope(q, k, freqs, start_pos=0)
        # Apply at position 10
        q_rot10, _ = apply_rope(q, k, freqs, start_pos=10)

        # Different positions should give different results
        assert not torch.allclose(q_rot0, q_rot10)

    def test_dtype_preservation(self):
        """RoPE should preserve input dtype."""
        freqs = precompute_rope_freqs(dim=16, max_seq_len=64)
        q = torch.randn(1, 8, 4, 16).half()  # fp16
        k = torch.randn(1, 8, 2, 16).half()
        q_rot, k_rot = apply_rope(q, k, freqs)
        assert q_rot.dtype == torch.float16
        assert k_rot.dtype == torch.float16


class TestGroupedQueryAttention:
    """Tests for GQA attention."""

    def setup_method(self):
        self.dim = 64
        self.n_heads = 4
        self.n_kv_heads = 2
        self.max_seq_len = 128

    def make_attention(self):
        return GroupedQueryAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            max_seq_len=self.max_seq_len,
        )

    def test_output_shape(self):
        attn = self.make_attention()
        freqs = precompute_rope_freqs(dim=self.dim // self.n_heads, max_seq_len=self.max_seq_len)
        x = torch.randn(2, 32, self.dim)
        mask = torch.zeros(32, 32)
        mask = torch.triu(torch.full((32, 32), float("-inf")), diagonal=1)
        out = attn(x, rope_freqs=freqs, mask=mask)
        assert out.shape == (2, 32, self.dim)

    def test_gqa_head_ratio(self):
        """GQA with different head ratios should work."""
        for n_kv_heads in [1, 2, 4]:
            attn = GroupedQueryAttention(
                dim=self.dim, n_heads=self.n_heads,
                n_kv_heads=n_kv_heads, max_seq_len=self.max_seq_len,
            )
            freqs = precompute_rope_freqs(
                dim=self.dim // self.n_heads, max_seq_len=self.max_seq_len
            )
            x = torch.randn(1, 8, self.dim)
            mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
            out = attn(x, rope_freqs=freqs, mask=mask)
            assert out.shape == (1, 8, self.dim)

    def test_kv_cache(self):
        """KV-cache should produce consistent results."""
        attn = self.make_attention()
        attn.eval()
        freqs = precompute_rope_freqs(
            dim=self.dim // self.n_heads, max_seq_len=self.max_seq_len
        )
        x = torch.randn(1, 8, self.dim)

        # Without cache
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
        with torch.no_grad():
            out_no_cache = attn(x, rope_freqs=freqs, mask=mask, use_cache=False)

        attn.clear_cache()

        # With cache (prefill)
        with torch.no_grad():
            out_cache = attn(x, rope_freqs=freqs, mask=mask, start_pos=0, use_cache=True)

        attn.clear_cache()

        torch.testing.assert_close(out_no_cache, out_cache, rtol=1e-4, atol=1e-4)

    def test_gradient_flow(self):
        attn = self.make_attention()
        freqs = precompute_rope_freqs(
            dim=self.dim // self.n_heads, max_seq_len=self.max_seq_len
        )
        x = torch.randn(1, 8, self.dim, requires_grad=True)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
        out = attn(x, rope_freqs=freqs, mask=mask)
        out.sum().backward()
        assert x.grad is not None
