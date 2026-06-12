"""MODEL-005: YaRN attention temperature (mscale) wired into attention.

YaRN doesn't only rescale RoPE frequencies — it also lowers the softmax
temperature so attention logits stay calibrated at extended context. The paper
sets sqrt(1/t) = 0.1*ln(s) + 1 (s = extension factor). We fold mscale**2 into
the SDPA scale (the logits pick up mscale from both q and k). These tests lock:
the math, the attention plumbing, and that only type="yarn" triggers it — every
other rope mode (none/ntk/linear) leaves attention temperature unchanged, so no
existing checkpoint changes behavior.
"""

import math

import pytest

from cola_coder.model.attention import GroupedQueryAttention
from cola_coder.model.config import ModelConfig, RoPEScalingConfig
from cola_coder.model.rope import yarn_attention_scale
from cola_coder.model.transformer import Transformer


class TestYarnAttentionScaleMath:
    def test_factor_one_is_identity(self):
        assert yarn_attention_scale(1.0) == 1.0

    def test_factor_below_one_is_identity(self):
        assert yarn_attention_scale(0.5) == 1.0

    def test_known_value_factor_eight(self):
        assert yarn_attention_scale(8.0) == pytest.approx(0.1 * math.log(8.0) + 1.0)

    def test_monotonic_increasing(self):
        assert yarn_attention_scale(2.0) < yarn_attention_scale(4.0) < yarn_attention_scale(16.0)


class TestAttentionAppliesLogitScale:
    def test_default_scale_unchanged(self):
        a = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=4, max_seq_len=128)
        assert a.scale == pytest.approx((64 // 8) ** -0.5)

    def test_logit_scale_multiplies(self):
        base = (64 // 8) ** -0.5
        a = GroupedQueryAttention(
            dim=64, n_heads=8, n_kv_heads=4, max_seq_len=128, attn_logit_scale=2.5,
        )
        assert a.scale == pytest.approx(base * 2.5)


def _cfg(scaling_type="none", factor=1.0):
    return ModelConfig(
        vocab_size=128, dim=64, n_layers=2, n_heads=8, n_kv_heads=4,
        max_seq_len=64,
        rope_scaling=RoPEScalingConfig(type=scaling_type, factor=factor),
    )


def _block_scale(model: Transformer) -> float:
    return model.blocks[0].attention.scale


class TestTransformerWiresYarnScale:
    BASE = (64 // 8) ** -0.5

    def test_no_scaling_uses_standard_attention(self):
        m = Transformer(_cfg("none", 1.0))
        assert _block_scale(m) == pytest.approx(self.BASE)

    def test_yarn_applies_mscale_squared(self):
        factor = 8.0
        m = Transformer(_cfg("yarn", factor))
        expected = self.BASE * (yarn_attention_scale(factor) ** 2)
        assert _block_scale(m) == pytest.approx(expected)
        assert _block_scale(m) > self.BASE  # temperature genuinely changed

    def test_yarn_factor_one_is_inert(self):
        # factor <= 1 means "no extension" → resolved to type none → standard.
        m = Transformer(_cfg("yarn", 1.0))
        assert _block_scale(m) == pytest.approx(self.BASE)

    def test_ntk_does_not_touch_attention_temperature(self):
        m = Transformer(_cfg("ntk", 8.0))
        assert _block_scale(m) == pytest.approx(self.BASE)

    def test_linear_does_not_touch_attention_temperature(self):
        m = Transformer(_cfg("linear", 8.0))
        assert _block_scale(m) == pytest.approx(self.BASE)

    def test_all_blocks_share_the_scale(self):
        m = Transformer(_cfg("yarn", 8.0))
        scales = {blk.attention.scale for blk in m.blocks}
        assert len(scales) == 1  # every layer got the same mscale**2
