"""MODEL-006: rope_scaling.original_max_seq_len must actually be honored.

The field was defined on RoPEScalingConfig but never read — Transformer hardcoded
`original_max_seq_len=config.max_seq_len` when building the YaRN freq table, so a
user extending a model trained at a DIFFERENT length than the current config's
max_seq_len was silently ignored. The 0 sentinel (default) means "use max_seq_len".
"""

import torch

from cola_coder.model.config import ModelConfig, RoPEScalingConfig
from cola_coder.model.rope import get_rope_freqs
from cola_coder.model.transformer import Transformer


def _cfg(original_max_seq_len: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=128, dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
        max_seq_len=512,
        rope_scaling=RoPEScalingConfig(
            type="yarn", factor=8.0, original_max_seq_len=original_max_seq_len,
        ),
    )


class TestGetRopeFreqsHonorsOriginalLen:
    def test_yarn_freqs_differ_by_original_len(self):
        a = get_rope_freqs(64, 4096, scaling_type="yarn", scaling_factor=8.0,
                           original_max_seq_len=512)
        b = get_rope_freqs(64, 4096, scaling_type="yarn", scaling_factor=8.0,
                           original_max_seq_len=2048)
        assert not torch.allclose(a, b)  # the partition thresholds shifted


class TestTransformerWiresOriginalLen:
    def test_explicit_original_len_changes_freqs(self):
        # Default (0 → max_seq_len=512) vs explicit 2048: end-to-end the freq
        # buffer must differ, proving the field is read (it wasn't before).
        default_model = Transformer(_cfg(0))
        custom_model = Transformer(_cfg(2048))
        assert not torch.allclose(default_model.rope_freqs, custom_model.rope_freqs)

    def test_zero_sentinel_equals_explicit_max_seq_len(self):
        # 0 must behave EXACTLY like passing max_seq_len (512) explicitly.
        zero_model = Transformer(_cfg(0))
        explicit_model = Transformer(_cfg(512))
        assert torch.allclose(zero_model.rope_freqs, explicit_model.rope_freqs)

    def test_non_yarn_unaffected_by_original_len(self):
        # For type "none" the freq table ignores original_max_seq_len entirely.
        def none_cfg(orig):
            return ModelConfig(
                vocab_size=128, dim=64, n_layers=1, n_heads=8, n_kv_heads=4,
                max_seq_len=512,
                rope_scaling=RoPEScalingConfig(type="none", original_max_seq_len=orig),
            )
        a = Transformer(none_cfg(0))
        b = Transformer(none_cfg(2048))
        assert torch.allclose(a.rope_freqs, b.rope_freqs)
