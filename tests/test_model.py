"""Tests for the transformer model architecture.

These tests verify that:
1. All tensor shapes are correct through the model
2. Gradients flow through all components
3. The model produces valid output for all configurations
"""

import torch

from cola_coder.model.config import ModelConfig, Config
from cola_coder.model.transformer import Transformer


def make_tiny_config() -> ModelConfig:
    """Create a minimal config for fast testing."""
    return ModelConfig(
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


class TestTransformerShapes:
    """Verify output shapes at each stage of the model."""

    def setup_method(self):
        self.config = make_tiny_config()
        self.model = Transformer(self.config)
        self.model.eval()

    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        logits = self.model(token_ids)
        assert logits.shape == (batch_size, seq_len, self.config.vocab_size)

    def test_single_token(self):
        """Model works with a single token input."""
        token_ids = torch.randint(0, self.config.vocab_size, (1, 1))
        logits = self.model(token_ids)
        assert logits.shape == (1, 1, self.config.vocab_size)

    def test_max_seq_len(self):
        """Model works at maximum sequence length."""
        token_ids = torch.randint(0, self.config.vocab_size, (1, self.config.max_seq_len))
        logits = self.model(token_ids)
        assert logits.shape == (1, self.config.max_seq_len, self.config.vocab_size)


class TestTransformerTraining:
    """Verify the model can train (gradients flow correctly)."""

    def setup_method(self):
        self.config = make_tiny_config()
        self.model = Transformer(self.config)

    def test_compute_loss(self):
        """Loss computation produces a scalar."""
        token_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        loss = self.model.compute_loss(token_ids)
        assert loss.shape == ()
        assert loss.item() > 0  # Loss should be positive

    def test_gradients_flow(self):
        """All parameters receive gradients."""
        token_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        loss = self.model.compute_loss(token_ids)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_loss_decreases(self):
        """Loss decreases after a few optimization steps."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        token_ids = torch.randint(0, self.config.vocab_size, (4, 32))

        initial_loss = self.model.compute_loss(token_ids).item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = self.model.compute_loss(token_ids)
            loss.backward()
            optimizer.step()

        final_loss = self.model.compute_loss(token_ids).item()
        assert final_loss < initial_loss, "Loss did not decrease after training steps"

    def test_no_nan_gradients(self):
        """No NaN values in gradients."""
        token_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        loss = self.model.compute_loss(token_ids)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"


class TestTransformerInference:
    """Verify KV-cache inference mode."""

    def setup_method(self):
        self.config = make_tiny_config()
        self.model = Transformer(self.config)
        self.model.eval()

    def test_kv_cache_matches_no_cache(self):
        """KV-cache inference produces same output as full forward pass."""
        token_ids = torch.randint(0, self.config.vocab_size, (1, 16))

        # Full forward pass (no cache)
        with torch.no_grad():
            logits_no_cache = self.model(token_ids, use_cache=False)

        self.model.clear_caches()

        # KV-cache forward pass (prefill)
        with torch.no_grad():
            logits_cache = self.model(token_ids, start_pos=0, use_cache=True)

        self.model.clear_caches()

        # Compare last token logits (the prediction for the next token)
        torch.testing.assert_close(
            logits_no_cache[:, -1, :],
            logits_cache[:, -1, :],
            rtol=1e-4, atol=1e-4,
        )

    def test_parameter_count(self):
        """Parameter count property works."""
        num_params = self.model.num_parameters
        assert num_params > 0
        assert isinstance(num_params, int)


class TestRoPEWiring:
    """Verify config.rope_theta and rope_scaling actually reach the model.

    Regression tests: previously the model always built its RoPE table with
    the default theta=10000 and ignored rope_scaling entirely, silently
    breaking long-context configs (4080_max theta=500K) and Stage 4 YaRN.
    """

    def test_rope_theta_reaches_model(self):
        from cola_coder.model.rope import get_rope_freqs

        config = make_tiny_config()
        config.rope_theta = 500000.0
        model = Transformer(config)

        expected = get_rope_freqs(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len * 2,
            theta=500000.0,
        )
        torch.testing.assert_close(model.rope_freqs, expected)

    def test_different_theta_produces_different_freqs(self):
        config_a = make_tiny_config()
        config_b = make_tiny_config()
        config_b.rope_theta = 500000.0
        freqs_a = Transformer(config_a).rope_freqs
        freqs_b = Transformer(config_b).rope_freqs
        assert not torch.allclose(freqs_a, freqs_b)

    def test_yarn_scaling_extends_freq_table(self):
        from cola_coder.model.config import RoPEScalingConfig

        config = make_tiny_config()
        config.rope_scaling = RoPEScalingConfig(type="yarn", factor=2.0)
        model = Transformer(config)
        # Table must cover the extended context (factor x) plus the 2x buffer
        assert model.rope_freqs.shape[0] == config.max_seq_len * 2 * 2

    def test_scaling_none_or_factor_1_is_identity(self):
        from cola_coder.model.config import RoPEScalingConfig

        base = Transformer(make_tiny_config()).rope_freqs
        config = make_tiny_config()
        config.rope_scaling = RoPEScalingConfig(type="yarn", factor=1.0)
        scaled = Transformer(config).rope_freqs
        torch.testing.assert_close(base, scaled)


class TestGradientCheckpointing:
    """Verify gradient checkpointing actually runs and matches plain backprop.

    Regression tests: previously enable_gradient_checkpointing() set a flag
    that nothing read — the forward pass never recomputed activations, so the
    documented ~60% VRAM saving (REQUIRED for 4080_max) silently never
    happened.
    """

    def _models(self):
        torch.manual_seed(0)
        config = make_tiny_config()
        plain = Transformer(config)
        ckpt = Transformer(config)
        ckpt.load_state_dict(plain.state_dict())
        ckpt.enable_gradient_checkpointing()
        return plain, ckpt

    def test_loss_identical(self):
        plain, ckpt = self._models()
        plain.train(), ckpt.train()
        token_ids = torch.randint(0, 256, (2, 32))
        torch.testing.assert_close(
            plain.compute_loss(token_ids), ckpt.compute_loss(token_ids),
        )

    def test_gradients_identical(self):
        plain, ckpt = self._models()
        plain.train(), ckpt.train()
        token_ids = torch.randint(0, 256, (2, 32))
        plain.compute_loss(token_ids).backward()
        ckpt.compute_loss(token_ids).backward()
        for (name, p_plain), (_, p_ckpt) in zip(
            plain.named_parameters(), ckpt.named_parameters()
        ):
            assert p_ckpt.grad is not None, f"No gradient for {name} with checkpointing"
            torch.testing.assert_close(
                p_plain.grad, p_ckpt.grad, rtol=1e-5, atol=1e-6,
                msg=lambda m: f"Gradient mismatch for {name}: {m}",
            )

    def test_kv_cache_inference_bypasses_checkpointing(self):
        _, ckpt = self._models()
        ckpt.eval()
        token_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = ckpt(token_ids, start_pos=0, use_cache=True)
        ckpt.clear_caches()
        assert logits.shape == (1, 8, 256)


class TestModelConfig:
    """Verify configuration loading and validation."""

    def test_tiny_config_params(self):
        """Tiny config produces expected parameter count."""
        config = make_tiny_config()
        assert config.head_dim == 16  # 64 / 4
        assert config.total_params > 0

    def test_from_yaml(self, tmp_path):
        """Config loads from YAML file."""
        yaml_content = """
model:
  vocab_size: 256
  dim: 64
  n_layers: 2
  n_heads: 4
  n_kv_heads: 2
  max_seq_len: 128
training:
  batch_size: 4
  max_steps: 100
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        config = Config.from_yaml(str(yaml_file))
        assert config.model.vocab_size == 256
        assert config.model.dim == 64
        assert config.training.batch_size == 4

    def test_config_summary(self):
        """Config summary produces a string."""
        config = Config()
        summary = config.summary()
        assert isinstance(summary, str)
        assert "Model" in summary
