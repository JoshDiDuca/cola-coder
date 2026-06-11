"""MODEL-002: ModelConfig.total_params must count ALL experts for MoE configs.

Before the fix, total_params used the dense FFN formula regardless of MoE, so a
MoE config reported ~active params instead of the true in-memory total (which
feeds VRAM estimates and model cards). Also locks the resolve_moe_layers move to
the torch-free config layer (re-exported from features/moe_layer).
"""

from cola_coder.model.config import ModelConfig, resolve_moe_layers


def _cfg(enabled: bool, **moe_over) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size=256, dim=64, n_layers=4,
        n_heads=4, n_kv_heads=2, max_seq_len=128,
    )
    cfg.moe.enabled = enabled
    cfg.moe.num_experts = moe_over.get("num_experts", 4)
    cfg.moe.num_shared_experts = moe_over.get("num_shared_experts", 1)
    cfg.moe.moe_layers = moe_over.get("moe_layers", "all")
    return cfg


class TestMoEParamCount:
    def test_dense_unchanged(self):
        dense = _cfg(enabled=False)
        # Sanity: dense count is positive and stable
        assert dense.total_params > 0

    def test_moe_counts_more_than_dense(self):
        dense = _cfg(enabled=False)
        moe = _cfg(enabled=True, num_experts=4, num_shared_experts=1)
        # 5 FFNs per layer instead of 1 → far more params (plus router gates)
        assert moe.total_params > dense.total_params

    def test_moe_ffn_multiplier_exact(self):
        dense = _cfg(enabled=False)
        moe = _cfg(enabled=True, num_experts=4, num_shared_experts=1)
        dim, hid, n = 64, _cfg(False).ffn_hidden_dim, 4
        dense_ffn = 3 * dim * hid
        router = 4 * dim
        # All 4 layers MoE: extra = per-layer (5*dense_ffn + router) - dense_ffn
        expected_delta = n * ((5 * dense_ffn + router) - dense_ffn)
        assert moe.total_params - dense.total_params == expected_delta

    def test_alternate_only_half_layers_moe(self):
        full = _cfg(enabled=True, moe_layers="all")
        alt = _cfg(enabled=True, moe_layers="alternate")
        # "alternate" (2 of 4 layers) must be between dense and all-MoE
        dense = _cfg(enabled=False)
        assert dense.total_params < alt.total_params < full.total_params

    def test_human_readable_reflects_moe(self):
        moe = _cfg(enabled=True, num_experts=8, num_shared_experts=1)
        assert moe.total_params_human.endswith(("M", "B"))


class TestResolveMoeLayersMoved:
    def test_config_resolver_works(self):
        assert resolve_moe_layers("all", 4) == {0, 1, 2, 3}
        assert resolve_moe_layers("alternate", 4) == {1, 3}
        assert resolve_moe_layers("0,2", 4) == {0, 2}

    def test_features_reexport_is_same_function(self):
        from cola_coder.features.moe_layer import resolve_moe_layers as feat_rml

        assert feat_rml is resolve_moe_layers
