"""Tests for ModelFingerprintGenerator (features/model_fingerprint.py)."""

from __future__ import annotations


from cola_coder.features.model_fingerprint import (
    FEATURE_ENABLED,
    LayerStats,
    ModelFingerprint,
    ModelFingerprintGenerator,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_state_dict(seed: float = 1.0) -> dict[str, list[float]]:
    """Simple fake state dict using plain Python lists."""
    return {
        "layer1.weight": [seed * 0.1, seed * 0.2, seed * -0.1, seed * 0.05],
        "layer1.bias": [seed * 0.0, seed * 0.01],
        "layer2.weight": [seed * 0.3, seed * -0.2, seed * 0.1, seed * -0.05],
        "layer2.bias": [seed * 0.02, seed * -0.01],
    }


META = {"num_layers": 2, "hidden_size": 64, "num_heads": 4}


class TestIsEnabled:
    def test_feature_enabled(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_returns_true(self):
        assert is_enabled() is True


class TestFingerprintGeneration:
    def test_generates_fingerprint(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict(), META)
        assert isinstance(fp, ModelFingerprint)

    def test_combined_hash_is_64_chars(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        assert len(fp.combined_hash) == 64

    def test_short_id_is_12_chars(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        assert len(fp.short_id) == 12

    def test_total_params_correct(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        # 4 + 2 + 4 + 2 = 12 params
        assert fp.total_params == 12

    def test_layer_count_correct(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        assert fp.layer_count == 4


class TestDeterminism:
    def test_same_weights_same_fingerprint(self):
        gen = ModelFingerprintGenerator()
        fp1 = gen.from_state_dict(make_state_dict(1.0), META)
        fp2 = gen.from_state_dict(make_state_dict(1.0), META)
        assert fp1.combined_hash == fp2.combined_hash

    def test_different_weights_different_fingerprint(self):
        gen = ModelFingerprintGenerator()
        fp1 = gen.from_state_dict(make_state_dict(1.0))
        fp2 = gen.from_state_dict(make_state_dict(2.0))
        assert fp1.combined_hash != fp2.combined_hash

    def test_different_metadata_different_arch_hash(self):
        gen = ModelFingerprintGenerator()
        fp1 = gen.from_state_dict(make_state_dict(), {"num_layers": 2})
        fp2 = gen.from_state_dict(make_state_dict(), {"num_layers": 4})
        assert fp1.architecture_hash != fp2.architecture_hash


class TestMatchMethod:
    def test_identical_fingerprints_match(self):
        gen = ModelFingerprintGenerator()
        fp1 = gen.from_state_dict(make_state_dict(1.0), META)
        fp2 = gen.from_state_dict(make_state_dict(1.0), META)
        assert fp1.matches(fp2)

    def test_different_fingerprints_no_match(self):
        gen = ModelFingerprintGenerator()
        fp1 = gen.from_state_dict(make_state_dict(1.0))
        fp2 = gen.from_state_dict(make_state_dict(2.0))
        assert not fp1.matches(fp2)


class TestToDict:
    def test_to_dict_has_required_keys(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        d = fp.to_dict()
        for key in ["short_id", "combined_hash", "total_params", "layer_count"]:
            assert key in d

    def test_to_dict_values_consistent(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        d = fp.to_dict()
        assert d["short_id"] == fp.short_id
        assert d["total_params"] == fp.total_params


class TestEdgeCases:
    def test_empty_state_dict(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict({})
        assert fp.total_params == 0
        assert fp.layer_count == 0

    def test_layer_stats_populated(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict())
        assert len(fp.layer_stats) == 4
        for ls in fp.layer_stats:
            assert isinstance(ls, LayerStats)
            assert ls.num_params > 0

    def test_metadata_stored(self):
        gen = ModelFingerprintGenerator()
        fp = gen.from_state_dict(make_state_dict(), META)
        assert fp.metadata == META
