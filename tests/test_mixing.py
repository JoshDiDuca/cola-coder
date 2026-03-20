"""Tests for data mixing optimization.

Tests verify:
1. MixingConfig creation and validation
2. MixingConfig normalization
3. Presets exist and have valid weights (sum to ~1.0)
4. suggest_proportions: source with higher loss should get higher weight
5. PerSourceTracker records and computes weights correctly
6. MixingOptimizer candidate generation
7. Edge cases (single source, zero weights, etc.)
"""

import math

import pytest

from cola_coder.data.mixing import (
    MixingConfig,
    MixingOptimizer,
    MIXING_PRESETS,
    PerSourceTracker,
)


# ---------------------------------------------------------------------------
# MixingConfig basics
# ---------------------------------------------------------------------------


class TestMixingConfig:
    def test_create_simple(self):
        config = MixingConfig(sources={"typescript": 0.6, "python": 0.4})
        assert config.sources["typescript"] == 0.6
        assert config.sources["python"] == 0.4

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError, match="at least one source"):
            MixingConfig(sources={})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            MixingConfig(sources={"ts": -0.5, "py": 1.0})

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="positive number"):
            MixingConfig(sources={"ts": 0.0, "py": 0.0})

    def test_normalize(self):
        config = MixingConfig(sources={"a": 2.0, "b": 3.0, "c": 5.0})
        normalized = config.normalize()
        assert abs(sum(normalized.sources.values()) - 1.0) < 1e-9
        assert abs(normalized.sources["a"] - 0.2) < 1e-9
        assert abs(normalized.sources["b"] - 0.3) < 1e-9
        assert abs(normalized.sources["c"] - 0.5) < 1e-9

    def test_is_normalized(self):
        config = MixingConfig(sources={"a": 0.5, "b": 0.5})
        assert config.is_normalized()

        config2 = MixingConfig(sources={"a": 2.0, "b": 3.0})
        assert not config2.is_normalized()

    def test_to_weight_list(self):
        config = MixingConfig(sources={"ts": 0.7, "py": 0.3})
        weights = config.to_weight_list(["py", "ts"])
        # Should be normalized: py=0.3/1.0, ts=0.7/1.0
        assert abs(weights[0] - 0.3) < 1e-9
        assert abs(weights[1] - 0.7) < 1e-9

    def test_to_weight_list_missing_source(self):
        config = MixingConfig(sources={"ts": 0.7, "py": 0.3})
        weights = config.to_weight_list(["ts", "java"])
        # java is missing, gets 0 weight; renormalized: ts=1.0, java=0.0
        assert abs(weights[0] - 1.0) < 1e-9
        assert abs(weights[1] - 0.0) < 1e-9

    def test_to_weight_list_no_overlap_falls_back_to_equal(self):
        config = MixingConfig(sources={"ts": 0.7, "py": 0.3})
        weights = config.to_weight_list(["java", "go"])
        # No overlap, falls back to equal weights
        assert abs(weights[0] - 0.5) < 1e-9
        assert abs(weights[1] - 0.5) < 1e-9

    def test_describe(self):
        config = MixingConfig(sources={"ts": 0.6, "py": 0.4})
        desc = config.describe()
        assert "ts" in desc
        assert "py" in desc
        assert "60%" in desc
        assert "40%" in desc

    def test_to_dict_and_from_dict(self):
        config = MixingConfig(sources={"ts": 0.5, "py": 0.3, "js": 0.2})
        d = config.to_dict()
        restored = MixingConfig.from_dict(d)
        assert restored.sources == config.sources


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_exist(self):
        expected = {"typescript_focused", "balanced_code", "quality_tiers", "equal"}
        assert expected.issubset(set(MIXING_PRESETS.keys()))

    @pytest.mark.parametrize("preset_name", list(MIXING_PRESETS.keys()))
    def test_preset_weights_sum_to_one(self, preset_name):
        config = MIXING_PRESETS[preset_name]
        total = sum(config.sources.values())
        assert abs(total - 1.0) < 0.01, (
            f"Preset {preset_name!r} weights sum to {total}, expected ~1.0"
        )

    @pytest.mark.parametrize("preset_name", list(MIXING_PRESETS.keys()))
    def test_preset_all_weights_positive(self, preset_name):
        config = MIXING_PRESETS[preset_name]
        for source, weight in config.sources.items():
            assert weight > 0, f"Preset {preset_name!r} has non-positive weight for {source!r}"

    def test_from_preset(self):
        config = MixingConfig.from_preset("typescript_focused")
        assert config.sources["typescript"] == 0.50
        assert config.sources["javascript"] == 0.25

    def test_from_preset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown mixing preset"):
            MixingConfig.from_preset("nonexistent_preset")


# ---------------------------------------------------------------------------
# PerSourceTracker
# ---------------------------------------------------------------------------


class TestPerSourceTracker:
    def test_record_and_mean(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 2.0)
        tracker.record("ts", 3.0)
        tracker.record("py", 1.5)

        means = tracker.mean_losses()
        assert abs(means["ts"] - 2.5) < 1e-9
        assert abs(means["py"] - 1.5) < 1e-9

    def test_get_weights_inverse_loss(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 3.0)
        tracker.record("py", 1.0)

        weights = tracker.get_weights(method="inverse_loss")
        # ts has higher loss, should get higher weight
        assert weights["ts"] > weights["py"]
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # Proportional: ts=3/4=0.75, py=1/4=0.25
        assert abs(weights["ts"] - 0.75) < 1e-9
        assert abs(weights["py"] - 0.25) < 1e-9

    def test_get_weights_softmax(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 3.0)
        tracker.record("py", 1.0)

        weights = tracker.get_weights(method="softmax")
        # ts has higher loss, should get higher weight
        assert weights["ts"] > weights["py"]
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_get_weights_equal(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 3.0)
        tracker.record("py", 1.0)

        weights = tracker.get_weights(method="equal")
        assert abs(weights["ts"] - 0.5) < 1e-9
        assert abs(weights["py"] - 0.5) < 1e-9

    def test_get_weights_no_data_raises(self):
        tracker = PerSourceTracker()
        with pytest.raises(ValueError, match="No losses recorded"):
            tracker.get_weights()

    def test_get_weights_unknown_method_raises(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 2.0)
        with pytest.raises(ValueError, match="Unknown weighting method"):
            tracker.get_weights(method="bogus")

    def test_reset(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 2.0)
        tracker.record("py", 1.5)
        tracker.reset()
        assert tracker.source_losses == {}

    def test_summary(self):
        tracker = PerSourceTracker()
        tracker.record("ts", 3.0)
        tracker.record("py", 1.0)
        summary = tracker.summary()
        assert "ts" in summary
        assert "py" in summary
        assert "3.0000" in summary
        assert "1.0000" in summary

    def test_summary_no_data(self):
        tracker = PerSourceTracker()
        assert "No losses recorded" in tracker.summary()


# ---------------------------------------------------------------------------
# MixingOptimizer
# ---------------------------------------------------------------------------


class TestMixingOptimizer:
    def test_suggest_proportions_inverse_loss(self):
        optimizer = MixingOptimizer()
        config = optimizer.suggest_proportions(
            {"typescript": 3.0, "javascript": 2.0, "python": 1.0}
        )
        # Higher loss = higher weight
        assert config.sources["typescript"] > config.sources["javascript"]
        assert config.sources["javascript"] > config.sources["python"]
        # Should sum to 1.0
        assert abs(sum(config.sources.values()) - 1.0) < 1e-9

    def test_suggest_proportions_softmax(self):
        optimizer = MixingOptimizer()
        config = optimizer.suggest_proportions(
            {"typescript": 3.0, "javascript": 2.0, "python": 1.0},
            method="softmax",
        )
        # Higher loss = higher weight
        assert config.sources["typescript"] > config.sources["javascript"]
        assert config.sources["javascript"] > config.sources["python"]
        # Should sum to 1.0
        assert abs(sum(config.sources.values()) - 1.0) < 1e-9

    def test_suggest_proportions_empty_raises(self):
        optimizer = MixingOptimizer()
        with pytest.raises(ValueError, match="must not be empty"):
            optimizer.suggest_proportions({})

    def test_suggest_proportions_unknown_method_raises(self):
        optimizer = MixingOptimizer()
        with pytest.raises(ValueError, match="Unknown method"):
            optimizer.suggest_proportions({"ts": 2.0}, method="bogus")

    def test_grid_search_raises_not_implemented(self):
        optimizer = MixingOptimizer()
        with pytest.raises(NotImplementedError):
            optimizer.grid_search(
                sources={"ts": "data/ts.npy"},
                eval_data="data/eval.npy",
            )

    def test_generate_candidates_single_source(self):
        optimizer = MixingOptimizer(candidates=5)
        candidates = optimizer.generate_candidates(["typescript"])
        assert len(candidates) == 1
        assert abs(candidates[0].sources["typescript"] - 1.0) < 1e-9

    def test_generate_candidates_two_sources(self):
        optimizer = MixingOptimizer(candidates=5)
        candidates = optimizer.generate_candidates(["ts", "py"])
        assert len(candidates) == 5
        for config in candidates:
            assert abs(sum(config.sources.values()) - 1.0) < 1e-9
            assert config.sources["ts"] > 0
            assert config.sources["py"] > 0

    def test_generate_candidates_three_sources(self):
        optimizer = MixingOptimizer(candidates=8)
        candidates = optimizer.generate_candidates(["ts", "py", "js"])
        assert len(candidates) == 8
        for config in candidates:
            assert abs(sum(config.sources.values()) - 1.0) < 1e-9

    def test_generate_candidates_empty_raises(self):
        optimizer = MixingOptimizer()
        with pytest.raises(ValueError, match="at least one source"):
            optimizer.generate_candidates([])


# ---------------------------------------------------------------------------
# Integration: MixingConfig + PerSourceTracker
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_tracker_to_optimizer(self):
        """Tracker losses feed into optimizer to produce a config."""
        tracker = PerSourceTracker()
        tracker.record("ts", 3.0)
        tracker.record("ts", 3.2)
        tracker.record("py", 1.8)
        tracker.record("py", 2.0)

        optimizer = MixingOptimizer()
        config = optimizer.suggest_proportions(tracker.mean_losses())

        # TS has higher loss, should get more weight
        assert config.sources["ts"] > config.sources["py"]
        assert config.is_normalized()

    def test_config_to_weight_list_for_combiner(self):
        """MixingConfig can produce ordered weight lists for DatasetCombiner."""
        config = MixingConfig.from_preset("typescript_focused")
        source_order = ["typescript", "javascript", "python"]
        weights = config.to_weight_list(source_order)

        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-9
        # TS should have highest weight
        assert weights[0] > weights[1] > weights[2]
