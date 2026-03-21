"""Tests for ModelSizeEstimator (features/model_size_estimator.py)."""

from __future__ import annotations


import pytest

from cola_coder.features.model_size_estimator import (
    ModelSizeEstimator,
    SizeReport,
    _DictModel,
)


# ---------------------------------------------------------------------------
# Fake ModelConfig dict (matches the expected attribute names)
# ---------------------------------------------------------------------------

TINY_CONFIG = {
    "name": "tiny",
    "dim": 512,
    "n_heads": 8,
    "n_kv_heads": 4,
    "n_layers": 8,
    "vocab_size": 32_000,
    "max_seq_len": 2048,
    "ffn_hidden_dim": 1365,  # ~8/3 * 512 rounded
}

SMALL_CONFIG = {
    "name": "small",
    "dim": 768,
    "n_heads": 12,
    "n_kv_heads": 6,
    "n_layers": 12,
    "vocab_size": 32_000,
    "max_seq_len": 2048,
    "ffn_hidden_dim": 2048,
}


@pytest.fixture()
def estimator() -> ModelSizeEstimator:
    return ModelSizeEstimator()


@pytest.fixture()
def tiny_model() -> _DictModel:
    return _DictModel(TINY_CONFIG)


@pytest.fixture()
def tiny_report(estimator, tiny_model) -> SizeReport:
    return estimator.estimate(tiny_model)


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.model_size_estimator import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestEstimateBasic:
    def test_returns_size_report(self, tiny_report):
        assert isinstance(tiny_report, SizeReport)

    def test_config_name(self, tiny_report):
        assert tiny_report.config_name == "tiny"

    def test_total_params_positive(self, tiny_report):
        assert tiny_report.params.total > 0

    def test_non_embedding_less_than_total(self, tiny_report):
        assert tiny_report.params.non_embedding < tiny_report.params.total

    def test_embedding_params(self, tiny_report):
        expected = TINY_CONFIG["vocab_size"] * TINY_CONFIG["dim"]
        assert tiny_report.params.embedding == expected

    def test_larger_model_more_params(self, estimator):
        tiny = estimator.estimate(_DictModel(TINY_CONFIG))
        small = estimator.estimate(_DictModel(SMALL_CONFIG))
        assert small.params.total > tiny.params.total


class TestMemoryEstimates:
    def test_all_precisions_present(self, tiny_report):
        for prec in ("fp32", "bf16", "fp16", "int8", "int4"):
            assert prec in tiny_report.memory

    def test_fp32_larger_than_bf16(self, tiny_report):
        assert tiny_report.memory["fp32"].weights_gb > tiny_report.memory["bf16"].weights_gb

    def test_int4_smallest(self, tiny_report):
        weights = {p: tiny_report.memory[p].weights_gb for p in tiny_report.memory}
        assert weights["int4"] < weights["fp16"]
        assert weights["int4"] < weights["fp32"]

    def test_training_larger_than_inference(self, tiny_report):
        for prec in ("fp32", "bf16"):
            m = tiny_report.memory[prec]
            assert m.total_training_gb > m.total_inference_gb

    def test_memory_values_positive(self, tiny_report):
        for prec, m in tiny_report.memory.items():
            assert m.weights_gb > 0, prec
            assert m.total_inference_gb > 0, prec


class TestEstimateFromDict:
    def test_from_dict(self, estimator):
        report = estimator.estimate_from_dict(TINY_CONFIG)
        assert isinstance(report, SizeReport)
        assert report.params.total > 0

    def test_missing_attr_raises(self):
        model = _DictModel({"dim": 512})
        with pytest.raises(AttributeError):
            model.n_heads  # missing key


class TestPrintTable:
    def test_print_table_does_not_crash(self, estimator, tiny_model, capsys):
        report = estimator.estimate(tiny_model)
        estimator.print_table(report)
        captured = capsys.readouterr()
        assert "tiny" in captured.out
        assert "fp32" in captured.out
        assert "bf16" in captured.out

    def test_print_table_shows_params(self, estimator, tiny_model, capsys):
        report = estimator.estimate(tiny_model)
        estimator.print_table(report)
        captured = capsys.readouterr()
        assert "M" in captured.out  # millions notation
