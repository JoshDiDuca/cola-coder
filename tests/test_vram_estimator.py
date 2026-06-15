"""Tests for the VRAM estimator's GPU-probe robustness.

The breakdown math (weights / optimizer / gradients / activations) is
GPU-independent. Only the "fits on this GPU?" check needs to probe CUDA.
A broken or partial CUDA install can make ``torch.cuda.is_available()``
return True while ``torch.cuda.get_device_properties(0)`` raises
``RuntimeError`` — that probe failure must never crash callers such as
``scripts/vram_estimate.py``; it should degrade to a GPU-less estimate
(the same shape as a machine with no CUDA at all).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cola_coder.features.vram_estimator import estimate_vram
from cola_coder.model.config import ModelConfig, TrainingConfig


def _model() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64,
    )


def _training() -> TrainingConfig:
    return TrainingConfig(batch_size=2, precision="bf16")


class TestVramEstimatorGpuProbe:
    def test_broken_cuda_probe_does_not_crash(self):
        # is_available() True but get_device_properties raises RuntimeError —
        # the classic broken-driver / partial-CUDA failure mode.
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="Broken GPU"
        ), patch(
            "torch.cuda.get_device_properties",
            side_effect=RuntimeError("CUDA driver version is insufficient"),
        ):
            est = estimate_vram(model_config=_model(), training_config=_training())

        # The estimate is still produced; only the fit check is skipped.
        assert est.total_training_gb > 0
        assert est.gpu_name is None
        assert est.gpu_vram_gb is None
        assert est.fits_training is None
        assert est.fits_inference is None

    def test_no_cuda_yields_none_fit_fields(self):
        with patch("torch.cuda.is_available", return_value=False):
            est = estimate_vram(model_config=_model(), training_config=_training())

        assert est.total_training_gb > 0
        assert est.gpu_name is None
        assert est.fits_training is None

    def test_breakdown_is_gpu_independent(self):
        # The numeric breakdown must be identical whether or not the GPU probe
        # succeeds — the probe only affects the gpu_*/fits_* fields.
        with patch("torch.cuda.is_available", return_value=False):
            no_gpu = estimate_vram(model_config=_model(), training_config=_training())
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_properties",
            side_effect=RuntimeError("boom"),
        ):
            broken = estimate_vram(model_config=_model(), training_config=_training())

        assert no_gpu.total_training_gb == broken.total_training_gb
        assert no_gpu.model_weights_gb == broken.model_weights_gb
        assert no_gpu.activations_gb == broken.activations_gb


class TestKvCacheQuantization:
    """kv_cache_bits projects int8/int4 KV-cache memory (long-context VRAM planning)."""

    def _est(self, bits: int):
        with patch("torch.cuda.is_available", return_value=False):
            return estimate_vram(model_config=_model(), training_config=_training(), kv_cache_bits=bits)

    def test_default_16_is_unchanged(self):
        with patch("torch.cuda.is_available", return_value=False):
            default = estimate_vram(model_config=_model(), training_config=_training())
        explicit16 = self._est(16)
        assert default.kv_cache_gb == explicit16.kv_cache_gb

    def test_int8_halves_kv_cache(self):
        assert self._est(8).kv_cache_gb == pytest.approx(self._est(16).kv_cache_gb / 2)

    def test_int4_quarters_kv_cache(self):
        assert self._est(4).kv_cache_gb == pytest.approx(self._est(16).kv_cache_gb / 4)

    def test_kv_quant_only_scales_kv_not_weights(self):
        # Quantizing the KV cache must not change the model-weights term.
        assert self._est(8).model_weights_gb == self._est(16).model_weights_gb
        # And lower-bit KV → lower total inference memory.
        assert self._est(8).total_inference_gb < self._est(16).total_inference_gb

    def test_invalid_bits_raise(self):
        with pytest.raises(ValueError):
            self._est(12)
