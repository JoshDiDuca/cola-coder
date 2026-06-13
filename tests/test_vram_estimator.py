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
