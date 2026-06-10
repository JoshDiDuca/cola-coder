"""Tests for the hardware profiler and Full Auto Pipeline wiring.

Covers:
- GPU capability detection (bf16 vs fp16 by compute capability)
- CPU-only fallback behavior
- VRAM tier → config mapping with estimator-driven fitting
- Auto-config generation (overrides applied, smoke mode, loadable by Config)
- Menu and script wiring (no orphan script, feature categorized)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cola_coder.features.hardware_profiler import (
    GPUInfo,
    HardwareProfile,
    TrainingRecommendation,
    generate_auto_config,
    profile_hardware,
    recommend_config,
)

PROJECT_ROOT = Path(__file__).parent.parent


def _gpu(vram_gb: float, cc: tuple[int, int] = (8, 9), name: str = "Test GPU") -> GPUInfo:
    return GPUInfo(index=0, name=name, total_vram_gb=vram_gb, compute_capability=cc)


def _profile(*gpus: GPUInfo) -> HardwareProfile:
    return HardwareProfile(
        has_cuda=bool(gpus), gpus=list(gpus), cpu_count=16, total_ram_gb=64.0,
    )


# ── GPUInfo ─────────────────────────────────────────────────────────────────

class TestGPUInfo:
    def test_ampere_supports_bf16(self):
        assert _gpu(16, cc=(8, 0)).supports_bf16
        assert _gpu(16, cc=(8, 9)).supports_bf16
        assert _gpu(16, cc=(9, 0)).supports_bf16

    def test_pre_ampere_no_bf16(self):
        assert not _gpu(10, cc=(7, 5)).supports_bf16
        assert not _gpu(10, cc=(6, 1)).supports_bf16

    def test_best_gpu_picks_most_vram(self):
        small = _gpu(10, name="RTX 3080")
        big = _gpu(16, name="RTX 4080 Super")
        big.index = 1
        profile = _profile(small, big)
        assert profile.best_gpu is big


# ── profile_hardware ────────────────────────────────────────────────────────

class TestProfileHardware:
    def test_cpu_only_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            profile = profile_hardware()
        assert profile.has_cuda is False
        assert profile.gpus == []
        assert profile.cpu_count >= 1
        assert profile.torch_version  # torch is installed in the test env

    def test_never_raises_on_broken_cuda(self):
        with patch("torch.cuda.is_available", side_effect=RuntimeError("driver")):
            profile = profile_hardware()
        assert profile.has_cuda is False


# ── recommend_config ────────────────────────────────────────────────────────

class TestRecommendConfig:
    def test_cpu_only_recommends_tiny_fp32(self):
        rec = recommend_config(_profile())
        assert rec.config_name == "tiny"
        assert rec.precision == "fp32"
        assert rec.gpu is None
        assert rec.warnings  # must warn about CPU speed

    def test_16gb_ampere_recommends_4080_max_bf16(self):
        rec = recommend_config(_profile(_gpu(16.0, cc=(8, 9), name="RTX 4080 Super")))
        assert rec.config_name == "4080_max"
        assert rec.precision == "bf16"

    def test_8gb_recommends_small(self):
        rec = recommend_config(_profile(_gpu(8.0)))
        assert rec.config_name == "small"

    def test_3_5gb_recommends_tiny(self):
        rec = recommend_config(_profile(_gpu(3.5)))
        assert rec.config_name == "tiny"

    def test_pre_ampere_gets_fp16(self):
        rec = recommend_config(_profile(_gpu(10.0, cc=(7, 5), name="RTX 3080")))
        assert rec.precision == "fp16"

    @pytest.mark.parametrize("vram", [4.0, 8.0, 12.0, 16.0, 24.0])
    def test_estimate_always_fits_budget(self, vram: float):
        rec = recommend_config(_profile(_gpu(vram)))
        if rec.estimated_vram_gb is not None:
            assert rec.estimated_vram_gb <= vram * 0.92 + 1e-6

    def test_effective_batch_preserved_when_halving(self):
        # Whatever adjustments happen, effective batch must stay >= the
        # base config's batch_size (halving doubles accumulation).
        rec = recommend_config(_profile(_gpu(16.0)))
        base = yaml.safe_load(
            (PROJECT_ROOT / "configs" / f"{rec.config_name}.yaml").read_text()
        )
        base_effective = (
            base["training"]["batch_size"] * base["training"]["gradient_accumulation"]
        )
        assert rec.batch_size * rec.gradient_accumulation >= base_effective

    def test_multi_gpu_notes_device_choice(self):
        gpu0 = _gpu(10.0, name="RTX 3080")
        gpu1 = _gpu(16.0, name="RTX 4080 Super")
        gpu1.index = 1
        rec = recommend_config(_profile(gpu0, gpu1))
        assert rec.gpu is gpu1
        assert any("GPU" in r and "1" in r for r in rec.reasons)


# ── generate_auto_config ────────────────────────────────────────────────────

class TestGenerateAutoConfig:
    def _rec(self, **overrides) -> TrainingRecommendation:
        defaults = dict(
            config_name="tiny",
            config_path=str(PROJECT_ROOT / "configs" / "tiny.yaml"),
            precision="bf16",
            batch_size=8,
            gradient_accumulation=4,
            gradient_checkpointing=True,
            estimated_vram_gb=5.0,
            gpu=_gpu(16.0),
        )
        defaults.update(overrides)
        return TrainingRecommendation(**defaults)

    def test_overrides_applied(self, tmp_path: Path):
        out = generate_auto_config(self._rec(), output_dir=tmp_path)
        raw = yaml.safe_load(out.read_text())
        assert raw["training"]["precision"] == "bf16"
        assert raw["training"]["batch_size"] == 8
        assert raw["training"]["gradient_accumulation"] == 4
        assert raw["training"]["gradient_checkpointing"] is True

    def test_output_dir_isolated_from_base(self, tmp_path: Path):
        out = generate_auto_config(self._rec(), output_dir=tmp_path)
        raw = yaml.safe_load(out.read_text())
        assert raw["checkpoint"]["output_dir"].endswith("_auto")

    def test_smoke_mode_shrinks_run(self, tmp_path: Path):
        out = generate_auto_config(self._rec(), output_dir=tmp_path, smoke=True)
        raw = yaml.safe_load(out.read_text())
        assert raw["training"]["max_steps"] == 30
        assert raw["training"]["warmup_steps"] == 5
        assert raw["training"]["batch_size"] <= 4
        assert raw["checkpoint"]["output_dir"].endswith("_smoke")
        assert out.name == "auto_tiny_smoke.yaml"

    def test_generated_config_loads_via_config_from_yaml(self, tmp_path: Path):
        from cola_coder.model.config import Config

        out = generate_auto_config(self._rec(), output_dir=tmp_path)
        cfg = Config.from_yaml(out)
        assert cfg.training.batch_size == 8
        assert cfg.training.precision == "bf16"

    def test_non_smoke_keeps_base_max_steps(self, tmp_path: Path):
        out = generate_auto_config(self._rec(), output_dir=tmp_path, smoke=False)
        raw = yaml.safe_load(out.read_text())
        base = yaml.safe_load((PROJECT_ROOT / "configs" / "tiny.yaml").read_text())
        assert raw["training"]["max_steps"] == base["training"]["max_steps"]


# ── Wiring ──────────────────────────────────────────────────────────────────

class TestWiring:
    def test_pipeline_menu_has_full_auto(self):
        from cola_coder.features.menus.pipeline_menu import PipelineMenu
        assert callable(getattr(PipelineMenu, "_full_auto", None))
        assert callable(getattr(PipelineMenu, "_unique_run_name", None))

    def test_master_menu_has_full_auto_entry(self):
        from cola_coder.features.master_menu import MasterMenu
        assert callable(getattr(MasterMenu, "full_auto_pipeline", None))

    def test_auto_pipeline_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "auto_pipeline.py").exists()

    def test_feature_is_categorized(self):
        from cola_coder.features.master_menu import _FEATURE_CATEGORIES
        all_stems = [s for stems in _FEATURE_CATEGORIES.values() for s in stems]
        assert "hardware_profiler" in all_stems
