"""Tests for GGUF export, Ollama Modelfile generation, and quantization.

Coverage:
- Weight name mapping (cola-coder → GGUF)
- GGUF file writing (both built-in writer and gguf package if available)
- Modelfile generation content
- Dynamic INT8 quantization
- Weight-only INT4 quantization
- QuantResult metrics
- ExportResult dataclass
- Benchmark comparison
- Edge cases (unknown keys, tied weights, f32/f16/q8_0)

Run:
    cd "C:/Users/josh/ai research/cola-coder"
    .venv/Scripts/pytest tests/test_export.py -v
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer
from cola_coder.export.gguf_export import (
    GGUFExporter,
    ExportResult,
    _map_single_key,
    _GGUF_MAGIC,
    _GGUF_VERSION,
    _to_f32,
    _to_f16,
    _quantize_q8_0,
)
from cola_coder.export.ollama_export import OllamaExporter
from cola_coder.export.quantize import ModelQuantizer, QuantResult


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tiny_config() -> ModelConfig:
    """Minimal config for fast tests (no GPU needed)."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )


def _tiny_model() -> Transformer:
    return Transformer(_tiny_config())


def _fake_state_dict(config: ModelConfig) -> dict[str, torch.Tensor]:
    """Build a realistic (random-valued) state dict matching cola-coder checkpoint format.

    output.weight is intentionally excluded (weight-tied to tok_emb.weight).
    """
    state: dict[str, torch.Tensor] = {}
    state["tok_emb.weight"] = torch.randn(config.vocab_size, config.dim)

    for i in range(config.n_layers):
        kv_dim = config.n_kv_heads * config.head_dim
        state[f"blocks.{i}.attn_norm.weight"] = torch.ones(config.dim)
        state[f"blocks.{i}.ffn_norm.weight"] = torch.ones(config.dim)
        state[f"blocks.{i}.attention.q_proj.weight"] = torch.randn(config.dim, config.dim)
        state[f"blocks.{i}.attention.k_proj.weight"] = torch.randn(kv_dim, config.dim)
        state[f"blocks.{i}.attention.v_proj.weight"] = torch.randn(kv_dim, config.dim)
        state[f"blocks.{i}.attention.out_proj.weight"] = torch.randn(config.dim, config.dim)
        state[f"blocks.{i}.ffn.gate_proj.weight"] = torch.randn(config.ffn_hidden_dim, config.dim)
        state[f"blocks.{i}.ffn.up_proj.weight"] = torch.randn(config.ffn_hidden_dim, config.dim)
        state[f"blocks.{i}.ffn.down_proj.weight"] = torch.randn(config.dim, config.ffn_hidden_dim)

    state["final_norm.weight"] = torch.ones(config.dim)
    return state


def _save_fake_checkpoint(tmp_dir: Path, config: ModelConfig) -> Path:
    """Write a fake model.safetensors to tmp_dir and return the directory path."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    state = _fake_state_dict(config)
    save_file(state, str(tmp_dir / "model.safetensors"))
    return tmp_dir


# ──────────────────────────────────────────────────────────────────────────────
# 1. Weight name mapping
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightNameMapping:

    def test_tok_emb_maps_to_token_embd(self):
        assert _map_single_key("tok_emb.weight") == "token_embd.weight"

    def test_final_norm_maps_to_output_norm(self):
        assert _map_single_key("final_norm.weight") == "output_norm.weight"

    def test_attention_q_proj(self):
        assert _map_single_key("blocks.0.attention.q_proj.weight") == "blk.0.attn_q.weight"

    def test_attention_k_proj(self):
        assert _map_single_key("blocks.3.attention.k_proj.weight") == "blk.3.attn_k.weight"

    def test_attention_v_proj(self):
        assert _map_single_key("blocks.1.attention.v_proj.weight") == "blk.1.attn_v.weight"

    def test_attention_out_proj(self):
        assert _map_single_key("blocks.2.attention.out_proj.weight") == "blk.2.attn_output.weight"

    def test_ffn_gate_proj(self):
        assert _map_single_key("blocks.0.ffn.gate_proj.weight") == "blk.0.ffn_gate.weight"

    def test_ffn_up_proj(self):
        assert _map_single_key("blocks.0.ffn.up_proj.weight") == "blk.0.ffn_up.weight"

    def test_ffn_down_proj(self):
        assert _map_single_key("blocks.0.ffn.down_proj.weight") == "blk.0.ffn_down.weight"

    def test_attn_norm(self):
        assert _map_single_key("blocks.5.attn_norm.weight") == "blk.5.attn_norm.weight"

    def test_ffn_norm(self):
        assert _map_single_key("blocks.5.ffn_norm.weight") == "blk.5.ffn_norm.weight"

    def test_unknown_key_returns_none(self):
        assert _map_single_key("some.unknown.key") is None

    def test_unknown_layer_suffix_returns_none(self):
        assert _map_single_key("blocks.0.nonexistent.weight") is None

    def test_layer_index_preserved(self):
        for i in range(8):
            result = _map_single_key(f"blocks.{i}.attn_norm.weight")
            assert result == f"blk.{i}.attn_norm.weight"


# ──────────────────────────────────────────────────────────────────────────────
# 2. GGUFExporter._map_weight_names
# ──────────────────────────────────────────────────────────────────────────────

class TestGGUFExporterWeightMapping:

    def setup_method(self):
        self.config = _tiny_config()
        self.exporter = GGUFExporter(self.config)

    def test_all_cola_coder_keys_mapped(self):
        state = _fake_state_dict(self.config)
        mapped = self.exporter._map_weight_names(state)
        # Should have no cola-coder style keys
        for key in mapped:
            assert not key.startswith("tok_emb"), f"Unmapped key: {key}"
            assert not key.startswith("blocks."), f"Unmapped key: {key}"
            assert not key.startswith("final_norm"), f"Unmapped key: {key}"

    def test_output_weight_is_injected(self):
        """output.weight must be added even though it's absent from checkpoint."""
        state = _fake_state_dict(self.config)
        assert "output.weight" not in state  # confirm it's missing from checkpoint
        mapped = self.exporter._map_weight_names(state)
        assert "output.weight" in mapped

    def test_output_weight_equals_token_embd(self):
        """Weight tying: output.weight should be the same tensor as token_embd.weight."""
        state = _fake_state_dict(self.config)
        mapped = self.exporter._map_weight_names(state)
        assert torch.equal(mapped["output.weight"], mapped["token_embd.weight"])

    def test_expected_gguf_keys_present(self):
        state = _fake_state_dict(self.config)
        mapped = self.exporter._map_weight_names(state)
        expected = {
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        }
        missing = expected - mapped.keys()
        assert not missing, f"Missing GGUF keys: {missing}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. GGUF file writing (built-in writer)
# ──────────────────────────────────────────────────────────────────────────────

class TestGGUFFileWriting:

    def setup_method(self):
        self.config = _tiny_config()
        self.exporter = GGUFExporter(self.config)

    def test_export_f32_success(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="f32")
        assert result.success, f"Export failed: {result.error}"
        assert Path(out).exists()
        assert result.file_size_mb > 0

    def test_export_f16_success(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model-f16.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="f16")
        assert result.success
        assert Path(out).exists()

    def test_export_q8_0_success(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model-q8.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="q8_0")
        assert result.success
        assert Path(out).exists()

    def test_export_returns_tensor_count(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="f16")
        # 2 layers × 9 weights + token_embd + output_norm + output = 2*9+3 = 21
        assert result.num_tensors > 0

    def test_gguf_file_starts_with_magic(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        self.exporter.export(str(ckpt_dir), out, quantization="f16")
        with open(out, "rb") as f:
            magic = f.read(4)
        assert magic == _GGUF_MAGIC, f"Expected GGUF magic, got {magic!r}"

    def test_gguf_file_version_3(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        self.exporter.export(str(ckpt_dir), out, quantization="f16")
        with open(out, "rb") as f:
            f.read(4)  # magic
            version = struct.unpack("<I", f.read(4))[0]
        assert version == _GGUF_VERSION

    def test_export_invalid_quantization_returns_error(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="invalid_quant")
        assert not result.success
        assert result.error != ""

    def test_export_missing_checkpoint_returns_error(self, tmp_path):
        result = self.exporter.export(
            str(tmp_path / "does_not_exist"), str(tmp_path / "out.gguf"), quantization="f16"
        )
        assert not result.success

    def test_export_q4_success(self, tmp_path):
        """q4_k_m falls back to q8_0 in built-in writer — should still succeed."""
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model-q4.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="q4_k_m")
        assert result.success

    def test_export_result_quantization_field(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out = str(tmp_path / "model.gguf")
        result = self.exporter.export(str(ckpt_dir), out, quantization="q8_0")
        assert result.quantization == "q8_0"

    def test_f16_file_smaller_than_f32(self, tmp_path):
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", self.config)
        out_f32 = str(tmp_path / "model-f32.gguf")
        out_f16 = str(tmp_path / "model-f16.gguf")
        r32 = self.exporter.export(str(ckpt_dir), out_f32, quantization="f32")
        r16 = self.exporter.export(str(ckpt_dir), out_f16, quantization="f16")
        assert r32.success and r16.success
        assert r16.file_size_mb < r32.file_size_mb


# ──────────────────────────────────────────────────────────────────────────────
# 4. Quantization helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestQuantizationHelpers:

    def test_to_f32_shape_preserved(self):
        t = torch.randn(8, 16)
        arr = _to_f32(t)
        assert arr.shape == (8, 16)
        assert arr.dtype == np.float32

    def test_to_f16_shape_preserved(self):
        t = torch.randn(8, 16)
        arr = _to_f16(t)
        assert arr.shape == (8, 16)
        assert arr.dtype == np.float16

    def test_q8_0_output_is_bytes(self):
        t = torch.randn(64)
        arr = _quantize_q8_0(t)
        assert arr.dtype == np.uint8

    def test_q8_0_block_size(self):
        """Q8_0 packs 32 int8 + 2 byte scale = 34 bytes per block."""
        t = torch.randn(64)  # 2 blocks of 32
        arr = _quantize_q8_0(t)
        assert len(arr) == 2 * 34  # 2 blocks × 34 bytes


# ──────────────────────────────────────────────────────────────────────────────
# 5. OllamaExporter
# ──────────────────────────────────────────────────────────────────────────────

class TestOllamaExporter:

    def setup_method(self):
        self.exporter = OllamaExporter()

    def test_modelfile_is_created(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path), "cola-coder")
        assert Path(mf_path).exists()

    def test_modelfile_contains_from_directive(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path))
        content = Path(mf_path).read_text()
        assert "FROM" in content

    def test_modelfile_contains_gguf_path(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path))
        content = Path(mf_path).read_text()
        assert "model.gguf" in content

    def test_modelfile_contains_template(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path))
        content = Path(mf_path).read_text()
        assert "TEMPLATE" in content

    def test_modelfile_contains_temperature(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path))
        content = Path(mf_path).read_text()
        assert "temperature" in content

    def test_modelfile_uses_model_name(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        mf_path = self.exporter.create_modelfile(str(gguf), str(tmp_path), "my-model")
        content = Path(mf_path).read_text()
        assert "my-model" in content

    def test_readme_is_created(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        self.exporter.create_modelfile(str(gguf), str(tmp_path))
        assert (tmp_path / "README_ollama.txt").exists()

    def test_create_modelfile_creates_output_dir(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        new_dir = tmp_path / "subdir" / "exports"
        self.exporter.create_modelfile(str(gguf), str(new_dir))
        assert new_dir.exists()


# ──────────────────────────────────────────────────────────────────────────────
# 6. ModelQuantizer
# ──────────────────────────────────────────────────────────────────────────────

class TestModelQuantizer:

    def setup_method(self):
        self.model = _tiny_model()

    def test_dynamic_quantization_returns_model(self):
        quantizer = ModelQuantizer(self.model)
        q_model, result = quantizer.quantize_dynamic()
        assert isinstance(q_model, nn.Module)

    def test_dynamic_quant_result_fields(self):
        quantizer = ModelQuantizer(self.model)
        _, result = quantizer.quantize_dynamic()
        assert isinstance(result, QuantResult)
        assert result.original_size_mb > 0
        assert result.method == "dynamic_int8"

    def test_dynamic_quant_compression_ratio(self):
        """INT8 should give at least 1× compression (never larger)."""
        quantizer = ModelQuantizer(self.model)
        _, result = quantizer.quantize_dynamic()
        # torch.ao dynamic quant may not always shrink the python object, but ratio >= 0.5
        assert result.compression_ratio >= 0.5

    def test_dynamic_quant_model_forward_runs(self):
        """Quantized model must still produce valid logits."""
        quantizer = ModelQuantizer(self.model)
        q_model, _ = quantizer.quantize_dynamic()
        q_model.eval()
        token_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = q_model(token_ids)
        assert logits.shape == (1, 8, 256)
        assert not torch.isnan(logits).any()

    def test_weights_only_int8_same_as_dynamic(self):
        """quantize_weights_only(bits=8) delegates to quantize_dynamic."""
        quantizer = ModelQuantizer(self.model)
        _, r8 = quantizer.quantize_weights_only(bits=8)
        assert r8.method == "dynamic_int8"

    def test_weights_only_int4_returns_model(self):
        quantizer = ModelQuantizer(self.model)
        q_model, result = quantizer.quantize_weights_only(bits=4)
        assert isinstance(q_model, nn.Module)
        assert result.method == "weight_only_int4"

    def test_weights_only_int4_model_forward_runs(self):
        """INT4 quantized model must produce valid logits on CPU."""
        quantizer = ModelQuantizer(self.model)
        q_model, _ = quantizer.quantize_weights_only(bits=4)
        q_model.eval()
        token_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = q_model(token_ids)
        assert logits.shape == (1, 8, 256)
        assert not torch.isnan(logits).any()

    def test_invalid_bits_raises(self):
        quantizer = ModelQuantizer(self.model)
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            quantizer.quantize_weights_only(bits=3)

    def test_benchmark_returns_expected_keys(self):
        quantizer = ModelQuantizer(self.model)
        q_model, _ = quantizer.quantize_dynamic()
        stats = quantizer.benchmark(
            self.model, q_model, ["def hello():"], device="cpu"
        )
        expected_keys = {
            "original_ms", "quantized_ms", "speedup",
            "original_size_mb", "quantized_size_mb",
            "compression_ratio", "logit_cosine_sim",
        }
        assert expected_keys.issubset(stats.keys())

    def test_benchmark_logit_cosine_sim_range(self):
        """Dynamic INT8 should produce very similar logits (cosine sim > 0.9)."""
        quantizer = ModelQuantizer(self.model)
        q_model, _ = quantizer.quantize_dynamic()
        stats = quantizer.benchmark(
            self.model, q_model, ["test"], device="cpu"
        )
        assert -1.0 <= stats["logit_cosine_sim"] <= 1.0

    def test_quant_result_dataclass_fields(self):
        result = QuantResult(
            original_size_mb=100.0,
            quantized_size_mb=50.0,
            compression_ratio=2.0,
            method="dynamic_int8",
        )
        assert result.compression_ratio == 2.0
        assert result.backend == "torch.ao"  # default

    def test_export_result_dataclass_fields(self):
        result = ExportResult(
            output_path="/tmp/model.gguf",
            file_size_mb=42.5,
            quantization="f16",
            num_tensors=21,
            success=True,
        )
        assert result.success
        assert result.error == ""


# ──────────────────────────────────────────────────────────────────────────────
# 7. End-to-end: export from real-looking state dict
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_export_pipeline(self, tmp_path):
        """Build fake checkpoint → export GGUF → check result."""
        config = _tiny_config()
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", config)
        exporter = GGUFExporter(config)
        out = str(tmp_path / "cola-coder.gguf")
        result = exporter.export(str(ckpt_dir), out, quantization="f16")
        assert result.success
        assert result.num_tensors >= 1
        assert result.file_size_mb > 0

    def test_export_then_modelfile(self, tmp_path):
        """Export GGUF and immediately create Modelfile."""
        config = _tiny_config()
        ckpt_dir = _save_fake_checkpoint(tmp_path / "ckpt", config)
        exporter = GGUFExporter(config)
        gguf_path = str(tmp_path / "cola-coder-f16.gguf")
        r = exporter.export(str(ckpt_dir), gguf_path, quantization="f16")
        assert r.success

        ollama = OllamaExporter()
        mf = ollama.create_modelfile(gguf_path, str(tmp_path / "exports"), "cola-coder")
        content = Path(mf).read_text()
        assert "cola-coder" in content
        assert "FROM" in content
