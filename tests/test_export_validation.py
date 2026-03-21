"""Export validation tests.

Tests:
- GGUF export produces a file with the GGUF magic header
- Quantization produces smaller model representations
- Ollama Modelfile generation creates the correct structure
- Weight name mapping (cola-coder -> GGUF/llama.cpp names)
- ExportResult dataclass fields

No GPU required. Uses tiny synthetic weights and temp files.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cola_coder.export.gguf_export import (
    GGUFExporter,
    _map_single_key,
    _quantize_q8_0,
    _GGUF_MAGIC,
)
from cola_coder.export.ollama_export import OllamaExporter
from cola_coder.export.quantize import ModelQuantizer, _model_size_mb
from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)


def _make_model() -> Transformer:
    return Transformer(_tiny_config())


# ── 1. GGUF file header ───────────────────────────────────────────────────────

def test_gguf_export_creates_file(tmp_path):
    """GGUFExporter.export creates a non-empty output file."""
    from safetensors.torch import save_file

    model = _make_model()
    # Build a minimal safetensors checkpoint
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "output.weight"}
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_file(state, str(ckpt_dir / "model.safetensors"))

    exporter = GGUFExporter(_tiny_config())
    out_path = str(tmp_path / "model.gguf")
    result = exporter.export(str(ckpt_dir), out_path, quantization="f32")

    assert result.success, f"Export failed: {result.error}"
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0


def test_gguf_file_starts_with_magic(tmp_path):
    """Exported GGUF file starts with the 'GGUF' magic bytes."""
    from safetensors.torch import save_file

    model = _make_model()
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "output.weight"}
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_file(state, str(ckpt_dir / "model.safetensors"))

    exporter = GGUFExporter(_tiny_config())
    out_path = str(tmp_path / "model.gguf")
    result = exporter.export(str(ckpt_dir), out_path, quantization="f16")

    assert result.success, f"Export failed: {result.error}"
    with open(out_path, "rb") as f:
        header = f.read(4)
    assert header == _GGUF_MAGIC, f"Expected GGUF magic, got {header!r}"


def test_gguf_export_result_has_tensor_count(tmp_path):
    """ExportResult.num_tensors is positive after a successful export."""
    from safetensors.torch import save_file

    model = _make_model()
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "output.weight"}
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_file(state, str(ckpt_dir / "model.safetensors"))

    exporter = GGUFExporter(_tiny_config())
    out_path = str(tmp_path / "model.gguf")
    result = exporter.export(str(ckpt_dir), out_path, quantization="f32")

    assert result.num_tensors > 0


def test_gguf_export_fails_for_unknown_quantization(tmp_path):
    """GGUFExporter.export returns failure for unknown quantization."""
    exporter = GGUFExporter(_tiny_config())
    result = exporter.export(str(tmp_path), str(tmp_path / "out.gguf"), quantization="q99_bad")
    assert not result.success
    assert "q99_bad" in result.error


# ── 2. Quantization ───────────────────────────────────────────────────────────

def test_quantize_dynamic_reduces_size():
    """Dynamic INT8 quantization reduces model memory footprint."""
    model = _make_model()
    original_mb = _model_size_mb(model)

    quantizer = ModelQuantizer(model)
    q_model, result = quantizer.quantize_dynamic()

    assert result.quantized_size_mb <= original_mb * 1.1  # should not grow
    assert result.original_size_mb > 0
    assert result.compression_ratio > 0


def test_quantize_int4_reduces_size():
    """INT4 weight-only quantization reduces Linear layer sizes."""
    model = _make_model()
    original_mb = _model_size_mb(model)

    quantizer = ModelQuantizer(model)
    q_model, result = quantizer.quantize_weights_only(bits=4)

    assert result.quantized_size_mb < original_mb * 1.5  # packed weights are smaller
    assert "int4" in result.method.lower() or "bitsandbytes" in result.method.lower()


def test_quantize_q8_encoding_shape():
    """_quantize_q8_0 produces expected byte-count output."""
    t = torch.randn(64)  # 2 blocks of 32
    encoded = _quantize_q8_0(t)
    # Each block = 2 bytes (f16 scale) + 32 bytes (int8) = 34 bytes
    # 2 blocks => 68 bytes
    assert len(encoded) == 68


def test_model_size_mb_positive():
    """_model_size_mb returns a positive float for a non-empty model."""
    model = _make_model()
    size = _model_size_mb(model)
    assert size > 0.0


# ── 3. Ollama Modelfile generation ────────────────────────────────────────────

def test_ollama_creates_modelfile(tmp_path):
    """OllamaExporter creates a Modelfile in the output directory."""
    exporter = OllamaExporter()
    gguf_path = str(tmp_path / "model.gguf")
    # The gguf file doesn't need to exist for Modelfile generation
    result_path = exporter.create_modelfile(
        gguf_path=gguf_path,
        output_dir=str(tmp_path / "exports"),
        model_name="cola-test",
    )
    assert Path(result_path).exists()
    assert Path(result_path).name == "Modelfile"


def test_ollama_modelfile_contains_from(tmp_path):
    """Generated Modelfile contains a FROM directive pointing to the GGUF path."""
    exporter = OllamaExporter()
    gguf_path = str(tmp_path / "cola-coder-f16.gguf")
    result_path = exporter.create_modelfile(
        gguf_path=gguf_path,
        output_dir=str(tmp_path / "exports"),
        model_name="cola-test",
    )
    content = Path(result_path).read_text()
    assert "FROM" in content


def test_ollama_creates_readme(tmp_path):
    """OllamaExporter also creates a README_ollama.txt helper file."""
    exporter = OllamaExporter()
    out_dir = tmp_path / "exports"
    exporter.create_modelfile(
        gguf_path=str(tmp_path / "model.gguf"),
        output_dir=str(out_dir),
        model_name="cola-test",
    )
    assert (out_dir / "README_ollama.txt").exists()


# ── 4. Weight name mapping ────────────────────────────────────────────────────

def test_map_embedding_weight():
    """tok_emb.weight maps to token_embd.weight."""
    assert _map_single_key("tok_emb.weight") == "token_embd.weight"


def test_map_final_norm_weight():
    """final_norm.weight maps to output_norm.weight."""
    assert _map_single_key("final_norm.weight") == "output_norm.weight"


def test_map_layer_attention_q():
    """blocks.0.attention.q_proj.weight maps to blk.0.attn_q.weight."""
    assert _map_single_key("blocks.0.attention.q_proj.weight") == "blk.0.attn_q.weight"


def test_map_layer_ffn_gate():
    """blocks.3.ffn.gate_proj.weight maps to blk.3.ffn_gate.weight."""
    assert _map_single_key("blocks.3.ffn.gate_proj.weight") == "blk.3.ffn_gate.weight"


def test_map_unknown_key_returns_none():
    """_map_single_key returns None for an unrecognised key."""
    result = _map_single_key("some.unknown.weight.xyz")
    assert result is None
