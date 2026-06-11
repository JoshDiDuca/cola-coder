"""TOOL-005: GGUF export must embed the tokenizer vocabulary.

Before this, the GGUF writers emitted tokenizer.ggml.model="llama" with bos/eos
ids but NO token list or merges — a GGUF with no vocabulary cannot be loaded by
llama.cpp at all, so every exported model was unusable. cola-coder uses
byte-level BPE, which maps to llama.cpp's "gpt2" tokenizer (not "llama").

These tests cover the pure vocab builder, the new GGUF array encoder, and an
end-to-end builtin export whose KV section is parsed back and verified.
"""

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from cola_coder.model.config import ModelConfig
from cola_coder.export.gguf_export import (
    GGUFExporter,
    build_gguf_vocab,
    _encode_kv,
    _GGUF_TYPE_ARRAY,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_INT32,
    _GGUF_MAGIC,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _train_tokenizer(tmp_path: Path) -> tuple[str, dict]:
    from cola_coder.tokenizer import train_tokenizer as tt

    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 200),
        vocab_size=320,
        output_path=out,
    )
    return out, json.loads(Path(out).read_text(encoding="utf-8"))


def _fake_state_dict(cfg: ModelConfig) -> dict:
    state = {"tok_emb.weight": torch.randn(cfg.vocab_size, cfg.dim)}
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    for i in range(cfg.n_layers):
        state[f"blocks.{i}.attn_norm.weight"] = torch.ones(cfg.dim)
        state[f"blocks.{i}.ffn_norm.weight"] = torch.ones(cfg.dim)
        state[f"blocks.{i}.attention.q_proj.weight"] = torch.randn(cfg.dim, cfg.dim)
        state[f"blocks.{i}.attention.k_proj.weight"] = torch.randn(kv_dim, cfg.dim)
        state[f"blocks.{i}.attention.v_proj.weight"] = torch.randn(kv_dim, cfg.dim)
        state[f"blocks.{i}.attention.out_proj.weight"] = torch.randn(cfg.dim, cfg.dim)
        state[f"blocks.{i}.ffn.gate_proj.weight"] = torch.randn(cfg.ffn_hidden_dim, cfg.dim)
        state[f"blocks.{i}.ffn.up_proj.weight"] = torch.randn(cfg.ffn_hidden_dim, cfg.dim)
        state[f"blocks.{i}.ffn.down_proj.weight"] = torch.randn(cfg.dim, cfg.ffn_hidden_dim)
    state["final_norm.weight"] = torch.ones(cfg.dim)
    return state


# --------------------------------------------------------------------------
# Minimal GGUF KV parser (enough to verify the tokenizer metadata)
# --------------------------------------------------------------------------

def _read_str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + n].decode("utf-8")
    return s, off + n


def _read_value(buf, off, vtype):
    if vtype == _GGUF_TYPE_STRING:
        return _read_str(buf, off)
    if vtype == 4:  # uint32
        (v,) = struct.unpack_from("<I", buf, off)
        return v, off + 4
    if vtype == 5:  # int32
        (v,) = struct.unpack_from("<i", buf, off)
        return v, off + 4
    if vtype == 6:  # float32
        (v,) = struct.unpack_from("<f", buf, off)
        return v, off + 4
    if vtype == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<I", buf, off)
        off += 4
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        items = []
        for _ in range(count):
            val, off = _read_value(buf, off, elem_type)
            items.append(val)
        return items, off
    raise AssertionError(f"parser doesn't handle vtype {vtype}")


def _parse_gguf_kv(path: Path) -> dict:
    buf = path.read_bytes()
    assert buf[:4] == _GGUF_MAGIC
    off = 4
    (_version,) = struct.unpack_from("<I", buf, off)
    off += 4
    (_n_tensors,) = struct.unpack_from("<Q", buf, off)
    off += 8
    (n_kv,) = struct.unpack_from("<Q", buf, off)
    off += 8
    kv = {}
    for _ in range(n_kv):
        key, off = _read_str(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        value, off = _read_value(buf, off, vtype)
        kv[key] = value
    return kv


# --------------------------------------------------------------------------
# 1. build_gguf_vocab
# --------------------------------------------------------------------------

class TestBuildVocab:
    def test_gpt2_model_and_token_count(self, tmp_path):
        _, tj = _train_tokenizer(tmp_path)
        vocab = build_gguf_vocab(tj)
        assert vocab["model"] == "gpt2"
        assert len(vocab["tokens"]) == len(tj["model"]["vocab"])
        assert len(vocab["token_types"]) == len(vocab["tokens"])
        assert vocab["merges"]  # byte-level BPE has merges

    def test_special_tokens_typed_control_and_ids(self, tmp_path):
        _, tj = _train_tokenizer(tmp_path)
        vocab = build_gguf_vocab(tj)
        v = tj["model"]["vocab"]
        # <|bos|>/<|eos|> are special → CONTROL (3); ids resolved correctly.
        assert vocab["bos_id"] == v["<|bos|>"]
        assert vocab["eos_id"] == v["<|eos|>"]
        assert vocab["token_types"][v["<|bos|>"]] == 3
        # A plain content token stays NORMAL (1).
        normal = next(tok for tok, i in v.items() if not tok.startswith("<|"))
        assert vocab["token_types"][v[normal]] == 1

    def test_no_vocab_returns_none(self):
        assert build_gguf_vocab({"model": {}}) is None
        assert build_gguf_vocab({}) is None


# --------------------------------------------------------------------------
# 2. _encode_kv array round-trip
# --------------------------------------------------------------------------

class TestArrayEncoding:
    def test_string_array_roundtrip(self):
        raw = _encode_kv("k", _GGUF_TYPE_ARRAY, (_GGUF_TYPE_STRING, ["a", "bb", "ccc"]))
        # key + vtype already consumed by parser; re-parse from after key+vtype.
        key, off = _read_str(raw, 0)
        (vtype,) = struct.unpack_from("<I", raw, off)
        off += 4
        assert key == "k" and vtype == _GGUF_TYPE_ARRAY
        items, _ = _read_value(raw, off, vtype)
        assert items == ["a", "bb", "ccc"]

    def test_int32_array_roundtrip(self):
        raw = _encode_kv("t", _GGUF_TYPE_ARRAY, (_GGUF_TYPE_INT32, [1, 3, 1]))
        _, off = _read_str(raw, 0)
        off += 4
        items, _ = _read_value(raw, off, _GGUF_TYPE_ARRAY)
        assert items == [1, 3, 1]


# --------------------------------------------------------------------------
# 3. End-to-end export embeds + parses back the vocab
# --------------------------------------------------------------------------

class TestExportEmbedsVocab:
    def _setup(self, tmp_path):
        tok_path, tj = _train_tokenizer(tmp_path)
        vocab_size = len(tj["model"]["vocab"])
        cfg = ModelConfig(vocab_size=vocab_size, dim=64, n_layers=2,
                          n_heads=4, n_kv_heads=2, max_seq_len=64)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        save_file(_fake_state_dict(cfg), str(ckpt / "model.safetensors"))
        # Record the tokenizer the way SFT/reasoning checkpoints do.
        (ckpt / "metadata.json").write_text(
            json.dumps({"tokenizer_path": tok_path}), encoding="utf-8"
        )
        return cfg, ckpt, vocab_size

    def test_gguf_contains_full_vocab(self, tmp_path):
        cfg, ckpt, vocab_size = self._setup(tmp_path)
        out = tmp_path / "model.gguf"
        result = GGUFExporter(cfg).export(str(ckpt), str(out), quantization="f16")
        assert result.success
        kv = _parse_gguf_kv(out)
        assert kv["tokenizer.ggml.model"] == "gpt2"
        assert len(kv["tokenizer.ggml.tokens"]) == vocab_size
        assert len(kv["tokenizer.ggml.token_type"]) == vocab_size
        assert len(kv["tokenizer.ggml.merges"]) > 0
        assert "tokenizer.ggml.bos_token_id" in kv

    def test_missing_tokenizer_still_succeeds_without_vocab(self, tmp_path):
        # No metadata.json / tokenizer → export must still succeed (warns), and
        # must NOT claim a token list it doesn't have.
        cfg, ckpt, _ = self._setup(tmp_path)
        (ckpt / "metadata.json").write_text(
            json.dumps({"tokenizer_path": str(tmp_path / "nope.json")}), encoding="utf-8"
        )
        out = tmp_path / "novocab.gguf"
        result = GGUFExporter(cfg).export(
            str(ckpt), str(out), quantization="f16",
            tokenizer_path=str(tmp_path / "nope.json"),
        )
        assert result.success
        kv = _parse_gguf_kv(out)
        assert "tokenizer.ggml.tokens" not in kv

    def test_vocab_mismatch_skips_embedding(self, tmp_path):
        # Model embedding rows != tokenizer vocab → must NOT embed a wrong vocab.
        tok_path, tj = _train_tokenizer(tmp_path)
        cfg = ModelConfig(vocab_size=len(tj["model"]["vocab"]) + 50, dim=64,
                          n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
        ckpt = tmp_path / "ckpt_mm"
        ckpt.mkdir()
        save_file(_fake_state_dict(cfg), str(ckpt / "model.safetensors"))
        out = tmp_path / "mm.gguf"
        result = GGUFExporter(cfg).export(
            str(ckpt), str(out), quantization="f16", tokenizer_path=tok_path
        )
        assert result.success
        kv = _parse_gguf_kv(out)
        assert "tokenizer.ggml.tokens" not in kv  # mismatch → not embedded
