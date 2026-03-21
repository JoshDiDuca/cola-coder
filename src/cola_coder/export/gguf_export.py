"""GGUF export for cola-coder models.

GGUF is the standard format used by llama.cpp and Ollama. Since cola-coder
uses the LLaMA 3 architecture (RoPE, GQA, SwiGLU, RMSNorm), its weights map
directly to llama.cpp's expected tensor names.

Weight name mapping (cola-coder → GGUF / llama.cpp):

  tok_emb.weight                        → token_embd.weight
  blocks.{i}.attn_norm.weight           → blk.{i}.attn_norm.weight
  blocks.{i}.ffn_norm.weight            → blk.{i}.ffn_norm.weight
  blocks.{i}.attention.q_proj.weight    → blk.{i}.attn_q.weight
  blocks.{i}.attention.k_proj.weight    → blk.{i}.attn_k.weight
  blocks.{i}.attention.v_proj.weight    → blk.{i}.attn_v.weight
  blocks.{i}.attention.out_proj.weight  → blk.{i}.attn_output.weight
  blocks.{i}.ffn.gate_proj.weight       → blk.{i}.ffn_gate.weight
  blocks.{i}.ffn.up_proj.weight         → blk.{i}.ffn_up.weight
  blocks.{i}.ffn.down_proj.weight       → blk.{i}.ffn_down.weight
  final_norm.weight                     → output_norm.weight
  [tok_emb.weight is also the output head, weight-tied — exported as output.weight]

The `gguf` package (pip install gguf) is optional. If unavailable, a lightweight
manual writer is used that produces a valid GGUF v3 file.
"""

from __future__ import annotations

import re
import struct
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from safetensors.torch import load_file

from ..model.config import ModelConfig

logger = logging.getLogger(__name__)

# Try to import the gguf package (pip install gguf)
try:
    import gguf  # type: ignore  # noqa: F401
    from gguf import GGUFWriter, GGMLQuantizationType  # type: ignore
    GGUF_PACKAGE_AVAILABLE = True
except ImportError:
    GGUF_PACKAGE_AVAILABLE = False
    logger.info("gguf package not found — using built-in GGUF writer")


# ──────────────────────────────────────────────────────────────────────────────
# Weight name mapping
# ──────────────────────────────────────────────────────────────────────────────

# Static renames (non-indexed weights)
_STATIC_MAP = {
    "tok_emb.weight": "token_embd.weight",
    "final_norm.weight": "output_norm.weight",
    # output.weight is tied to tok_emb.weight and not stored in checkpoints.
    # We re-add it during export pointing at the same tensor.
}

# Regex for per-layer weights: blocks.{i}.<suffix>
_LAYER_PATTERN = re.compile(r"^blocks\.(\d+)\.(.+)$")

# Per-layer suffix → GGUF suffix
_LAYER_SUFFIX_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attention.q_proj.weight": "attn_q.weight",
    "attention.k_proj.weight": "attn_k.weight",
    "attention.v_proj.weight": "attn_v.weight",
    "attention.out_proj.weight": "attn_output.weight",
    "ffn.gate_proj.weight": "ffn_gate.weight",
    "ffn.up_proj.weight": "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
}


def _map_single_key(key: str) -> Optional[str]:
    """Map one cola-coder weight name to its GGUF equivalent.

    Returns None if the key is unknown (will be skipped with a warning).
    """
    # Static renames
    if key in _STATIC_MAP:
        return _STATIC_MAP[key]

    # Per-layer renames
    m = _LAYER_PATTERN.match(key)
    if m:
        layer_idx = m.group(1)
        suffix = m.group(2)
        gguf_suffix = _LAYER_SUFFIX_MAP.get(suffix)
        if gguf_suffix is not None:
            return f"blk.{layer_idx}.{gguf_suffix}"
        logger.warning("Unknown layer suffix '%s' — skipping weight '%s'", suffix, key)
        return None

    logger.warning("Unknown weight key '%s' — skipping", key)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExportResult:
    """Summary of a completed GGUF export."""
    output_path: str
    file_size_mb: float
    quantization: str
    num_tensors: int
    success: bool
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Quantization helpers (pure NumPy/torch — no external deps)
# ──────────────────────────────────────────────────────────────────────────────

def _to_f32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.float().cpu().numpy()


def _to_f16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.half().cpu().numpy()


def _quantize_q8_0(tensor: torch.Tensor) -> np.ndarray:
    """Block-wise Q8_0 quantization (compatible with llama.cpp Q8_0).

    Each block of 32 float32 values is scaled to int8 using the block's max
    absolute value.  The layout stored in GGUF is float16 scale + 32 × int8
    per block (34 bytes / block).
    """
    BLOCK_SIZE = 32
    data = tensor.float().cpu()
    flat = data.reshape(-1)
    n = flat.shape[0]

    # Pad to multiple of BLOCK_SIZE
    pad = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])

    n_blocks = flat.shape[0] // BLOCK_SIZE
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)

    scales = blocks.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-8)
    quant = (blocks / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    # Pack: (float16 scale, 32 × int8) per block
    scales_f16 = scales.half().numpy()  # shape (n_blocks,)
    quant_np = quant.numpy()             # shape (n_blocks, 32)

    # Build byte array matching llama.cpp q8_0 block layout
    out = bytearray()
    for i in range(n_blocks):
        out += scales_f16[i].tobytes()          # 2 bytes: f16 scale
        out += quant_np[i].tobytes()            # 32 bytes: int8 × 32
    return np.frombuffer(bytes(out), dtype=np.uint8)


def _quantize_q4_k_m(tensor: torch.Tensor) -> np.ndarray:
    """Approximate Q4_K_M via Q8_0 super-blocks (simplified).

    True Q4_K_M requires the complex K-quant format defined in llama.cpp.
    This implementation produces a Q8_0-compatible encoding rather than
    trying to replicate the exact K-quant binary layout without the gguf
    package.  When the `gguf` package IS available we delegate to its writer
    which handles this correctly.

    For offline use without the gguf package this gives a working (slightly
    larger) file that llama.cpp can load as Q8_0.
    """
    return _quantize_q8_0(tensor)


def _quantize_q5_k_m(tensor: torch.Tensor) -> np.ndarray:
    """Approximate Q5_K_M — same note as Q4_K_M above."""
    return _quantize_q8_0(tensor)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in minimal GGUF v3 writer (used when gguf package is absent)
# ──────────────────────────────────────────────────────────────────────────────

# GGUF magic and version constants
_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3

# Value type codes from the GGUF spec
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# Tensor type codes
_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_Q8_0 = 8


def _encode_string(s: str) -> bytes:
    """Encode a UTF-8 string as GGUF: uint64 length + bytes."""
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _encode_kv(key: str, value_type: int, value: object) -> bytes:
    """Encode a GGUF key-value metadata entry."""
    out = _encode_string(key)
    out += struct.pack("<I", value_type)
    if value_type == _GGUF_TYPE_UINT32:
        out += struct.pack("<I", int(value))
    elif value_type == _GGUF_TYPE_INT32:
        out += struct.pack("<i", int(value))
    elif value_type == _GGUF_TYPE_FLOAT32:
        out += struct.pack("<f", float(value))
    elif value_type == _GGUF_TYPE_BOOL:
        out += struct.pack("<?", bool(value))
    elif value_type == _GGUF_TYPE_STRING:
        out += _encode_string(str(value))
    elif value_type == _GGUF_TYPE_UINT64:
        out += struct.pack("<Q", int(value))
    elif value_type == _GGUF_TYPE_FLOAT64:
        out += struct.pack("<d", float(value))
    else:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")
    return out


def _write_gguf_manual(
    output_path: str,
    tensors: dict[str, np.ndarray],
    tensor_types: dict[str, int],
    metadata: dict,
    quant_str: str,
) -> None:
    """Write a valid GGUF v3 file without the gguf package.

    Format (GGUF v3):
        magic (4B) | version (4B) | n_tensors (8B) | n_kv (8B)
        [kv pairs...]
        [tensor info: name, n_dims, dims, type, offset]
        [tensor data (32-byte aligned)]
    """
    # ── Build KV metadata ────────────────────────────────────────────────────
    kv_bytes = bytearray()
    kv_count = 0
    for k, (vtype, v) in metadata.items():
        kv_bytes += _encode_kv(k, vtype, v)
        kv_count += 1

    # ── Build tensor info ────────────────────────────────────────────────────
    # First pass: compute data offsets
    tensor_info_bytes = bytearray()
    tensor_data_parts: list[bytes] = []
    data_offset = 0

    # Sort tensors for deterministic output
    tensor_names = sorted(tensors.keys())
    for name in tensor_names:
        arr = tensors[name]
        ttype = tensor_types.get(name, _GGML_TYPE_F32)
        shape = arr.shape

        info = bytearray()
        info += _encode_string(name)
        info += struct.pack("<I", len(shape))
        for dim in shape:
            info += struct.pack("<Q", int(dim))
        info += struct.pack("<I", ttype)
        info += struct.pack("<Q", data_offset)
        tensor_info_bytes += info

        raw = arr.tobytes()
        tensor_data_parts.append(raw)

        # Advance offset with 32-byte alignment padding
        aligned = (len(raw) + 31) & ~31
        data_offset += aligned

    # ── Calculate where tensor data starts (for alignment) ───────────────────
    # Header: magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24 bytes
    header_size = 24 + len(kv_bytes) + len(tensor_info_bytes)
    # Data section must be 32-byte aligned
    data_section_start = (header_size + 31) & ~31
    data_section_padding = data_section_start - header_size

    # ── Re-build tensor info with corrected absolute offsets ─────────────────
    # (The offsets above are relative; we need absolute byte positions)
    tensor_info_bytes = bytearray()
    abs_offset = 0
    for name in tensor_names:
        arr = tensors[name]
        ttype = tensor_types.get(name, _GGML_TYPE_F32)
        shape = arr.shape

        info = bytearray()
        info += _encode_string(name)
        info += struct.pack("<I", len(shape))
        for dim in shape:
            info += struct.pack("<Q", int(dim))
        info += struct.pack("<I", ttype)
        info += struct.pack("<Q", abs_offset)
        tensor_info_bytes += info

        raw = tensor_data_parts[tensor_names.index(name)]
        aligned = (len(raw) + 31) & ~31
        abs_offset += aligned

    # ── Write file ───────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        # Header
        f.write(_GGUF_MAGIC)
        f.write(struct.pack("<I", _GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_names)))
        f.write(struct.pack("<Q", kv_count))

        # KV metadata
        f.write(bytes(kv_bytes))

        # Tensor info
        f.write(bytes(tensor_info_bytes))

        # Padding to align data section
        if data_section_padding > 0:
            f.write(b"\x00" * data_section_padding)

        # Tensor data
        for name in tensor_names:
            raw = tensor_data_parts[tensor_names.index(name)]
            f.write(raw)
            # Align to 32 bytes
            pad = (32 - len(raw) % 32) % 32
            if pad:
                f.write(b"\x00" * pad)


# ──────────────────────────────────────────────────────────────────────────────
# Main exporter
# ──────────────────────────────────────────────────────────────────────────────

class GGUFExporter:
    """Export cola-coder models to GGUF format for llama.cpp / Ollama.

    Uses the `gguf` Python package when available (pip install gguf), otherwise
    falls back to a built-in minimal GGUF v3 writer.
    """

    SUPPORTED_QUANTIZATIONS = ("f32", "f16", "q8_0", "q4_k_m", "q5_k_m")

    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    # ── Public API ────────────────────────────────────────────────────────────

    def export(
        self,
        checkpoint_path: str,
        output_path: str,
        quantization: str = "f16",
    ) -> ExportResult:
        """Export a cola-coder safetensors checkpoint to GGUF.

        Args:
            checkpoint_path: Path to a directory containing model.safetensors,
                             or directly to a model.safetensors file, or to a
                             "latest" pointer file.
            output_path: Destination .gguf file path.
            quantization: One of "f32", "f16", "q8_0", "q4_k_m", "q5_k_m".

        Returns:
            ExportResult with success flag, file size, and tensor count.
        """
        if quantization not in self.SUPPORTED_QUANTIZATIONS:
            return ExportResult(
                output_path=output_path,
                file_size_mb=0.0,
                quantization=quantization,
                num_tensors=0,
                success=False,
                error=(
                    f"Unknown quantization '{quantization}'. "
                    f"Supported: {self.SUPPORTED_QUANTIZATIONS}"
                ),
            )

        try:
            safetensors_path = self._resolve_checkpoint_path(checkpoint_path)
            logger.info("Loading weights from %s", safetensors_path)
            state_dict = load_file(str(safetensors_path), device="cpu")

            mapped = self._map_weight_names(state_dict)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if GGUF_PACKAGE_AVAILABLE:
                num_tensors = self._write_gguf_with_package(
                    mapped, output_path, quantization
                )
            else:
                num_tensors = self._write_gguf_builtin(
                    mapped, output_path, quantization
                )

            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(
                "Exported %d tensors to %s (%.1f MB)", num_tensors, output_path, size_mb
            )
            return ExportResult(
                output_path=output_path,
                file_size_mb=round(size_mb, 2),
                quantization=quantization,
                num_tensors=num_tensors,
                success=True,
            )

        except Exception as exc:
            logger.exception("GGUF export failed")
            return ExportResult(
                output_path=output_path,
                file_size_mb=0.0,
                quantization=quantization,
                num_tensors=0,
                success=False,
                error=str(exc),
            )

    def _map_weight_names(self, state_dict: dict) -> dict:
        """Map cola-coder tensor names to GGUF / llama.cpp names.

        Also injects the output.weight tensor (tied to token_embd.weight since
        output.weight is not stored in checkpoints due to weight tying).
        """
        mapped: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            gguf_key = _map_single_key(key)
            if gguf_key is not None:
                mapped[gguf_key] = tensor

        # Re-add output.weight — it's tied to tok_emb.weight and excluded from
        # the checkpoint.  llama.cpp needs it as a separate (or shared) entry.
        if "token_embd.weight" in mapped and "output.weight" not in mapped:
            mapped["output.weight"] = mapped["token_embd.weight"]

        return mapped

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Path:
        """Resolve checkpoint_path to the actual model.safetensors file."""
        p = Path(checkpoint_path)
        # Direct .safetensors file
        if p.suffix == ".safetensors" and p.exists():
            return p
        # "latest" pointer file (text file containing actual checkpoint dir)
        if p.name == "latest" and p.is_file():
            p = Path(p.read_text().strip())
        # Directory containing model.safetensors
        if p.is_dir():
            candidate = p / "model.safetensors"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Cannot find model.safetensors from checkpoint path: {checkpoint_path}"
        )

    def _build_gguf_metadata(self) -> dict:
        """Build GGUF metadata dict for the model config.

        Returns a dict: key → (gguf_value_type, value)
        """
        cfg = self.config
        return {
            "general.architecture": (_GGUF_TYPE_STRING, "llama"),
            "general.name": (_GGUF_TYPE_STRING, "cola-coder"),
            "llama.context_length": (_GGUF_TYPE_UINT32, cfg.max_seq_len),
            "llama.embedding_length": (_GGUF_TYPE_UINT32, cfg.dim),
            "llama.block_count": (_GGUF_TYPE_UINT32, cfg.n_layers),
            "llama.feed_forward_length": (_GGUF_TYPE_UINT32, cfg.ffn_hidden_dim),
            "llama.rope.dimension_count": (_GGUF_TYPE_UINT32, cfg.head_dim),
            "llama.rope.freq_base": (_GGUF_TYPE_FLOAT32, cfg.rope_theta),
            "llama.attention.head_count": (_GGUF_TYPE_UINT32, cfg.n_heads),
            "llama.attention.head_count_kv": (_GGUF_TYPE_UINT32, cfg.n_kv_heads),
            "llama.attention.layer_norm_rms_epsilon": (_GGUF_TYPE_FLOAT32, 1e-5),
            "tokenizer.ggml.model": (_GGUF_TYPE_STRING, "llama"),
            "tokenizer.ggml.bos_token_id": (_GGUF_TYPE_UINT32, 1),
            "tokenizer.ggml.eos_token_id": (_GGUF_TYPE_UINT32, 2),
        }

    def _quantize_tensor(
        self, tensor: torch.Tensor, method: str
    ) -> tuple[np.ndarray, int]:
        """Convert a tensor to the target quantization format.

        Returns (numpy_array, ggml_type_code).
        """
        if method == "f32":
            return _to_f32(tensor), _GGML_TYPE_F32
        elif method == "f16":
            return _to_f16(tensor), _GGML_TYPE_F16
        elif method == "q8_0":
            return _quantize_q8_0(tensor), _GGML_TYPE_Q8_0
        elif method in ("q4_k_m", "q5_k_m"):
            # Without the gguf package we fall back to q8_0 for these
            return _quantize_q8_0(tensor), _GGML_TYPE_Q8_0
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def _write_gguf_builtin(
        self,
        mapped: dict[str, torch.Tensor],
        output_path: str,
        quantization: str,
    ) -> int:
        """Write GGUF using the built-in writer (no external deps)."""
        metadata = self._build_gguf_metadata()
        tensors_np: dict[str, np.ndarray] = {}
        tensor_types: dict[str, int] = {}

        for name, tensor in mapped.items():
            # Norm/embedding tensors always stay in f32 for accuracy
            if self._is_norm_or_embed(name):
                arr, ttype = _to_f32(tensor), _GGML_TYPE_F32
            else:
                arr, ttype = self._quantize_tensor(tensor, quantization)
            tensors_np[name] = arr
            tensor_types[name] = ttype

        _write_gguf_manual(output_path, tensors_np, tensor_types, metadata, quantization)
        return len(tensors_np)

    def _write_gguf_with_package(
        self,
        mapped: dict[str, torch.Tensor],
        output_path: str,
        quantization: str,
    ) -> int:
        """Write GGUF using the official gguf Python package."""
        writer = GGUFWriter(output_path, arch="llama")

        # Add metadata
        cfg = self.config
        writer.add_name("cola-coder")
        writer.add_context_length(cfg.max_seq_len)
        writer.add_embedding_length(cfg.dim)
        writer.add_block_count(cfg.n_layers)
        writer.add_feed_forward_length(cfg.ffn_hidden_dim)
        writer.add_rope_dimension_count(cfg.head_dim)
        writer.add_rope_freq_base(cfg.rope_theta)
        writer.add_head_count(cfg.n_heads)
        writer.add_head_count_kv(cfg.n_kv_heads)
        writer.add_layer_norm_rms_eps(1e-5)

        # Map quantization string to GGMLQuantizationType
        quant_map = {
            "f32": GGMLQuantizationType.F32,
            "f16": GGMLQuantizationType.F16,
            "q8_0": GGMLQuantizationType.Q8_0,
            "q4_k_m": GGMLQuantizationType.Q4_K,
            "q5_k_m": GGMLQuantizationType.Q5_K,
        }
        quant_type = quant_map.get(quantization, GGMLQuantizationType.F16)

        for name, tensor in mapped.items():
            data = tensor.float().cpu().numpy()
            if self._is_norm_or_embed(name):
                writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.F32)
            else:
                writer.add_tensor(name, data, raw_dtype=quant_type)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        return len(mapped)

    @staticmethod
    def _is_norm_or_embed(name: str) -> bool:
        """Return True for tensors that should not be quantized (norms, embeddings)."""
        return (
            name.endswith("_norm.weight")
            or name == "token_embd.weight"
        )
