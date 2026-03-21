"""Checkpoint Converter: convert between safetensors, PyTorch, and HuggingFace formats.

Supported conversions:
    safetensors → pytorch (.pt)   — extract model weights as a raw state dict
    pytorch     → safetensors     — save a .pt state dict as safetensors
    cola-coder  → huggingface     — emit a HuggingFace-compatible directory
                                    (config.json + model.safetensors)

Weight-tying note: cola-coder ties ``tok_emb.weight`` and ``output.weight``.
``output.weight`` is *excluded* from the saved file; this converter handles that
invariant correctly in both directions.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if checkpoint conversion is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {
    "safetensors": "HuggingFace safetensors (.safetensors)",
    "pytorch": "PyTorch state dict (.pt)",
    "huggingface": "HuggingFace model directory (config.json + model.safetensors)",
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ConversionResult:
    """Outcome of a conversion operation."""

    source_path: str
    target_path: str
    source_format: str
    target_format: str
    tensor_count: int
    notes: list[str]

    @property
    def success(self) -> bool:
        return bool(self.target_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_safetensors(path: Path) -> dict[str, Any]:
    """Load a safetensors file into a {name: tensor} dict."""
    try:
        from safetensors.torch import load_file  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("safetensors package is required: pip install safetensors") from e
    return dict(load_file(str(path)))


def _save_safetensors(tensors: dict[str, Any], path: Path) -> None:
    """Save a {name: tensor} dict as a safetensors file."""
    try:
        from safetensors.torch import save_file  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("safetensors package is required: pip install safetensors") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def _load_pytorch(path: Path) -> dict[str, Any]:
    """Load a .pt state dict."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _save_pytorch(tensors: dict[str, Any], path: Path) -> None:
    """Save a state dict as a .pt file."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, str(path))


def _flat_config(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict with dotted keys. E.g. {"a": {"b": 1}} -> {"a.b": 1}."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = k if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            result.update(_flat_config(v, key))
        else:
            result[key] = v
    return result


def _strip_compile_prefix(state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Strip ``_orig_mod.`` prefix added by torch.compile."""
    prefix = "_orig_mod."
    stripped: dict[str, Any] = {}
    found = False
    for k, v in state.items():
        if k.startswith(prefix):
            stripped[k[len(prefix):]] = v
            found = True
        else:
            stripped[k] = v
    return stripped, found


def _detect_format(path: Path) -> str:
    """Detect checkpoint format from file extension / directory contents."""
    if path.is_dir():
        if (path / "config.json").exists() and (path / "model.safetensors").exists():
            return "huggingface"
        # cola-coder checkpoint directory
        st_files = list(path.glob("*.safetensors"))
        if st_files:
            return "safetensors"
        pt_files = list(path.glob("*.pt"))
        if pt_files:
            return "pytorch"
        return "unknown"
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".pt", ".pth", ".bin"}:
        return "pytorch"
    return "unknown"


def _build_hf_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert cola-coder metadata.json to a HuggingFace config.json."""
    raw = metadata.get("config", metadata)
    cfg = raw.get("model", raw)

    d_model = cfg.get("d_model", 768)
    n_heads = cfg.get("n_heads", 12)
    n_kv_heads = cfg.get("n_kv_heads", n_heads)
    n_layers = cfg.get("n_layers", 12)
    d_ffn = cfg.get("d_ffn", d_model * 4)
    vocab_size = cfg.get("vocab_size", 32_000)
    max_seq_len = cfg.get("max_seq_len", 2048)

    return {
        "model_type": "cola_coder",
        "architectures": ["ColaCoderForCausalLM"],
        "hidden_size": d_model,
        "intermediate_size": d_ffn,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "num_hidden_layers": n_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_seq_len,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }


# ---------------------------------------------------------------------------
# CheckpointConverter
# ---------------------------------------------------------------------------


class CheckpointConverter:
    """Convert cola-coder checkpoints between formats.

    Usage::

        from cola_coder.features.checkpoint_converter import CheckpointConverter

        converter = CheckpointConverter()

        # safetensors → pytorch
        result = converter.convert(
            source="checkpoints/tiny/step_00020000",
            target_format="pytorch",
            output="checkpoints/tiny/step_00020000/model.pt",
        )

        # cola-coder → huggingface
        result = converter.convert(
            source="checkpoints/tiny/step_00020000",
            target_format="huggingface",
            output="exports/tiny-hf",
        )
    """

    def convert(
        self,
        source: str | Path,
        target_format: str,
        output: str | Path | None = None,
    ) -> ConversionResult:
        """Convert *source* checkpoint to *target_format*.

        Parameters
        ----------
        source:
            Path to source checkpoint (file or directory).
        target_format:
            One of ``"safetensors"``, ``"pytorch"``, ``"huggingface"``.
        output:
            Output path.  If omitted, a sensible default is chosen next to
            the source.

        Returns
        -------
        ConversionResult
        """
        fmt = target_format.lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported target format: {target_format!r}. "
                f"Choose from: {list(SUPPORTED_FORMATS.keys())}"
            )

        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source checkpoint not found: {src}")

        src_fmt = _detect_format(src)
        notes: list[str] = []

        if fmt == "pytorch":
            return self._to_pytorch(src, src_fmt, output, notes)
        elif fmt == "safetensors":
            return self._to_safetensors(src, src_fmt, output, notes)
        else:
            return self._to_huggingface(src, src_fmt, output, notes)

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def _to_pytorch(
        self,
        src: Path,
        src_fmt: str,
        output: str | Path | None,
        notes: list[str],
    ) -> ConversionResult:
        tensors = self._load_tensors(src, src_fmt, notes)
        dest = Path(output) if output else src.parent / (src.stem + "_weights.pt")
        _save_pytorch(tensors, dest)
        return ConversionResult(
            source_path=str(src),
            target_path=str(dest),
            source_format=src_fmt,
            target_format="pytorch",
            tensor_count=len(tensors),
            notes=notes,
        )

    def _to_safetensors(
        self,
        src: Path,
        src_fmt: str,
        output: str | Path | None,
        notes: list[str],
    ) -> ConversionResult:
        tensors = self._load_tensors(src, src_fmt, notes)
        dest = Path(output) if output else src.parent / (src.stem + "_converted.safetensors")
        _save_safetensors(tensors, dest)
        return ConversionResult(
            source_path=str(src),
            target_path=str(dest),
            source_format=src_fmt,
            target_format="safetensors",
            tensor_count=len(tensors),
            notes=notes,
        )

    def _to_huggingface(
        self,
        src: Path,
        src_fmt: str,
        output: str | Path | None,
        notes: list[str],
    ) -> ConversionResult:
        tensors = self._load_tensors(src, src_fmt, notes)

        # Add output.weight (tied to tok_emb.weight)
        # Clone to avoid shared-memory error in safetensors when both keys are present.
        if "tok_emb.weight" in tensors and "output.weight" not in tensors:
            tensors["lm_head.weight"] = tensors["tok_emb.weight"].clone()
            notes.append("Added lm_head.weight (cloned from tok_emb.weight for safetensors compatibility)")

        # Read metadata for config
        metadata: dict[str, Any] = {}
        meta_path = src / "metadata.json" if src.is_dir() else src.parent / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        hf_config = _build_hf_config(metadata)

        dest_dir = Path(output) if output else src.parent / (src.stem + "-hf")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        model_dest = dest_dir / "model.safetensors"
        _save_safetensors(tensors, model_dest)

        # Save config
        (dest_dir / "config.json").write_text(
            json.dumps(hf_config, indent=2), encoding="utf-8"
        )

        notes.append(
            "HuggingFace config.json written with model_type='cola_coder'. "
            "You may need a custom modeling_cola_coder.py to load this with transformers."
        )

        return ConversionResult(
            source_path=str(src),
            target_path=str(dest_dir),
            source_format=src_fmt,
            target_format="huggingface",
            tensor_count=len(tensors),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_tensors(
        self,
        src: Path,
        src_fmt: str,
        notes: list[str],
    ) -> dict[str, Any]:
        """Load tensors from *src*, handling directories and format variants."""
        if src.is_dir():
            st_files = list(src.glob("*.safetensors"))
            pt_files = [p for p in src.glob("*.pt") if "optimizer" not in p.name]
            if st_files:
                tensors: dict[str, Any] = {}
                for f in st_files:
                    tensors.update(_load_safetensors(f))
                tensors, had_prefix = _strip_compile_prefix(tensors)
                if had_prefix:
                    notes.append("Stripped _orig_mod. prefix (model was torch.compiled)")
                return tensors
            if pt_files:
                tensors = _load_pytorch(pt_files[0])
                tensors, had_prefix = _strip_compile_prefix(tensors)
                if had_prefix:
                    notes.append("Stripped _orig_mod. prefix (model was torch.compiled)")
                return tensors
            raise FileNotFoundError(f"No .safetensors or .pt files found in {src}")
        else:
            if src.suffix.lower() == ".safetensors":
                tensors = _load_safetensors(src)
            else:
                tensors = _load_pytorch(src)
            tensors, had_prefix = _strip_compile_prefix(tensors)
            if had_prefix:
                notes.append("Stripped _orig_mod. prefix (model was torch.compiled)")
            return tensors
