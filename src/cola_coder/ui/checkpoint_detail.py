"""Deep per-checkpoint inspection helpers for the local UI/dashboard.

Pure library module (stdlib only — no torch, no safetensors) that reports the
ground-truth architecture and metadata of a single checkpoint WITHOUT loading
any weights into memory. It reads only the safetensors JSON header (the 8-byte
little-endian header-length prefix + the header JSON) to count tensors and sum
parameter counts, and parses the ``metadata.json`` / ``moe_config.json``
sidecars. Complements ``status.list_checkpoints`` (which only enumerates dirs).

All functions are best-effort and never raise — on failure they return an
``{"error": ...}`` dict.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

# Tensor key fragments that indicate an upcycled MoE checkpoint.
_MOE_TENSOR_HINTS = ("ffn.experts.", "shared_experts.", "router.gate")


def _read_json(path: Path) -> dict | None:
    """Parse a JSON sidecar. Returns None if missing/unreadable/not an object."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _read_safetensors_header(path: Path) -> dict | None:
    """Read JUST the JSON header of a .safetensors file (no weight bytes).

    Layout: 8-byte little-endian unsigned int N, then N bytes of UTF-8 JSON
    mapping tensor_name -> {dtype, shape, data_offsets}. Returns the parsed
    header dict (which may include the ``__metadata__`` key) or None on failure.
    """
    try:
        with open(path, "rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                return None
            n = int.from_bytes(length_bytes, "little")
            if n <= 0:
                return None
            header_bytes = handle.read(n)
            if len(header_bytes) != n:
                return None
    except OSError:
        return None
    try:
        header = json.loads(header_bytes.decode("utf-8", errors="replace"))
    except (ValueError, TypeError):
        return None
    return header if isinstance(header, dict) else None


def _accumulate_header(
    header: dict,
    dtypes: set[str],
    tensor_keys: set[str],
) -> tuple[int, int]:
    """Sum params and count tensors for one header. Skips ``__metadata__``.

    Returns (num_params, tensor_count) for this header. Mutates ``dtypes`` and
    ``tensor_keys`` in place.
    """
    num_params = 0
    tensor_count = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            continue
        tensor_keys.add(name)
        tensor_count += 1
        dtype = spec.get("dtype")
        if isinstance(dtype, str):
            dtypes.add(dtype)
        shape = spec.get("shape")
        if isinstance(shape, list) and all(isinstance(d, int) for d in shape):
            # Empty shape -> scalar tensor with prod([]) == 1.
            num_params += prod(shape)
    return num_params, tensor_count


def checkpoint_detail(path: str) -> dict:
    """Inspect one checkpoint directory (or a path to its .safetensors). Returns:
      {"path": str,
       "metadata": dict | None,        # parsed metadata.json if present
       "is_moe": bool,                 # moe_config.json present OR expert tensor keys present
       "moe_config": dict | None,      # parsed moe_config.json if present
       "has_training_state": bool,     # training_state.pt present (presence only, never parse)
       "num_params": int,              # summed from safetensors header shapes (0 if unreadable)
       "tensor_count": int,
       "dtypes": list[str],            # distinct dtypes seen in the header
       "files": list[str]}             # filenames present in the dir
    On any failure (missing dir / no safetensors / bad header) return {"error": "..."}. Never raise.
    """
    target = Path(path)

    # Resolve the directory and the set of safetensors shards to inspect.
    if target.is_file() and target.suffix == ".safetensors":
        ckpt_dir = target.parent
        shards = [target]
    elif target.is_dir():
        ckpt_dir = target
        try:
            shards = sorted(
                p for p in ckpt_dir.iterdir()
                if p.is_file() and p.suffix == ".safetensors"
            )
        except OSError as exc:
            return {"error": str(exc)}
        if not shards:
            return {"error": f"no safetensors in {path}"}
    else:
        return {"error": f"path not found: {path}"}

    try:
        files = sorted(p.name for p in ckpt_dir.iterdir() if p.is_file())
    except OSError:
        files = sorted(p.name for p in shards)

    metadata = _read_json(ckpt_dir / "metadata.json")
    moe_config = _read_json(ckpt_dir / "moe_config.json")
    has_training_state = (ckpt_dir / "training_state.pt").is_file()

    num_params = 0
    tensor_count = 0
    dtypes: set[str] = set()
    tensor_keys: set[str] = set()
    read_any = False
    for shard in shards:
        header = _read_safetensors_header(shard)
        if header is None:
            continue
        read_any = True
        params, count = _accumulate_header(header, dtypes, tensor_keys)
        num_params += params
        tensor_count += count

    if not read_any:
        return {"error": f"unreadable safetensors header in {path}"}

    is_moe = moe_config is not None or any(
        hint in key for key in tensor_keys for hint in _MOE_TENSOR_HINTS
    )

    return {
        "path": str(ckpt_dir),
        "metadata": metadata,
        "is_moe": is_moe,
        "moe_config": moe_config,
        "has_training_state": has_training_state,
        "num_params": num_params,
        "tensor_count": tensor_count,
        "dtypes": sorted(dtypes),
        "files": files,
    }
