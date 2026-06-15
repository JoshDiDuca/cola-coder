"""Checkpoint-health endpoint helper for the local UI.

Mirrors the CLI ``scripts/checkpoint_info.py``: resolves a single checkpoint
directory under ``checkpoints/<model>/step_<step>``, reads its ``metadata.json``
sidecar (loss, step, config stem), stats the files on disk for total size, and
counts tensors from the safetensors JSON header WITHOUT loading any weights.

Pure library module (stdlib only — no torch, no safetensors). Best-effort: on a
missing directory or absent weight file it returns an ``{"error": ...}`` dict and
never raises. Does NOT trust the ``latest`` pointer — the caller passes an explicit
``model`` + ``step`` which resolve straight to the ``step_*`` directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_step_dir(step: str) -> str:
    """Return the ``step_<zero-padded>`` directory name for a step argument.

    Accepts either the bare numeric string (``"8500"``) or the already-formatted
    directory name (``"step_00008500"``). Numeric input is zero-padded to the 8-digit
    convention used by the trainer; non-numeric input is returned unchanged so an
    odd directory name still resolves if it exists verbatim.
    """
    raw = step.strip()
    if raw.startswith("step_"):
        return raw
    return f"step_{int(raw):08d}" if raw.isdigit() else raw


def _read_metadata(ckpt_dir: Path) -> dict | None:
    """Parse ``metadata.json`` in the checkpoint dir. None if missing/unreadable."""
    meta_path = ckpt_dir / "metadata.json"
    try:
        raw = meta_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _config_stem_from_metadata(metadata: dict | None) -> str | None:
    """Best-effort extraction of a config stem (e.g. ``"small_react_best"``).

    Looks for an explicit ``config_stem``/``config_name`` key first, then falls
    back to the basename (sans suffix) of any ``config_path``/``config_file``.
    """
    if metadata is None:
        return None
    for key in ("config_stem", "config_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("config_path", "config_file"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    return None


def _count_tensors_from_header(path: Path) -> int | None:
    """Count tensors from a .safetensors JSON header (no weight bytes read).

    Layout: 8-byte little-endian unsigned int N, then N bytes of UTF-8 JSON
    mapping tensor_name -> spec. Returns the tensor count (excluding the
    ``__metadata__`` key) or None if anything about the header is uncertain.
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
    if not isinstance(header, dict):
        return None
    return sum(1 for key in header if key != "__metadata__")


def checkpoint_health(model: str, step: str) -> dict:
    """Inspect the health of one checkpoint at ``checkpoints/<model>/step_<step>``.

    ``step`` may be the bare numeric string (``"8500"``) or the full directory name
    (``"step_00008500"``). Returns a dict matching ``schemas.CheckpointHealth``:

      {"path": str, "model": str, "step": int, "loss": float | None,
       "size_mb": float, "num_tensors": int | None, "files": list[str],
       "config_stem": str | None, "ok": bool}

    On a missing directory return ``{"error": "..."}``. Never raises.
    """
    step_dir_name = _normalize_step_dir(step)
    ckpt_dir = Path("checkpoints") / model / step_dir_name

    if not ckpt_dir.is_dir():
        return {"error": f"checkpoint not found: {ckpt_dir}"}

    try:
        entries = [p for p in ckpt_dir.iterdir() if p.is_file()]
    except OSError as exc:
        logger.warning("failed to list %s: %s", ckpt_dir, exc)
        return {"error": str(exc)}

    files = sorted(p.name for p in entries)

    total_bytes = 0
    for entry in entries:
        try:
            total_bytes += entry.stat().st_size
        except OSError:
            continue
    size_mb = round(total_bytes / 1e6, 2)

    safetensors = sorted(p for p in entries if p.suffix == ".safetensors")
    ok = bool(safetensors)

    # Count tensors only when there is exactly one shard and its header parses
    # cleanly; multi-shard or uncertain headers leave num_tensors as None.
    num_tensors: int | None = None
    if len(safetensors) == 1:
        num_tensors = _count_tensors_from_header(safetensors[0])

    metadata = _read_metadata(ckpt_dir)
    loss: float | None = None
    resolved_step: int | None = None
    if metadata is not None:
        meta_loss = metadata.get("loss")
        if isinstance(meta_loss, (int, float)):
            loss = float(meta_loss)
        meta_step = metadata.get("step")
        if isinstance(meta_step, int):
            resolved_step = meta_step

    # Prefer the metadata step; fall back to parsing the directory name.
    if resolved_step is None:
        suffix = step_dir_name.split("_", 1)[-1]
        resolved_step = int(suffix) if suffix.isdigit() else 0

    return {
        "path": str(ckpt_dir),
        "model": model,
        "step": resolved_step,
        "loss": loss,
        "size_mb": size_mb,
        "num_tensors": num_tensors,
        "files": files,
        "config_stem": _config_stem_from_metadata(metadata),
        "ok": ok,
    }
