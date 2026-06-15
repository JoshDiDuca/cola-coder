"""Training-provenance endpoint helper for the local UI.

Scans ``checkpoints/<model>/training_manifest.yaml`` and surfaces one
provenance row per model: the config + key hyperparameters each checkpoint was
trained with. Pure stdlib + PyYAML (no torch / safetensors) — it never loads
weights. Complements :mod:`cola_coder.ui.checkpoint_detail` (single-checkpoint
architecture inspection) and ``status.list_checkpoints`` (dir enumeration).

The manifest is written by ``train.py`` and is nested under ``model`` /
``training`` / ``progress`` sections (see
``checkpoints/small_react_best/training_manifest.yaml``). This module flattens
the fields the UI cares about onto a single ``TrainingManifest`` row, leaving
anything the manifest does not contain as ``None``.

``latest_step`` is resolved by scanning the model dir's ``step_*`` subdirs
directly (the project's ``latest`` pointer file can be stale — see
``.claude/rules/checkpoints.md``), NOT trusted from any pointer.

All functions are best-effort and never raise — on a fatal error they return an
``{"error": ...}`` dict; per-model parse failures are skipped so one bad
manifest never hides the others. An empty ``checkpoints/`` yields ``count: 0``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .schemas import JsonValue

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "training_manifest.yaml"


def _to_int(value: object) -> int | None:
    """Coerce a manifest scalar to int, or None if it isn't an int-like number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _to_float(value: object) -> float | None:
    """Coerce a manifest scalar to float, or None if it isn't numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_str(value: object) -> str | None:
    """Return the value as a str when it is one, else None."""
    return value if isinstance(value, str) else None


def _section(parsed: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    """Return ``parsed[key]`` when it is a mapping, else an empty dict."""
    sub = parsed.get(key)
    return sub if isinstance(sub, dict) else {}


def _latest_step(model_dir: Path) -> int | None:
    """Newest ``step_<n>`` under ``model_dir`` by numeric step, or None.

    Scans ``step_*`` dirs directly rather than trusting the ``latest`` pointer
    file, which can be stale after a from-scratch restart.
    """
    try:
        steps = [
            int(d.name.split("_", 1)[1])
            for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith("step_") and d.name.split("_", 1)[1].isdigit()
        ]
    except OSError:
        return None
    return max(steps) if steps else None


def _read_manifest(manifest_path: Path) -> dict[str, JsonValue] | None:
    """Parse one ``training_manifest.yaml`` into a mapping, or None on failure."""
    try:
        raw_text = manifest_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("could not read manifest %s: %s", manifest_path, exc)
        return None
    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        logger.warning("could not parse manifest %s: %s", manifest_path, exc)
        return None
    return parsed if isinstance(parsed, dict) else None


def _build_row(model_dir: Path, manifest_path: Path) -> dict[str, JsonValue] | None:
    """Flatten one manifest into a ``TrainingManifest`` dict, or None if unparseable."""
    parsed = _read_manifest(manifest_path)
    if parsed is None:
        return None

    model = _section(parsed, "model")
    training = _section(parsed, "training")

    try:
        mtime = manifest_path.stat().st_mtime
    except OSError:
        mtime = 0.0

    return {
        "model": model_dir.name,
        "path": str(manifest_path),
        # The manifest has no explicit config-name field; surface the tool that
        # wrote it (``cola-coder/train.py``) so provenance isn't blank, else None.
        "config": _to_str(parsed.get("config")) or _to_str(parsed.get("tool")),
        "dim": _to_int(model.get("dim")),
        "n_layers": _to_int(model.get("n_layers")),
        "n_heads": _to_int(model.get("n_heads")),
        "seq_len": _to_int(model.get("max_seq_len")),
        "batch_size": _to_int(training.get("batch_size")),
        "learning_rate": _to_float(training.get("learning_rate")),
        "max_steps": _to_int(training.get("max_steps")),
        "latest_step": _latest_step(model_dir),
        "created_at": _to_str(parsed.get("created")),
        "mtime": mtime,
    }


def training_manifests(ckpt_root: str = "checkpoints") -> dict:
    """Scan ``ckpt_root/<model>/training_manifest.yaml`` and return provenance rows.

    Returns ``{"manifests": [...], "count": N}`` sorted by model name. Models
    without a manifest are omitted; an unparseable manifest is skipped (logged).
    Robust to a missing ``ckpt_root`` (empty list). Never raises — on a fatal
    filesystem error returns ``{"error": ...}``.
    """
    root = Path(ckpt_root)
    try:
        model_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    except FileNotFoundError:
        return {"manifests": [], "count": 0}
    except OSError as exc:
        return {"error": str(exc)}

    rows: list[dict[str, JsonValue]] = []
    for model_dir in model_dirs:
        manifest_path = model_dir / _MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        row = _build_row(model_dir, manifest_path)
        if row is not None:
            rows.append(row)

    return {"manifests": rows, "count": len(rows)}
