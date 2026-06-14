"""Side-by-side diff of two YAML configs for the local UI/dashboard.

Pure library module that diffs TWO YAML configs (e.g. ``configs/small.yaml`` vs
``configs/4080_max.yaml``) by flattened dotted key, so the UI can show what
changed. It does NOT re-parse YAML itself: it delegates to the existing
``ui.configs.read_config`` for each side (DRY) and computes the diff from the
``parsed`` dicts.

All functions are best-effort and never raise — on failure they return an
``{"error": ...}`` dict.
"""

from __future__ import annotations

from typing import Any

from .configs import read_config


def compare_configs(path_a: str, path_b: str) -> dict:
    """Diff two YAML config files by flattened dotted key. Returns:
      {"a": {"path": str, "parsed": dict},
       "b": {"path": str, "parsed": dict},
       "changed": [ {"key": str, "a": <any>, "b": <any>} ],   # keys present in BOTH with different values, sorted by key
       "only_a": list[str],   # dotted keys only in A, sorted
       "only_b": list[str]}   # dotted keys only in B, sorted
    Flatten nested dicts to dotted keys (e.g. "model.dim", "training.batch_size"); lists compared as whole values.
    Uses ui.configs.read_config for each side (DRY). If EITHER side errors, return
    {"error": "...", "a": <read_or_error>, "b": <read_or_error>}. Never raise.
    """
    try:
        read_a = read_config(path_a)
    except Exception as exc:  # defensive: read_config promises not to raise
        read_a = {"error": str(exc)}
    try:
        read_b = read_config(path_b)
    except Exception as exc:
        read_b = {"error": str(exc)}

    if "error" in read_a or "error" in read_b:
        if "error" in read_a and "error" in read_b:
            message = f"a: {read_a['error']}; b: {read_b['error']}"
        elif "error" in read_a:
            message = f"a: {read_a['error']}"
        else:
            message = f"b: {read_b['error']}"
        return {"error": message, "a": read_a, "b": read_b}

    flat_a = _flatten(read_a.get("parsed"))
    flat_b = _flatten(read_b.get("parsed"))

    changed: list[dict] = []
    for key in sorted(flat_a.keys() & flat_b.keys()):
        if flat_a[key] != flat_b[key]:
            changed.append({"key": key, "a": flat_a[key], "b": flat_b[key]})

    only_a = sorted(flat_a.keys() - flat_b.keys())
    only_b = sorted(flat_b.keys() - flat_a.keys())

    return {
        "a": {"path": path_a, "parsed": read_a.get("parsed")},
        "b": {"path": path_b, "parsed": read_b.get("parsed")},
        "changed": changed,
        "only_a": only_a,
        "only_b": only_b,
    }


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys.

    Recurses into dicts only — lists and scalars are leaf values (compared as
    whole). A non-dict top-level value (or ``None``) yields ``{}``. Deterministic:
    iteration order does not affect the resulting dict.
    """
    if not isinstance(d, dict):
        return {}

    flat: dict[str, Any] = {}
    for key, value in d.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat
