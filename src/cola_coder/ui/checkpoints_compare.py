"""Side-by-side comparison of two checkpoints for the local UI/dashboard.

Pure library module (stdlib only — no torch, no safetensors) that diffs the
architecture/parameter/metadata of TWO checkpoints WITHOUT loading any weights.
It does NOT re-parse safetensors itself: it delegates to the existing
``ui.checkpoint_detail.checkpoint_detail`` for each side (DRY) and computes the
diff from those two result dicts.

All functions are best-effort and never raise — on failure they return an
``{"error": ...}`` dict.
"""

from __future__ import annotations

from .checkpoint_detail import checkpoint_detail


def compare_checkpoints(path_a: str, path_b: str) -> dict:
    """Compare two checkpoint dirs. Returns:
      {"a": <checkpoint_detail(path_a)>,
       "b": <checkpoint_detail(path_b)>,
       "diff": {"num_params_delta": int,        # b - a
                "tensor_count_delta": int,       # b - a
                "is_moe_changed": bool,
                "metadata_changed_keys": list[str],   # keys whose value differs between a.metadata and b.metadata
                "dtypes_only_a": list[str], "dtypes_only_b": list[str]}}
    Calls the existing ui.checkpoint_detail.checkpoint_detail for each side (DRY — do NOT re-parse
    safetensors yourself). If EITHER side returns an {"error": ...}, return
    {"error": "...", "a": <detail_or_error>, "b": <detail_or_error>} (so the UI can show which side failed).
    Never raise.
    """
    try:
        detail_a = checkpoint_detail(path_a)
    except Exception as exc:  # defensive: checkpoint_detail promises not to raise
        detail_a = {"error": str(exc)}
    try:
        detail_b = checkpoint_detail(path_b)
    except Exception as exc:
        detail_b = {"error": str(exc)}

    if "error" in detail_a or "error" in detail_b:
        if "error" in detail_a and "error" in detail_b:
            message = f"a: {detail_a['error']}; b: {detail_b['error']}"
        elif "error" in detail_a:
            message = f"a: {detail_a['error']}"
        else:
            message = f"b: {detail_b['error']}"
        return {"error": message, "a": detail_a, "b": detail_b}

    diff = {
        "num_params_delta": _int(detail_b.get("num_params")) - _int(detail_a.get("num_params")),
        "tensor_count_delta": (
            _int(detail_b.get("tensor_count")) - _int(detail_a.get("tensor_count"))
        ),
        "is_moe_changed": bool(detail_a.get("is_moe")) != bool(detail_b.get("is_moe")),
        "metadata_changed_keys": _metadata_changed_keys(
            detail_a.get("metadata"), detail_b.get("metadata")
        ),
        "dtypes_only_a": sorted(_dtype_set(detail_a) - _dtype_set(detail_b)),
        "dtypes_only_b": sorted(_dtype_set(detail_b) - _dtype_set(detail_a)),
    }

    return {"a": detail_a, "b": detail_b, "diff": diff}


def _int(value: object) -> int:
    """Coerce a (possibly missing/None) numeric field to int, defaulting to 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _dtype_set(detail: dict) -> set[str]:
    """Extract the distinct dtype strings from a checkpoint_detail result."""
    dtypes = detail.get("dtypes")
    if not isinstance(dtypes, list):
        return set()
    return {d for d in dtypes if isinstance(d, str)}


def _metadata_changed_keys(meta_a: object, meta_b: object) -> list[str]:
    """Sorted union-diff of two metadata dicts.

    Returns keys present in both with differing values, plus keys present in only
    one of the two. ``None`` metadata is treated as an empty dict.
    """
    a = meta_a if isinstance(meta_a, dict) else {}
    b = meta_b if isinstance(meta_b, dict) else {}
    changed: set[str] = set()
    for key in a.keys() | b.keys():
        if key not in a or key not in b or a[key] != b[key]:
            changed.add(key)
    return sorted(changed)
