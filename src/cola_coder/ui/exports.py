"""Export overview helpers for the local cola-coder UI/dashboard.

Read-only summary of model export options (the formats that
``scripts/export_model.py`` supports) plus any artifacts that have already been
produced (``*.gguf`` files, an Ollama ``Modelfile``, INT8 ``.pt`` dumps). Pure
library module (no Rich, no CLI). All functions are best-effort and never raise
— on failure they return an ``{"error": ...}`` dict.
"""

from __future__ import annotations

from pathlib import Path

# Supported export formats, mirroring scripts/export_model.py's --action choices
# (_ACTION_MAP / _MENU_CHOICES). Keep these keys in sync with that script — they
# are the real, supported actions, not invented ones.
_FORMATS: list[dict] = [
    {
        "key": "gguf-f16",
        "label": "GGUF (F16)",
        "desc": "Full-precision 16-bit GGUF for llama.cpp / Ollama.",
    },
    {
        "key": "gguf-q8",
        "label": "GGUF (Q8_0)",
        "desc": "8-bit quantized GGUF — near-lossless, ~2x smaller.",
    },
    {
        "key": "gguf-q4",
        "label": "GGUF (Q4_K_M)",
        "desc": "4-bit K-quant GGUF — smallest, fast CPU inference.",
    },
    {
        "key": "ollama",
        "label": "Ollama Modelfile",
        "desc": "F16 GGUF plus a Modelfile for `ollama create`.",
    },
    {
        "key": "quantize",
        "label": "INT8 (PyTorch)",
        "desc": "Dynamic INT8 quantization saved as a .pt state dict.",
    },
    {
        "key": "benchmark",
        "label": "Benchmark",
        "desc": "Compare original vs INT8 latency/size (produces no artifact).",
    },
]

# Filenames / extensions an export run can produce, mapped to a format hint.
_GGUF_EXT = ".gguf"
_INT8_EXT = ".pt"
_MODELFILE_NAME = "Modelfile"


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _format_for(path: Path) -> str:
    """Best-effort format hint for a produced artifact, from its name."""
    name = path.name.lower()
    if name.endswith(_GGUF_EXT):
        return "gguf"
    if name == _MODELFILE_NAME.lower():
        return "ollama"
    if name.endswith(_INT8_EXT):
        return "int8"
    return "other"


def _list_checkpoints(ckpt_root: Path) -> list[dict]:
    """Enumerate ckpt_root/<model>/step_* dirs, newest-first.

    Mirrors status.list_checkpoints' step_* scanning (parse the int after
    "step_", tolerant of non-step dirs). Each entry: {model, name, step, path}.
    """
    out: list[dict] = []
    try:
        models = [d for d in ckpt_root.iterdir() if d.is_dir()]
    except OSError:
        return out

    for model_dir in models:
        try:
            step_dirs = [
                d
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            ]
        except OSError:
            continue
        for step_dir in step_dirs:
            step = _to_int(step_dir.name.split("_", 1)[1])
            if step is None:
                continue
            out.append(
                {
                    "model": model_dir.name,
                    "name": step_dir.name,
                    "step": step,
                    "path": str(step_dir),
                }
            )

    # newest-first: highest step first, grouped by model name
    out.sort(key=lambda c: (c["model"], -(c["step"] or 0)))
    return out


def _scan_existing(root: Path) -> list[dict]:
    """Recursively find produced export artifacts under ``root``, newest-first.

    Matches ``*.gguf`` files, an Ollama ``Modelfile``, and INT8 ``*.pt`` dumps.
    Each entry: {path, format, size_bytes, mtime}. Returns [] if none/missing.
    """
    out: list[dict] = []
    try:
        candidates = [p for p in root.rglob("*") if p.is_file()]
    except OSError:
        return out

    for path in candidates:
        name = path.name.lower()
        is_gguf = name.endswith(_GGUF_EXT)
        is_modelfile = name == _MODELFILE_NAME.lower()
        is_int8 = name.endswith(_INT8_EXT)
        if not (is_gguf or is_modelfile or is_int8):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append(
            {
                "path": str(path),
                "format": _format_for(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )

    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def export_overview(root: str = ".") -> dict:
    """Summarize export options + outputs. Returns:

      {"checkpoints": [ {"model": str, "name": str, "step": int | None, "path": str} ],
       "formats": [ {"key": str, "label": str, "desc": str} ],
       "existing": [ {"path": str, "format": str, "size_bytes": int, "mtime": float} ]}

    ``checkpoints`` enumerates checkpoints/<model>/step_* like
    status.list_checkpoints (top-level dense + sft + moe etc.), newest-first.
    ``formats`` are the real supported export formats from export_model.py.
    ``existing`` scans for produced artifacts (``*.gguf``, ``Modelfile``, INT8
    ``*.pt``) under ``root``; [] if none. On any failure returns {"error": ...}.
    Never raises.
    """
    try:
        base = Path(root)
        checkpoints = _list_checkpoints(base / "checkpoints")
        existing = _scan_existing(base)
        return {
            "checkpoints": checkpoints,
            "formats": [dict(fmt) for fmt in _FORMATS],
            "existing": existing,
        }
    except Exception as exc:  # never raise
        return {"error": str(exc)}
