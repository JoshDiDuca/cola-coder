"""Environment-check endpoint helper for the local UI.

Structured (non-log) mirror of the CLI ``scripts/env_check.py``: reports the
Python version, PyTorch version + CUDA availability, primary GPU name/VRAM, key
dependency presence, ``HF_TOKEN`` presence, the active venv path, and free disk
space on the project drive. Robust to a missing ``torch`` (or any other
dependency): a guarded import failure becomes a check with ``ok=False`` and a
``None`` field, never a raised exception. Does not print — returns data.
"""

from __future__ import annotations

import importlib.metadata as _meta
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Dependencies whose mere presence we report (distribution name -> label).
_KEY_DEPS: tuple[tuple[str, str], ...] = (
    ("safetensors", "safetensors"),
    ("tokenizers", "tokenizers (HF)"),
    ("pydantic", "pydantic"),
    ("fastapi", "fastapi"),
)


def _item(name: str, ok: bool, value: str, detail: str | None = None) -> dict:
    """Build one ``EnvCheckItem`` dict."""
    return {"name": name, "ok": ok, "value": value, "detail": detail}


def _dep_version(dist: str) -> str | None:
    """Return an installed distribution's version, or None if absent."""
    try:
        return _meta.version(dist)
    except _meta.PackageNotFoundError:
        return None
    except Exception as exc:  # defensive: corrupt metadata, etc.
        logger.warning("could not read version for %s: %s", dist, exc)
        return None


def env_check() -> dict:
    """Run the structured environment battery; return an ``EnvCheckReport`` dict.

    Never raises. On an unexpected top-level failure returns ``{"error": ...}``.
    """
    try:
        return _build_report()
    except Exception as exc:  # belt-and-braces: this fn must never raise
        logger.exception("env_check failed")
        return {"error": str(exc)}


def _build_report() -> dict:
    checks: list[dict] = []

    # ── Python version (requires 3.10+) ──────────────────────────────────────
    vi = sys.version_info
    python_version = f"{vi.major}.{vi.minor}.{vi.micro}"
    py_ok = vi >= (3, 10)
    checks.append(
        _item(
            "Python version",
            py_ok,
            python_version,
            None if py_ok else "requires 3.10+",
        )
    )

    # ── PyTorch + CUDA + GPU (all guarded behind the torch import) ────────────
    torch_version: str | None = None
    cuda_available = False
    gpu_name: str | None = None
    vram_gb: float | None = None
    try:
        import torch  # noqa: PLC0415

        torch_version = str(torch.__version__)
        checks.append(_item("PyTorch", True, torch_version, None))

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_ver = torch.version.cuda or "unknown"
            count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            vram_gb = round(props.total_memory / (1024**3), 1)
            checks.append(
                _item(
                    "CUDA",
                    True,
                    f"v{cuda_ver}",
                    f"{count} GPU(s)",
                )
            )
            checks.append(
                _item(
                    "GPU",
                    vram_gb >= 8.0,
                    gpu_name,
                    f"{vram_gb:.1f} GB VRAM",
                )
            )
        else:
            checks.append(
                _item("CUDA", False, "", "not available — training would use CPU")
            )
    except ImportError:
        checks.append(_item("PyTorch", False, "", "torch not installed"))
    except Exception as exc:
        logger.warning("torch/CUDA probe failed: %s", exc)
        checks.append(_item("CUDA", False, "", f"could not query CUDA: {exc}"))

    # ── Key dependencies ─────────────────────────────────────────────────────
    for dist, label in _KEY_DEPS:
        ver = _dep_version(dist)
        checks.append(
            _item(label, ver is not None, ver or "", None if ver else "not installed")
        )

    # ── HF_TOKEN presence (masked) ───────────────────────────────────────────
    token = os.environ.get("HF_TOKEN", "")
    hf_token_set = bool(token) and len(token) > 8
    if hf_token_set:
        masked = f"{token[:4]}...{token[-4:]}"
        checks.append(_item("HF_TOKEN", True, masked, "set"))
    elif token:
        checks.append(_item("HF_TOKEN", False, "", "set but very short — may be invalid"))
    else:
        checks.append(_item("HF_TOKEN", False, "", "not set — needed for gated datasets"))

    # ── venv path ────────────────────────────────────────────────────────────
    venv = os.environ.get("VIRTUAL_ENV") or sys.prefix
    checks.append(_item("venv", bool(venv), venv, None))

    # ── Disk space on the project drive ──────────────────────────────────────
    probe = _PROJECT_ROOT
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        disk_ok = free_gb >= 10.0
        checks.append(
            _item(
                "Disk (project drive)",
                disk_ok,
                f"{free_gb:.1f} GB free",
                f"{total_gb:.1f} GB total"
                + ("" if disk_ok else " — under 10 GB free"),
            )
        )
    except OSError as exc:
        logger.warning("disk usage probe failed: %s", exc)
        checks.append(_item("Disk (project drive)", False, "", f"could not check: {exc}"))

    passed = sum(1 for c in checks if c["ok"])
    failed = len(checks) - passed

    return {
        "python_version": python_version,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "hf_token_set": hf_token_set,
        "passed": passed,
        "failed": failed,
        "ok": failed == 0,
        "checks": checks,
    }
