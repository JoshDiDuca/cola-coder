"""Environment/system info helpers for the local cola-coder UI/dashboard.

Pure library module (no Rich, no CLI) that summarizes the Python/platform
environment, key package versions, GPUs (via ``nvidia-smi``), and disk usage.

CRITICAL: this module is intentionally fast and light — it must NOT import
``torch`` (or any other heavy ML package). Package versions are read from
installed package metadata via ``importlib.metadata`` without importing the
packages themselves, so this never loads CUDA or contends with a live trainer.

All functions are best-effort and never raise: a failure in one field leaves
that field empty/None while the rest of the call succeeds.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# Packages we report versions for. Names are distribution names as known to
# importlib.metadata (NOT imported as modules).
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "fastapi",
    "numpy",
    "safetensors",
)


def _to_int(value: str) -> int | None:
    try:
        return int(float(value.replace(",", "")))
    except (ValueError, AttributeError):
        return None


def _package_versions() -> dict:
    """Resolve installed versions via importlib.metadata, never importing them."""
    versions: dict = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
        except Exception:
            # Defensive: metadata corruption / unexpected errors are non-fatal.
            versions[name] = None
    return versions


def _query_gpus() -> list[dict]:
    """Best-effort GPU list via nvidia-smi. Empty list if unavailable."""
    gpus: list[dict] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return gpus
    if proc.returncode != 0 or not proc.stdout.strip():
        return gpus

    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0] or None,
                "mem_total_mb": _to_int(parts[1]),
                "mem_used_mb": _to_int(parts[2]),
                "util_pct": _to_int(parts[3]),
            }
        )
    return gpus


def _disk_usage(root: str) -> dict:
    """Best-effort disk usage for ``root`` via shutil.disk_usage."""
    result = {
        "path": root,
        "total_bytes": None,
        "free_bytes": None,
        "used_bytes": None,
    }
    try:
        usage = shutil.disk_usage(root)
    except (OSError, ValueError):
        return result
    result["total_bytes"] = usage.total
    result["free_bytes"] = usage.free
    result["used_bytes"] = usage.used
    return result


def system_info(root: str = ".") -> dict:
    """Best-effort environment summary. Returns:
      {"python_version": str,
       "platform": str,
       "packages": dict,        # {"torch": ver|None, "transformers": ver|None, "tokenizers": ver|None,
                                #  "fastapi": ver|None, "numpy": ver|None, "safetensors": ver|None} via importlib.metadata
       "gpus": [ {"name": str, "mem_total_mb": int|None, "mem_used_mb": int|None, "util_pct": int|None} ],
       "disk": {"path": str, "total_bytes": int|None, "free_bytes": int|None, "used_bytes": int|None}}
    Each field independently best-effort: a failure in one (e.g. nvidia-smi absent) leaves that field
    empty/None, never crashes the whole call. Never raise. Only return {"error": ...} on a truly
    catastrophic failure.
    """
    try:
        try:
            python_version = sys.version.split()[0]
        except Exception:
            python_version = ""

        try:
            platform_str = platform.platform()
        except Exception:
            platform_str = ""

        return {
            "python_version": python_version,
            "platform": platform_str,
            "packages": _package_versions(),
            "gpus": _query_gpus(),
            "disk": _disk_usage(root),
        }
    except Exception as exc:  # pragma: no cover - truly catastrophic only
        return {"error": str(exc)}
