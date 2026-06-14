"""Status helpers for the local cola-coder UI/dashboard.

Pure library module (no Rich, no CLI) that reads training progress from the
log/err files, queries the GPU via nvidia-smi, and enumerates checkpoints.
All functions are best-effort and never raise.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

# Matches a "pretty" log line such as:
#   03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl      5.0 lr 6.00e-04     1,813 tok/s
_LOG_LINE_RE = re.compile(
    r"step\s+([\d,]+)\s*\(\s*([\d.]+)%\)"
    r".*?loss\s+([\d.]+)"
    r".*?ppl\s+([\d.]+)"
    r".*?([\d,]+)\s*(?:tok/s)?\s*$"
)

# Matches a tqdm bar such as:
#   Training:   2%|x| 2515/150000 [04:40<700:27:01, 17.10s/it]
_TQDM_RE = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*([\d.]+)s/it"
)


def _split_lines(text: str) -> list[str]:
    """Split on both carriage returns and newlines (tqdm uses \\r)."""
    return [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _to_int(value: str) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None


def _is_training_alive() -> bool:
    """True if a python process whose cmdline contains 'train.py' is running."""
    try:
        import psutil
    except ImportError:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("train.py" in str(part) for part in cmdline):
            return True
    return False


def _empty_status(alive: bool) -> dict:
    return {
        "alive": alive,
        "step": None,
        "total_steps": None,
        "progress_pct": None,
        "loss": None,
        "ppl": None,
        "tok_per_s": None,
        "s_per_it": None,
        "last_log_line": None,
    }


def _parse_log(text: str) -> dict | None:
    """Parse the most recent pretty step line from a .log file."""
    match_line = None
    match_obj = None
    for line in _split_lines(text):
        m = _LOG_LINE_RE.search(line)
        if m:
            match_line = line
            match_obj = m
    if match_obj is None:
        return None
    return {
        "step": _to_int(match_obj.group(1)),
        "total_steps": None,
        "progress_pct": _to_float(match_obj.group(2)),
        "loss": _to_float(match_obj.group(3)),
        "ppl": _to_float(match_obj.group(4)),
        "tok_per_s": _to_float(match_obj.group(5)),
        "s_per_it": None,
        "last_log_line": match_line.strip(),
    }


def _parse_err(text: str) -> dict | None:
    """Parse the most recent tqdm bar from a .err file."""
    match_line = None
    match_obj = None
    for line in _split_lines(text):
        m = _TQDM_RE.search(line)
        if m:
            match_line = line
            match_obj = m
    if match_obj is None:
        return None
    step = _to_int(match_obj.group(1))
    total = _to_int(match_obj.group(2))
    pct = None
    if step is not None and total:
        pct = round(step / total * 100, 4)
    return {
        "step": step,
        "total_steps": total,
        "progress_pct": pct,
        "loss": None,
        "ppl": None,
        "tok_per_s": None,
        "s_per_it": _to_float(match_obj.group(3)),
        "last_log_line": match_line.strip(),
    }


def get_training_status(
    log_path: str = "train_small_react_best.log",
    err_path: str = "train_small_react_best.err",
) -> dict:
    """Read training progress from the log/err files.

    Returns a dict with keys: alive, step, total_steps, progress_pct, loss,
    ppl, tok_per_s, s_per_it, last_log_line. Never raises.
    """
    alive = _is_training_alive()
    status = _empty_status(alive)

    log_text = _read_text(log_path)
    if log_text:
        parsed = _parse_log(log_text)
        if parsed is not None:
            status.update(parsed)
            status["alive"] = alive
            return status

    err_text = _read_text(err_path)
    if err_text:
        parsed = _parse_err(err_text)
        if parsed is not None:
            status.update(parsed)
            status["alive"] = alive
            return status

    return status


def get_system_status() -> dict:
    """Best-effort GPU status via nvidia-smi. Returns all-None on failure."""
    result = {
        "gpu_name": None,
        "gpu_util_pct": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_power_w": None,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,"
                "memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if proc.returncode != 0 or not proc.stdout.strip():
        return result

    first_line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 5:
        return result
    result["gpu_name"] = parts[0] or None
    result["gpu_util_pct"] = _to_float(parts[1])
    result["gpu_mem_used_mb"] = _to_float(parts[2])
    result["gpu_mem_total_mb"] = _to_float(parts[3])
    result["gpu_power_w"] = _to_float(parts[4])
    return result


def list_checkpoints(ckpt_root: str = "checkpoints") -> list[dict]:
    """Enumerate checkpoints under ckpt_root/<model>/step_*.

    Each entry: {model, name, step, loss, path, mtime}, sorted by (model, step).
    Robust to a missing ckpt_root (returns []). Never raises.
    """
    root = Path(ckpt_root)
    out: list[dict] = []
    try:
        models = [d for d in root.iterdir() if d.is_dir()]
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
            loss = None
            meta_path = step_dir / "metadata.json"
            meta_text = _read_text(str(meta_path))
            if meta_text:
                try:
                    loss = json.loads(meta_text).get("loss")
                except (ValueError, AttributeError):
                    loss = None
            try:
                mtime = step_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            out.append(
                {
                    "model": model_dir.name,
                    "name": step_dir.name,
                    "step": step,
                    "loss": loss,
                    "path": str(step_dir),
                    "mtime": mtime,
                }
            )

    out.sort(key=lambda c: (c["model"], c["step"]))
    return out
