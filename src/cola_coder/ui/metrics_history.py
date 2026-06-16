"""Time-series helpers for the local cola-coder UI/dashboard.

Pure library module (no Rich, no CLI) that parses the FULL training .log into a
downsampled time series for charting a loss curve + throughput. Mirrors the
line-parsing approach of ``status.py`` so the extracted numbers agree exactly.
All functions are best-effort and never raise.
"""

from __future__ import annotations

import re
from math import ceil
from pathlib import Path

# Matches a "pretty" log line such as:
#   03:12:20 step   2,500 ( 1.7%) loss 1.6057 ppl      5.0 lr 6.00e-04     1,813 tok/s
#   08:38:21 step 16,200 (10.8%) loss 1.2492 ppl 3.5 lr 6e-04 11,738 tok/s | ETA 338h (11:37)
# Field extraction mirrors status._LOG_LINE_RE, extended to also capture lr. The tok/s
# count is anchored on the literal "tok/s" (NOT end-of-line) so the trainer's trailing
# "| ETA …" suffix doesn't break the match and freeze the metrics chart (BUG-138).
_LOG_LINE_RE = re.compile(
    r"step\s+([\d,]+)\s*\(\s*([\d.]+)%\)"
    r".*?loss\s+([\d.]+)"
    r".*?ppl\s+([\d.]+)"
    r".*?lr\s+([\d.eE+-]+)"
    r".*?([\d,]+)\s*tok/s"
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


def _downsample(points: list[dict], max_points: int) -> list[dict]:
    """Uniform-stride downsample, always keeping the first and last points."""
    count = len(points)
    if count <= max_points:
        return points
    stride = ceil(count / max_points)
    kept = points[::stride]
    if kept[-1] is not points[-1]:
        # Always include the last point; reserve a slot so we never exceed
        # max_points (drop the trailing strided sample if the cap is full).
        if len(kept) >= max_points:
            kept = kept[: max_points - 1]
        kept.append(points[-1])
    return kept


def training_history(
    log_path: str = "train_small_react_best.log",
    max_points: int = 500,
) -> dict:
    """Parse all step lines from the training .log into a downsampled time series.

    Returns::

        {"points": [ {"step": int, "loss": float | None, "ppl": float | None,
                      "lr": float | None, "tok_s": float | None} ],
         "count": int}

    ``points`` is chronological and downsampled to at most ``max_points`` entries
    (first & last always kept). ``count`` is the total number of valid step lines
    found BEFORE downsampling. Malformed/partial lines are skipped; commas in
    numbers (e.g. "5,200") are stripped. On any failure (e.g. missing file) returns
    ``{"error": "..."}``. Never raises.
    """
    text = _read_text(log_path)
    if text is None:
        return {"error": f"could not read log: {log_path}"}

    points: list[dict] = []
    try:
        for line in _split_lines(text):
            m = _LOG_LINE_RE.search(line)
            if not m:
                continue
            step = _to_int(m.group(1))
            if step is None:
                continue
            points.append(
                {
                    "step": step,
                    "loss": _to_float(m.group(3)),
                    "ppl": _to_float(m.group(4)),
                    "lr": _to_float(m.group(5)),
                    "tok_s": _to_float(m.group(6)),
                }
            )
    except (re.error, ValueError):
        return {"error": "failed to parse log"}

    count = len(points)
    return {"points": _downsample(points, max_points), "count": count}
