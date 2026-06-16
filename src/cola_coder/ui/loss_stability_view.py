"""Training loss-stability meter for the dashboard (MODEL-055 / UI-102).

Applies ZClip's z-score spike idea (research-log 2026-06-16) to the OBSERVABLE
loss curve instead of the (unlogged) gradient norm: read the loss series the
trainer already writes, EMA-smooth it, classify the trend, and flag step-to-step
loss SPIKES via the z-score of the loss deltas. Surfaces an at-a-glance
stability verdict so someone watching a multi-day run sees instability early —
a cluster of loss spikes precedes many divergences.

Pure read over ``metrics_history.training_history`` (DRY — no re-parsing of the
log). MAIN-SAFE: no model, no GPU, never touches the trainer.
"""

from __future__ import annotations

import statistics
from typing import Literal

from cola_coder.ui.metrics_history import training_history

# Minimum loss points before the meter is meaningful (else "insufficient_data").
_MIN_POINTS = 6
# Only the most recent window of loss points is analysed (recent behavior matters).
_WINDOW = 60
# A loss delta whose z-score exceeds this is a spike.
_Z_THRESH = 3.0
# Deltas in the last this-many steps count toward the "recent" spike verdict.
_RECENT = 5
# Relative change (second-half mean vs first-half mean) for a trend call.
_TREND_EPS = 0.01

Trend = Literal["improving", "flat", "worsening", "unknown"]
Verdict = Literal["stable", "watch", "spiking", "insufficient_data"]


def _empty(points_used: int = 0) -> dict:
    return {
        "current_loss": None,
        "ema_loss": None,
        "trend": "unknown",
        "spike_count": 0,
        "recent_max_z": None,
        "verdict": "insufficient_data",
        "points_used": points_used,
    }


def compute_loss_stability(
    losses: list[float],
    z_thresh: float = _Z_THRESH,
    ema_alpha: float = 0.3,
) -> dict:
    """Classify a loss series' stability — pure function over the loss values.

    Returns the ``LossStability`` shape: current/EMA loss, trend
    (improving/flat/worsening), the count of spike deltas, the max recent
    z-score, and an overall verdict. Uses only the most recent ``_WINDOW`` points.
    """
    clean = [float(x) for x in losses if x is not None]
    if len(clean) < _MIN_POINTS:
        return _empty(len(clean))

    window = clean[-_WINDOW:]
    n = len(window)
    current = window[-1]

    ema = window[0]
    for value in window[1:]:
        ema = ema_alpha * value + (1.0 - ema_alpha) * ema

    half = n // 2
    first_mean = statistics.fmean(window[:half])
    second_mean = statistics.fmean(window[half:])
    rel = (second_mean - first_mean) / (abs(first_mean) or 1.0)
    if rel < -_TREND_EPS:
        trend: Trend = "improving"
    elif rel > _TREND_EPS:
        trend = "worsening"
    else:
        trend = "flat"

    # Spike detection on step-to-step loss deltas (positive = loss jumped up).
    deltas = [window[i] - window[i - 1] for i in range(1, n)]
    mean_d = statistics.fmean(deltas)
    std_d = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0

    spike_count = 0
    recent_max_z = 0.0
    last_spikes = 0
    for idx, d in enumerate(deltas):
        z = (d - mean_d) / std_d if std_d > 1e-12 else 0.0
        is_recent = idx >= len(deltas) - _RECENT
        if is_recent:
            recent_max_z = max(recent_max_z, z)
        if d > 0 and z > z_thresh:
            spike_count += 1
            if is_recent:
                last_spikes += 1

    if last_spikes > 0:
        verdict: Verdict = "spiking"
    elif trend == "worsening":
        verdict = "watch"
    else:
        verdict = "stable"

    return {
        "current_loss": round(current, 4),
        "ema_loss": round(ema, 4),
        "trend": trend,
        "spike_count": spike_count,
        "recent_max_z": round(recent_max_z, 3),
        "verdict": verdict,
        "points_used": n,
    }


def loss_stability(log_path: str = "train_small_react_best.log") -> dict:
    """Read the loss series from the training log and classify its stability.

    Returns the ``LossStability`` dict, or ``{"error": str}`` if the log can't be
    read. An empty/short series yields a valid ``insufficient_data`` verdict
    (not an error). Never raises.
    """
    history = training_history(log_path)
    if "error" in history:
        return {"error": history["error"]}
    points = history.get("points", [])
    losses = [p["loss"] for p in points if isinstance(p, dict) and p.get("loss") is not None]
    return compute_loss_stability(losses)
