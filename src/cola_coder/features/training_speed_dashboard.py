"""Training Speed Dashboard: track and display training throughput metrics.

Monitors steps/sec, tokens/sec, GPU utilization, and ETA during training.
Self-contained — no external dependencies beyond the standard library and
optional pynvml for GPU utilization.
"""

import time
from dataclasses import dataclass
from collections import deque
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Optional GPU utilization via pynvml
# ---------------------------------------------------------------------------

def _gpu_utilization() -> float:
    """Return GPU utilization as a fraction 0.0–1.0, or 0.0 if unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu / 100.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# SpeedMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeedMetrics:
    step: int
    steps_per_second: float
    tokens_per_second: float
    gpu_utilization: float       # 0.0–1.0
    eta_seconds: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# TrainingSpeedDashboard
# ---------------------------------------------------------------------------

class TrainingSpeedDashboard:
    """Track training speed and render a text dashboard.

    Args:
        total_steps: Total number of training steps (used for ETA).
        window: Number of recent measurements used for rolling-average speed.
    """

    def __init__(self, total_steps: int, window: int = 50) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self._total_steps = total_steps
        self._window = window

        self._start_time: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None
        self._last_tokens: int = 0

        # Rolling window of (timestamp, step, cumulative_tokens) tuples
        self._samples: deque = deque(maxlen=window)
        self._history: list[SpeedMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, step: int, tokens_processed: int) -> None:
        """Record a training step.

        Args:
            step: Current step index (0-based).
            tokens_processed: Cumulative tokens processed so far, *or* tokens
                in this single step — both work because speed is derived from
                deltas between consecutive calls.
        """
        now = time.monotonic()

        if self._start_time is None:
            self._start_time = now

        self._samples.append((now, step, tokens_processed))

        metrics = self._compute_metrics(now, step, tokens_processed)
        self._history.append(metrics)

        self._last_time = now
        self._last_step = step
        self._last_tokens = tokens_processed

    def current_speed(self) -> SpeedMetrics:
        """Return the most recent SpeedMetrics, computing on demand if needed."""
        if self._history:
            return self._history[-1]
        # Nothing recorded yet — return zeroed metrics
        return SpeedMetrics(
            step=0,
            steps_per_second=0.0,
            tokens_per_second=0.0,
            gpu_utilization=0.0,
            eta_seconds=float(self._total_steps),
            elapsed_seconds=0.0,
        )

    def estimate_eta(self) -> float:
        """Return estimated seconds remaining until total_steps is reached."""
        return self.current_speed().eta_seconds

    def format_dashboard(self) -> str:
        """Render a human-readable text dashboard."""
        m = self.current_speed()
        elapsed_str = _format_duration(m.elapsed_seconds)
        eta_str = _format_duration(m.eta_seconds)
        gpu_pct = m.gpu_utilization * 100.0
        progress_pct = (m.step / self._total_steps * 100.0) if self._total_steps > 0 else 0.0

        bar_width = 30
        filled = int(bar_width * m.step / self._total_steps) if self._total_steps > 0 else 0
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        lines = [
            "=== Training Speed Dashboard ===",
            f"  Progress : {bar} {m.step}/{self._total_steps} ({progress_pct:.1f}%)",
            f"  Steps/s  : {m.steps_per_second:.2f}",
            f"  Tokens/s : {m.tokens_per_second:,.0f}",
            f"  GPU util : {gpu_pct:.1f}%",
            f"  Elapsed  : {elapsed_str}",
            f"  ETA      : {eta_str}",
            "================================",
        ]
        return "\n".join(lines)

    def history(self) -> list[SpeedMetrics]:
        """Return all recorded SpeedMetrics in chronological order."""
        return list(self._history)

    def summary(self) -> dict:
        """Return a summary dict of aggregate statistics."""
        if not self._history:
            return {
                "total_steps_recorded": 0,
                "avg_steps_per_second": 0.0,
                "avg_tokens_per_second": 0.0,
                "peak_steps_per_second": 0.0,
                "peak_tokens_per_second": 0.0,
                "avg_gpu_utilization": 0.0,
                "total_elapsed_seconds": 0.0,
                "final_eta_seconds": float(self._total_steps),
            }

        steps_per_sec = [m.steps_per_second for m in self._history]
        tokens_per_sec = [m.tokens_per_second for m in self._history]
        gpu_utils = [m.gpu_utilization for m in self._history]
        last = self._history[-1]

        return {
            "total_steps_recorded": len(self._history),
            "avg_steps_per_second": sum(steps_per_sec) / len(steps_per_sec),
            "avg_tokens_per_second": sum(tokens_per_sec) / len(tokens_per_sec),
            "peak_steps_per_second": max(steps_per_sec),
            "peak_tokens_per_second": max(tokens_per_sec),
            "avg_gpu_utilization": sum(gpu_utils) / len(gpu_utils),
            "total_elapsed_seconds": last.elapsed_seconds,
            "final_eta_seconds": last.eta_seconds,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, now: float, step: int, tokens_processed: int) -> SpeedMetrics:
        elapsed = now - self._start_time if self._start_time is not None else 0.0

        steps_per_second = 0.0
        tokens_per_second = 0.0

        if len(self._samples) >= 2:
            oldest_time, oldest_step, oldest_tokens = self._samples[0]
            newest_time, newest_step, newest_tokens = self._samples[-1]

            dt = newest_time - oldest_time
            if dt > 0:
                d_steps = newest_step - oldest_step
                d_tokens = newest_tokens - oldest_tokens

                steps_per_second = max(0.0, d_steps / dt)
                tokens_per_second = max(0.0, d_tokens / dt)

        # ETA based on whole-run average speed when rolling window is sparse
        if steps_per_second > 0:
            remaining_steps = max(0, self._total_steps - step - 1)
            eta_seconds = remaining_steps / steps_per_second
        elif elapsed > 0 and step > 0:
            avg_sps = step / elapsed
            remaining_steps = max(0, self._total_steps - step - 1)
            eta_seconds = remaining_steps / avg_sps if avg_sps > 0 else 0.0
        else:
            eta_seconds = 0.0

        gpu_util = _gpu_utilization()

        return SpeedMetrics(
            step=step,
            steps_per_second=steps_per_second,
            tokens_per_second=tokens_per_second,
            gpu_utilization=gpu_util,
            eta_seconds=eta_seconds,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as h:mm:ss or mm:ss."""
    seconds = max(0.0, seconds)
    total_s = int(seconds)
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
