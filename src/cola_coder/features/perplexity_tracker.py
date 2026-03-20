"""Perplexity Tracker: tracks model perplexity (exp(loss)) over training steps.

Records perplexity values per training step, supports per-language breakdowns
(Python, TypeScript, JavaScript), computes moving averages, and alerts on
divergence (sudden perplexity spikes indicating training instability).

Perplexity scale reference:
  > 1000  : barely learning
  100-1000 : early training / small model
  30-100   : decent small model
  10-30    : good small model on familiar data
  < 10     : excellent or check for overfitting
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class PerplexityRecord:
    """A single perplexity measurement."""
    step: int
    loss: float
    perplexity: float
    language: Optional[str] = None


class PerplexityTracker:
    """Tracks perplexity (exp(loss)) over training steps.

    Supports overall tracking and per-language breakdowns. Provides moving
    average smoothing and divergence (spike) detection.

    Usage::

        tracker = PerplexityTracker()
        tracker.update(loss=2.5, step=100, language="python")
        ppl = tracker.get_perplexity()
        ma  = tracker.get_moving_average(window=50)
        spiked = tracker.check_divergence(threshold=2.0)
    """

    def __init__(self) -> None:
        # All records in insertion order
        self._records: list[PerplexityRecord] = []
        # Per-language records: language -> list[PerplexityRecord]
        self._lang_records: dict[str, list[PerplexityRecord]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, loss: float, step: int, language: Optional[str] = None) -> None:
        """Record a perplexity measurement.

        Args:
            loss: Cross-entropy loss value.
            step: Training step number.
            language: Optional language tag (e.g. "python", "typescript", "javascript").
        """
        ppl = math.exp(loss)
        record = PerplexityRecord(step=step, loss=loss, perplexity=ppl, language=language)
        self._records.append(record)
        if language is not None:
            self._lang_records[language].append(record)

    # ------------------------------------------------------------------
    # Perplexity queries
    # ------------------------------------------------------------------

    def get_perplexity(self, step: Optional[int] = None) -> Optional[float]:
        """Return perplexity at a given step, or the most recent measurement.

        Args:
            step: If provided, find the last record at or before this step.
                  If None, return the most recent perplexity.

        Returns:
            Perplexity value, or None if no records exist.
        """
        if not self._records:
            return None
        if step is None:
            return self._records[-1].perplexity
        # Return the perplexity of the last record whose step <= requested step
        match = None
        for r in self._records:
            if r.step <= step:
                match = r
        return match.perplexity if match is not None else None

    def get_moving_average(self, window: int = 100) -> Optional[float]:
        """Return the moving average of perplexity over the last `window` records.

        Args:
            window: Number of most recent records to average.

        Returns:
            Mean perplexity over the window, or None if no records exist.
        """
        if not self._records:
            return None
        recent = self._records[-window:]
        return sum(r.perplexity for r in recent) / len(recent)

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def check_divergence(self, threshold: float = 2.0) -> bool:
        """Detect whether the most recent perplexity is a spike (divergence).

        A spike is defined as the latest perplexity being more than
        `threshold` times larger than the moving average of the preceding
        measurements.

        Args:
            threshold: Ratio above which the latest perplexity is considered
                       a spike. Default 2.0 means a 2× jump triggers an alert.

        Returns:
            True if a divergence spike is detected, False otherwise.
        """
        if len(self._records) < 2:
            return False
        latest = self._records[-1].perplexity
        # Baseline: moving average of everything *except* the latest record
        preceding = self._records[:-1]
        baseline = sum(r.perplexity for r in preceding) / len(preceding)
        return latest > baseline * threshold

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict of overall perplexity statistics.

        Returns:
            Dict with keys:
              - current_perplexity: most recent perplexity (or None)
              - min_perplexity: lowest recorded perplexity (or None)
              - max_perplexity: highest recorded perplexity (or None)
              - mean_perplexity: mean over all records (or None)
              - total_steps: total number of measurements recorded
              - moving_average_100: moving average over last 100 records
        """
        if not self._records:
            return {
                "current_perplexity": None,
                "min_perplexity": None,
                "max_perplexity": None,
                "mean_perplexity": None,
                "total_steps": 0,
                "moving_average_100": None,
            }
        ppls = [r.perplexity for r in self._records]
        return {
            "current_perplexity": self._records[-1].perplexity,
            "min_perplexity": min(ppls),
            "max_perplexity": max(ppls),
            "mean_perplexity": sum(ppls) / len(ppls),
            "total_steps": len(self._records),
            "moving_average_100": self.get_moving_average(window=100),
        }

    def per_language_summary(self) -> dict[str, dict]:
        """Return a per-language breakdown of perplexity statistics.

        Returns:
            Dict mapping language name -> summary dict with the same keys
            as `summary()`, but scoped to that language's records only.
            Languages with no records are excluded.
        """
        result: dict[str, dict] = {}
        for lang, records in self._lang_records.items():
            if not records:
                continue
            ppls = [r.perplexity for r in records]
            result[lang] = {
                "current_perplexity": records[-1].perplexity,
                "min_perplexity": min(ppls),
                "max_perplexity": max(ppls),
                "mean_perplexity": sum(ppls) / len(ppls),
                "total_steps": len(records),
            }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def perplexity_to_level(ppl: float) -> tuple[str, str]:
        """Return (description, severity_color) for a perplexity value.

        Args:
            ppl: Perplexity value.

        Returns:
            Tuple of (human-readable description, color string).
        """
        if ppl > 1000:
            return "barely learning", "red"
        elif ppl > 200:
            return "early training", "red"
        elif ppl > 100:
            return "below average", "yellow"
        elif ppl > 50:
            return "reasonable", "yellow"
        elif ppl > 20:
            return "good", "green"
        elif ppl > 10:
            return "very good", "green"
        else:
            return "excellent / check for overfitting", "cyan"

    def __repr__(self) -> str:
        n = len(self._records)
        current = self._records[-1].perplexity if self._records else None
        langs = list(self._lang_records.keys())
        return (
            f"PerplexityTracker(records={n}, current_ppl={current}, languages={langs})"
        )
