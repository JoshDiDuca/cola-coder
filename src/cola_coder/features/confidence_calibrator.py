"""Confidence Calibrator — Feature 92

Maps raw model logit scores to well-calibrated probabilities.

Key capabilities
----------------
- Temperature scaling: divide logits by T before softmax so that the
  resulting probabilities better match empirical accuracy.
- Expected Calibration Error (ECE): bucket predictions by confidence,
  compare average confidence to average accuracy in each bucket, and
  compute the weighted mean absolute difference.
- Reliability diagram data: return (confidence_bins, accuracy_bins,
  counts) so callers can plot calibration curves.

All computations are pure Python / math — no PyTorch dependency required
so tests run without GPU.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if confidence calibration is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Math helpers (no torch dependency)
# ---------------------------------------------------------------------------


def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Compute softmax over *logits* with optional *temperature* scaling."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [x / temperature for x in logits]
    max_v = max(scaled)
    exps = [math.exp(x - max_v) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _logsoftmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Compute log-softmax over *logits* with optional *temperature* scaling."""
    probs = _softmax(logits, temperature)
    return [math.log(max(p, 1e-45)) for p in probs]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    """Statistics for a single confidence bucket."""

    lower: float
    upper: float
    count: int = 0
    total_confidence: float = 0.0
    total_correct: float = 0.0

    @property
    def avg_confidence(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_confidence / self.count

    @property
    def accuracy(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_correct / self.count

    @property
    def calibration_gap(self) -> float:
        """|avg_confidence - accuracy|."""
        return abs(self.avg_confidence - self.accuracy)


@dataclass
class CalibrationResult:
    """Full calibration analysis output."""

    ece: float
    temperature: float
    n_bins: int
    bins: list[CalibrationBin]
    n_samples: int
    overconfidence: float  # fraction of samples where confidence > accuracy
    underconfidence: float  # fraction where confidence < accuracy

    def summary(self) -> dict[str, float]:
        return {
            "ece": self.ece,
            "temperature": self.temperature,
            "n_samples": float(self.n_samples),
            "overconfidence_frac": self.overconfidence,
            "underconfidence_frac": self.underconfidence,
        }


# ---------------------------------------------------------------------------
# Core calibrator
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Calibrate model output confidence using temperature scaling + ECE."""

    def __init__(self, temperature: float = 1.0, n_bins: int = 10) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.temperature = temperature
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------

    def logits_to_probs(self, logits: list[float]) -> list[float]:
        """Convert raw logits to calibrated probabilities."""
        return _softmax(logits, self.temperature)

    def logits_to_log_probs(self, logits: list[float]) -> list[float]:
        """Convert raw logits to calibrated log-probabilities."""
        return _logsoftmax(logits, self.temperature)

    def top_confidence(self, logits: list[float]) -> tuple[int, float]:
        """Return (argmax_index, max_probability) after temperature scaling."""
        probs = self.logits_to_probs(logits)
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return idx, probs[idx]

    # ------------------------------------------------------------------
    # ECE
    # ------------------------------------------------------------------

    def compute_ece(
        self,
        confidences: list[float],
        correctness: list[float],
    ) -> CalibrationResult:
        """Compute Expected Calibration Error.

        Parameters
        ----------
        confidences:
            Per-sample predicted confidence in [0, 1].
        correctness:
            Per-sample indicator of whether prediction was correct (0 or 1,
            or a soft value in [0, 1]).

        Returns
        -------
        CalibrationResult
        """
        if len(confidences) != len(correctness):
            raise ValueError("confidences and correctness must have the same length")
        n = len(confidences)
        if n == 0:
            return CalibrationResult(
                ece=0.0,
                temperature=self.temperature,
                n_bins=self.n_bins,
                bins=[],
                n_samples=0,
                overconfidence=0.0,
                underconfidence=0.0,
            )

        # Build bins
        bin_edges = [i / self.n_bins for i in range(self.n_bins + 1)]
        bins: list[CalibrationBin] = [
            CalibrationBin(lower=bin_edges[i], upper=bin_edges[i + 1])
            for i in range(self.n_bins)
        ]

        over = 0
        under = 0
        for conf, corr in zip(confidences, correctness):
            # Clip confidence to [0, 1]
            conf = max(0.0, min(1.0, conf))
            b_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[b_idx].count += 1
            bins[b_idx].total_confidence += conf
            bins[b_idx].total_correct += corr
            if conf > corr:
                over += 1
            elif conf < corr:
                under += 1

        ece = sum(
            (b.count / n) * b.calibration_gap for b in bins if b.count > 0
        )

        return CalibrationResult(
            ece=ece,
            temperature=self.temperature,
            n_bins=self.n_bins,
            bins=bins,
            n_samples=n,
            overconfidence=over / n,
            underconfidence=under / n,
        )

    # ------------------------------------------------------------------
    # Temperature search
    # ------------------------------------------------------------------

    def find_temperature(
        self,
        confidences: list[float],
        correctness: list[float],
        temps: Optional[list[float]] = None,
    ) -> float:
        """Grid-search for the temperature that minimises ECE.

        Parameters
        ----------
        confidences:
            Per-sample max-prob predictions.
        correctness:
            Per-sample correctness indicators.
        temps:
            Candidate temperature values to try.  Defaults to a log-spaced
            grid from 0.1 to 5.0.

        Returns
        -------
        float
            The temperature with the lowest ECE.
        """
        if temps is None:
            temps = [0.1 * 2**i for i in range(6)]  # 0.1 .. 3.2

        best_t = self.temperature
        best_ece = float("inf")
        original_t = self.temperature

        for t in temps:
            self.temperature = t
            result = self.compute_ece(confidences, correctness)
            if result.ece < best_ece:
                best_ece = result.ece
                best_t = t

        self.temperature = original_t  # restore
        return best_t

    # ------------------------------------------------------------------
    # Reliability diagram data
    # ------------------------------------------------------------------

    def reliability_diagram_data(
        self,
        confidences: list[float],
        correctness: list[float],
    ) -> dict[str, list[float]]:
        """Return data for a reliability (calibration) diagram.

        Returns a dict with keys ``bin_centers``, ``accuracies``,
        ``avg_confidences``, and ``counts``.
        """
        result = self.compute_ece(confidences, correctness)
        centers = [(b.lower + b.upper) / 2 for b in result.bins]
        accuracies = [b.accuracy for b in result.bins]
        avg_confs = [b.avg_confidence for b in result.bins]
        counts = [float(b.count) for b in result.bins]
        return {
            "bin_centers": centers,
            "accuracies": accuracies,
            "avg_confidences": avg_confs,
            "counts": counts,
        }
