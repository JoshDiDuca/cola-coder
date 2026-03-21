"""Training Plateau Detector: detect plateaus in loss history and suggest fixes.

Analyses a sequence of training loss values and:
  - Detects plateau regions using rolling-average comparison
  - Computes trend slope to distinguish improvement from stagnation
  - Detects oscillation (loss bouncing without net improvement)
  - Suggests remediation actions based on plateau characteristics

Suggested actions:
  - reduce_lr: learning rate is too large (oscillation) or annealing needed
  - increase_batch: gradient noise may be preventing escape
  - change_data_mix: model may have exhausted useful signal in current data
  - add_regularisation: possible overfitting when train loss plateaus but was falling
  - wait: short plateau, may be temporary

For a TS dev: like watching a performance metric that has flatlined and
diagnosing whether it's a true plateau or just noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------


class PlateauAction(str, Enum):
    REDUCE_LR = "reduce_lr"
    INCREASE_BATCH = "increase_batch"
    CHANGE_DATA_MIX = "change_data_mix"
    ADD_REGULARISATION = "add_regularisation"
    WAIT = "wait"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlateauRegion:
    """A detected plateau segment."""

    start_step: int
    end_step: int
    mean_loss: float
    loss_std: float  # standard deviation within the region

    @property
    def length(self) -> int:
        return self.end_step - self.start_step + 1

    @property
    def is_oscillating(self) -> bool:
        """High std relative to mean suggests oscillation."""
        if self.mean_loss == 0:
            return False
        return (self.loss_std / self.mean_loss) > 0.02


@dataclass
class PlateauReport:
    """Results of plateau detection."""

    is_plateau: bool
    plateaus: list[PlateauRegion] = field(default_factory=list)
    current_trend: float = 0.0  # negative = improving, positive = worsening
    is_oscillating: bool = False
    suggested_actions: list[PlateauAction] = field(default_factory=list)
    window_size: int = 20

    @property
    def longest_plateau(self) -> PlateauRegion | None:
        if not self.plateaus:
            return None
        return max(self.plateaus, key=lambda p: p.length)

    @property
    def summary(self) -> str:
        status = "PLATEAU" if self.is_plateau else "improving"
        actions = [a.value for a in self.suggested_actions] if self.suggested_actions else ["none"]
        return (
            f"Status: {status}, trend={self.current_trend:+.4f}/step, "
            f"oscillating={self.is_oscillating}, "
            f"actions: {', '.join(actions)}"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PlateauDetector:
    """Detect training plateaus in a sequence of loss values."""

    def __init__(
        self,
        window: int = 20,
        plateau_threshold: float = 1e-4,
        min_plateau_len: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        window:
            Rolling window size for computing averages.
        plateau_threshold:
            Maximum average per-step improvement to be considered a plateau.
        min_plateau_len:
            Minimum consecutive steps below threshold to declare a plateau.
        """
        self.window = window
        self.plateau_threshold = plateau_threshold
        self.min_plateau_len = min_plateau_len

    def detect(self, loss_history: list[float]) -> PlateauReport:
        """Analyse *loss_history* and return a :class:`PlateauReport`."""
        n = len(loss_history)
        if n < max(self.window, self.min_plateau_len * 2):
            return PlateauReport(
                is_plateau=False,
                window_size=self.window,
                suggested_actions=[PlateauAction.WAIT],
            )

        rolling_avgs = self._rolling_average(loss_history, self.window)
        trend = self._compute_trend(rolling_avgs[-self.window :])
        plateaus = self._find_plateaus(rolling_avgs)
        oscillating = self._detect_oscillation(loss_history[-self.window :])

        is_plateau = len(plateaus) > 0 and plateaus[-1].end_step >= len(rolling_avgs) - 5

        actions = self._suggest_actions(
            is_plateau=is_plateau,
            trend=trend,
            oscillating=oscillating,
            plateaus=plateaus,
        )

        return PlateauReport(
            is_plateau=is_plateau,
            plateaus=plateaus,
            current_trend=trend,
            is_oscillating=oscillating,
            suggested_actions=actions,
            window_size=self.window,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_average(values: list[float], window: int) -> list[float]:
        avgs: list[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            avgs.append(sum(values[start : i + 1]) / (i - start + 1))
        return avgs

    @staticmethod
    def _compute_trend(values: list[float]) -> float:
        """Linear regression slope over the last window of rolling averages."""
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return num / denom

    def _find_plateaus(self, rolling_avgs: list[float]) -> list[PlateauRegion]:
        """Find contiguous segments where improvement is below threshold."""
        plateaus: list[PlateauRegion] = []
        n = len(rolling_avgs)
        in_plateau = False
        start = 0

        for i in range(1, n):
            improvement = rolling_avgs[i - 1] - rolling_avgs[i]
            if improvement < self.plateau_threshold:
                if not in_plateau:
                    in_plateau = True
                    start = i - 1
            else:
                if in_plateau:
                    length = i - start
                    if length >= self.min_plateau_len:
                        region_vals = rolling_avgs[start:i]
                        plateaus.append(self._make_region(start, i - 1, region_vals))
                    in_plateau = False

        # Handle plateau that extends to end
        if in_plateau:
            length = n - start
            if length >= self.min_plateau_len:
                region_vals = rolling_avgs[start:]
                plateaus.append(self._make_region(start, n - 1, region_vals))

        return plateaus

    @staticmethod
    def _make_region(start: int, end: int, vals: list[float]) -> PlateauRegion:
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = variance ** 0.5
        return PlateauRegion(start_step=start, end_step=end, mean_loss=mean, loss_std=std)

    @staticmethod
    def _detect_oscillation(recent: list[float]) -> bool:
        """Detect if loss is oscillating (alternating up/down) without improvement."""
        if len(recent) < 4:
            return False
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        sign_changes = sum(
            1 for i in range(1, len(changes)) if (changes[i] > 0) != (changes[i - 1] > 0)
        )
        return sign_changes / max(len(changes), 1) > 0.6

    @staticmethod
    def _suggest_actions(
        is_plateau: bool,
        trend: float,
        oscillating: bool,
        plateaus: list[PlateauRegion],
    ) -> list[PlateauAction]:
        actions: list[PlateauAction] = []

        if not is_plateau and trend < -1e-5:
            return [PlateauAction.NONE]

        if oscillating:
            actions.append(PlateauAction.REDUCE_LR)

        if is_plateau:
            if plateaus and plateaus[-1].length > 50:
                actions.append(PlateauAction.CHANGE_DATA_MIX)
            if not oscillating:
                actions.append(PlateauAction.REDUCE_LR)
            if plateaus and plateaus[-1].length > 20:
                actions.append(PlateauAction.INCREASE_BATCH)

        if not actions:
            actions.append(PlateauAction.WAIT)

        return actions
