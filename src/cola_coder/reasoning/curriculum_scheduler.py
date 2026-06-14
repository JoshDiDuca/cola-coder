"""MODEL-042: verifier-effort Easy→Hard (E2H) curriculum scheduler.

2026 curriculum RL (E2H, arXiv:2506.06632) shows easy→hard scheduling helps but
mastered tasks must be FADED OUT or they cause overfitting. The hard part is a
DIFFICULTY metric that is model-relative and EVOLVES as the model learns — which
cola-coder uniquely has: the sandbox verifier's per-problem pass-rate (EVAL-026).

This scheduler tracks each problem's verified pass-rate across epochs and, between
epochs, (1) re-tags its difficulty tier from the LATEST measured rate (so IDEA-020's
per-difficulty entropy floors + curriculum temperature use measured, not static,
difficulty), and (2) FADES OUT problems the model has mastered (rate ≥ threshold for
a streak of epochs), keeping at least ``min_active`` so the set never empties — the
freed budget then concentrates on the frontier (mid pass-rate) problems.

Pure logic — no torch/model. The GRPO trainer calls ``record`` per step,
``end_epoch`` after each epoch, then re-tags + filters with ``tier_for`` / ``active``.
"""

from __future__ import annotations

from typing import Callable, Iterable


class VerifierEffortCurriculum:
    """Track per-problem verified pass-rate and drive an E2H fade-out schedule.

    Args:
        mastery_threshold: pass-rate at/above which a problem counts as mastered.
        mastery_streak: consecutive epochs at mastery before a problem is faded.
        min_active: never fade below this many active problems (keep the hardest).
        easy_below: latest-rate boundary for the "easy" tier (>= this).
        hard_above: latest-rate boundary for the "hard" tier (<= this).
    """

    def __init__(
        self,
        mastery_threshold: float = 0.9,
        mastery_streak: int = 2,
        min_active: int = 4,
        easy_below: float = 0.8,
        hard_above: float = 0.2,
    ) -> None:
        if not 0.0 < mastery_threshold <= 1.0:
            raise ValueError("mastery_threshold must be in (0, 1]")
        if mastery_streak < 1:
            raise ValueError("mastery_streak must be >= 1")
        self.mastery_threshold = mastery_threshold
        self.mastery_streak = mastery_streak
        self.min_active = max(1, min_active)
        self.easy_below = easy_below
        self.hard_above = hard_above
        self._history: dict[str, list[float]] = {}   # key -> per-epoch mean rates
        self._epoch_sum: dict[str, float] = {}        # current-epoch accumulators
        self._epoch_cnt: dict[str, int] = {}

    def record(self, key: str, pass_rate: float) -> None:
        """Record one step's verified pass-rate for a problem (within the epoch)."""
        self._epoch_sum[key] = self._epoch_sum.get(key, 0.0) + pass_rate
        self._epoch_cnt[key] = self._epoch_cnt.get(key, 0) + 1

    def end_epoch(self) -> None:
        """Finalize this epoch's mean pass-rate per problem into history."""
        for key, total in self._epoch_sum.items():
            mean = total / max(1, self._epoch_cnt[key])
            self._history.setdefault(key, []).append(mean)
        self._epoch_sum.clear()
        self._epoch_cnt.clear()

    def latest_rate(self, key: str) -> float | None:
        rates = self._history.get(key)
        return rates[-1] if rates else None

    def tier_for(self, key: str) -> str:
        """Difficulty tier from the latest measured pass-rate (medium if unseen)."""
        rate = self.latest_rate(key)
        if rate is None:
            return "medium"
        if rate >= self.easy_below:
            return "easy"
        if rate <= self.hard_above:
            return "hard"
        return "medium"

    def is_mastered(self, key: str) -> bool:
        """True once the problem has held >= mastery_threshold for the streak length."""
        rates = self._history.get(key, [])
        if len(rates) < self.mastery_streak:
            return False
        return all(r >= self.mastery_threshold for r in rates[-self.mastery_streak:])

    def active(self, problems: Iterable, key_fn: Callable[[object], str]) -> list:
        """E2H fade-out: drop mastered problems, keeping at least ``min_active``.

        If fading would drop below ``min_active``, the LEAST-mastered of the would-be-
        faded problems (lowest latest rate) are re-included to fill the floor — so the
        set keeps the hardest material and never empties.
        """
        problems = list(problems)
        kept = [p for p in problems if not self.is_mastered(key_fn(p))]
        if len(kept) >= self.min_active or len(problems) <= self.min_active:
            return kept if kept else problems[: self.min_active]
        # Re-include the least-mastered faded problems to reach min_active.
        faded = [p for p in problems if self.is_mastered(key_fn(p))]
        faded.sort(key=lambda p: self.latest_rate(key_fn(p)) or 0.0)
        need = self.min_active - len(kept)
        return kept + faded[:need]
