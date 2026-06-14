"""IDEA-013: entropy-gated closed-loop clip controller for GRPO/RLVR.

Entropy collapse is the dominant RLVR failure mode (the policy converges to a
near-deterministic argmax, killing exploration). MODEL-037 made policy entropy
OBSERVABLE; this closes the loop — a controller that adjusts the DAPO clip-higher
bound from the live entropy signal:

- When measured entropy falls below a target floor, RAISE ``clip_high`` (DAPO
  clip-higher) so low-probability tokens have more room to grow → more exploration
  → higher entropy. This is the empirically-grounded direction (DAPO uses
  clip_high 0.28 > clip_low 0.2 precisely to counteract collapse).
- When entropy is healthy, relax ``clip_high`` back to its base (no windup).

VERIFIER-AWARE (cola-coder's rare asset): only inject exploration when the verifier
isn't already satisfied. If the sandbox pass-rate is at/above ``pass_rate_ceiling``,
the policy is solving the task — don't spend entropy budget exploring away from
working solutions. This couples RL exploration to executable success, which the
RLVR entropy papers (no verifier) cannot do.

Proportional control only (no integral term) so there is no windup and the mapping
from entropy deficit → clip_high is transparent and bounded. ``clip_low`` is held
fixed; ``clip_high`` is the modulated DAPO lever.

Pure logic — no torch, no model. The GRPOTrainer calls ``update`` once per
non-skipped step and feeds the returned ``clip_high`` into the next step's clip.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EntropyClipController:
    """Adjust the GRPO clip-higher bound from the measured policy entropy.

    Args:
        target_entropy: Entropy floor (nats). Below it, exploration is injected.
        clip_low: Fixed lower clip (1 - clip_low); not modulated.
        clip_high: Base upper clip (1 + clip_high) — the relaxed/healthy value.
        max_clip_high: Hard cap on the raised upper clip.
        gain: Proportional gain (clip_high units per nat of entropy deficit).
        pass_rate_ceiling: If the group pass-rate is >= this, treat the verifier as
            satisfied and do NOT inject exploration (relax to base). 1.0 disables
            the gate (always control on entropy alone).
        difficulty_floors: IDEA-020 — optional per-difficulty entropy floors, e.g.
            ``{"easy": 0.3, "medium": 0.7, "hard": 1.2}``. When ``update`` receives a
            ``difficulty``, the floor for that tier is used instead of
            ``target_entropy`` (which remains the fallback). Hard problems (need
            search) get a HIGHER floor → more clip-higher exploration; easy/solved
            tiers get a LOWER floor → exploit. The GRPO curriculum is staged
            (easy→hard), so consecutive steps share a tier and the next-step clip is
            appropriate. None (default) = single ``target_entropy`` for all tiers.
    """

    target_entropy: float
    clip_low: float = 0.2
    clip_high: float = 0.28
    max_clip_high: float = 0.40
    gain: float = 0.5
    pass_rate_ceiling: float = 0.9
    difficulty_floors: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.target_entropy < 0:
            raise ValueError(f"target_entropy must be >= 0, got {self.target_entropy}")
        if self.max_clip_high < self.clip_high:
            raise ValueError(
                f"max_clip_high ({self.max_clip_high}) < base clip_high "
                f"({self.clip_high})"
            )
        if self.gain < 0:
            raise ValueError(f"gain must be >= 0, got {self.gain}")
        if self.difficulty_floors is not None:
            for tier, floor in self.difficulty_floors.items():
                if floor < 0:
                    raise ValueError(
                        f"difficulty_floors[{tier!r}] must be >= 0, got {floor}"
                    )
        # Current (modulated) upper clip — starts at base.
        self._current_high = self.clip_high

    @property
    def current_clip_high(self) -> float:
        return self._current_high

    def floor_for(self, difficulty: str | None) -> float:
        """The entropy floor for a difficulty tier (target_entropy fallback)."""
        if self.difficulty_floors and difficulty in self.difficulty_floors:
            return self.difficulty_floors[difficulty]
        return self.target_entropy

    def update(
        self,
        measured_entropy: float,
        pass_rate: float = 0.0,
        difficulty: str | None = None,
    ) -> tuple[float, float]:
        """Return ``(clip_low, clip_high)`` for the NEXT GRPO step.

        Raises clip_high proportionally to the deficit below this tier's entropy
        floor, but only when the verifier is unsatisfied; otherwise relaxes to base.
        """
        deficit = self.floor_for(difficulty) - measured_entropy
        verifier_satisfied = pass_rate >= self.pass_rate_ceiling
        if deficit <= 0.0 or verifier_satisfied:
            self._current_high = self.clip_high  # healthy / solved → relax to base
        else:
            raised = self.clip_high + self.gain * deficit
            self._current_high = min(raised, self.max_clip_high)
        return self.clip_low, self._current_high
