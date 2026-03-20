"""Generation Constraints: modify logits before sampling to enforce constraints.

Provides a composable constraint system that operates on raw logits at each
generation step. Constraints are pure logit transformers — they receive the
current logit vector and the list of already-generated token IDs, and return
a modified logit vector.

For a TS dev: think of this as a middleware pipeline (like Express.js) where
each constraint is a middleware that can modify the "request" (logits) before
the final "handler" (sampling) runs.

Design follows the plan in research/suggested-features/40-generation-constraints.plan.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import torch

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """Abstract base for all logit-modifying constraints.

    Each constraint receives:
        logits           -- 1-D float tensor of shape (vocab_size,)
        generated_tokens -- list of token IDs generated so far

    and returns a modified logit tensor (same shape).  Constraints must NOT
    sample; they only transform the distribution.
    """

    @abstractmethod
    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        """Apply this constraint and return the modified logits."""
        ...


# ---------------------------------------------------------------------------
# MaxLengthConstraint
# ---------------------------------------------------------------------------


class MaxLengthConstraint(Constraint):
    """Force the model to emit EOS once the generated sequence reaches max_tokens.

    Sets all logits to -inf except the EOS token, guaranteeing the next
    sampled token is EOS when the limit is hit.

    Args:
        max_tokens: Maximum number of tokens before forcing EOS.
        eos_token_id: The EOS token index in the vocabulary (default 0).
    """

    def __init__(self, max_tokens: int, eos_token_id: int = 0) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        if len(generated_tokens) >= self.max_tokens:
            # Force EOS: mask every token except EOS to -inf
            forced = torch.full_like(logits, float("-inf"))
            forced[self.eos_token_id] = 0.0  # arbitrary finite value
            return forced
        return logits


# ---------------------------------------------------------------------------
# BannedTokenConstraint
# ---------------------------------------------------------------------------


class BannedTokenConstraint(Constraint):
    """Permanently set logits to -inf for a set of banned token IDs.

    This is a hard constraint — banned tokens can never be sampled regardless
    of context.  For soft penalties, use RepetitionPenalty instead.

    Args:
        token_ids: Iterable of vocabulary indices to ban.
    """

    def __init__(self, token_ids: Sequence[int]) -> None:
        self.token_ids = list(token_ids)

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        if not self.token_ids:
            return logits
        vocab_size = logits.shape[0]
        # Build index tensor on the same device as logits
        ids = torch.tensor(
            [t for t in self.token_ids if 0 <= t < vocab_size],
            dtype=torch.long,
            device=logits.device,
        )
        if ids.numel() > 0:
            logits[ids] = float("-inf")
        return logits


# ---------------------------------------------------------------------------
# RequiredTokenConstraint
# ---------------------------------------------------------------------------


class RequiredTokenConstraint(Constraint):
    """Force a specific token at a specific generation position.

    At position ``by_position``, every token except the required one is set
    to -inf, guaranteeing that token is chosen.  Outside that position the
    logits are returned unchanged.

    Args:
        token_ids:   Mapping from position -> required token ID, OR a list of
                     token IDs where index == position.  Accepts both forms:
                       - dict: {0: 50256, 5: 198}
                       - list: [50256, None, None, None, None, 198]  (None = no constraint)
        by_position: When token_ids is a plain list of token IDs (not a dict),
                     interpret the list as the sequence of required tokens
                     starting at position 0 (None entries are skipped).
                     Ignored when token_ids is already a dict.
    """

    def __init__(
        self,
        token_ids: Sequence[Optional[int]] | dict,
        by_position: bool = True,
    ) -> None:
        if isinstance(token_ids, dict):
            self._map: dict[int, int] = {int(k): int(v) for k, v in token_ids.items()}
        else:
            # List form: index is the position, value is the required token (None = skip)
            self._map = {
                pos: int(tid)
                for pos, tid in enumerate(token_ids)
                if tid is not None
            }

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        pos = len(generated_tokens)
        required_id = self._map.get(pos)
        if required_id is None:
            return logits
        vocab_size = logits.shape[0]
        if not (0 <= required_id < vocab_size):
            return logits
        # Force required token
        forced = torch.full_like(logits, float("-inf"))
        forced[required_id] = 0.0
        return forced


# ---------------------------------------------------------------------------
# RepetitionPenalty
# ---------------------------------------------------------------------------


class RepetitionPenalty(Constraint):
    """Penalise tokens that appear in the recent generation window.

    Follows the Ctrl/RepetitionPenalty convention:
        logit /= penalty   if logit > 0
        logit *= penalty   if logit < 0

    This pushes the token's probability down without hard-banning it.

    Args:
        penalty: Penalty factor (> 1.0 = penalise, 1.0 = no-op, < 1.0 = boost).
        window:  Number of most-recent tokens to consider (0 = all).
    """

    def __init__(self, penalty: float = 1.3, window: int = 64) -> None:
        if penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {penalty}")
        self.penalty = penalty
        self.window = window

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        if self.penalty == 1.0 or not generated_tokens:
            return logits

        recent = generated_tokens if self.window == 0 else generated_tokens[-self.window :]
        vocab_size = logits.shape[0]
        unique_ids = torch.tensor(
            list({t for t in recent if 0 <= t < vocab_size}),
            dtype=torch.long,
            device=logits.device,
        )
        if unique_ids.numel() == 0:
            return logits

        score = logits[unique_ids]
        # Standard repetition penalty from the HuggingFace implementation
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        logits[unique_ids] = score
        return logits


# ---------------------------------------------------------------------------
# TemperatureSchedule
# ---------------------------------------------------------------------------


class TemperatureSchedule(Constraint):
    """Linearly anneal temperature from start_temp to end_temp over total_steps.

    Applies temperature scaling to the logits (divides by current temperature).
    High temperature = flatter distribution (more creative).
    Low temperature  = peakier distribution (more deterministic).

    The ``apply`` method uses an internal step counter that increments on every
    call.  You can also call ``get_temperature(step)`` directly and apply the
    scaling yourself.

    Args:
        start_temp:  Temperature at step 0.
        end_temp:    Temperature at step total_steps.
        total_steps: Number of steps for the full annealing schedule.
    """

    def __init__(
        self,
        start_temp: float = 1.0,
        end_temp: float = 0.1,
        total_steps: int = 200,
    ) -> None:
        if start_temp <= 0 or end_temp <= 0:
            raise ValueError("Temperatures must be > 0")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self._step: int = 0

    def get_temperature(self, step: int) -> float:
        """Return the temperature for a given absolute step index."""
        t = min(step, self.total_steps) / self.total_steps  # in [0, 1]
        return self.start_temp + t * (self.end_temp - self.start_temp)

    def reset(self) -> None:
        """Reset the internal step counter (call between independent generations)."""
        self._step = 0

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        temp = self.get_temperature(self._step)
        self._step += 1
        if abs(temp - 1.0) < 1e-6:
            return logits
        return logits / temp


# ---------------------------------------------------------------------------
# ConstraintChain
# ---------------------------------------------------------------------------


class ConstraintChain(Constraint):
    """Apply a sequence of constraints in order, left to right.

    Equivalent to function composition:
        result = c_n( ... c_2( c_1(logits, tokens) ) ... )

    Args:
        constraints: Ordered list of Constraint instances.
    """

    def __init__(self, constraints: List[Constraint]) -> None:
        self.constraints = list(constraints)

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        for constraint in self.constraints:
            logits = constraint.apply(logits, generated_tokens)
        return logits

    def append(self, constraint: Constraint) -> "ConstraintChain":
        """Return a new chain with constraint appended (non-mutating)."""
        return ConstraintChain(self.constraints + [constraint])


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def apply_constraints(
    logits: torch.Tensor,
    constraints: List[Constraint],
    generated: List[int],
) -> torch.Tensor:
    """Apply a list of constraints to logits and return the result.

    Equivalent to ``ConstraintChain(constraints).apply(logits, generated)``
    but avoids constructing a ConstraintChain object when you just want a
    one-shot application.

    Args:
        logits:      1-D float tensor of shape (vocab_size,).
        constraints: List of Constraint instances to apply in order.
        generated:   Token IDs generated so far.

    Returns:
        Modified logit tensor (same shape as input).
    """
    for constraint in constraints:
        logits = constraint.apply(logits, generated)
    return logits
