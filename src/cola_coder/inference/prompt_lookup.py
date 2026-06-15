"""Prompt-lookup (n-gram) speculative drafting + offline acceptance analysis (INFER-035).

Prompt Lookup Decoding (PLD, apoorvumang 2023) is a *draft-free*, single-model
speculative-decoding drafter: instead of running a small draft model, it proposes
the next few tokens by string-matching the most-recently-generated n-gram against
EARLIER occurrences in the running context (prompt + tokens emitted so far) and
copying the tokens that followed that earlier match. It is lossless — the target
model still verifies every proposed token, so the output is byte-identical to greedy
decoding; only the wall-clock changes. REST (NAACL 2024) generalises the same idea to
an external (context, continuation) datastore. Code generation is an unusually good
fit: identifiers, imports, and boilerplate recur verbatim within a file, so the
running buffer is itself a high-yield draft source.

This module is the OFFLINE half. It contains:

* :class:`PromptLookupDrafter` — the pure drafter (what tokens would PLD propose at a
  given context?). No model, no logits — it is exact string matching over token ids.
* :func:`analyze_acceptance` — replays a KNOWN full token sequence (a recorded
  generation trace, or any reference completion) and, at each position, asks the
  drafter what it would propose, then counts how many of those proposed tokens match
  the ground-truth continuation. That accepted-length distribution is exactly the
  statistic the 2026 speculative-decoding benchmark literature says you must measure
  BEFORE wiring a drafter into the hot path: acceptance rate and mean accepted length
  are the first-order predictors of speedup, and PLD's tend to be high-acceptance but
  low-diversity, so this lets you size the win on a real corpus first.

MAIN-SAFE: pure-Python integer/string analysis over token-id lists. No torch, no model
load, no GPU, no checkpoint, no network. It mirrors the upstream PLD reference
(``max_ngram_size`` down to 1, ``num_pred_tokens`` continuation) so the offline numbers
predict a faithful live implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Drafter configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PromptLookupConfig:
    """Knobs for the prompt-lookup drafter (mirrors the PLD reference defaults).

    Attributes:
        max_ngram_size: Longest suffix n-gram to try matching. The drafter searches
            from this size down to 1; longer matches are more specific (fewer false
            continuations) but rarer, so trying a descending ladder maximises hit rate.
        num_pred_tokens: Maximum number of continuation tokens to propose per draft.
        min_ngram_size: Shortest suffix n-gram the drafter is allowed to match on.
            ``1`` reproduces the upstream behaviour; raising it suppresses the noisy
            single-token matches that rarely survive verification.
    """

    max_ngram_size: int = 3
    num_pred_tokens: int = 10
    min_ngram_size: int = 1

    def __post_init__(self) -> None:
        if self.max_ngram_size < 1:
            raise ValueError(f"max_ngram_size must be >= 1, got {self.max_ngram_size}")
        if self.min_ngram_size < 1:
            raise ValueError(f"min_ngram_size must be >= 1, got {self.min_ngram_size}")
        if self.min_ngram_size > self.max_ngram_size:
            raise ValueError(
                f"min_ngram_size ({self.min_ngram_size}) must be <= "
                f"max_ngram_size ({self.max_ngram_size})"
            )
        if self.num_pred_tokens < 1:
            raise ValueError(f"num_pred_tokens must be >= 1, got {self.num_pred_tokens}")


class PromptLookupDrafter:
    """Draft-free PLD drafter: propose next tokens by matching the context's own n-grams.

    The drafter holds no model state; :meth:`draft` is a pure function of the supplied
    context. It is reused both live (a real generator would call :meth:`draft` each
    step) and offline (the acceptance analyzer drives it over a recorded trace).
    """

    def __init__(self, config: PromptLookupConfig | None = None) -> None:
        """Create a drafter.

        Args:
            config: Drafting knobs; defaults to the PLD reference configuration.
        """
        self._config = config or PromptLookupConfig()

    @property
    def config(self) -> PromptLookupConfig:
        """The drafting configuration in effect."""
        return self._config

    def draft(self, context: list[int]) -> list[int]:
        """Propose continuation tokens for ``context`` by prompt n-gram lookup.

        Tries suffix n-grams from ``max_ngram_size`` down to ``min_ngram_size``. For a
        size ``n`` it takes the last ``n`` tokens as a needle and scans EARLIER
        positions for the most recent identical window; the tokens immediately
        following that earlier window become the draft. The most recent earlier match
        is preferred (locality: nearby code is the best predictor of nearby code).

        Args:
            context: Token ids generated/seen so far (prompt + completion). The suffix
                is the search needle; matches are sought strictly before it.

        Returns:
            Up to ``num_pred_tokens`` proposed token ids, or an empty list when no
            earlier n-gram of any tried size matches (the live decoder then falls back
            to a normal single-token step).
        """
        cfg = self._config
        n_ctx = len(context)
        # Need at least one token before the needle to have a continuation to copy.
        for size in range(min(cfg.max_ngram_size, n_ctx - 1), cfg.min_ngram_size - 1, -1):
            if size < 1:
                break
            needle = context[n_ctx - size :]
            match_end = self._find_latest_match_end(context, needle)
            if match_end is None:
                continue
            draft = context[match_end : match_end + cfg.num_pred_tokens]
            if draft:
                return draft
        return []

    @staticmethod
    def _find_latest_match_end(context: list[int], needle: list[int]) -> int | None:
        """Return the end index of the LATEST occurrence of ``needle`` before its own suffix.

        Searches occurrences of ``needle`` that start strictly before the final
        (suffix) occurrence and returns the index just past the most recent such match
        — i.e. where its copied continuation begins. ``None`` when there is no earlier
        occurrence.
        """
        size = len(needle)
        # The needle itself sits at context[-size:]; only search positions before it.
        last_start = len(context) - size
        for start in range(last_start - 1, -1, -1):
            if context[start : start + size] == needle:
                return start + size
        return None


# ── Offline acceptance analysis ───────────────────────────────────────────────


@dataclass(frozen=True)
class StepAcceptance:
    """One verification step of a simulated speculative decode over a reference trace.

    Attributes:
        position: Index in the reference sequence of the first token being predicted
            this step (the position the live decoder would be emitting).
        drafted: Number of tokens the drafter proposed (0 when it found no match).
        accepted: Number of leading drafted tokens that matched the ground-truth
            continuation (prefix-match length; PLD accepts a prefix and stops at the
            first divergence).
    """

    position: int
    drafted: int
    accepted: int


@dataclass(frozen=True)
class AcceptanceReport:
    """Aggregate acceptance statistics from replaying a drafter over a reference trace.

    All ratios are in ``[0, 1]``; ``mean_accepted_length`` and ``speedup_estimate`` are
    in ``[0, num_pred_tokens]`` and ``[1, num_pred_tokens + 1]`` respectively.

    Attributes:
        total_tokens: Reference tokens that were subject to prediction (sequence length
            minus the seed prefix that primes the context).
        decode_steps: Number of verification steps the simulated decode took. With a
            perfect drafter this approaches ``total_tokens / (num_pred_tokens + 1)``;
            with no matches it equals ``total_tokens`` (one token per step).
        baseline_steps: Steps a non-speculative decode would take (== ``total_tokens``),
            i.e. the reference point ``speedup_estimate`` is measured against.
        total_drafted: Sum of tokens proposed across all steps.
        total_accepted: Sum of drafted tokens accepted across all steps.
        draft_hit_rate: Fraction of steps where the drafter proposed at least one token.
        acceptance_rate: ``total_accepted / total_drafted`` — of the tokens PLD
            proposed, how many survived verification (the classic acceptance α).
        mean_accepted_length: Average accepted tokens per step (the speedup-relevant
            statistic: how many tokens the target model gets "for free" per step).
        speedup_estimate: ``baseline_steps / decode_steps`` — idealised step-count
            speedup assuming verification of a draft is one model step (the standard
            speculative-decoding upper bound; ignores per-step overhead, which the
            benchmark literature warns erodes the realised gain).
        steps: Per-step detail, in trace order.
    """

    total_tokens: int
    decode_steps: int
    baseline_steps: int
    total_drafted: int
    total_accepted: int
    draft_hit_rate: float
    acceptance_rate: float
    mean_accepted_length: float
    speedup_estimate: float
    steps: list[StepAcceptance] = field(default_factory=list)


def analyze_acceptance(
    tokens: list[int],
    drafter: PromptLookupDrafter | None = None,
    *,
    seed_len: int = 1,
) -> AcceptanceReport:
    """Replay a drafter over a known token sequence and measure draft acceptance.

    Simulates a *lossless* speculative decode: starting from a seed prefix, at each
    step the drafter proposes a continuation from the context emitted so far, and we
    accept the longest prefix of that draft that matches the ground-truth next tokens
    (the reference trace). Whether or not a draft is accepted, the step also commits
    the one correctly-predicted token the target model would produce — so every step
    advances by ``accepted + 1`` tokens, exactly as live speculative decoding does
    (verify N drafted, plus the model's own next token). The result is the acceptance
    distribution this drafter WOULD achieve on this corpus, with no model in the loop.

    Args:
        tokens: The full reference token sequence (prompt + completion, or any trace).
        drafter: Drafter to evaluate; defaults to a reference-configured drafter.
        seed_len: Tokens to treat as the initial primed context before prediction
            begins (must be >= 1 and < ``len(tokens)``).

    Returns:
        An :class:`AcceptanceReport` aggregating the replay.

    Raises:
        ValueError: If ``tokens`` is too short or ``seed_len`` is out of range.
    """
    drafter = drafter or PromptLookupDrafter()
    n = len(tokens)
    if seed_len < 1:
        raise ValueError(f"seed_len must be >= 1, got {seed_len}")
    if n <= seed_len:
        raise ValueError(
            f"tokens length ({n}) must exceed seed_len ({seed_len}) to have anything to predict"
        )

    total_tokens = n - seed_len
    steps: list[StepAcceptance] = []
    total_drafted = 0
    total_accepted = 0
    steps_with_draft = 0

    pos = seed_len
    while pos < n:
        context = tokens[:pos]
        draft = drafter.draft(context)
        # Ground-truth tokens the model will actually emit from this position onward.
        truth = tokens[pos:n]
        accepted = _common_prefix_len(draft, truth)

        steps.append(StepAcceptance(position=pos, drafted=len(draft), accepted=accepted))
        total_drafted += len(draft)
        total_accepted += accepted
        if draft:
            steps_with_draft += 1

        # Commit accepted drafted tokens PLUS the one token the target model produces
        # this step (the bonus token of lossless speculative decoding). Clamp to n so
        # a draft running past the end of the trace doesn't overshoot.
        pos = min(pos + accepted + 1, n)

    decode_steps = len(steps)
    draft_hit_rate = steps_with_draft / decode_steps if decode_steps else 0.0
    acceptance_rate = total_accepted / total_drafted if total_drafted else 0.0
    mean_accepted_length = total_accepted / decode_steps if decode_steps else 0.0
    speedup_estimate = total_tokens / decode_steps if decode_steps else 1.0

    return AcceptanceReport(
        total_tokens=total_tokens,
        decode_steps=decode_steps,
        baseline_steps=total_tokens,
        total_drafted=total_drafted,
        total_accepted=total_accepted,
        draft_hit_rate=draft_hit_rate,
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_length,
        speedup_estimate=speedup_estimate,
        steps=steps,
    )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-id lists."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


# ── Adaptive draft length (IDEA-006 / MODEL-044) ──────────────────────────────


@dataclass
class AdaptiveDraftLength:
    """Closed-loop controller for the PLD draft length γ (IDEA-006, MODEL-044).

    A fixed draft length wastes verification compute where the model rarely accepts
    drafts and under-drafts where it almost always does. On code, realized acceptance
    is bimodal (boilerplate runs vs. genuine decision points), so adapting γ from a
    running acceptance signal beats any single fixed value.

    Pure and model-free: :meth:`update` is fed only ``(accepted, drafted)`` counts from
    each verification step (exactly what :func:`analyze_acceptance` already computes), so
    it can be unit-tested in isolation and dropped into a future generate-loop integration
    without touching model code. Maintains an EMA of the per-step acceptance RATIO and
    nudges γ up after high-acceptance steps, down after low-acceptance ones, clamped to
    ``[min_pred_tokens, max_pred_tokens]``.

    Attributes:
        min_pred_tokens: Lower clamp for γ (>= 1).
        max_pred_tokens: Upper clamp for γ (>= min_pred_tokens).
        ema_alpha: Smoothing for the acceptance-ratio EMA, in (0, 1]; higher = more
            reactive to the most recent step.
        grow_threshold: EMA acceptance ratio at/above which γ grows by one.
        shrink_threshold: EMA acceptance ratio at/below which γ shrinks by one.
        initial_pred_tokens: Starting γ (defaults to ``min_pred_tokens`` when unset).
    """

    min_pred_tokens: int = 1
    max_pred_tokens: int = 10
    ema_alpha: float = 0.3
    grow_threshold: float = 0.7
    shrink_threshold: float = 0.3
    initial_pred_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.min_pred_tokens < 1:
            raise ValueError(f"min_pred_tokens must be >= 1, got {self.min_pred_tokens}")
        if self.max_pred_tokens < self.min_pred_tokens:
            raise ValueError(
                f"max_pred_tokens ({self.max_pred_tokens}) must be >= "
                f"min_pred_tokens ({self.min_pred_tokens})"
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if not 0.0 <= self.shrink_threshold <= self.grow_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= shrink_threshold <= grow_threshold <= 1, "
                f"got shrink={self.shrink_threshold}, grow={self.grow_threshold}"
            )
        start = self.min_pred_tokens if self.initial_pred_tokens is None else self.initial_pred_tokens
        self._draft_length = max(self.min_pred_tokens, min(self.max_pred_tokens, start))
        self._ema: float | None = None

    @property
    def draft_length(self) -> int:
        """The number of tokens to draft on the NEXT step (current γ)."""
        return self._draft_length

    @property
    def acceptance_ema(self) -> float | None:
        """The current acceptance-ratio EMA, or ``None`` before the first update."""
        return self._ema

    def update(self, accepted: int, drafted: int) -> int:
        """Record one verification step's outcome and return the new draft length.

        Args:
            accepted: Number of drafted tokens accepted under greedy verification (0..drafted).
            drafted: Number of tokens that were drafted this step. ``0`` means the drafter
                proposed nothing (no n-gram match); such a step carries no acceptance signal
                and leaves γ and the EMA unchanged.

        Returns:
            The updated draft length to use on the next step.
        """
        if drafted < 0 or accepted < 0 or accepted > drafted:
            raise ValueError(
                f"require 0 <= accepted <= drafted, got accepted={accepted}, drafted={drafted}"
            )
        if drafted == 0:
            return self._draft_length  # no signal: a no-match step doesn't move γ

        ratio = accepted / drafted
        self._ema = ratio if self._ema is None else (
            self.ema_alpha * ratio + (1.0 - self.ema_alpha) * self._ema
        )
        if self._ema >= self.grow_threshold:
            self._draft_length = min(self.max_pred_tokens, self._draft_length + 1)
        elif self._ema <= self.shrink_threshold:
            self._draft_length = max(self.min_pred_tokens, self._draft_length - 1)
        return self._draft_length
