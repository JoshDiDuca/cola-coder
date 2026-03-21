"""Inference Profiler: measure latency breakdown during code generation.

Tracks four timing phases:
  - tokenization  : text -> token IDs
  - prefill       : processing the prompt (first forward pass)
  - decode        : generating each new token
  - total         : wall-clock end-to-end

Also derives: tokens/sec, time-to-first-token (TTFT), inter-token latency.

Usage::

    profiler = InferenceProfiler()

    with profiler.run() as ctx:
        ids = ctx.record_tokenization(tokenize, text)
        ctx.record_prefill_start()
        first_token = model_prefill(ids)
        ctx.record_first_token()
        for token in model_decode():
            ctx.record_token()

    print(profiler.last_result())

For a TS dev: it's like a Performance.mark / Performance.measure wrapper
but specialised for LLM inference phases.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    """Timing profile for a single inference call."""

    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0

    prompt_tokens: int = 0
    generated_tokens: int = 0
    inter_token_latencies_ms: list[float] = field(default_factory=list)

    @property
    def time_to_first_token_ms(self) -> float:
        """Time from start until the first new token is produced (ms)."""
        return self.tokenization_ms + self.prefill_ms

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput in tokens/sec (excludes prompt processing)."""
        if self.decode_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.decode_ms / 1000.0)

    @property
    def avg_inter_token_latency_ms(self) -> float:
        """Average ms between consecutive decode tokens."""
        if not self.inter_token_latencies_ms:
            return 0.0
        return sum(self.inter_token_latencies_ms) / len(self.inter_token_latencies_ms)

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"total={self.total_ms:.1f}ms  "
            f"tok={self.tokenization_ms:.1f}ms  "
            f"prefill={self.prefill_ms:.1f}ms  "
            f"decode={self.decode_ms:.1f}ms  "
            f"TTFT={self.time_to_first_token_ms:.1f}ms  "
            f"tok/s={self.tokens_per_second:.1f}  "
            f"gen_tokens={self.generated_tokens}"
        )


# ---------------------------------------------------------------------------
# Context object used inside "with profiler.run() as ctx:"
# ---------------------------------------------------------------------------


class _ProfileContext:
    """Internal context object handed to the caller."""

    def __init__(self) -> None:
        self._wall_start: float = 0.0
        self._tok_start: float = 0.0
        self._tok_end: float = 0.0
        self._prefill_start: float = 0.0
        self._first_token_time: float = 0.0
        self._decode_start: float = 0.0
        self._last_token_time: float = 0.0
        self._token_times: list[float] = []
        self.profile = InferenceProfile()

    # ── Phase markers ─────────────────────────────────────────────────

    def start_wall(self) -> None:
        self._wall_start = time.perf_counter()

    def record_tokenization(self, tokenize_fn, text: str):
        """Call *tokenize_fn(text)*, timing it, and return the result."""
        self._tok_start = time.perf_counter()
        result = tokenize_fn(text)
        self._tok_end = time.perf_counter()
        self.profile.tokenization_ms = (self._tok_end - self._tok_start) * 1000
        if isinstance(result, (list, tuple)):
            self.profile.prompt_tokens = len(result)
        return result

    def set_tokenization_ms(self, ms: float, prompt_tokens: int = 0) -> None:
        """Manually set tokenization time (when you time it yourself)."""
        self.profile.tokenization_ms = ms
        self.profile.prompt_tokens = prompt_tokens

    def record_prefill_start(self) -> None:
        """Mark the start of prefill (first forward pass)."""
        self._prefill_start = time.perf_counter()

    def record_first_token(self) -> None:
        """Mark when the first new token is produced."""
        now = time.perf_counter()
        self._first_token_time = now
        self._decode_start = now
        self._last_token_time = now
        if self._prefill_start > 0:
            self.profile.prefill_ms = (now - self._prefill_start) * 1000
        self.profile.generated_tokens = 1
        self._token_times.append(now)

    def record_token(self) -> None:
        """Mark each subsequent decode token."""
        now = time.perf_counter()
        if self._last_token_time > 0:
            latency = (now - self._last_token_time) * 1000
            self.profile.inter_token_latencies_ms.append(latency)
        self._last_token_time = now
        self._token_times.append(now)
        self.profile.generated_tokens += 1

    def finish_wall(self) -> None:
        now = time.perf_counter()
        self.profile.total_ms = (now - self._wall_start) * 1000
        if self._decode_start > 0 and self._last_token_time > 0:
            self.profile.decode_ms = (self._last_token_time - self._decode_start) * 1000


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class InferenceProfiler:
    """Context-manager-based inference profiler.

    Keeps a history of all runs so you can compare across checkpoints or
    prompt lengths.

    Example::

        profiler = InferenceProfiler()

        with profiler.run() as ctx:
            tokens = ctx.record_tokenization(tokenizer.encode, prompt)
            ctx.record_prefill_start()
            # ... first forward pass ...
            ctx.record_first_token()
            for _ in range(max_new_tokens):
                # ... decode step ...
                ctx.record_token()

        print(profiler.last_result().summary())
    """

    def __init__(self) -> None:
        self._history: list[InferenceProfile] = []

    @contextmanager
    def run(self) -> Generator[_ProfileContext, None, None]:
        """Context manager that yields a _ProfileContext.

        Timing starts when the ``with`` block is entered; the completed
        InferenceProfile is stored in history on exit.
        """
        ctx = _ProfileContext()
        ctx.start_wall()
        try:
            yield ctx
        finally:
            ctx.finish_wall()
            self._history.append(ctx.profile)

    # ------------------------------------------------------------------
    # Results API
    # ------------------------------------------------------------------

    def last_result(self) -> InferenceProfile | None:
        """Return the most recent profile, or None if nothing recorded yet."""
        return self._history[-1] if self._history else None

    def history(self) -> list[InferenceProfile]:
        """Return all recorded profiles (oldest first)."""
        return list(self._history)

    def average(self) -> InferenceProfile | None:
        """Return an averaged InferenceProfile across all runs."""
        if not self._history:
            return None
        n = len(self._history)
        avg = InferenceProfile(
            tokenization_ms=sum(p.tokenization_ms for p in self._history) / n,
            prefill_ms=sum(p.prefill_ms for p in self._history) / n,
            decode_ms=sum(p.decode_ms for p in self._history) / n,
            total_ms=sum(p.total_ms for p in self._history) / n,
            prompt_tokens=round(sum(p.prompt_tokens for p in self._history) / n),
            generated_tokens=round(sum(p.generated_tokens for p in self._history) / n),
        )
        return avg

    def reset(self) -> None:
        """Clear history."""
        self._history.clear()

    def __repr__(self) -> str:
        return f"InferenceProfiler(runs={len(self._history)})"
