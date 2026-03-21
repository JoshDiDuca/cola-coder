"""Tokenizer Debugger: inspect and debug tokenization decisions.

Works with any tokenizer that implements a simple protocol:
    tokenizer.encode(text) -> list[int]
    tokenizer.decode(ids)  -> str
    tokenizer.id_to_token(id) -> str   (optional; used for display)

Provides:
- Visual token boundary display (with customizable delimiters)
- Side-by-side tokenization comparison for similar strings
- Worst-case examples (strings with most tokens per character)
- Merge decision tracing for BPE-style tokenizers
- Vocabulary statistics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the tokenizer debugger feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal duck-type interface for tokenizer objects."""

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenBoundary:
    """Information about a single token in a string."""

    token_id: int
    token_str: str  # Decoded text of this token
    start: int  # Character offset in the original string
    end: int  # Exclusive end character offset
    length: int  # Characters covered


@dataclass
class TokenizationView:
    """Full breakdown of tokenizing a text."""

    text: str
    token_ids: list[int]
    boundaries: list[TokenBoundary]
    n_tokens: int
    n_chars: int
    chars_per_token: float
    visual: str  # Human-readable annotated string

    def as_records(self) -> list[dict]:
        return [
            {
                "token_id": b.token_id,
                "token_str": b.token_str,
                "start": b.start,
                "end": b.end,
            }
            for b in self.boundaries
        ]


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two tokenizations."""

    text_a: str
    text_b: str
    ids_a: list[int]
    ids_b: list[int]
    n_tokens_a: int
    n_tokens_b: int
    shared_ids: list[int]
    diff_ids_a: list[int]  # IDs in A but not B
    diff_ids_b: list[int]  # IDs in B but not A
    efficiency_delta: float  # (chars/token_b) - (chars/token_a)

    def summary(self) -> str:
        return (
            f"Comparison: '{self.text_a[:30]}' ({self.n_tokens_a} tok) vs "
            f"'{self.text_b[:30]}' ({self.n_tokens_b} tok), "
            f"shared={len(self.shared_ids)}, delta={self.efficiency_delta:+.2f} char/tok"
        )


@dataclass
class WorstCaseExample:
    """A string that has unusually many tokens per character."""

    text: str
    n_tokens: int
    n_chars: int
    tokens_per_char: float
    token_ids: list[int]


# ---------------------------------------------------------------------------
# Debugger
# ---------------------------------------------------------------------------

class TokenizerDebugger:
    """Debug and inspect tokenizer behaviour."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        id_to_token: Optional[Callable[[int], str]] = None,
        boundary_open: str = "[",
        boundary_close: str = "]",
    ) -> None:
        """
        Parameters
        ----------
        tokenizer:
            An object with encode(text)->list[int] and decode(ids)->str.
        id_to_token:
            Optional function mapping token_id → token string.
            If not provided, each token is decoded individually via tokenizer.decode.
        boundary_open / boundary_close:
            Delimiters used when rendering token boundaries visually.
        """
        self.tokenizer = tokenizer
        self._id_to_token = id_to_token
        self.boundary_open = boundary_open
        self.boundary_close = boundary_close

    def _tok_str(self, token_id: int) -> str:
        if self._id_to_token is not None:
            return self._id_to_token(token_id)
        return self.tokenizer.decode([token_id])

    def analyze(self, text: str) -> TokenizationView:
        """Tokenize *text* and return a detailed view of token boundaries."""
        ids = self.tokenizer.encode(text)
        boundaries: list[TokenBoundary] = []
        pos = 0

        # Reconstruct character positions by decoding tokens one by one
        for tid in ids:
            tok_str = self._tok_str(tid)
            # Normalise whitespace markers (e.g. Ġ → space)
            display_str = tok_str.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
            start = pos
            end = start + len(display_str)
            boundaries.append(
                TokenBoundary(
                    token_id=tid,
                    token_str=display_str,
                    start=start,
                    end=end,
                    length=len(display_str),
                )
            )
            pos = end

        n_chars = len(text)
        n_tokens = len(ids)
        cpt = n_chars / n_tokens if n_tokens > 0 else 0.0

        # Build visual string: [token1][token2]...
        visual_parts = [
            f"{self.boundary_open}{b.token_str}{self.boundary_close}"
            for b in boundaries
        ]
        visual = "".join(visual_parts)

        return TokenizationView(
            text=text,
            token_ids=ids,
            boundaries=boundaries,
            n_tokens=n_tokens,
            n_chars=n_chars,
            chars_per_token=cpt,
            visual=visual,
        )

    def compare(self, text_a: str, text_b: str) -> ComparisonResult:
        """Compare tokenization of two strings."""
        ids_a = self.tokenizer.encode(text_a)
        ids_b = self.tokenizer.encode(text_b)

        set_a, set_b = set(ids_a), set(ids_b)
        shared = sorted(set_a & set_b)
        diff_a = sorted(set_a - set_b)
        diff_b = sorted(set_b - set_a)

        cpt_a = len(text_a) / len(ids_a) if ids_a else 0.0
        cpt_b = len(text_b) / len(ids_b) if ids_b else 0.0

        return ComparisonResult(
            text_a=text_a,
            text_b=text_b,
            ids_a=ids_a,
            ids_b=ids_b,
            n_tokens_a=len(ids_a),
            n_tokens_b=len(ids_b),
            shared_ids=shared,
            diff_ids_a=diff_a,
            diff_ids_b=diff_b,
            efficiency_delta=cpt_b - cpt_a,
        )

    def find_worst_cases(
        self,
        candidates: list[str],
        top_n: int = 5,
        metric: str = "tokens_per_char",
    ) -> list[WorstCaseExample]:
        """Find strings in *candidates* with the worst tokenization efficiency.

        Parameters
        ----------
        metric:
            "tokens_per_char" (higher = worse) or "chars_per_token" (lower = worse).
        """
        results = []
        for text in candidates:
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            n_tokens = len(ids)
            n_chars = len(text)
            tpc = n_tokens / n_chars if n_chars > 0 else 0.0
            results.append(
                WorstCaseExample(
                    text=text,
                    n_tokens=n_tokens,
                    n_chars=n_chars,
                    tokens_per_char=tpc,
                    token_ids=ids,
                )
            )

        if metric == "tokens_per_char":
            results.sort(key=lambda x: x.tokens_per_char, reverse=True)
        else:
            results.sort(key=lambda x: x.n_chars / x.n_tokens if x.n_tokens else 0)

        return results[:top_n]

    def vocab_statistics(self, vocab: dict[str, int]) -> dict[str, Any]:
        """Compute statistics over a vocabulary mapping {token_str: token_id}.

        Returns a dict with keys: vocab_size, mean_token_len, max_token_len,
        min_token_len, single_char_fraction, numeric_fraction, whitespace_fraction.
        """
        if not vocab:
            return {"vocab_size": 0}

        tokens = list(vocab.keys())
        lengths = [len(t) for t in tokens]
        n = len(tokens)
        single_char = sum(1 for t in tokens if len(t) == 1) / n
        numeric = sum(1 for t in tokens if re.fullmatch(r"\d+", t.strip())) / n
        whitespace = sum(1 for t in tokens if not t.strip()) / n

        return {
            "vocab_size": n,
            "mean_token_len": sum(lengths) / n,
            "max_token_len": max(lengths),
            "min_token_len": min(lengths),
            "single_char_fraction": single_char,
            "numeric_fraction": numeric,
            "whitespace_fraction": whitespace,
        }

    def highlight_merges(self, text: str, n_steps: int = 5) -> list[str]:
        """Simulate BPE merge steps by showing progressively merged token sequences.

        This is a simplified simulation: starts from character-level and merges
        the most frequent adjacent pair at each step.

        Returns a list of *n_steps* token sequence strings (one per merge step).
        """
        # Start from character tokenization
        tokens: list[str] = list(text)
        stages = ["".join(f"{self.boundary_open}{c}{self.boundary_close}" for c in tokens)]

        for _ in range(min(n_steps - 1, len(tokens) - 1)):
            if len(tokens) <= 1:
                break
            # Count adjacent pairs
            pair_counts: dict[tuple[str, str], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Merge the most frequent pair
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            merged = best_pair[0] + best_pair[1]
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            stage_str = "".join(
                f"{self.boundary_open}{t}{self.boundary_close}" for t in tokens
            )
            stages.append(stage_str)

        return stages
