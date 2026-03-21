"""Perplexity Analyzer: compute per-token and per-line perplexity for generated text.

Perplexity is the exponential of the average cross-entropy loss — a measure of how
"surprised" the model is by each token.  Lower perplexity = more confident.

Think of it like TypeScript's type inference confidence: a ``never`` type means the
model has no idea; a perfectly typed expression maps to perplexity ≈ 1.

The analyzer highlights which tokens the model was most/least confident about,
making it useful for debugging model quality and identifying hard regions in code.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if perplexity analysis is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Protocols / type stubs (avoid hard torch dependency at import time)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    import torch


class _TokenizerProtocol(Protocol):
    """Minimal tokenizer interface required by PerplexityAnalyzer."""

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TokenPerplexity:
    """Perplexity for a single token."""

    token_id: int
    token_str: str
    log_prob: float  # log probability (negative = less likely)
    perplexity: float  # exp(-log_prob)


@dataclass
class LinePerplexity:
    """Perplexity aggregated over one line of code."""

    line_number: int
    text: str
    mean_perplexity: float
    token_count: int


@dataclass
class PerplexityReport:
    """Full report from PerplexityAnalyzer.analyze()."""

    text: str
    tokens: list[TokenPerplexity] = field(default_factory=list)
    lines: list[LinePerplexity] = field(default_factory=list)
    mean_perplexity: float = 0.0
    median_perplexity: float = 0.0
    min_perplexity: float = 0.0
    max_perplexity: float = 0.0
    most_confident: list[TokenPerplexity] = field(default_factory=list)
    least_confident: list[TokenPerplexity] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            f"Perplexity Analysis — {len(self.tokens)} tokens",
            f"  Mean:   {self.mean_perplexity:.2f}",
            f"  Median: {self.median_perplexity:.2f}",
            f"  Min:    {self.min_perplexity:.2f}  (most confident)",
            f"  Max:    {self.max_perplexity:.2f}  (least confident)",
            "",
            "Top 5 most confident tokens (model is sure about these):",
        ]
        for tp in self.most_confident[:5]:
            lines.append(f"  {tp.token_str!r:20s}  ppl={tp.perplexity:.2f}")
        lines.append("")
        lines.append("Top 5 least confident tokens (model struggled here):")
        for tp in self.least_confident[:5]:
            lines.append(f"  {tp.token_str!r:20s}  ppl={tp.perplexity:.2f}")
        if self.lines:
            lines.append("")
            lines.append("Per-line perplexity:")
            for lp in sorted(self.lines, key=lambda x: x.mean_perplexity, reverse=True)[:5]:
                lines.append(
                    f"  line {lp.line_number:3d} ({lp.token_count:2d} tok)  "
                    f"ppl={lp.mean_perplexity:.2f}  {lp.text[:60]!r}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: compute log-probs without storing full logit tensors
# ---------------------------------------------------------------------------


def _compute_token_log_probs(
    model: object,
    input_ids: "torch.Tensor",
) -> "torch.Tensor":
    """Run a forward pass and return per-token log-probabilities.

    Returns shape ``[seq_len - 1]`` — log prob of each token given its prefix.
    The first token has no context so is excluded.

    Parameters
    ----------
    model:
        Any PyTorch model that accepts ``input_ids`` as a positional argument
        and returns logits of shape ``[batch, seq_len, vocab_size]``.
    input_ids:
        Token IDs, shape ``[1, seq_len]``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    with torch.no_grad():
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        if isinstance(logits, tuple):
            logits = logits[0]

    # Shift: predict token[i] from tokens[0:i]
    logits = logits[0, :-1, :]  # [seq_len-1, vocab_size]
    targets = input_ids[0, 1:]  # [seq_len-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(targets)), targets]  # [seq_len-1]
    return token_log_probs


# ---------------------------------------------------------------------------
# PerplexityAnalyzer
# ---------------------------------------------------------------------------


class PerplexityAnalyzer:
    """Compute per-token and per-line perplexity for a text given a model.

    Usage::

        from cola_coder.features.perplexity_analyzer import PerplexityAnalyzer

        analyzer = PerplexityAnalyzer()
        report = analyzer.analyze(model, code_text, tokenizer)
        print(report.summary())
    """

    def __init__(self, top_k: int = 10) -> None:
        """
        Parameters
        ----------
        top_k:
            How many most- and least-confident tokens to include in the report.
        """
        self.top_k = top_k

    def analyze(
        self,
        model: object,
        text: str,
        tokenizer: _TokenizerProtocol,
        device: str = "cpu",
    ) -> PerplexityReport:
        """Compute perplexity for *text* using *model* and *tokenizer*.

        Parameters
        ----------
        model:
            Trained model.  Must accept ``input_ids`` tensor and return logits.
        text:
            Source text to analyse.
        tokenizer:
            Tokenizer with ``encode(str) -> list[int]`` and
            ``decode(list[int]) -> str`` methods.
        device:
            Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns
        -------
        PerplexityReport
        """
        import torch

        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            return PerplexityReport(text=text)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Ensure model is in eval mode
        if hasattr(model, "eval"):
            model.eval()  # type: ignore[union-attr]
        if hasattr(model, "to"):
            model.to(device)  # type: ignore[union-attr]

        token_log_probs = _compute_token_log_probs(model, input_ids)
        log_probs_list = token_log_probs.tolist()

        # Build per-token records (index 0 = token_ids[1], i.e., second token)
        token_records: list[TokenPerplexity] = []
        for i, lp in enumerate(log_probs_list):
            tid = token_ids[i + 1]
            try:
                tok_str = tokenizer.decode([tid])
            except Exception:
                tok_str = f"<{tid}>"
            ppl = math.exp(-lp)
            token_records.append(
                TokenPerplexity(
                    token_id=tid,
                    token_str=tok_str,
                    log_prob=float(lp),
                    perplexity=ppl,
                )
            )

        # Global statistics
        perplexities = [t.perplexity for t in token_records]
        mean_ppl = sum(perplexities) / len(perplexities)
        sorted_ppls = sorted(perplexities)
        mid = len(sorted_ppls) // 2
        median_ppl = (
            sorted_ppls[mid]
            if len(sorted_ppls) % 2
            else (sorted_ppls[mid - 1] + sorted_ppls[mid]) / 2
        )

        sorted_tokens = sorted(token_records, key=lambda t: t.perplexity)
        most_confident = sorted_tokens[: self.top_k]
        least_confident = sorted_tokens[-self.top_k :][::-1]

        # Per-line perplexity
        line_records = self._compute_line_perplexity(text, token_ids, token_records)

        return PerplexityReport(
            text=text,
            tokens=token_records,
            lines=line_records,
            mean_perplexity=mean_ppl,
            median_perplexity=median_ppl,
            min_perplexity=min(perplexities),
            max_perplexity=max(perplexities),
            most_confident=most_confident,
            least_confident=least_confident,
        )

    def analyze_from_log_probs(
        self,
        text: str,
        token_ids: list[int],
        log_probs: list[float],
        tokenizer: _TokenizerProtocol,
    ) -> PerplexityReport:
        """Build a PerplexityReport from pre-computed log-probabilities.

        Useful when you already have log-probs from a forward pass and don't
        want to run inference again.

        Parameters
        ----------
        text:
            Original text (for display).
        token_ids:
            Full token ID sequence (length N).
        log_probs:
            Log-probabilities for tokens[1:] (length N-1).
        tokenizer:
            Tokenizer for decoding token IDs to strings.
        """
        if len(log_probs) == 0:
            return PerplexityReport(text=text)

        token_records: list[TokenPerplexity] = []
        for i, lp in enumerate(log_probs):
            tid = token_ids[i + 1]
            try:
                tok_str = tokenizer.decode([tid])
            except Exception:
                tok_str = f"<{tid}>"
            ppl = math.exp(-float(lp))
            token_records.append(
                TokenPerplexity(token_id=tid, token_str=tok_str, log_prob=float(lp), perplexity=ppl)
            )

        perplexities = [t.perplexity for t in token_records]
        mean_ppl = sum(perplexities) / len(perplexities)
        sorted_ppls = sorted(perplexities)
        mid = len(sorted_ppls) // 2
        median_ppl = (
            sorted_ppls[mid]
            if len(sorted_ppls) % 2
            else (sorted_ppls[mid - 1] + sorted_ppls[mid]) / 2
        )
        sorted_tokens = sorted(token_records, key=lambda t: t.perplexity)

        line_records = self._compute_line_perplexity(text, token_ids, token_records)

        return PerplexityReport(
            text=text,
            tokens=token_records,
            lines=line_records,
            mean_perplexity=mean_ppl,
            median_perplexity=median_ppl,
            min_perplexity=min(perplexities),
            max_perplexity=max(perplexities),
            most_confident=sorted_tokens[: self.top_k],
            least_confident=sorted_tokens[-self.top_k :][::-1],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_line_perplexity(
        self,
        text: str,
        token_ids: list[int],
        token_records: list[TokenPerplexity],
    ) -> list[LinePerplexity]:
        """Approximate per-line perplexity by partitioning tokens by newlines.

        This is an approximation: we split the decoded text by newlines and
        assign tokens sequentially to lines.  It won't be pixel-perfect for
        all tokenizers but is close enough for diagnostics.
        """
        text_lines = text.splitlines()
        if not text_lines or not token_records:
            return []

        line_records: list[LinePerplexity] = []
        rec_idx = 0
        for lineno, line_text in enumerate(text_lines, start=1):
            line_ppls: list[float] = []
            chars_so_far = 0
            # Consume tokens roughly corresponding to this line
            # Heuristic: count characters decoded
            target_chars = len(line_text) + 1  # +1 for \n
            while rec_idx < len(token_records) and chars_so_far < target_chars:
                tok = token_records[rec_idx]
                line_ppls.append(tok.perplexity)
                chars_so_far += max(1, len(tok.token_str))
                rec_idx += 1

            if line_ppls:
                mean_ppl = sum(line_ppls) / len(line_ppls)
                line_records.append(
                    LinePerplexity(
                        line_number=lineno,
                        text=line_text,
                        mean_perplexity=mean_ppl,
                        token_count=len(line_ppls),
                    )
                )

        return line_records
