"""Loss Component Analyzer.

Break down training loss by multiple dimensions:
  - Per-token: which vocabulary items are hardest to predict
  - Per-position: which sequence positions have highest loss
  - Per-category: group tokens by category (keyword, identifier, operator, etc.)

Helps identify training blind spots and guide data curation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Token categories
# ---------------------------------------------------------------------------

# Maps token strings to semantic categories for Python-like code
_PYTHON_KEYWORDS = frozenset(
    "def class return import from as if else elif for while in not and or "
    "is None True False try except finally with yield lambda pass break "
    "continue raise assert del async await".split()
)
_OPERATORS = frozenset("+ - * / // % ** & | ^ ~ << >> == != < > <= >= = += -= "
                       "*= /= //= %= **= &= |= ^= <<= >>= -> : , . ; @ ( ) [ ] { }".split())


def categorize_token(token_str: str) -> str:
    """Assign a category string to a token."""
    if token_str in _PYTHON_KEYWORDS:
        return "keyword"
    if token_str in _OPERATORS:
        return "operator"
    if token_str.strip() == "":
        return "whitespace"
    if token_str.startswith('"') or token_str.startswith("'"):
        return "string_literal"
    if token_str.lstrip("-").replace(".", "").isdigit():
        return "numeric_literal"
    if token_str.isidentifier():
        return "identifier"
    return "other"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TokenLoss(NamedTuple):
    """Loss information for a single token occurrence."""

    token_id: int
    token_str: str
    position: int
    loss: float
    category: str


@dataclass
class PositionStats:
    """Aggregated loss statistics for a sequence position."""

    position: int
    mean_loss: float
    count: int
    min_loss: float
    max_loss: float

    @property
    def std_loss(self) -> float:
        """Cannot compute std from mean only — placeholder."""
        return 0.0


@dataclass
class TokenStats:
    """Aggregated loss statistics for a token type."""

    token_id: int
    token_str: str
    mean_loss: float
    count: int
    total_loss: float


@dataclass
class CategoryStats:
    """Aggregated loss statistics for a token category."""

    category: str
    mean_loss: float
    count: int
    total_loss: float
    worst_tokens: list[str]  # top tokens by mean loss within this category


@dataclass
class LossBreakdown:
    """Full loss breakdown report."""

    total_tokens: int = 0
    mean_loss: float = 0.0
    # Per-position statistics (list indexed by position)
    per_position: list[PositionStats] = field(default_factory=list)
    # Per-token statistics (dict keyed by token_id)
    per_token: dict[int, TokenStats] = field(default_factory=dict)
    # Per-category statistics
    per_category: dict[str, CategoryStats] = field(default_factory=dict)
    # Hardest tokens (highest mean loss)
    hardest_tokens: list[TokenStats] = field(default_factory=list)
    # Hardest positions
    hardest_positions: list[PositionStats] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"LossBreakdown: {self.total_tokens} tokens, mean_loss={self.mean_loss:.4f}",
        ]
        if self.hardest_tokens:
            top = self.hardest_tokens[:3]
            lines.append("Hardest tokens: " + ", ".join(
                f"'{t.token_str}'({t.mean_loss:.3f})" for t in top
            ))
        if self.per_category:
            lines.append("Category losses: " + ", ".join(
                f"{cat}={stats.mean_loss:.3f}"
                for cat, stats in sorted(self.per_category.items(), key=lambda x: -x[1].mean_loss)[:4]
            ))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LossComponentAnalyzer:
    """Analyze training loss broken down by token, position, and category.

    Parameters
    ----------
    top_k:
        Number of hardest tokens/positions to include in summaries.
    vocab:
        Optional mapping from token_id to token string.  If not provided,
        token IDs are used as strings.
    """

    def __init__(
        self,
        top_k: int = 10,
        vocab: dict[int, str] | None = None,
    ) -> None:
        self.top_k = top_k
        self.vocab = vocab or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        token_ids: list[int],
        losses: list[float],
    ) -> LossBreakdown:
        """Analyze per-token losses.

        Parameters
        ----------
        token_ids:
            Sequence of token IDs (length N).
        losses:
            Per-token cross-entropy losses (length N).  Must match token_ids.
        """
        if len(token_ids) != len(losses):
            raise ValueError(
                f"token_ids length {len(token_ids)} != losses length {len(losses)}"
            )
        if not token_ids:
            return LossBreakdown()

        records = self._build_records(token_ids, losses)
        return self._aggregate(records)

    def analyze_batch(
        self,
        batch_token_ids: list[list[int]],
        batch_losses: list[list[float]],
    ) -> LossBreakdown:
        """Analyze a batch of sequences."""
        all_records: list[TokenLoss] = []
        for token_ids, losses in zip(batch_token_ids, batch_losses):
            all_records.extend(self._build_records(token_ids, losses))
        if not all_records:
            return LossBreakdown()
        return self._aggregate(all_records)

    def hardest_positions(self, breakdown: LossBreakdown, top_k: int | None = None) -> list[PositionStats]:
        """Return positions sorted by mean loss descending."""
        k = top_k or self.top_k
        return sorted(breakdown.per_position, key=lambda p: -p.mean_loss)[:k]

    def hardest_tokens(self, breakdown: LossBreakdown, top_k: int | None = None) -> list[TokenStats]:
        """Return token types sorted by mean loss descending."""
        k = top_k or self.top_k
        return sorted(breakdown.per_token.values(), key=lambda t: -t.mean_loss)[:k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _token_str(self, token_id: int) -> str:
        return self.vocab.get(token_id, str(token_id))

    def _build_records(
        self, token_ids: list[int], losses: list[float]
    ) -> list[TokenLoss]:
        records: list[TokenLoss] = []
        for pos, (tid, loss) in enumerate(zip(token_ids, losses)):
            tok_str = self._token_str(tid)
            cat = categorize_token(tok_str)
            records.append(TokenLoss(
                token_id=tid,
                token_str=tok_str,
                position=pos,
                loss=loss,
                category=cat,
            ))
        return records

    def _aggregate(self, records: list[TokenLoss]) -> LossBreakdown:
        total = len(records)
        mean_loss = sum(r.loss for r in records) / total if total else 0.0

        # Per-position: group by position
        pos_groups: dict[int, list[float]] = defaultdict(list)
        for r in records:
            pos_groups[r.position].append(r.loss)

        per_position = []
        for pos in sorted(pos_groups):
            vals = pos_groups[pos]
            per_position.append(PositionStats(
                position=pos,
                mean_loss=sum(vals) / len(vals),
                count=len(vals),
                min_loss=min(vals),
                max_loss=max(vals),
            ))

        # Per-token
        tok_groups: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for r in records:
            tok_groups[r.token_id].append((r.token_str, r.loss))

        per_token: dict[int, TokenStats] = {}
        for tid, entries in tok_groups.items():
            losses_here = [e[1] for e in entries]
            tok_str = entries[0][0]
            total_loss = sum(losses_here)
            per_token[tid] = TokenStats(
                token_id=tid,
                token_str=tok_str,
                mean_loss=total_loss / len(losses_here),
                count=len(losses_here),
                total_loss=total_loss,
            )

        # Per-category
        cat_groups: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
        for r in records:
            cat_groups[r.category].append((r.token_id, r.token_str, r.loss))

        per_category: dict[str, CategoryStats] = {}
        for cat, entries in cat_groups.items():
            losses_here = [e[2] for e in entries]
            total_loss = sum(losses_here)
            mean = total_loss / len(losses_here)
            # Find worst tokens in this category
            tok_loss: dict[str, list[float]] = defaultdict(list)
            for _, tok_str, loss in entries:
                tok_loss[tok_str].append(loss)
            worst = sorted(tok_loss, key=lambda t: -sum(tok_loss[t]) / len(tok_loss[t]))
            per_category[cat] = CategoryStats(
                category=cat,
                mean_loss=mean,
                count=len(losses_here),
                total_loss=total_loss,
                worst_tokens=worst[:5],
            )

        # Top-k
        hardest_tokens = sorted(per_token.values(), key=lambda t: -t.mean_loss)[: self.top_k]
        hardest_positions = sorted(per_position, key=lambda p: -p.mean_loss)[: self.top_k]

        return LossBreakdown(
            total_tokens=total,
            mean_loss=round(mean_loss, 6),
            per_position=per_position,
            per_token=per_token,
            per_category=per_category,
            hardest_tokens=hardest_tokens,
            hardest_positions=hardest_positions,
        )
