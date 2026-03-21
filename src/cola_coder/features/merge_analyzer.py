"""Tokenizer Merge Analyzer — Feature 96

Analyse BPE (Byte-Pair Encoding) merge history to understand which merges
matter most for code tokenization quality.

Key capabilities
----------------
- Load merge rules from a ``merges.txt``-format file or a raw list of
  ``(a, b)`` pairs.
- Compute **merge frequency** given a corpus of token sequences.
- Rank merges by impact (frequency × token-length improvement).
- Reconstruct a lightweight **merge tree** to show derivation chains.
- Generate vocabulary improvement suggestions (merges with low usage).

No tokenizer library required — all analysis works on pre-tokenized
text or raw merge-rule lists.

Feature toggle: set FEATURE_ENABLED = False to disable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the merge analyzer is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MergeRule:
    """A single BPE merge rule: merge token *a* and *b* into *a+b*."""

    a: str
    b: str
    rank: int  # 0-based index in the merge list (lower = earlier = more common)

    @property
    def result(self) -> str:
        return self.a + self.b

    def __repr__(self) -> str:
        return f"MergeRule({self.a!r} + {self.b!r} → {self.result!r}, rank={self.rank})"


@dataclass
class MergeStats:
    """Usage statistics for a single merge rule."""

    rule: MergeRule
    frequency: int = 0  # how many times this pair appeared in the corpus
    savings: int = 0  # tokens saved (frequency × 1 token each merge)

    @property
    def impact_score(self) -> float:
        """Frequency × token length of result (longer merges = more impact)."""
        return self.frequency * len(self.rule.result)


@dataclass
class MergeAnalysisReport:
    """Full report from a merge analysis run."""

    total_merges: int
    total_tokens_in_corpus: int
    top_merges: list[MergeStats]  # ranked by impact
    low_impact_merges: list[MergeStats]  # candidates for vocabulary pruning
    coverage: float  # fraction of merges actually seen in corpus

    def as_dict(self) -> dict:
        return {
            "total_merges": self.total_merges,
            "total_tokens_in_corpus": self.total_tokens_in_corpus,
            "top_merges_count": len(self.top_merges),
            "low_impact_count": len(self.low_impact_merges),
            "coverage": self.coverage,
        }


# ---------------------------------------------------------------------------
# Merge tree node
# ---------------------------------------------------------------------------


@dataclass
class MergeTreeNode:
    """Node in the BPE merge derivation tree."""

    token: str
    left: Optional["MergeTreeNode"] = None
    right: Optional["MergeTreeNode"] = None
    rank: Optional[int] = None  # rank of the merge that created this node

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def leaves(self) -> list[str]:
        if self.is_leaf:
            return [self.token]
        out: list[str] = []
        if self.left:
            out.extend(self.left.leaves())
        if self.right:
            out.extend(self.right.leaves())
        return out


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _parse_merges_txt(text: str) -> list[tuple[str, str]]:
    """Parse a ``merges.txt`` file (HuggingFace tokenizer format)."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


class MergeAnalyzer:
    """Analyse BPE merge rules for code tokenization quality."""

    def __init__(self) -> None:
        self._rules: list[MergeRule] = []
        self._rule_index: dict[tuple[str, str], MergeRule] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Load merge rules from a list of ``(a, b)`` tuples."""
        self._rules = [
            MergeRule(a=a, b=b, rank=i) for i, (a, b) in enumerate(pairs)
        ]
        self._rule_index = {(r.a, r.b): r for r in self._rules}

    def load_from_text(self, merges_txt: str) -> None:
        """Load merge rules from a ``merges.txt``-format string."""
        pairs = _parse_merges_txt(merges_txt)
        self.load_from_pairs(pairs)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse_corpus(
        self,
        token_sequences: list[list[str]],
        top_n: int = 20,
        low_impact_threshold: int = 1,
    ) -> MergeAnalysisReport:
        """Compute merge frequency and impact over a token corpus.

        Parameters
        ----------
        token_sequences:
            List of token sequences (each a list of string tokens).
        top_n:
            Number of top merges to include in the report.
        low_impact_threshold:
            Merges with frequency <= this value are considered low-impact.

        Returns
        -------
        MergeAnalysisReport
        """
        # Count bigram frequencies in the corpus
        pair_counts: Counter[tuple[str, str]] = Counter()
        total_tokens = 0
        for seq in token_sequences:
            total_tokens += len(seq)
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1

        # Map to merge stats
        stats_map: dict[tuple[str, str], MergeStats] = {}
        for rule in self._rules:
            key = (rule.a, rule.b)
            freq = pair_counts.get(key, 0)
            stats_map[key] = MergeStats(
                rule=rule,
                frequency=freq,
                savings=freq,
            )

        all_stats = list(stats_map.values())
        seen = sum(1 for s in all_stats if s.frequency > 0)
        coverage = seen / len(all_stats) if all_stats else 0.0

        sorted_by_impact = sorted(
            all_stats, key=lambda s: s.impact_score, reverse=True
        )
        low_impact = [s for s in all_stats if s.frequency <= low_impact_threshold]

        return MergeAnalysisReport(
            total_merges=len(self._rules),
            total_tokens_in_corpus=total_tokens,
            top_merges=sorted_by_impact[:top_n],
            low_impact_merges=low_impact,
            coverage=coverage,
        )

    # ------------------------------------------------------------------
    # Merge tree
    # ------------------------------------------------------------------

    def build_merge_tree(self, token: str) -> MergeTreeNode:
        """Build a derivation tree for *token* using loaded merge rules.

        Recursively splits the token back into its component pieces.
        Leaf nodes are byte-level characters or sub-tokens not formed by
        any known merge.
        """
        # Try to find a rule whose result is this token
        for rule in self._rules:
            if rule.result == token:
                left = self.build_merge_tree(rule.a)
                right = self.build_merge_tree(rule.b)
                return MergeTreeNode(
                    token=token, left=left, right=right, rank=rule.rank
                )
        # No rule found → leaf
        return MergeTreeNode(token=token)

    # ------------------------------------------------------------------
    # Vocabulary suggestions
    # ------------------------------------------------------------------

    def suggest_removals(
        self, report: MergeAnalysisReport, max_suggestions: int = 10
    ) -> list[str]:
        """Return merge-result tokens that are low-impact and could be removed."""
        return [
            s.rule.result
            for s in report.low_impact_merges[:max_suggestions]
        ]

    def find_code_relevant_merges(self, keywords: Optional[list[str]] = None) -> list[MergeRule]:
        """Return merge rules whose result appears in common code patterns.

        Parameters
        ----------
        keywords:
            List of code-specific token strings to look for.  Defaults to a
            small built-in set of common code tokens.
        """
        if keywords is None:
            keywords = [
                "def", "return", "import", "class", "self", "for", "if",
                "elif", "else", "None", "True", "False", "in", "not",
                "and", "or", "->", "**", "//",
            ]
        kw_set = set(keywords)
        return [r for r in self._rules if r.result in kw_set]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_rules(self) -> int:
        return len(self._rules)

    @property
    def vocabulary(self) -> list[str]:
        """All unique tokens that appear as merge *results*."""
        return [r.result for r in self._rules]
