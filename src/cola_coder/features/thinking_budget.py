"""Thinking Budget Controller: dynamically allocate token budgets for chain-of-thought reasoning.

Controls the <think>...</think> section token budget based on estimated problem complexity.
Simple problems get a small budget (e.g. 128 tokens); hard problems get a larger one
(e.g. 1024 tokens). Tracks consumption and can signal when to stop thinking.

For a TS dev: like a rate-limiter for reasoning — the model has a "credit" of thinking
tokens proportional to how hard the problem is. Easy problems get fewer credits.

CLI flag: --thinking-budget
"""

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Heuristic signal sets (from the research plan)
# ---------------------------------------------------------------------------

_ALGO_KEYWORDS = {
    "sort", "search", "binary", "graph", "tree", "heap", "hash",
    "dynamic", "recursion", "backtrack", "greedy", "optimal",
    "complexity", "O(n", "O(log", "matrix", "dp", "memoize",
    "recursive", "iterate", "algorithm", "traverse", "bfs", "dfs",
    "shortest", "path", "cycle", "topological", "partition",
}

_DS_KEYWORDS = {
    "linkedlist", "linked list", "stack", "queue", "trie",
    "segment tree", "fenwick", "disjoint", "union-find", "adjacency",
    "topological", "balanced", "avl", "b-tree", "red-black",
    "priority queue", "deque", "circular",
}

_LANG_COMPLEXITY_KEYWORDS = {
    "generics", "union types", "conditional types", "mapped types",
    "intersection", "infer", "template literal", "decorator",
    "covariant", "contravariant", "discriminated",
    "parser", "lexer", "ast", "abstract syntax", "grammar", "combinator",
    "macro", "trait", "lifetime", "borrow",
}

# Budget table: score range -> token budget
_BUDGET_TABLE = [
    (0.0, 0.2, 64),    # trivial
    (0.2, 0.4, 128),   # easy
    (0.4, 0.6, 256),   # medium
    (0.6, 0.8, 512),   # hard
    (0.8, 1.0, 1024),  # very hard
]


def _score_to_budget(score: float, complexity_scale: float = 1.0) -> int:
    """Map a 0-1 difficulty score to a token budget, scaled by complexity_scale."""
    for lo, hi, budget in _BUDGET_TABLE:
        if lo <= score < hi:
            return max(1, int(budget * complexity_scale))
    # score == 1.0 lands here
    return max(1, int(1024 * complexity_scale))


# ---------------------------------------------------------------------------
# Standalone estimate_complexity function
# ---------------------------------------------------------------------------

def estimate_complexity(prompt: str) -> float:
    """Estimate prompt complexity as a float in [0.0, 1.0].

    Uses the weighted signal approach from the research plan:
      - Prompt length (tokens)          weight 0.25
      - Algorithmic keyword presence    weight 0.30
      - Cyclomatic keyword density      weight 0.20
      - Data structure keywords         weight 0.15
      - Explicit complexity hints       weight 0.10

    Args:
        prompt: The user prompt or problem description.

    Returns:
        Complexity score in [0.0, 1.0]. Higher = harder.
    """
    tokens = prompt.split()
    token_count = max(len(tokens), 1)
    lower = prompt.lower()

    # --- length signal ---
    length_score = min(token_count / 512, 1.0)

    # --- algorithmic keywords ---
    algo_hits = sum(1 for kw in _ALGO_KEYWORDS if kw in lower)
    algo_score = min(algo_hits / 5, 1.0)

    # --- language / advanced type system keywords ---
    lang_hits = sum(1 for kw in _LANG_COMPLEXITY_KEYWORDS if kw in lower)
    # Treat these as additional algo signal (blended in)
    algo_score = min(algo_score + lang_hits / 4, 1.0)

    # --- cyclomatic keyword density ---
    control_matches = re.findall(r"\b(for|while|if|else|switch|case|catch|try)\b", lower)
    control_density = len(control_matches) / token_count
    control_score = min(control_density * 20, 1.0)

    # --- data structure keywords ---
    ds_hits = sum(1 for kw in _DS_KEYWORDS if kw in lower)
    ds_score = min(ds_hits / 3, 1.0)

    # --- explicit complexity / optimization hints ---
    has_complexity_hint = bool(
        re.search(r"O\s*\(", prompt)
        or re.search(r"\boptimize\b", lower)
        or re.search(r"\boptimal\b", lower)
        or re.search(r"\bperformance\b", lower)
        or re.search(r"\befficient\b", lower)
    )
    explicit_hint = 1.0 if has_complexity_hint else 0.0

    score = (
        0.25 * length_score
        + 0.30 * algo_score
        + 0.20 * control_score
        + 0.15 * ds_score
        + 0.10 * explicit_hint
    )

    return float(min(max(score, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThinkingBudgetConfig:
    """Configuration for the thinking budget controller.

    Attributes:
        min_tokens: Minimum token budget regardless of score.
        max_tokens: Hard cap on token budget.
        complexity_scale: Multiplier applied to the base budget from the table.
            Values > 1.0 give more headroom; < 1.0 tighten the budget.
        warn_threshold: Fraction of budget at which to issue a soft warning (0-1).
        repetition_window: Sliding window size (tokens) for repetition detection.
        repetition_ngram: N-gram size for repetition detection.
        repetition_threshold: How many times an n-gram must appear to be "stuck".
    """
    min_tokens: int = 64
    max_tokens: int = 1024
    complexity_scale: float = 1.0
    warn_threshold: float = 0.8
    repetition_window: int = 64
    repetition_ngram: int = 4
    repetition_threshold: int = 3


# ---------------------------------------------------------------------------
# ThinkingBudgetController
# ---------------------------------------------------------------------------

class ThinkingBudgetController:
    """Controls the thinking token budget for a generation session.

    Usage:
        config = ThinkingBudgetConfig()
        controller = ThinkingBudgetController(config)
        budget = controller.estimate_budget(prompt)

        # During generation loop:
        while generating:
            if not controller.should_continue_thinking(tokens_used, budget):
                force_close_think_tag()
    """

    def __init__(self, config: ThinkingBudgetConfig) -> None:
        self.config = config
        self._total_estimated: int = 0
        self._total_used: int = 0
        self._call_count: int = 0
        self._budget_exceeded_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def estimate_budget(self, prompt: str, complexity: float | None = None) -> int:
        """Estimate the thinking token budget for a prompt.

        Args:
            prompt: The user prompt or problem description.
            complexity: Optional pre-computed complexity score in [0.0, 1.0].
                If None, complexity is computed via estimate_complexity().

        Returns:
            Token budget (integer, clamped to [min_tokens, max_tokens]).
        """
        if complexity is None:
            complexity = estimate_complexity(prompt)

        raw_budget = _score_to_budget(complexity, self.config.complexity_scale)
        budget = max(self.config.min_tokens, min(raw_budget, self.config.max_tokens))

        self._total_estimated += budget
        self._call_count += 1
        return budget

    def should_continue_thinking(self, tokens_used: int, budget: int) -> bool:
        """Return True if thinking should continue, False if budget is exhausted.

        Args:
            tokens_used: Number of thinking tokens consumed so far.
            budget: The budget returned by estimate_budget().

        Returns:
            True  → keep thinking
            False → stop (budget exhausted)
        """
        if tokens_used >= budget:
            self._budget_exceeded_count += 1
            return False
        return True

    def adjust_budget(self, initial_budget: int, partial_output: str) -> int:
        """Dynamically adjust the budget based on partial thinking output so far.

        Strategy:
        - If the partial output looks like it's making progress (unique sentences,
          diverse vocabulary), allow up to max_tokens.
        - If repetition is detected in the text, shrink the budget to current usage
          plus a small grace margin (force an early stop).
        - Otherwise return the initial budget unchanged.

        Args:
            initial_budget: The budget originally assigned.
            partial_output: The thinking text generated so far.

        Returns:
            Adjusted budget (integer).
        """
        if not partial_output:
            return initial_budget

        # Rough token count from whitespace split
        words = partial_output.split()
        tokens_so_far = len(words)

        # Detect text-level repetition: check if any 4-word phrase repeats 3+ times
        if self._text_repetition_detected(words):
            # Shrink budget to current usage + small grace (16 tokens)
            adjusted = max(self.config.min_tokens, tokens_so_far + 16)
            return min(adjusted, self.config.max_tokens)

        # If the output is already dense (long sentences, varied vocab), give a
        # small extension bonus — up to 10% over initial, still capped at max.
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio > 0.6 and tokens_so_far > initial_budget * 0.5:
            bonus = int(initial_budget * 0.1)
            return min(initial_budget + bonus, self.config.max_tokens)

        return initial_budget

    def classify_complexity(self, prompt: str) -> str:
        """Classify a prompt as 'simple', 'medium', or 'complex'.

        Uses the same heuristics as estimate_complexity() but returns a
        human-readable label.

        Args:
            prompt: The user prompt or problem description.

        Returns:
            One of: 'simple', 'medium', 'complex'
        """
        score = estimate_complexity(prompt)
        if score < 0.4:
            return "simple"
        elif score < 0.65:
            return "medium"
        else:
            return "complex"

    def summary(self) -> dict:
        """Return usage statistics across all calls to this controller instance.

        Returns:
            Dict with keys:
              - call_count: number of estimate_budget() calls
              - total_estimated_tokens: sum of all budgets returned
              - total_used_tokens: sum reported via record_usage()
              - budget_exceeded_count: times should_continue_thinking() returned False
              - avg_budget: average budget per call
        """
        avg = self._total_estimated / max(self._call_count, 1)
        return {
            "call_count": self._call_count,
            "total_estimated_tokens": self._total_estimated,
            "total_used_tokens": self._total_used,
            "budget_exceeded_count": self._budget_exceeded_count,
            "avg_budget": round(avg, 1),
        }

    def record_usage(self, tokens_used: int) -> None:
        """Record actual token usage for a completed thinking session.

        Call this after each generation completes to populate summary() stats.

        Args:
            tokens_used: Actual think tokens consumed.
        """
        self._total_used += tokens_used

    # ------------------------------------------------------------------
    # Repetition detection (token-id level, mirroring the research plan)
    # ------------------------------------------------------------------

    @staticmethod
    def is_repetition(
        token_ids: list[int],
        window: int = 64,
        ngram: int = 4,
        threshold: int = 3,
    ) -> bool:
        """Detect if a token sequence has a repeating n-gram (model is stuck).

        Sliding window n-gram overlap check over the last `window` tokens.
        If any n-gram appears >= `threshold` times, the model is looping.

        Args:
            token_ids: List of generated token IDs (thinking section).
            window: How many recent tokens to examine.
            ngram: N-gram size.
            threshold: Repetition count that signals a loop.

        Returns:
            True if a repeating loop is detected.
        """
        if len(token_ids) < window:
            return False
        recent = token_ids[-window:]
        counts: dict[tuple, int] = {}
        for i in range(len(recent) - ngram + 1):
            gram = tuple(recent[i : i + ngram])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _text_repetition_detected(self, words: list[str]) -> bool:
        """Check for repeated 4-word phrases in the word list (text-level analog)."""
        ngram = 4
        threshold = self.config.repetition_threshold
        window_size = min(self.config.repetition_window, len(words))
        if len(words) < ngram:
            return False
        recent = words[-window_size:]
        counts: dict[tuple, int] = {}
        for i in range(len(recent) - ngram + 1):
            gram = tuple(recent[i : i + ngram])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= threshold:
                return True
        return False
