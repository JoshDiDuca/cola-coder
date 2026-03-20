"""Multi-Step Reasoning: structured think → plan → code pipeline for code generation.

Introduces a three-phase generation pipeline with distinct structured sections:
  <think>...</think>  — problem understanding and analysis
  <plan>...</plan>    — structured solution outline (pseudocode-level)
  <code>...</code>    — full implementation

Inspired by "Chain of Code" (Li et al., 2023) which shows explicit intermediate
representations improve code generation accuracy by 8-15% on hard problems.

For a TS dev: like TypeScript's strict mode for reasoning — each phase has a
clearly-typed contract (think: analysis, plan: algorithm, code: implementation).
Human expert programmers follow this pattern naturally; this makes it explicit.

CLI flag: --multi-step-reasoning
"""

import re
from dataclasses import dataclass
from math import prod
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Heuristic keyword sets (shared with ProblemDecomposer and step estimator)
# ---------------------------------------------------------------------------

# Keywords suggesting algorithmic complexity
_ALGO_KEYWORDS = {
    "sort", "search", "binary", "graph", "tree", "heap", "hash",
    "dynamic", "recursion", "backtrack", "greedy", "optimal",
    "complexity", "matrix", "dp", "memoize", "recursive", "iterate",
    "algorithm", "traverse", "bfs", "dfs", "shortest", "path", "cycle",
    "topological", "partition", "divide", "conquer",
}

# Keywords suggesting distinct architectural concerns that can be decomposed
_DECOMPOSE_SIGNALS = {
    "authentication": "Implement authentication logic",
    "auth": "Implement authentication logic",
    "rate limiting": "Implement rate limiting",
    "rate-limiting": "Implement rate limiting",
    "database": "Implement database layer",
    "db": "Implement database layer",
    "cache": "Implement caching layer",
    "caching": "Implement caching layer",
    "api": "Design API interface",
    "rest": "Design REST endpoints",
    "graphql": "Design GraphQL schema",
    "validation": "Implement input validation",
    "error handling": "Implement error handling",
    "error-handling": "Implement error handling",
    "logging": "Add logging and observability",
    "test": "Write tests",
    "testing": "Write tests",
    "pagination": "Implement pagination",
    "search": "Implement search functionality",
    "filter": "Implement filtering logic",
    "websocket": "Implement WebSocket handling",
    "connection pooling": "Implement connection pooling",
    "connection-pooling": "Implement connection pooling",
    "middleware": "Implement middleware layer",
    "router": "Implement routing logic",
    "serialization": "Implement serialization / deserialization",
    "encryption": "Implement encryption",
    "hashing": "Implement hashing",
    "queue": "Implement queue / message passing",
    "worker": "Implement worker / background job",
    "scheduler": "Implement scheduling logic",
    "migration": "Write database migration",
}

# Complexity signal multipliers for step estimation
_STEP_BASE = 3         # minimum steps for any problem
_STEP_PER_ALGO_KW = 1  # extra steps per algorithmic keyword hit
_STEP_SIMPLE_MAX = 4   # cap for simple problems
_STEP_MEDIUM_MAX = 7   # cap for medium problems

# Patterns that suggest structural delimiters in a problem description
_CONJUNCTION_PATTERN = re.compile(
    r"\b(and|with|plus|including|also|as well as|,)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# ReasoningStep dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain.

    Attributes:
        step_number: 1-based index of this step in its chain.
        description: Short label for what this step accomplishes (e.g. "Understand the problem").
        content:     The actual reasoning text for this step.
        confidence:  Confidence in this step's conclusion, in [0.0, 1.0].
    """

    step_number: int
    description: str
    content: str
    confidence: float  # [0.0, 1.0]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )


# ---------------------------------------------------------------------------
# ReasoningChain
# ---------------------------------------------------------------------------

class ReasoningChain:
    """An ordered chain of ReasoningStep objects that models structured CoT.

    Analogy for TS devs: think of this as an immutable log of typed "reasoning
    events", similar to Redux actions — each step is dispatched in order, and
    the chain gives you a typed view of the full reasoning trace.

    Usage::

        chain = ReasoningChain()
        chain.add_step("Understand the problem", "Need to implement binary search", 0.95)
        chain.add_step("Design the algorithm", "Use iterative low/high approach", 0.9)
        prompt = chain.to_prompt()
    """

    def __init__(self) -> None:
        self._steps: list[ReasoningStep] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_step(self, description: str, content: str, confidence: float) -> ReasoningStep:
        """Append a new step to the chain and return it.

        Args:
            description: Short label for this step (e.g. "Understand the problem").
            content:     The reasoning text for this step.
            confidence:  Confidence score in [0.0, 1.0].

        Returns:
            The newly-created ReasoningStep (1-based step_number assigned automatically).
        """
        step = ReasoningStep(
            step_number=len(self._steps) + 1,
            description=description,
            content=content,
            confidence=confidence,
        )
        self._steps.append(step)
        return step

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_steps(self) -> list[ReasoningStep]:
        """Return all reasoning steps in insertion order."""
        return list(self._steps)

    def get_conclusion(self) -> Optional[ReasoningStep]:
        """Return the final step (the conclusion), or None if the chain is empty."""
        if not self._steps:
            return None
        return self._steps[-1]

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def total_confidence(self) -> float:
        """Return the product of all step confidences (joint confidence of the chain).

        Returns 0.0 for an empty chain (no evidence).

        Like TypeScript's intersection types: every step must hold for the
        conclusion to hold — so confidence compounds multiplicatively.
        """
        if not self._steps:
            return 0.0
        return float(prod(s.confidence for s in self._steps))

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_prompt(self) -> str:
        """Format the reasoning chain as a CoT prompt string.

        The output follows the think → plan → code structure used in training:

            <think>
            Step 1 — Understand the problem:
            Need to implement binary search
            ...
            </think>

        Returns:
            Formatted string ready to be prepended to a generation prompt.
        """
        if not self._steps:
            return "<think>\n</think>"

        lines: list[str] = ["<think>"]
        for step in self._steps:
            lines.append(
                f"Step {step.step_number} — {step.description} "
                f"[confidence: {step.confidence:.2f}]:"
            )
            lines.append(step.content)
            lines.append("")
        lines.append("</think>")
        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a summary dict for logging / debugging.

        Returns:
            Dict with keys:
              - num_steps: total number of steps
              - total_confidence: product of all step confidences
              - steps: list of step summary dicts (step_number, description, confidence)
              - conclusion: description of the final step, or None
        """
        conclusion = self.get_conclusion()
        return {
            "num_steps": len(self._steps),
            "total_confidence": self.total_confidence(),
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "confidence": s.confidence,
                }
                for s in self._steps
            ],
            "conclusion": conclusion.description if conclusion else None,
        }

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return (
            f"ReasoningChain(steps={len(self._steps)}, "
            f"total_confidence={self.total_confidence():.4f})"
        )


# ---------------------------------------------------------------------------
# ProblemDecomposer
# ---------------------------------------------------------------------------

class ProblemDecomposer:
    """Decomposes complex programming problems into sub-problems using heuristics.

    This is the static / rule-based "plan bootstrapper" — it does not require a
    model and is used to:
      1. Generate initial sub-problem lists for the planning phase.
      2. Estimate the number of reasoning steps a problem needs.

    For a TS dev: think of it as a TypeScript compiler's diagnostic pass —
    it reads the problem text and emits a list of structured concerns, each of
    which becomes a reasoning step in the chain.
    """

    def decompose(self, problem: str) -> list[str]:
        """Break a problem description into a list of sub-problems.

        Strategy:
          1. Scan for known architectural concern keywords (auth, caching, etc.)
             and emit a dedicated sub-problem for each one found.
          2. If conjunctions (", and, with, plus, ...") separate clauses,
             split into clause-level sub-problems.
          3. Always include a fallback "Core implementation" task.
          4. Deduplicate while preserving order.

        Args:
            problem: Natural-language description of the programming task.

        Returns:
            List of sub-problem strings (at least one entry).
        """
        lower = problem.lower()
        sub_problems: list[str] = []
        seen: set[str] = set()

        def _add(item: str) -> None:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                sub_problems.append(item.strip())

        # 1. Keyword-based architectural concerns
        for keyword, label in _DECOMPOSE_SIGNALS.items():
            if keyword in lower:
                _add(label)

        # 2. Conjunction-based clause splitting (only when no keyword hit dominates)
        if len(sub_problems) < 2:
            # Split on ", " or conjunctions to find discrete clauses
            clauses = re.split(r",\s*|\s+(?:and|with|plus|including)\s+", problem, flags=re.IGNORECASE)
            clauses = [c.strip() for c in clauses if len(c.strip()) > 8]
            for clause in clauses:
                _add(f"Implement: {clause}")

        # 3. Always ensure a core implementation task is present
        if not sub_problems:
            _add(f"Implement: {problem.strip()}")

        return sub_problems

    def estimate_steps(self, problem: str) -> int:
        """Estimate the number of reasoning steps needed for a problem.

        Heuristic:
          - Start at _STEP_BASE (3).
          - Add 1 per algorithmic keyword hit (sorted, search, dp, graph, …),
            up to a cap that depends on problem length.
          - Add 1 per distinct architectural concern found (auth, caching, …).
          - Simple one-liner problems are capped at _STEP_SIMPLE_MAX.
          - Longer / multi-concern problems can reach _STEP_MEDIUM_MAX+.

        Args:
            problem: Natural-language description of the programming task.

        Returns:
            Estimated step count (integer >= 1).
        """
        lower = problem.lower()
        word_count = len(problem.split())

        # Algorithmic keyword hits
        algo_hits = sum(1 for kw in _ALGO_KEYWORDS if kw in lower)

        # Architectural concern hits
        concern_hits = sum(1 for kw in _DECOMPOSE_SIGNALS if kw in lower)

        # Conjunction count as a proxy for multi-part problems
        conjunction_hits = len(_CONJUNCTION_PATTERN.findall(problem))

        steps = _STEP_BASE + (algo_hits * _STEP_PER_ALGO_KW) + concern_hits + (conjunction_hits // 2)

        # Simple / short problems get a tighter cap
        if word_count <= 6:
            steps = min(steps, _STEP_SIMPLE_MAX)
        elif word_count <= 20 and algo_hits == 0 and concern_hits == 0:
            steps = min(steps, _STEP_SIMPLE_MAX)

        return max(1, steps)


# ---------------------------------------------------------------------------
# format_chain_of_thought
# ---------------------------------------------------------------------------

def format_chain_of_thought(steps: list[str]) -> str:
    """Format a list of reasoning step strings into a CoT prompt block.

    Each step is wrapped with a numbered header inside a <think>...</think>
    block, ready to prepend to a generation prompt.

    Args:
        steps: Ordered list of step description strings. Each may already start
               with "Step N:" or be a plain description — both are accepted.

    Returns:
        Formatted CoT string, e.g.::

            <think>
            Step 1: analyze
            Step 2: implement
            </think>

    Example::

        >>> formatted = format_chain_of_thought(["analyze the problem", "write the code"])
        >>> "Step 1" in formatted
        True
    """
    if not steps:
        return "<think>\n</think>"

    lines: list[str] = ["<think>"]
    for i, step in enumerate(steps, start=1):
        text = step.strip()
        # If the step already carries a "Step N:" prefix, preserve it as-is.
        # Otherwise, add the numbered prefix.
        if re.match(r"^Step\s+\d+", text, re.IGNORECASE):
            lines.append(text)
        else:
            lines.append(f"Step {i}: {text}")
    lines.append("</think>")
    return "\n".join(lines)
