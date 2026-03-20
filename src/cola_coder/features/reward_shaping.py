"""Reward Shaping: composable partial-credit reward function for RLHF/GRPO training.

Combines syntax correctness, style quality, completeness, and brevity into a
single scalar reward. Replaces binary pass/fail with dense signals that give
partial credit, enabling meaningful policy gradients even on hard problems.

Total reward = sum(weight_i * score_i) for all components.
Default components: syntax (0.30), completeness (0.25), style (0.25), brevity (0.20).
"""

import ast
import re
import keyword
from dataclasses import dataclass
from typing import Callable, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RewardComponent:
    """A single named reward component with its weight and computed score."""
    name: str
    weight: float
    score: float

    @property
    def weighted_score(self) -> float:
        return self.weight * self.score


@dataclass
class RewardResult:
    """The result of a full reward computation."""
    total: float
    components: list[RewardComponent]
    details: dict


# ---------------------------------------------------------------------------
# RewardShaper
# ---------------------------------------------------------------------------

class RewardShaper:
    """Computes a composable scalar reward for generated Python code.

    Default weights:
        syntax:       0.30  (can the code be parsed?)
        completeness: 0.25  (does it address the prompt?)
        style:        0.25  (PEP-8-ish quality signals)
        brevity:      0.20  (penalise unnecessarily long code)

    All weights are re-normalised to sum to 1.0 before use.
    """

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "syntax": 0.30,
        "completeness": 0.25,
        "style": 0.25,
        "brevity": 0.20,
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Args:
            weights: Dict mapping component name -> weight.  Keys must include
                     at least the built-in names if you want to override them.
                     Extra keys are accepted when add_custom_component is used.
                     Weights are normalised internally so they don't need to
                     sum to exactly 1.0.
        """
        self._weights: dict[str, float] = dict(self._DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)

        # Custom scorer registry: name -> (weight, callable)
        self._custom: dict[str, tuple[float, Callable[[str], float]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_reward(self, code: str, prompt: str = "") -> RewardResult:
        """Compute the composite reward for *code* (optionally guided by *prompt*).

        Args:
            code:   The generated Python code string.
            prompt: The original user prompt / task description.

        Returns:
            RewardResult with total scalar, per-component breakdown, and details.
        """
        details: dict = {}

        # Built-in scores
        syn_score = self.syntax_reward(code)
        comp_score = self.completeness_reward(code, prompt)
        style_score = self.style_reward(code)
        brev_score = self.brevity_reward(code)

        raw_components = [
            ("syntax",       self._weights.get("syntax", 0.30),       syn_score),
            ("completeness", self._weights.get("completeness", 0.25),  comp_score),
            ("style",        self._weights.get("style", 0.25),         style_score),
            ("brevity",      self._weights.get("brevity", 0.20),       brev_score),
        ]

        # Custom components
        for name, (weight, scorer_fn) in self._custom.items():
            try:
                score = float(scorer_fn(code))
                score = max(0.0, min(1.0, score))
            except Exception:
                score = 0.0
            raw_components.append((name, weight, score))

        # Normalise weights so they sum to 1.0
        total_weight = sum(w for _, w, _ in raw_components) or 1.0
        components: list[RewardComponent] = []
        for name, weight, score in raw_components:
            norm_w = weight / total_weight
            components.append(RewardComponent(name=name, weight=norm_w, score=score))

        total = sum(c.weighted_score for c in components)
        total = max(0.0, min(1.0, total))

        details["raw_scores"] = {c.name: c.score for c in components}
        details["weights"] = {c.name: c.weight for c in components}
        details["code_length"] = len(code)
        details["prompt"] = prompt

        return RewardResult(total=total, components=components, details=details)

    def syntax_reward(self, code: str) -> float:
        """Return a [0, 1] score based on Python syntax validity.

        1.0  — parses cleanly with ast.parse
        0.5  — fails parse but passes a lightweight bracket-balance check
        0.0  — bracket imbalance or empty
        """
        if not code.strip():
            return 0.0
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return _naive_bracket_balance(code)

    def completeness_reward(self, code: str, prompt: str) -> float:
        """Return a [0, 1] score for how well *code* addresses *prompt*.

        Heuristics (language-agnostic, no external calls):
        - Keyword overlap between prompt tokens and code identifiers
        - Presence of a function/class definition (structural completeness)
        - Non-trivial body (at least one return/yield/assign)
        """
        if not code.strip():
            return 0.0

        score = 0.0

        # Structural: does it define something?
        has_def = bool(re.search(r"\bdef\s+\w+", code) or re.search(r"\bclass\s+\w+", code))
        if has_def:
            score += 0.35

        # Non-trivial body
        has_body = bool(
            re.search(r"\breturn\b", code)
            or re.search(r"\byield\b", code)
            or re.search(r"\w+\s*=\s*\S", code)
        )
        if has_body:
            score += 0.25

        # Keyword overlap with prompt
        if prompt.strip():
            prompt_tokens = set(_tokenize_words(prompt.lower()))
            code_tokens = set(_tokenize_words(code.lower()))
            # Remove Python keywords from code tokens so we only count semantic overlap
            code_tokens -= set(keyword.kwlist)
            overlap = prompt_tokens & code_tokens
            # Normalise: full overlap of meaningful words = 0.40 bonus
            if prompt_tokens:
                overlap_ratio = len(overlap) / max(len(prompt_tokens), 1)
                score += 0.40 * min(overlap_ratio * 2.0, 1.0)  # *2 so ~50% overlap = full bonus
        else:
            # No prompt supplied — give a neutral partial bonus
            score += 0.20

        return min(score, 1.0)

    def style_reward(self, code: str) -> float:
        """Return a [0, 1] score for code style quality.

        Checks (each adds to the score):
        - Has type annotations on function parameters / return
        - Uses descriptive names (>= 3 chars for non-loop vars)
        - Docstring present in first function/class
        - Not excessively wide lines (<=100 chars)
        - Has a blank line between top-level definitions
        """
        if not code.strip():
            return 0.0

        lines = code.splitlines()
        score = 0.0

        # Type annotations
        has_annotations = bool(
            re.search(r"def\s+\w+\s*\([^)]*:\s*\w", code)   # param annotations
            or re.search(r"\)\s*->\s*\w", code)              # return annotation
        )
        if has_annotations:
            score += 0.25

        # Descriptive names: identifiers used as params/variables should be >= 3 chars
        # (excluding loop variables i, j, k, x, y, z and _ prefix)
        param_names = re.findall(r"def\s+\w+\s*\(([^)]*)\)", code)
        all_params: list[str] = []
        for group in param_names:
            for p in group.split(","):
                name = p.strip().split(":")[0].strip().split("=")[0].strip()
                if name and not name.startswith("*") and not name.startswith("_"):
                    all_params.append(name)
        if all_params:
            long_enough = sum(1 for n in all_params if len(n) >= 3)
            score += 0.20 * (long_enough / len(all_params))
        else:
            score += 0.10  # neutral when no params detectable

        # Docstring
        has_docstring = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))
        if has_docstring:
            score += 0.20

        # Line length
        long_lines = sum(1 for ln in lines if len(ln) > 100)
        line_penalty = long_lines / max(len(lines), 1)
        score += 0.20 * (1.0 - min(line_penalty * 5.0, 1.0))

        # Blank line between top-level defs (PEP-8: two blank lines)
        top_level_defs = [i for i, ln in enumerate(lines) if re.match(r"^(def|class)\s+", ln)]
        if len(top_level_defs) >= 2:
            has_blank_sep = any(
                lines[top_level_defs[k] - 1].strip() == ""
                for k in range(1, len(top_level_defs))
                if top_level_defs[k] > 0
            )
            if has_blank_sep:
                score += 0.15
        else:
            score += 0.15  # single definition: not penalised

        return min(score, 1.0)

    def brevity_reward(self, code: str) -> float:
        """Return a [0, 1] score that penalises unnecessarily long code.

        A reasonable single-function implementation is 5-30 lines.
        < 5 lines  → 0.70 (might be too terse / incomplete)
        5-30 lines → 1.00 (sweet spot)
        31-60      → linear decay to 0.60
        61-120     → linear decay to 0.30
        > 120      → 0.10
        """
        if not code.strip():
            return 0.0

        # Count non-blank, non-comment lines
        content_lines = [
            ln for ln in code.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        n = len(content_lines)

        if n < 5:
            return 0.70
        if n <= 30:
            return 1.00
        if n <= 60:
            return 1.0 - (n - 30) / 30.0 * 0.40   # 1.0 → 0.60
        if n <= 120:
            return 0.60 - (n - 60) / 60.0 * 0.30   # 0.60 → 0.30
        return 0.10

    def add_custom_component(
        self,
        name: str,
        weight: float,
        scorer_fn: Callable[[str], float],
    ) -> None:
        """Register a custom reward component.

        Args:
            name:      Unique identifier for the component.
            weight:    Relative weight (will be normalised alongside built-in weights).
            scorer_fn: Callable that accepts a code string and returns a float in [0, 1].
        """
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        self._custom[name] = (weight, scorer_fn)
        # Also store in _weights for consistency
        self._weights[name] = weight


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _naive_bracket_balance(code: str) -> float:
    """Return 0.5 if brackets balance, 0.0 otherwise (used as syntax fallback)."""
    stack: list[str] = []
    pairs = {")": "(", "}": "{", "]": "["}
    in_string = False
    string_char = ""
    i = 0
    while i < len(code):
        ch = code[i]
        if in_string:
            if ch == "\\" and i + 1 < len(code):
                i += 2  # skip escaped char
                continue
            if ch == string_char:
                in_string = False
        elif ch in ('"', "'"):
            # Detect triple quotes
            triple = code[i:i+3]
            if triple in ('"""', "'''"):
                end = code.find(triple, i + 3)
                i = end + 3 if end != -1 else len(code)
                continue
            in_string = True
            string_char = ch
        elif ch in "({[":
            stack.append(ch)
        elif ch in ")}]":
            if not stack or stack[-1] != pairs[ch]:
                return 0.0
            stack.pop()
        i += 1
    return 0.5 if not stack else 0.0


def _tokenize_words(text: str) -> list[str]:
    """Extract word tokens (alpha sequences, length >= 2) from text."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 2]
