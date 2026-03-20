"""Ensemble Generation: generate multiple candidates and pick the best.

Instead of a single generation, produce N candidates with different
sampling parameters and select the best based on scoring criteria.
This improves reliability at the cost of N× compute.

For a TS dev: like running the same prompt N times and picking the best
result — similar to best-of-N sampling or self-consistency prompting.
"""

from dataclasses import dataclass, field
from typing import Callable
import re

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class GenerationCandidate:
    """A single generation candidate with its score."""
    text: str
    score: float
    params: dict = field(default_factory=dict)
    scores_breakdown: dict[str, float] = field(default_factory=dict)

    def __lt__(self, other: "GenerationCandidate") -> bool:
        return self.score < other.score


@dataclass
class EnsembleConfig:
    """Configuration for ensemble generation."""
    num_candidates: int = 5
    temperature_range: tuple[float, float] = (0.3, 1.0)
    top_k_range: tuple[int, int] = (10, 100)
    top_p_range: tuple[float, float] = (0.8, 0.95)
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "length": 0.1,
        "syntax": 0.3,
        "completeness": 0.3,
        "diversity": 0.1,
        "no_repetition": 0.2,
    })


class CodeScorer:
    """Score generated code on multiple quality dimensions."""

    def score(self, code: str, prompt: str = "") -> dict[str, float]:
        """Score code on multiple dimensions, each 0.0 to 1.0."""
        return {
            "length": self._score_length(code),
            "syntax": self._score_syntax(code),
            "completeness": self._score_completeness(code),
            "diversity": self._score_diversity(code),
            "no_repetition": self._score_no_repetition(code),
        }

    def weighted_score(self, code: str, weights: dict[str, float], prompt: str = "") -> float:
        """Compute weighted aggregate score."""
        scores = self.score(code, prompt)
        total = 0.0
        weight_sum = 0.0
        for key, weight in weights.items():
            if key in scores:
                total += scores[key] * weight
                weight_sum += weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def _score_length(self, code: str) -> float:
        """Score based on reasonable length (not too short, not too long)."""
        length = len(code.strip())
        if length < 10:
            return 0.1
        if length < 50:
            return 0.5
        if length > 5000:
            return 0.5  # Probably too long
        return 1.0

    def _score_syntax(self, code: str) -> float:
        """Score based on basic syntax validity (brace/paren matching)."""
        score = 1.0

        # Check brace balance
        if code.count("{") != code.count("}"):
            score -= 0.3

        # Check paren balance
        if code.count("(") != code.count(")"):
            score -= 0.2

        # Check bracket balance
        if code.count("[") != code.count("]"):
            score -= 0.2

        # Check for unclosed strings (simple heuristic)
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        if single_quotes % 2 != 0:
            score -= 0.15
        if double_quotes % 2 != 0:
            score -= 0.15

        return max(0.0, score)

    def _score_completeness(self, code: str) -> float:
        """Score based on code completeness indicators."""
        score = 0.5  # Base score
        code_lower = code.lower()

        # Has a return statement (for functions)
        if "return " in code_lower:
            score += 0.2

        # Has function/class definition
        if any(kw in code_lower for kw in ["function ", "class ", "const ", "let ", "var "]):
            score += 0.15

        # Ends with proper terminator
        stripped = code.rstrip()
        if stripped and stripped[-1] in "};)":
            score += 0.15

        return min(1.0, score)

    def _score_diversity(self, code: str) -> float:
        """Score based on token diversity (not just repeating the same thing)."""
        words = code.split()
        if len(words) < 3:
            return 0.3
        unique_ratio = len(set(words)) / len(words)
        return min(1.0, unique_ratio * 1.2)  # Scale up slightly

    def _score_no_repetition(self, code: str) -> float:
        """Score based on absence of repetitive patterns."""
        lines = code.splitlines()
        if len(lines) < 3:
            return 1.0

        # Check for repeated lines
        repeated = 0
        for i in range(1, len(lines)):
            if lines[i].strip() and lines[i].strip() == lines[i-1].strip():
                repeated += 1

        if repeated > len(lines) * 0.3:
            return 0.2
        if repeated > 0:
            return 0.7

        # Check for repeated multi-line blocks
        for block_size in range(2, min(5, len(lines) // 2)):
            for i in range(len(lines) - block_size * 2 + 1):
                block1 = "\n".join(lines[i:i+block_size])
                block2 = "\n".join(lines[i+block_size:i+block_size*2])
                if block1.strip() and block1.strip() == block2.strip():
                    return 0.3

        return 1.0


class EnsembleGenerator:
    """Generate multiple candidates and select the best."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.scorer = CodeScorer()
        self._last_candidates: list[GenerationCandidate] = []

    @property
    def last_candidates(self) -> list[GenerationCandidate]:
        """All candidates from the most recent generation, sorted by score."""
        return sorted(self._last_candidates, reverse=True)

    def generate_params_list(self) -> list[dict]:
        """Generate a list of diverse sampling parameter sets."""
        import random
        params_list = []
        cfg = self.config

        for i in range(cfg.num_candidates):
            # Spread parameters across the configured ranges
            t = i / max(1, cfg.num_candidates - 1)  # 0.0 to 1.0

            temp = cfg.temperature_range[0] + t * (cfg.temperature_range[1] - cfg.temperature_range[0])
            top_k = int(cfg.top_k_range[0] + t * (cfg.top_k_range[1] - cfg.top_k_range[0]))
            top_p = cfg.top_p_range[0] + t * (cfg.top_p_range[1] - cfg.top_p_range[0])

            params_list.append({
                "temperature": round(temp, 2),
                "top_k": top_k,
                "top_p": round(top_p, 2),
            })

        return params_list

    def select_best(
        self,
        candidates: list[str],
        params_list: list[dict] | None = None,
        prompt: str = "",
        custom_scorer: Callable[[str], float] | None = None,
    ) -> GenerationCandidate:
        """Score and select the best candidate.

        Args:
            candidates: List of generated code strings
            params_list: Sampling params used for each candidate
            prompt: Original prompt (for context-aware scoring)
            custom_scorer: Optional custom scoring function

        Returns:
            The best GenerationCandidate
        """
        if params_list is None:
            params_list = [{}] * len(candidates)

        scored: list[GenerationCandidate] = []
        for text, params in zip(candidates, params_list):
            if custom_scorer:
                score = custom_scorer(text)
                breakdown = {"custom": score}
            else:
                breakdown = self.scorer.score(text, prompt)
                score = self.scorer.weighted_score(
                    text, self.config.scoring_weights, prompt
                )

            scored.append(GenerationCandidate(
                text=text,
                score=score,
                params=params,
                scores_breakdown=breakdown,
            ))

        self._last_candidates = scored
        return max(scored)

    def print_candidates_report(self) -> None:
        """Print a summary of all candidates from the last generation."""
        from cola_coder.cli import cli

        candidates = self.last_candidates
        if not candidates:
            cli.warn("No candidates to report")
            return

        cli.header("Ensemble Results", f"{len(candidates)} candidates")
        for i, c in enumerate(candidates):
            status = "BEST" if i == 0 else f"#{i+1}"
            preview = c.text[:60].replace("\n", " ") + "..."
            cli.info(
                f"[{status}] score={c.score:.2f}",
                f"temp={c.params.get('temperature', '?')} | {preview}"
            )
            if c.scores_breakdown:
                parts = [f"{k}={v:.2f}" for k, v in c.scores_breakdown.items()]
                cli.dim(f"    {' | '.join(parts)}")
