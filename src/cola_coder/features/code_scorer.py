"""Comprehensive code quality scoring system.

Replaces binary pass/fail (quality_filter.py) with continuous 0.0-1.0 scores
so training samples can be *weighted* instead of just accepted or rejected.

The core idea maps to something familiar from TypeScript land:
    pass/fail  ≡  TypeScript error: "does not compile"
    0.0-1.0    ≡  ESLint severity levels: off / warn / error with numeric weight

Why continuous scores?
    In machine learning, "sample weighting" lets you say "learn a lot from this
    example" or "learn a little from this example" without throwing anything away.
    A file that's not great but isn't garbage still teaches the model something —
    it just shouldn't count as much as an excellent file.

    High-quality code → weight 2.0  (model trains harder on it)
    Average code      → weight 1.0  (baseline)
    Poor code         → weight 0.3  (barely contributes)
    Garbage           → weight 0.0  (excluded)

How this fits the pipeline:
    Stream from HuggingFace
        → quality_filter.py  (hard reject the truly awful stuff)
        → code_scorer.py     (assign weights to everything that made it through)
        → Tokenizer
        → Weighted training loop

Feature toggle pattern (project-wide convention):
    Set FEATURE_ENABLED = False to disable scoring and fall back to flat weights.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field

# HeuristicQualityScorer lives in the filters package.  We reuse its sub-scores
# rather than reimplementing them.  If the import fails (e.g. in an isolated
# test environment) we fall back gracefully.
try:
    from cola_coder.data.filters.quality_classifier import HeuristicQualityScorer
    _HEURISTIC_AVAILABLE = True
except ImportError:
    _HEURISTIC_AVAILABLE = False
    HeuristicQualityScorer = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if code scoring is active.

    When False, CodeScorer.score_to_weight() returns 1.0 for everything,
    effectively disabling weighting without changing the rest of the pipeline.
    """
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeQualityScore:
    """The full quality assessment for one code file.

    Think of this like a TypeScript compiler result: it gives you not just
    "pass/fail" but a breakdown of *which* rules fired and *how badly*.

    Attributes:
        overall:     Composite score 0.0-1.0.  Higher = better.
        breakdown:   Dict of signal_name -> sub-score (each 0.0-1.0).
                     Useful for debugging why a file scored the way it did.
        tier:        Human-readable bucket: "excellent", "good", "average",
                     "poor", or "reject".
        issues:      List of specific problems found (like ESLint error messages).
        suggestions: Actionable improvements the author could make.
    """

    overall: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    tier: str = "reject"
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Clamp overall to [0, 1] in case of floating-point drift
        self.overall = max(0.0, min(1.0, self.overall))


# ---------------------------------------------------------------------------
# Tier and weight tables
# ---------------------------------------------------------------------------

# Tier thresholds (lower bound inclusive)
_TIERS: list[tuple[float, str]] = [
    (0.8, "excellent"),
    (0.6, "good"),
    (0.4, "average"),
    (0.2, "poor"),
    (0.0, "reject"),
]

# Training weight per tier.  "reject" gets 0.0 so it never contributes to
# gradient updates (but we still score it so callers know *why* it's rejected).
_TIER_WEIGHTS: dict[str, float] = {
    "excellent": 2.0,
    "good":      1.5,
    "average":   1.0,
    "poor":      0.3,
    "reject":    0.0,
}


def _tier_for_score(score: float) -> str:
    """Map a 0-1 score to a tier name."""
    for threshold, name in _TIERS:
        if score >= threshold:
            return name
    return "reject"


# ---------------------------------------------------------------------------
# Language detection helpers (mirror quality_filter.py logic)
# ---------------------------------------------------------------------------

def _looks_like_python(code: str) -> bool:
    """Quick heuristic: does this look like Python source?"""
    header = code[:2000]
    js_ts_signals = ["const ", "let ", "=> ", "require(", "export ", "interface ", "type "]
    if sum(1 for s in js_ts_signals if s in header) >= 2:
        return False
    py_indicators = ["def ", "self.", "if __name__", "elif ", "except ", "print(", "#!/usr/bin"]
    return sum(1 for ind in py_indicators if ind in header) >= 2


def _looks_like_js_ts(code: str) -> bool:
    """Quick heuristic: does this look like JavaScript/TypeScript source?"""
    header = code[:2000]
    indicators = [
        "function ", "const ", "let ", "var ", "=> ", "require(",
        "import ", "export ", "interface ", "type ", "async ",
    ]
    return sum(1 for ind in indicators if ind in header) >= 2


def _detect_language(code: str, language: str) -> str:
    """Return a canonical language string from hint or auto-detection.

    Returns one of: "python", "typescript", "javascript", or "unknown".
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        return "python"
    if lang in ("typescript", "ts", "tsx"):
        return "typescript"
    if lang in ("javascript", "js", "jsx"):
        return "javascript"

    # Auto-detect
    if _looks_like_python(code):
        return "python"
    if _looks_like_js_ts(code):
        return "typescript"  # treat JS and TS the same for scoring purposes
    return "unknown"


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class CodeScorer:
    """Assign continuous 0.0-1.0 quality scores to code files.

    Combines:
      - All checks from quality_filter.py (converted to continuous scores)
      - The HeuristicQualityScorer sub-scores from quality_classifier.py
      - New signals: modernness, error handling, security

    Usage:
        scorer = CodeScorer()
        result = scorer.score(code_text, language="python")
        weight = scorer.score_to_weight(result)
        # weight is in 0.1-2.0 range; use it in your training loop

    The scorer is stateless — every `score()` call is independent.
    It is safe to call from multiple threads (no shared mutable state).
    """

    # Composite weights.  Must sum to 1.0.
    # Think of this like CSS specificity rules: each signal contributes
    # exactly this fraction to the final score.
    _WEIGHTS: dict[str, float] = {
        "length":         0.05,
        "line_quality":   0.05,
        "structure":      0.15,
        "naming":         0.12,
        "comments":       0.10,
        "documentation":  0.10,
        "complexity":     0.08,
        "formatting":     0.05,
        "duplication":    0.08,
        "syntax":         0.10,
        "modernness":     0.05,
        "error_handling": 0.04,
        "security":       0.03,
    }

    def __init__(self) -> None:
        # Optionally reuse HeuristicQualityScorer for its naming/structure
        # sub-scores.  If unavailable, we compute those ourselves.
        self._heuristic = HeuristicQualityScorer() if _HEURISTIC_AVAILABLE else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, code: str, language: str = "") -> CodeQualityScore:
        """Score a single code file.

        Args:
            code:     Raw source code as a string.
            language: Optional language hint ("python", "typescript", etc.).
                      If omitted, auto-detected from content.

        Returns:
            CodeQualityScore with overall (0-1), breakdown by signal,
            tier ("excellent"/"good"/"average"/"poor"/"reject"),
            issues (problems found), and suggestions (how to improve).
        """
        if not code or not code.strip():
            return CodeQualityScore(
                overall=0.0,
                breakdown={},
                tier="reject",
                issues=["empty file"],
                suggestions=["add actual code content"],
            )

        lang = _detect_language(code, language)
        issues: list[str] = []
        suggestions: list[str] = []

        # Compute each signal
        breakdown: dict[str, float] = {
            "length":         self._score_length(code),
            "line_quality":   self._score_line_quality(code),
            "structure":      self._score_structure(code, lang),
            "naming":         self._score_naming(code, lang),
            "comments":       self._score_comments(code, lang),
            "documentation":  self._score_documentation(code, lang),
            "complexity":     self._score_complexity(code),
            "formatting":     self._score_formatting(code),
            "duplication":    self._score_duplication(code),
            "syntax":         self._score_syntax(code, lang),
            "modernness":     self._score_modernness(code, lang),
            "error_handling": self._score_error_handling(code, lang),
            "security":       self._score_security(code),
        }

        # Weighted average
        overall = sum(
            breakdown[key] * weight
            for key, weight in self._WEIGHTS.items()
        )
        overall = max(0.0, min(1.0, overall))

        # Collect issues and suggestions from low-scoring signals
        self._collect_feedback(breakdown, lang, issues, suggestions)

        tier = _tier_for_score(overall)

        return CodeQualityScore(
            overall=overall,
            breakdown=breakdown,
            tier=tier,
            issues=issues,
            suggestions=suggestions,
        )

    def score_batch(
        self,
        codes: list[str],
        language: str = "",
    ) -> list[CodeQualityScore]:
        """Score multiple code files.

        Straightforward loop — kept separate from `score()` so callers
        can swap this for a parallelised version later without changing
        the call site.  (Like making a function async in TS: the interface
        stays the same, the internals change.)

        Args:
            codes:    List of raw source code strings.
            language: Optional language hint applied to every file.

        Returns:
            List of CodeQualityScore in the same order as `codes`.
        """
        return [self.score(code, language=language) for code in codes]

    def score_to_weight(self, score: CodeQualityScore) -> float:
        """Convert a CodeQualityScore to a training sample weight.

        The weight controls how much this sample contributes to gradient
        updates.  In PyTorch terms this is the per-sample weight passed to
        a weighted loss or a WeightedRandomSampler.

        Mapping:
            "excellent" (0.8+)  → 2.0   (train hard on this)
            "good"      (0.6+)  → 1.5
            "average"   (0.4+)  → 1.0   (baseline)
            "poor"      (0.2+)  → 0.3   (barely contributes)
            "reject"    (<0.2)  → 0.0   (excluded from loss)

        When FEATURE_ENABLED is False, always returns 1.0.

        Within each tier we do a small linear interpolation so the weight
        curve is smooth rather than a staircase:

            weight = tier_weight * lerp_factor

        where lerp_factor nudges the weight toward the next tier's weight
        based on where in the tier range the score falls.

        Args:
            score: A CodeQualityScore returned by `score()`.

        Returns:
            Float in [0.0, 2.0].
        """
        if not FEATURE_ENABLED:
            return 1.0

        tier = score.tier
        overall = score.overall

        base_weight = _TIER_WEIGHTS[tier]

        if tier == "reject":
            return 0.0

        # Smooth interpolation within tier to avoid hard jumps.
        # Find the tier's lower and upper bounds.
        tier_bounds = {
            "excellent": (0.8, 1.0),
            "good":      (0.6, 0.8),
            "average":   (0.4, 0.6),
            "poor":      (0.2, 0.4),
        }
        lo, hi = tier_bounds.get(tier, (0.0, 1.0))
        t = (overall - lo) / (hi - lo) if hi > lo else 0.5  # 0.0 at bottom of tier, 1.0 at top

        # Interpolate toward the next tier's weight
        next_weight_map = {
            "excellent": 2.0,   # already at max
            "good":      2.0,   # nudge toward excellent
            "average":   1.5,   # nudge toward good
            "poor":      1.0,   # nudge toward average
        }
        next_weight = next_weight_map.get(tier, base_weight)

        weight = base_weight + t * (next_weight - base_weight)
        return max(0.0, min(2.0, weight))

    # ------------------------------------------------------------------
    # Individual signal scorers
    # ------------------------------------------------------------------
    # Each returns a float in [0.0, 1.0].  Higher = better.
    # Design principle: be generous — these nudge scores, they don't
    # gatekeep.  Hard rejection already happened in quality_filter.py.

    def _score_length(self, code: str) -> float:
        """Score based on file length (line count).

        Sweet spot: 10-500 lines.  Very short files are boilerplate,
        very long files are usually auto-generated or bundled.

        Like a TypeScript file: a 5-line file probably isn't useful,
        a 10,000-line file is probably a generated bundle.
        """
        lines = code.splitlines()
        n = len(lines)

        if n < 5:
            return 0.1   # trivially short
        if n < 10:
            return 0.4
        if n <= 300:
            return 1.0   # sweet spot
        if n <= 500:
            return 0.9
        if n <= 1000:
            return 0.7
        if n <= 2000:
            return 0.5
        if n <= 5000:
            return 0.3
        return 0.1       # extremely long — probably generated

    def _score_line_quality(self, code: str) -> float:
        """Score average and maximum line length.

        Minified/bundled code has very long lines.
        Normal source averages 30-80 chars; max under 120 is ideal.
        """
        lines = code.splitlines()
        if not lines:
            return 0.0

        lengths = [len(line) for line in lines]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)

        # Hard red flags (minified code)
        if max_length > 500:
            return 0.1
        if avg_length > 200:
            return 0.15

        # Score avg length
        if avg_length < 10:
            score_avg = 0.3   # lines are trivially short
        elif avg_length <= 80:
            score_avg = 1.0
        elif avg_length <= 100:
            score_avg = 0.8
        elif avg_length <= 120:
            score_avg = 0.6
        else:
            score_avg = 0.4

        # Score max length
        if max_length <= 100:
            score_max = 1.0
        elif max_length <= 120:
            score_max = 0.9
        elif max_length <= 200:
            score_max = 0.7
        elif max_length <= 300:
            score_max = 0.5
        else:
            score_max = 0.3

        return (score_avg * 0.6 + score_max * 0.4)

    def _score_structure(self, code: str, language: str) -> float:
        """Score code structure: functions, classes, imports.

        Well-structured code is organised into functions and classes rather
        than being a wall of top-level statements.

        If HeuristicQualityScorer is available, delegates to it so we stay
        consistent with the existing classifier.
        """
        if self._heuristic is not None:
            lines = code.splitlines()
            return self._heuristic._score_structure(code, lines, language)

        # Fallback implementation
        lines = code.splitlines()
        func_count = 0
        class_count = 0
        import_count = 0

        for line in lines:
            stripped = line.strip()
            if language == "python":
                if stripped.startswith("def "):
                    func_count += 1
                elif stripped.startswith("class "):
                    class_count += 1
                elif stripped.startswith(("import ", "from ")):
                    import_count += 1
            else:  # JS/TS/unknown
                if re.match(r"(export\s+)?(async\s+)?function\s+\w+", stripped):
                    func_count += 1
                elif re.match(r"(export\s+)?(default\s+)?class\s+\w+", stripped):
                    class_count += 1
                elif stripped.startswith("import "):
                    import_count += 1
                # Arrow functions: const foo = (args) =>
                if re.match(r"(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?\(", stripped):
                    func_count += 1

        total_lines = max(len(lines), 1)
        has_structure = func_count > 0 or class_count > 0

        if not has_structure:
            return 0.3 if total_lines < 20 else 0.2

        func_density = func_count / total_lines * 100
        if 3 <= func_density <= 15:
            base = 1.0
        elif 1 <= func_density < 3 or 15 < func_density <= 25:
            base = 0.7
        else:
            base = 0.4

        if class_count > 0:
            base = min(1.0, base + 0.1)
        if import_count > 0:
            base = min(1.0, base + 0.05)

        return base

    def _score_naming(self, code: str, language: str) -> float:
        """Score naming convention consistency.

        Looks for: snake_case (Python), camelCase (JS/TS), PascalCase (classes),
        UPPER_CASE (constants).  Penalises single-character names and
        inconsistent mixing.

        Delegates to HeuristicQualityScorer when available.
        """
        if self._heuristic is not None:
            return self._heuristic._score_naming(code, language)

        # Fallback: same logic as quality_filter.check_naming_quality
        name_pattern = re.compile(
            r'(?:def |function |const |let |var |export (?:const |let |function )?)([a-zA-Z_]\w*)'
        )
        names = name_pattern.findall(code[:10000])

        if len(names) < 5:
            return 0.5   # not enough data

        acceptable_short = {"i", "j", "k", "n", "x", "y", "e", "fn", "cb", "db", "id", "ok"}
        meaningful = [n for n in names if n.lower() not in acceptable_short]

        if not meaningful:
            return 0.5

        avg_len = sum(len(n) for n in meaningful) / len(meaningful)

        snake_count  = sum(1 for n in meaningful if re.match(r'^[a-z][a-z0-9_]*$', n))
        camel_count  = sum(1 for n in meaningful if re.match(r'^[a-z][a-zA-Z0-9]*$', n))
        pascal_count = sum(1 for n in meaningful if re.match(r'^[A-Z][a-zA-Z0-9]*$', n))
        upper_count  = sum(1 for n in meaningful if re.match(r'^[A-Z][A-Z0-9_]+$', n))
        single_char  = sum(1 for n in meaningful if len(n) <= 1)

        total = len(meaningful)

        if single_char / total > 0.5:
            return 0.2   # too many single-char names

        max_convention = max(snake_count, camel_count)
        consistency = (max_convention + pascal_count + upper_count) / total

        if avg_len < 3.0:
            return 0.25

        if consistency > 0.8:
            score = 1.0
        elif consistency > 0.6:
            score = 0.7
        elif consistency > 0.4:
            score = 0.5
        else:
            score = 0.3

        return score

    def _score_comments(self, code: str, language: str) -> float:
        """Score comment ratio: some comments = good, none or wall-of-text = bad.

        Sweet spot: 5-25% of characters are in comments.

        Delegates to HeuristicQualityScorer when available.
        """
        if self._heuristic is not None:
            lines = code.splitlines()
            return self._heuristic._score_comment_ratio(code, lines, language)

        # Fallback
        total_chars = max(len(code), 1)
        comment_chars = 0

        for line in code.splitlines():
            stripped = line.strip()
            if (stripped.startswith("#") or stripped.startswith("//")
                    or stripped.startswith("/*") or stripped.startswith("*")):
                comment_chars += len(line)

        ratio = comment_chars / total_chars

        if 0.05 <= ratio <= 0.25:
            return 1.0
        if 0.02 <= ratio < 0.05 or 0.25 < ratio <= 0.40:
            return 0.6
        if ratio < 0.02:
            return 0.3
        return 0.2   # > 40% comments — probably a license dump

    def _score_documentation(self, code: str, language: str) -> float:
        """Score docstring / JSDoc presence and quantity.

        More docstrings on more functions = higher score.

        Delegates to HeuristicQualityScorer when available.
        """
        if self._heuristic is not None:
            return self._heuristic._score_docstrings(code, language)

        # Fallback
        has_docs = '"""' in code or "'''" in code or "/**" in code

        if not has_docs:
            lines = code.splitlines()
            # Require at least one doc line per 50 lines
            if len(lines) > 50:
                return 0.2
            return 0.4   # short files without docs are borderline ok

        # Count documentation blocks
        doc_count = code.count('"""') // 2 + code.count("/**")
        if doc_count >= 5:
            return 1.0
        if doc_count >= 3:
            return 0.9
        if doc_count >= 1:
            return 0.7
        return 0.4

    def _score_complexity(self, code: str) -> float:
        """Score complexity density: number of control-flow keywords per line.

        Too many per line → spaghetti code.
        Too few → probably declarations-only (not penalised, just neutral).

        Delegates to HeuristicQualityScorer when available.
        """
        if self._heuristic is not None:
            lines = code.splitlines()
            return self._heuristic._score_complexity_density(code, lines)

        lines = code.splitlines()
        if not lines:
            return 0.0

        complexity_keywords = re.findall(
            r"\b(if|else|elif|for|while|switch|case|try|catch|except|finally)\b",
            code,
        )
        density = len(complexity_keywords) / max(len(lines), 1)

        if density == 0:
            return 0.5   # no control flow
        if density <= 0.15:
            return 1.0
        if density <= 0.25:
            return 0.7
        if density <= 0.40:
            return 0.4
        return 0.2

    def _score_formatting(self, code: str) -> float:
        """Score formatting quality: blank line usage and indentation consistency.

        Looks for:
        - Reasonable blank-line ratio (5-25%)
        - Consistent indentation character (spaces XOR tabs, not mixed)
        """
        lines = code.splitlines()
        if not lines:
            return 0.0

        # Blank line ratio (delegate to HeuristicQualityScorer if available)
        if self._heuristic is not None:
            blank_score = self._heuristic._score_blank_lines(lines)
        else:
            blank_count = sum(1 for line in lines if not line.strip())
            ratio = blank_count / len(lines)
            if 0.05 <= ratio <= 0.25:
                blank_score = 1.0
            elif 0.01 <= ratio < 0.05 or 0.25 < ratio <= 0.35:
                blank_score = 0.6
            elif ratio < 0.01:
                blank_score = 0.3
            else:
                blank_score = 0.2

        # Indentation consistency
        indented_lines = [line for line in lines if line and line[0] in (" ", "\t")]
        if not indented_lines:
            indent_score = 0.8   # no indented lines → probably flat script
        else:
            tab_lines   = sum(1 for line in indented_lines if line.startswith("\t"))
            space_lines = sum(1 for line in indented_lines if line.startswith(" "))
            total_indented = len(indented_lines)

            # Mixed tabs and spaces is bad (Python 3 rejects it; editors hate it)
            mixed_ratio = min(tab_lines, space_lines) / total_indented
            if mixed_ratio > 0.05:
                indent_score = 0.3   # significant mixing
            elif mixed_ratio > 0.01:
                indent_score = 0.6
            else:
                indent_score = 1.0

        return blank_score * 0.6 + indent_score * 0.4

    def _score_duplication(self, code: str) -> float:
        """Score internal code duplication (copy-paste detection).

        Counts non-trivial lines that appear more than once.
        High duplication ratio → generated code or lazy copy-paste.

        Based on quality_filter.check_no_obvious_copy_paste.
        """
        lines = [line.strip() for line in code.splitlines()
                 if line.strip() and len(line.strip()) > 10]

        if len(lines) < 20:
            return 0.8   # too short to judge reliably

        counts = Counter(lines)
        duplicate_lines = sum(count - 1 for count in counts.values() if count > 1)
        ratio = duplicate_lines / len(lines)

        # ratio=0   → perfect (no duplicates)
        # ratio=0.3 → 30% duplicate → borderline
        # ratio=0.5 → clearly generated/copy-pasted

        if ratio <= 0.05:
            return 1.0
        if ratio <= 0.15:
            return 0.85
        if ratio <= 0.25:
            return 0.65
        if ratio <= 0.35:
            return 0.45
        if ratio <= 0.50:
            return 0.25
        return 0.1

    def _score_syntax(self, code: str, language: str) -> float:
        """Score syntax validity.

        For Python: try AST parsing — if it fails, score is low.
        For JS/TS: check brace balance as a proxy for syntax health.
        For unknown: return 0.7 (neutral, can't tell).
        """
        if language == "python":
            try:
                ast.parse(code)
                return 1.0
            except SyntaxError as exc:
                # Give partial credit if we can identify the line number
                # (the file is partly valid; only the tail is broken)
                total_lines = max(code.count("\n") + 1, 1)
                if exc.lineno is not None:
                    fraction_valid = exc.lineno / total_lines
                    # Even if 90% is valid, a syntax error is a hard fail for training
                    return min(0.3, fraction_valid * 0.3)
                return 0.1
            except Exception:
                return 0.0

        if language in ("typescript", "javascript"):
            open_braces  = code.count("{")
            close_braces = code.count("}")
            imbalance = abs(open_braces - close_braces)

            if open_braces == 0 and close_braces == 0:
                return 0.7   # no braces at all — might be valid

            relative_imbalance = imbalance / max(open_braces, close_braces, 1)

            if relative_imbalance <= 0.02:
                return 1.0
            if relative_imbalance <= 0.05:
                return 0.85
            if relative_imbalance <= 0.10:
                return 0.65
            if relative_imbalance <= 0.20:
                return 0.40
            return 0.15

        return 0.7   # unknown language — neutral

    def _score_modernness(self, code: str, language: str) -> float:
        """Score use of modern language patterns vs deprecated idioms.

        Python modern patterns (higher score):
            - f-strings (f"...") instead of %-formatting or .format()
            - Type hints on functions
            - Walrus operator  (:=)
            - Match/case (Python 3.10+)
            - async/await

        Python deprecated patterns (lower score):
            - %-style string formatting  ("hello %s" % name)
            - Old-style super() call:  super(ClassName, self)
            - print statement (Python 2)
            - map/filter/reduce used without itertools

        JS/TS modern patterns:
            - const / let instead of var
            - Arrow functions (=>)
            - Template literals (`...`)
            - Optional chaining (?.)
            - Nullish coalescing (??)
            - async/await instead of .then()/.catch() chains

        JS/TS deprecated:
            - var declarations
            - .then()/.catch() callback chains (instead of async/await)
            - ==  instead of ===
            - arguments keyword

        Returns a score from 0.0 (all deprecated) to 1.0 (all modern).
        """
        if language == "python":
            return self._score_modernness_python(code)

        if language in ("typescript", "javascript"):
            return self._score_modernness_js_ts(code)

        # Unknown language: neutral
        return 0.6

    def _score_modernness_python(self, code: str) -> float:
        """Python-specific modernness scoring."""
        modern_points = 0.0
        deprecated_points = 0.0

        # --- Modern patterns ---

        # f-strings
        fstring_count = len(re.findall(r'\bf"', code)) + len(re.findall(r"\bf'", code))
        if fstring_count > 0:
            modern_points += min(fstring_count * 0.5, 2.0)

        # Type hints on function defs: def foo(x: int) -> str:
        typed_funcs = len(re.findall(r'def \w+\([^)]*:\s*\w', code))
        return_hints = len(re.findall(r'\) -> \w', code))
        if typed_funcs > 0 or return_hints > 0:
            modern_points += min((typed_funcs + return_hints) * 0.4, 2.0)

        # Walrus operator
        if ":=" in code:
            modern_points += 1.0

        # match/case (Python 3.10+)
        if re.search(r'^\s*match\s+\w', code, re.MULTILINE):
            modern_points += 1.5

        # async/await
        if "async def" in code or "await " in code:
            modern_points += 1.0

        # dataclasses
        if "@dataclass" in code:
            modern_points += 0.5

        # pathlib instead of os.path
        if "from pathlib" in code or "import pathlib" in code:
            modern_points += 0.5

        # --- Deprecated patterns ---

        # %-style formatting: "hello %s" % something
        pct_format = len(re.findall(r'%[sdrfoxX]', code))
        deprecated_points += min(pct_format * 0.4, 2.0)

        # Old-style super: super(ClassName, self)
        old_super = len(re.findall(r'super\(\w+,\s*\w+\)', code))
        deprecated_points += min(old_super * 0.5, 1.5)

        # print without parens (Python 2)
        py2_print = len(re.findall(r'^print [^(]', code, re.MULTILINE))
        deprecated_points += min(py2_print * 0.3, 1.0)

        # has_key() (Python 2 dict method)
        deprecated_points += len(re.findall(r'\.has_key\(', code)) * 0.5

        # Normalise to 0-1
        total = modern_points + deprecated_points
        if total == 0:
            return 0.6   # neutral — no signals either way

        return max(0.0, min(1.0, modern_points / (total + 1.0)))

    def _score_modernness_js_ts(self, code: str) -> float:
        """JavaScript/TypeScript-specific modernness scoring."""
        modern_points = 0.0
        deprecated_points = 0.0

        # --- Modern ---

        # const/let declarations
        const_let = len(re.findall(r'\b(const|let)\b', code))
        modern_points += min(const_let * 0.2, 2.0)

        # Arrow functions
        arrow_funcs = len(re.findall(r'=>', code))
        modern_points += min(arrow_funcs * 0.2, 2.0)

        # Template literals
        template_lits = len(re.findall(r'`[^`]*`', code))
        modern_points += min(template_lits * 0.3, 1.5)

        # Optional chaining
        opt_chain = len(re.findall(r'\?\.',  code))
        modern_points += min(opt_chain * 0.3, 1.5)

        # Nullish coalescing
        null_coalesce = len(re.findall(r'\?\?', code))
        modern_points += min(null_coalesce * 0.3, 1.0)

        # async/await
        if "async " in code or "await " in code:
            modern_points += 1.0

        # TypeScript generics (signal of proper TS usage)
        ts_generics = len(re.findall(r'<[A-Z]\w*>', code))
        modern_points += min(ts_generics * 0.2, 1.0)

        # TypeScript interfaces / types
        if re.search(r'\binterface\s+\w+', code) or re.search(r'\btype\s+\w+\s*=', code):
            modern_points += 0.5

        # --- Deprecated ---

        # var declarations
        var_count = len(re.findall(r'\bvar\b', code))
        deprecated_points += min(var_count * 0.3, 2.0)

        # .then()/.catch() chains (callback-style promise handling)
        then_count = len(re.findall(r'\.then\(', code))
        deprecated_points += min(then_count * 0.2, 1.5)

        # == (loose equality) instead of ===
        loose_eq = len(re.findall(r'(?<!=)==(?!=)', code))
        deprecated_points += min(loose_eq * 0.2, 1.0)

        # arguments keyword (old-style variadic)
        if re.search(r'\barguments\b', code):
            deprecated_points += 0.5

        # Normalise
        total = modern_points + deprecated_points
        if total == 0:
            return 0.6

        return max(0.0, min(1.0, modern_points / (total + 1.0)))

    def _score_error_handling(self, code: str, language: str) -> float:
        """Score error handling quality.

        Good error handling:
        - Has try/except (Python) or try/catch (JS/TS) blocks
        - Catches specific exception types (not bare `except:`)
        - Uses custom exception classes (Python) or custom Error subclasses (JS)
        - Provides descriptive error messages (not just `pass` or `throw err`)

        Poor error handling:
        - Bare `except:` (catches everything including KeyboardInterrupt)
        - Empty catch blocks  (swallows errors silently)
        - Single-letter exception variables (`except e:` rather than `except ValueError as e:`)
        """
        if language == "python":
            return self._score_error_handling_python(code)
        if language in ("typescript", "javascript"):
            return self._score_error_handling_js_ts(code)
        return 0.5   # unknown language

    def _score_error_handling_python(self, code: str) -> float:
        """Python error handling scoring."""
        score = 0.5   # neutral baseline

        # Count try/except blocks
        try_count    = len(re.findall(r'\btry\s*:', code))
        except_count = len(re.findall(r'\bexcept\b', code))

        if try_count == 0:
            # No error handling at all — might be fine for a utility script
            lines = code.splitlines()
            if len(lines) < 20:
                return 0.6   # short scripts don't need it
            return 0.35

        score += 0.2   # has try/except

        # Bare except (no exception type specified)
        bare_except = len(re.findall(r'\bexcept\s*:', code))
        specific_except = except_count - bare_except
        if bare_except > 0 and specific_except == 0:
            score -= 0.3   # ALL excepts are bare — bad practice

        # Specific exception types mentioned
        if re.search(r'\bexcept\s+\w+Error\b', code):
            score += 0.1
        if re.search(r'\bexcept\s+\(', code):
            score += 0.05   # tuple of exception types

        # Custom exceptions (class FooError(Exception):)
        if re.search(r'\bclass\s+\w+(Error|Exception|Warning)\b', code):
            score += 0.15

        # finally block (resource cleanup)
        if re.search(r'\bfinally\s*:', code):
            score += 0.05

        # raise with a message (not bare `raise`)
        raise_with_msg = len(re.findall(r'\braise\s+\w+\([^)]+\)', code))
        if raise_with_msg > 0:
            score += min(raise_with_msg * 0.05, 0.1)

        # Logging errors (better than silently swallowing)
        if re.search(r'\b(logging|logger)\.(error|exception|warning)\b', code):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _score_error_handling_js_ts(self, code: str) -> float:
        """JavaScript/TypeScript error handling scoring."""
        score = 0.5

        try_count = len(re.findall(r'\btry\s*\{', code))

        if try_count == 0:
            lines = code.splitlines()
            if len(lines) < 20:
                return 0.6
            return 0.35

        score += 0.2

        # Empty catch blocks: catch (e) {}
        empty_catch = len(re.findall(r'catch\s*\(\w+\)\s*\{\s*\}', code))
        if empty_catch > 0:
            score -= 0.2 * empty_catch   # each empty catch is bad

        # Typed catches (TypeScript): catch (e: unknown) or instanceof narrowing
        if re.search(r'catch\s*\(\w+\s*:\s*\w+\)', code):
            score += 0.1
        if re.search(r'instanceof\s+Error\b', code):
            score += 0.1

        # Custom error classes
        if re.search(r'\bclass\s+\w+(Error|Exception)\s+extends\s+Error\b', code):
            score += 0.15

        # finally block
        if re.search(r'\bfinally\s*\{', code):
            score += 0.05

        # throw new Error with a message
        if re.search(r'\bthrow\s+new\s+\w+(Error)?\s*\([^)]+\)', code):
            score += 0.05

        # console.error (better than swallowing)
        if re.search(r'\bconsole\.(error|warn)\b', code):
            score += 0.05

        # Promise reject handling
        if re.search(r'\.catch\(', code):
            score += 0.03

        return max(0.0, min(1.0, score))

    def _score_security(self, code: str) -> float:
        """Score for absence of security anti-patterns.

        Detects:
        - Hardcoded secrets (API keys, passwords, tokens)
        - eval() / exec() usage
        - SQL string concatenation (injection risk)
        - Command injection patterns (shell=True with user input)

        Returns 1.0 if clean, lower if issues found.
        This is a PENALTY scorer — starts at 1.0 and deducts.
        """
        score = 1.0

        # Check only a sample to keep this fast
        sample = code[:8000]

        # Hardcoded secrets (same patterns as quality_filter.py)
        secret_patterns = [
            re.compile(r'(?:api[_-]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}', re.IGNORECASE),
            re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}', re.IGNORECASE),
            re.compile(r'(?:secret|token)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}', re.IGNORECASE),
            re.compile(r'(?:aws_access_key_id|aws_secret_access_key)\s*[:=]', re.IGNORECASE),
            re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),
            re.compile(r'sk-[a-zA-Z0-9]{32,}'),
            re.compile(r'ghp_[a-zA-Z0-9]{36}'),
        ]
        for pattern in secret_patterns:
            if pattern.search(sample):
                score -= 0.5
                break   # one hit is enough to flag

        # eval() / exec() — very dangerous
        eval_exec = len(re.findall(r'\beval\s*\(', sample))
        exec_calls = len(re.findall(r'\bexec\s*\(', sample))
        dangerous_calls = eval_exec + exec_calls
        if dangerous_calls > 0:
            score -= min(dangerous_calls * 0.15, 0.3)

        # SQL injection patterns: "SELECT ... " + variable
        sql_concat = len(re.findall(
            r'(?:SELECT|INSERT|UPDATE|DELETE|WHERE)\s+[^"\']+["\'\s]*\+', sample, re.IGNORECASE,
        ))
        if sql_concat > 0:
            score -= min(sql_concat * 0.1, 0.2)

        # subprocess with shell=True (potential command injection)
        if re.search(r'\bsubprocess\b.*\bshell\s*=\s*True\b', sample, re.DOTALL):
            score -= 0.1

        # os.system() with string concatenation
        if re.search(r'\bos\.system\s*\([^)]*\+', sample):
            score -= 0.1

        return max(0.0, score)

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    def _collect_feedback(
        self,
        breakdown: dict[str, float],
        language: str,
        issues: list[str],
        suggestions: list[str],
    ) -> None:
        """Populate issues and suggestions from low signal scores.

        Threshold: signals scoring below 0.5 generate an issue entry.
        Signals below 0.3 generate both an issue and a suggestion.
        """
        thresholds = {
            "length":         (0.5, "file is too short or too long"),
            "line_quality":   (0.5, "lines are too long (possible minification)"),
            "structure":      (0.5, "code lacks functions or classes"),
            "naming":         (0.5, "inconsistent or very short identifier names"),
            "comments":       (0.4, "comment ratio is too low or too high"),
            "documentation":  (0.5, "missing docstrings / JSDoc"),
            "complexity":     (0.5, "very high control-flow density (hard to read)"),
            "formatting":     (0.4, "inconsistent indentation or excessive blank lines"),
            "duplication":    (0.5, "high internal copy-paste duplication detected"),
            "syntax":         (0.6, "syntax errors or structural issues found"),
            "modernness":     (0.4, "deprecated language patterns detected"),
            "error_handling": (0.4, "poor or missing error handling"),
            "security":       (0.7, "potential security issue detected"),
        }

        suggestion_map = {
            "length":         "aim for 10-500 lines of meaningful code",
            "line_quality":   "keep lines under 100 characters",
            "structure":      "organise code into functions and classes",
            "naming":         "use descriptive names (>= 3 chars); follow snake_case / camelCase",
            "comments":       "add inline comments explaining the 'why', target 5-25% coverage",
            "documentation":  "add docstrings to all public functions and classes",
            "complexity":     "extract complex conditionals into named functions",
            "formatting":     "use consistent spaces (not tabs) and keep blank-line ratio 5-25%",
            "duplication":    "extract repeated logic into reusable functions",
            "syntax":         "fix syntax errors so the file is parseable",
            "modernness":     (
                "use f-strings and type hints (Python) or const/let and arrow functions (JS/TS)"
            ),
            "error_handling": "add specific try/except blocks and custom exception types",
            "security":       "remove hardcoded secrets; avoid eval(); use parameterised queries",
        }

        for key, (threshold, issue_text) in thresholds.items():
            sig_score = breakdown.get(key, 1.0)
            if sig_score < threshold:
                issues.append(f"{key}: {issue_text} (score={sig_score:.2f})")
            if sig_score < 0.3 and key in suggestion_map:
                suggestions.append(suggestion_map[key])


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def score_to_weight(score: CodeQualityScore) -> float:
    """Module-level wrapper around CodeScorer.score_to_weight().

    Convenience function so callers don't need to instantiate CodeScorer
    just to convert an existing score to a weight.

    Args:
        score: A CodeQualityScore produced by CodeScorer.score().

    Returns:
        Training weight in [0.0, 2.0].
    """
    return CodeScorer().score_to_weight(score)
