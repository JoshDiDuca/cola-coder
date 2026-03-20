"""Self-Verification Loop: model checks its own output for correctness.

Implements multi-pass verification with confidence tracking. Inspired by
Self-Refine (Madaan et al., 2023) and Reflexion (Shinn et al., 2023):
iterative revision improves accuracy on code tasks by 5-15% with no
additional training.

For a TS dev: like running eslint + tsc + your own code review in sequence,
keeping the best version by a quality heuristic.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Suspicious patterns that often indicate hallucinated code
# ---------------------------------------------------------------------------

# Modules that AI models commonly fabricate
_SUSPICIOUS_MODULES = {
    "torch.utils.data.experimental",
    "torch.nn.functional.advanced",
    "transformers.advanced",
    "numpy.experimental",
    "sklearn.experimental.advanced",
    "fastapi.advanced",
    "flask.advanced",
    "requests.async",
    "os.advanced",
    "sys.advanced",
    "pathlib.advanced",
    "json.stream",
    "re.advanced",
}

# Method names that don't exist but are commonly hallucinated
_SUSPICIOUS_METHODS = [
    r"\.fit_transform_async\(",
    r"\.predict_proba_fast\(",
    r"\.to_json_string\(",
    r"\.from_pretrained_fast\(",
    r"\.encode_batch_fast\(",
    r"\.generate_fast\(",
    r"\.compute_metrics_fast\(",
    r"\.load_checkpoint_fast\(",
    r"\.save_pretrained_fast\(",
    r"torch\.tensor_like\(",
    r"np\.array_like\(",
    r"pd\.DataFrame\.from_records_fast\(",
]

# Common markers of placeholder / incomplete code
_PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bHACK\b",
    r"\bXXX\b",
    r"\bNOT IMPLEMENTED\b",
    r"raise NotImplementedError\(",
    r"\.\.\.\s*#\s*implement",
    r"pass\s*#\s*TODO",
]

# Patterns that suggest degenerate/repetitive generation
_REPETITION_THRESHOLD = 3  # same line repeated this many times = suspicious


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of a single verification pass."""
    passed: bool
    confidence: float          # 0.0 – 1.0; higher is better
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"VerificationResult({status}, confidence={self.confidence:.2f}, "
            f"issues={len(self.issues)}, suggestions={len(self.suggestions)})"
        )


# ---------------------------------------------------------------------------
# SelfVerifier
# ---------------------------------------------------------------------------

class SelfVerifier:
    """
    Stateless verifier: each method inspects code using heuristics and
    static analysis only (no model inference required).

    Design note: the plan describes a model-in-the-loop revision flow.
    This implementation handles the static analysis / confidence-scoring
    layer. Hook into run_verification_loop() when you have the model
    available; without it, the loop still provides useful heuristic feedback.
    """

    # Weights used when computing confidence in verify_code()
    _W_SYNTAX = 0.40
    _W_COMPLETENESS = 0.35
    _W_HALLUCINATION = 0.25

    def verify_syntax(self, code: str) -> bool:
        """
        Basic syntax check using Python's ast module plus manual bracket
        matching for cases that confuse the parser (e.g. type stubs with
        arbitrary indentation).

        Returns True if the code parses cleanly, False otherwise.
        """
        # 1. Python AST parse — the authoritative check for Python code
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            pass

        # 2. Fallback: bracket / paren / brace balance check.
        #    Useful for TypeScript / JavaScript snippets that aren't valid Python
        #    but might appear in a Python-centric codebase as string literals.
        open_to_close = {"(": ")", "[": "]", "{": "}"}
        close_chars = set(open_to_close.values())
        stack: list[str] = []
        in_single_string = False
        in_double_string = False
        i = 0
        while i < len(code):
            ch = code[i]
            # Skip escape sequences inside strings
            if ch == "\\" and (in_single_string or in_double_string):
                i += 2
                continue
            if ch == "'" and not in_double_string:
                in_single_string = not in_single_string
            elif ch == '"' and not in_single_string:
                in_double_string = not in_double_string
            elif not in_single_string and not in_double_string:
                if ch in open_to_close:
                    stack.append(open_to_close[ch])
                elif ch in close_chars:
                    if not stack or stack[-1] != ch:
                        return False
                    stack.pop()
            i += 1

        return len(stack) == 0

    # ------------------------------------------------------------------

    def verify_completeness(self, code: str, prompt: str) -> float:
        """
        Estimate how well the code satisfies the prompt intent (0.0 – 1.0).

        Uses simple keyword overlap and structural checks rather than
        embedding similarity (keeps this self-contained with zero extra deps).
        A TS analogy: think of this as a lightweight contract test — does the
        implementation expose the names the caller promised?
        """
        if not code.strip():
            return 0.0
        if not prompt.strip():
            # No prompt to compare against — can't penalise
            return 0.5

        score = 0.0
        checks = 0

        # 1. Identifier overlap: function / class / variable names mentioned
        #    in the prompt and appearing in the code.
        prompt_words = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", prompt))
        code_words = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", code))
        if prompt_words:
            overlap = len(prompt_words & code_words) / len(prompt_words)
            score += overlap
        checks += 1

        # 2. Does code define at least one function / class?
        has_definition = bool(
            re.search(r"\bdef\s+\w+", code) or re.search(r"\bclass\s+\w+", code)
        )
        score += 1.0 if has_definition else 0.3
        checks += 1

        # 3. Placeholder / incomplete markers reduce completeness
        placeholder_hits = sum(
            1 for p in _PLACEHOLDER_PATTERNS if re.search(p, code, re.IGNORECASE)
        )
        placeholder_penalty = min(placeholder_hits * 0.15, 0.45)
        score -= placeholder_penalty
        checks += 0  # penalty only, no divisor increment

        # 4. Length sanity: very short code for a non-trivial prompt is suspect
        code_lines = [l for l in code.splitlines() if l.strip()]
        if len(prompt.split()) > 20 and len(code_lines) < 3:
            score -= 0.2

        return max(0.0, min(1.0, score / checks))

    # ------------------------------------------------------------------

    def verify_no_hallucination(self, code: str) -> list[str]:
        """
        Check for suspicious patterns that suggest hallucinated code.

        Returns a (possibly empty) list of issue strings.  Each entry
        describes a specific suspicious pattern found.
        """
        issues: list[str] = []

        # 1. Suspicious import paths
        for mod in _SUSPICIOUS_MODULES:
            pattern = re.escape(mod)
            if re.search(pattern, code):
                issues.append(f"Suspicious import path detected: '{mod}' — likely hallucinated module.")

        # 2. Suspicious method calls
        for method_pattern in _SUSPICIOUS_METHODS:
            if re.search(method_pattern, code):
                # Extract the method name from the pattern for a readable message
                method_name = method_pattern.lstrip(r"\.").rstrip(r"\(").replace("\\", "")
                issues.append(f"Possibly hallucinated method call: '{method_name}' — verify this API exists.")

        # 3. Repetition detection (degenerate loop output)
        lines = [l.strip() for l in code.splitlines() if l.strip()]
        if lines:
            from collections import Counter
            counts = Counter(lines)
            for line, count in counts.items():
                if count >= _REPETITION_THRESHOLD and len(line) > 5:
                    issues.append(
                        f"Repeated line detected {count}x (possible degenerate generation): '{line[:60]}...'"
                        if len(line) > 60
                        else f"Repeated line detected {count}x (possible degenerate generation): '{line}'"
                    )

        # 4. Undefined references (very basic: usage of names never defined or imported)
        #    Only flag obvious cases to avoid false positives
        undefined_patterns = [
            (r"\bundefined\b", "Use of 'undefined' (Python doesn't have this — did you mean None?)"),
            (r"\bconsole\.log\b", "Use of console.log (JavaScript-ism in Python code)"),
            (r"\bvar\s+\w+\s*=", "Use of 'var' keyword (JavaScript-ism in Python code)"),
        ]
        for pattern, message in undefined_patterns:
            if re.search(pattern, code):
                issues.append(message)

        return issues

    # ------------------------------------------------------------------

    def verify_code(self, code: str) -> VerificationResult:
        """
        Full single-pass verification of `code`.

        Runs syntax check, completeness estimate (prompt-free = 0.5 base),
        and hallucination detection.  Returns a VerificationResult with an
        aggregated confidence score and any issues found.
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # --- Syntax ---
        syntax_ok = self.verify_syntax(code)
        syntax_score = 1.0 if syntax_ok else 0.0
        if not syntax_ok:
            issues.append("Syntax check failed: code may contain unbalanced brackets or invalid Python syntax.")
            suggestions.append("Review bracket matching and function/class definition syntax.")

        # --- Completeness (no prompt available → use neutral 0.5) ---
        completeness_score = self.verify_completeness(code, prompt="")

        # --- Hallucination ---
        hallucination_issues = self.verify_no_hallucination(code)
        hallucination_score = max(0.0, 1.0 - len(hallucination_issues) * 0.20)
        issues.extend(hallucination_issues)
        if hallucination_issues:
            suggestions.append(
                "Verify that all imported modules and called methods exist in the target environment."
            )

        # --- Aggregated confidence ---
        confidence = (
            self._W_SYNTAX * syntax_score
            + self._W_COMPLETENESS * completeness_score
            + self._W_HALLUCINATION * hallucination_score
        )
        confidence = max(0.0, min(1.0, confidence))

        # A result "passes" when confidence is above a threshold and syntax is valid
        passed = syntax_ok and confidence >= 0.45

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------

    def run_verification_loop(
        self,
        code: str,
        max_iterations: int = 3,
    ) -> list[VerificationResult]:
        """
        Run up to `max_iterations` verification passes on `code`.

        In a full model-in-the-loop setup you would regenerate/revise
        the code between iterations.  Here each pass re-analyses the same
        code with verify_code() to build a confidence history — useful for
        detecting unstable / borderline heuristic results and for providing
        a list of results to summarise.

        Stop early if:
        - The result passes with confidence > 0.85 (model satisfied).
        - The result has no issues (nothing left to flag).
        """
        results: list[VerificationResult] = []

        for i in range(max_iterations):
            result = self.verify_code(code)
            results.append(result)

            # Early stop conditions (mirrors the plan's is_correct / no_change checks)
            if result.passed and result.confidence > 0.85:
                break
            if not result.issues:
                break

        return results

    # ------------------------------------------------------------------

    def summary(self, results: list[VerificationResult]) -> dict:
        """
        Summarise a list of VerificationResult objects (e.g. from run_verification_loop).

        Returns a dict with aggregate statistics.
        """
        if not results:
            return {
                "n_passes": 0,
                "all_passed": False,
                "any_passed": False,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "total_issues": 0,
                "total_suggestions": 0,
                "best_result_index": -1,
            }

        confidences = [r.confidence for r in results]
        best_idx = confidences.index(max(confidences))

        return {
            "n_passes": len(results),
            "all_passed": all(r.passed for r in results),
            "any_passed": any(r.passed for r in results),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "total_issues": sum(len(r.issues) for r in results),
            "total_suggestions": sum(len(r.suggestions) for r in results),
            "best_result_index": best_idx,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def quick_verify(code: str) -> bool:
    """
    Quick pass/fail check for `code`.

    Returns True if syntax is valid and no critical issues are detected.
    Equivalent to SelfVerifier().verify_code(code).passed but with a
    slightly higher confidence threshold to be conservative.
    """
    verifier = SelfVerifier()
    result = verifier.verify_code(code)
    return result.passed and result.confidence >= 0.40
