"""Code Complexity Scorer: measure code complexity for quality filtering.

Computes complexity metrics like cyclomatic complexity, nesting depth,
and line count. Used for:
- Data quality filtering (exclude trivially simple or overly complex code)
- Curriculum learning (train on easy examples first, then harder ones)
- Benchmark difficulty estimation

For a TS dev: like ESLint's complexity rule but as a standalone scorer
that categorizes code into difficulty buckets.
"""

import re
from dataclasses import dataclass

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a piece of code."""
    cyclomatic: int = 1  # Cyclomatic complexity (decision points + 1)
    max_nesting: int = 0  # Maximum nesting depth
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    comment_ratio: float = 0.0  # Fraction of lines that are comments
    avg_line_length: float = 0.0

    @property
    def difficulty_bucket(self) -> int:
        """Assign to a difficulty bucket 1-5.

        1: Trivial (hello world, single function)
        2: Simple (few functions, low branching)
        3: Medium (multiple functions, moderate branching)
        4: Complex (classes, deep nesting, high branching)
        5: Very complex (many classes, deep nesting, high cyclomatic)
        """
        score = 0

        # Cyclomatic complexity contribution
        if self.cyclomatic <= 2:
            score += 1
        elif self.cyclomatic <= 5:
            score += 2
        elif self.cyclomatic <= 10:
            score += 3
        elif self.cyclomatic <= 20:
            score += 4
        else:
            score += 5

        # Nesting depth contribution
        if self.max_nesting <= 1:
            score += 1
        elif self.max_nesting <= 3:
            score += 2
        elif self.max_nesting <= 5:
            score += 3
        else:
            score += 4

        # Size contribution
        if self.line_count <= 10:
            score += 1
        elif self.line_count <= 30:
            score += 2
        elif self.line_count <= 100:
            score += 3
        else:
            score += 4

        # Function/class count
        if self.function_count + self.class_count <= 1:
            score += 1
        elif self.function_count + self.class_count <= 3:
            score += 2
        else:
            score += 3

        # Map aggregate score to 1-5 bucket
        if score <= 5:
            return 1
        elif score <= 8:
            return 2
        elif score <= 11:
            return 3
        elif score <= 14:
            return 4
        return 5

    def summary(self) -> str:
        return (
            f"Complexity: cyclomatic={self.cyclomatic}, "
            f"nesting={self.max_nesting}, lines={self.line_count}, "
            f"functions={self.function_count}, classes={self.class_count}, "
            f"difficulty={self.difficulty_bucket}/5"
        )


class ComplexityScorer:
    """Score code complexity using heuristic metrics."""

    def score(self, code: str, language: str = "typescript") -> ComplexityMetrics:
        """Compute complexity metrics for a code string.

        Args:
            code: Source code to analyze
            language: "typescript", "javascript", or "python"

        Returns:
            ComplexityMetrics with all computed values
        """
        lines = code.splitlines()
        metrics = ComplexityMetrics()
        metrics.line_count = len(lines)

        if not lines:
            return metrics

        # Cyclomatic complexity (count decision points)
        metrics.cyclomatic = self._cyclomatic_complexity(code, language)

        # Max nesting depth
        metrics.max_nesting = self._max_nesting(code, language)

        # Count functions and classes
        metrics.function_count = self._count_functions(code, language)
        metrics.class_count = self._count_classes(code, language)

        # Count imports
        metrics.import_count = self._count_imports(code, language)

        # Comment ratio
        metrics.comment_ratio = self._comment_ratio(code, language)

        # Average line length
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            metrics.avg_line_length = sum(len(l) for l in non_empty) / len(non_empty)

        return metrics

    def _cyclomatic_complexity(self, code: str, language: str) -> int:
        """Count decision points (if, for, while, case, etc.) + 1."""
        cc = 1  # Base complexity

        if language == "python":
            patterns = [
                r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b",
                r"\band\b", r"\bor\b", r"\bexcept\b",
            ]
        else:
            patterns = [
                r"\bif\b", r"\belse\s+if\b", r"\bfor\b", r"\bwhile\b",
                r"\bcase\b", r"\bcatch\b", r"&&", r"\|\|",
                r"\?\.", r"\?\s",  # Optional chaining, ternary
            ]

        for pattern in patterns:
            cc += len(re.findall(pattern, code))

        return cc

    def _max_nesting(self, code: str, language: str) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0

        if language == "python":
            # Python: track indentation level
            for line in code.splitlines():
                stripped = line.lstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(stripped)
                depth = indent // 4  # Assume 4-space indent
                max_depth = max(max_depth, depth)
        else:
            # JS/TS: track brace nesting
            in_string = False
            string_char = ""
            for char in code:
                if in_string:
                    if char == string_char and (not code or True):  # Simplified
                        in_string = False
                    continue
                if char in ('"', "'", "`"):
                    in_string = True
                    string_char = char
                elif char == "{":
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == "}":
                    current_depth = max(0, current_depth - 1)

        return max_depth

    def _count_functions(self, code: str, language: str) -> int:
        if language == "python":
            return len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", code, re.MULTILINE))
        return len(re.findall(
            r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>)",
            code
        ))

    def _count_classes(self, code: str, language: str) -> int:
        return len(re.findall(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+", code, re.MULTILINE))

    def _count_imports(self, code: str, language: str) -> int:
        if language == "python":
            return len(re.findall(r"^\s*(?:import|from)\s+", code, re.MULTILINE))
        return len(re.findall(r"^\s*import\s+", code, re.MULTILINE))

    def _comment_ratio(self, code: str, language: str) -> float:
        lines = code.splitlines()
        if not lines:
            return 0.0

        comment_count = 0
        for line in lines:
            stripped = line.strip()
            if language == "python":
                if stripped.startswith("#"):
                    comment_count += 1
            else:
                if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                    comment_count += 1

        return comment_count / len(lines)

    def filter_by_complexity(
        self,
        code_samples: list[str],
        min_bucket: int = 1,
        max_bucket: int = 5,
        language: str = "typescript",
    ) -> list[tuple[str, ComplexityMetrics]]:
        """Filter code samples by difficulty bucket.

        Returns:
            List of (code, metrics) tuples within the bucket range
        """
        results = []
        for code in code_samples:
            metrics = self.score(code, language)
            if min_bucket <= metrics.difficulty_bucket <= max_bucket:
                results.append((code, metrics))
        return results
