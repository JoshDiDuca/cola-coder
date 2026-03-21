"""Code Complexity Scorer: measure code complexity for quality filtering.

Computes complexity metrics like cyclomatic complexity, cognitive complexity,
nesting depth, function length, and line count.  Used for:
- Data quality filtering (exclude trivially simple or overly complex code)
- Curriculum learning (train on easy examples first, then harder ones)
- Benchmark difficulty estimation

For a TS dev: like ESLint's complexity rule but as a standalone scorer
that categorizes code into difficulty buckets.

Enhancements over v1:
- cognitive_complexity: penalises nested branching more than flat branching
- nesting_depth_per_function: per-function max nesting
- avg_function_length: mean lines per function
- max_function_length: longest function in lines
- Works for both Python and TypeScript/JavaScript
"""

import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the complexity scorer feature is active."""
    return FEATURE_ENABLED


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a piece of code."""

    cyclomatic: int = 1  # Cyclomatic complexity (decision points + 1)
    cognitive_complexity: int = 0  # Cognitive complexity (nesting-penalised)
    max_nesting: int = 0  # Maximum nesting depth
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    comment_ratio: float = 0.0  # Fraction of lines that are comments
    avg_line_length: float = 0.0
    avg_function_length: float = 0.0  # Mean lines per function
    max_function_length: int = 0  # Longest function in lines
    nesting_depth_per_function: list[int] = field(default_factory=list)

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
            f"cognitive={self.cognitive_complexity}, "
            f"nesting={self.max_nesting}, lines={self.line_count}, "
            f"functions={self.function_count}, classes={self.class_count}, "
            f"avg_fn_len={self.avg_function_length:.1f}, "
            f"max_fn_len={self.max_function_length}, "
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
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            metrics.avg_line_length = sum(len(ln) for ln in non_empty) / len(non_empty)

        # Cognitive complexity
        metrics.cognitive_complexity = self._cognitive_complexity(code, language)

        # Per-function length stats
        fn_lengths = self._function_lengths(code, language)
        if fn_lengths:
            metrics.avg_function_length = sum(fn_lengths) / len(fn_lengths)
            metrics.max_function_length = max(fn_lengths)

        # Per-function nesting depth
        metrics.nesting_depth_per_function = self._nesting_per_function(code, language)

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

    # ------------------------------------------------------------------
    # New metrics (v2)
    # ------------------------------------------------------------------

    def _cognitive_complexity(self, code: str, language: str) -> int:
        """Compute cognitive complexity — nesting penalises more than flat branches.

        Each decision point adds (1 + current_nesting_level) to the score,
        making deeply nested branches cost more than flat ones.
        """
        score = 0
        if language == "python":
            branch_re = re.compile(
                r"^\s*(if|elif|for|while|except|with|match)\b", re.MULTILINE
            )
            for line in code.splitlines():
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                nesting = indent // 4
                if branch_re.match(line):
                    score += 1 + nesting
                # logical operators add flat +1
                score += len(re.findall(r"\band\b|\bor\b", stripped))
        else:
            # JS/TS: track brace depth as nesting proxy
            depth = 0
            flat_branch_re = re.compile(
                r"\b(if|else\s+if|for|while|catch|switch)\b"
            )
            for line in code.splitlines():
                matches = flat_branch_re.findall(line)
                for _ in matches:
                    score += 1 + depth
                score += len(re.findall(r"&&|\|\|", line))
                depth += line.count("{") - line.count("}")
                depth = max(0, depth)
        return score

    def _function_lengths(self, code: str, language: str) -> list[int]:
        """Return a list of line-counts for each function/method body."""
        lines = code.splitlines()
        lengths: list[int] = []

        if language == "python":
            fn_start_re = re.compile(r"^\s*(?:async\s+)?def\s+\w+")
            starts: list[int] = [i for i, ln in enumerate(lines) if fn_start_re.match(ln)]
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
                # Find actual body end by looking for de-indent
                base_indent = len(lines[start]) - len(lines[start].lstrip())
                body_end = start + 1
                for j in range(start + 1, end):
                    ln = lines[j]
                    if not ln.strip():
                        continue
                    ln_indent = len(ln) - len(ln.lstrip())
                    if ln_indent <= base_indent and fn_start_re.match(ln):
                        break
                    body_end = j + 1
                lengths.append(body_end - start)
        else:
            # JS/TS: count brace-delimited function bodies
            fn_re = re.compile(
                r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?[\w(])"
            )
            i = 0
            while i < len(lines):
                if fn_re.search(lines[i]):
                    start = i
                    depth = 0
                    found_open = False
                    j = i
                    while j < len(lines):
                        depth += lines[j].count("{") - lines[j].count("}")
                        if "{" in lines[j]:
                            found_open = True
                        if found_open and depth <= 0:
                            lengths.append(j - start + 1)
                            i = j
                            break
                        j += 1
                    else:
                        lengths.append(len(lines) - start)
                i += 1
        return lengths

    def _nesting_per_function(self, code: str, language: str) -> list[int]:
        """Return max nesting depth for each detected function."""
        lines = code.splitlines()
        depths: list[int] = []

        if language == "python":
            fn_start_re = re.compile(r"^\s*(?:async\s+)?def\s+\w+")
            starts: list[int] = [i for i, ln in enumerate(lines) if fn_start_re.match(ln)]
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
                base_indent = len(lines[start]) - len(lines[start].lstrip())
                max_d = 0
                for ln in lines[start + 1 : end]:
                    if not ln.strip():
                        continue
                    ind = len(ln) - len(ln.lstrip())
                    rel = max(0, (ind - base_indent) // 4)
                    max_d = max(max_d, rel)
                depths.append(max_d)
        else:
            fn_re = re.compile(
                r"(?:function\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?[\w(])"
            )
            i = 0
            while i < len(lines):
                if fn_re.search(lines[i]):
                    depth = 0
                    max_d = 0
                    found_open = False
                    base_depth: int | None = None
                    j = i
                    while j < len(lines):
                        opens = lines[j].count("{")
                        closes = lines[j].count("}")
                        if opens > 0 and not found_open:
                            found_open = True
                            depth += opens - closes
                            base_depth = depth
                        else:
                            depth += opens - closes
                        if found_open and base_depth is not None:
                            rel = max(0, depth - base_depth)
                            max_d = max(max_d, rel)
                        if found_open and depth <= 0:
                            depths.append(max_d)
                            i = j
                            break
                        j += 1
                    else:
                        depths.append(max_d)
                i += 1
        return depths

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
