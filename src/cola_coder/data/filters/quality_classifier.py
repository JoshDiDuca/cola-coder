"""FineWeb-Edu style quality classifier for code.

Two-phase approach:
Phase 1 (expensive, one-time): Score ~10k code samples with an LLM (Claude/GPT)
Phase 2 (cheap, reusable): Train a small classifier on Phase 1 data
Phase 3 (fast): Use classifier to score millions of files

The classifier is a small model (DistilBERT-sized) fine-tuned for
code quality scoring. Input: code text. Output: quality score 1-5.

Heuristic fallback is always available and requires no ML dependencies.
Think of it like having a TypeScript type-checker that can run without
a language server — basic but useful.
"""

from __future__ import annotations

import re
from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter


# ---------------------------------------------------------------------------
# Heuristic quality scorer (always available, no dependencies)
# ---------------------------------------------------------------------------

class HeuristicQualityScorer:
    """Fast heuristic code quality scoring (no ML needed).

    Combines multiple signals into a quality score.  Each signal returns
    a 0.0-1.0 sub-score, and the final score is a weighted average.

    This is the "always works" fallback.  Think of it like ESLint's
    built-in rules vs. a trained code-review model — less nuanced but
    zero setup cost.
    """

    # Weights for each signal (must sum to ~1.0)
    _WEIGHTS: dict[str, float] = {
        "comment_ratio": 0.15,
        "structure": 0.20,
        "naming": 0.15,
        "line_length": 0.10,
        "docstrings": 0.10,
        "complexity_density": 0.10,
        "blank_line_ratio": 0.05,
        "length_penalty": 0.15,
    }

    def score(self, code: str, language: str = "") -> float:
        """Score code 0.0-1.0 based on heuristics.

        Higher = better quality.  Roughly maps to:
            0.0-0.2 = garbage/noise
            0.2-0.4 = poor quality
            0.4-0.6 = average
            0.6-0.8 = good
            0.8-1.0 = excellent
        """
        if not code or not code.strip():
            return 0.0

        lines = code.splitlines()
        if len(lines) < 2:
            return 0.05

        scores: dict[str, float] = {
            "comment_ratio": self._score_comment_ratio(code, lines, language),
            "structure": self._score_structure(code, lines, language),
            "naming": self._score_naming(code, language),
            "line_length": self._score_line_length(lines),
            "docstrings": self._score_docstrings(code, language),
            "complexity_density": self._score_complexity_density(code, lines),
            "blank_line_ratio": self._score_blank_lines(lines),
            "length_penalty": self._score_length(lines),
        }

        total = sum(
            scores[k] * self._WEIGHTS[k] for k in self._WEIGHTS
        )
        return max(0.0, min(1.0, total))

    # -- Individual signal scorers --

    def _score_comment_ratio(
        self, code: str, lines: list[str], language: str
    ) -> float:
        """Score comment ratio.  Some comments = good, too many/few = bad."""
        total_chars = len(code)
        if total_chars == 0:
            return 0.0

        comment_chars = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                comment_chars += len(line)
            elif stripped.startswith("/*") or stripped.startswith("*"):
                comment_chars += len(line)

        ratio = comment_chars / total_chars

        # Sweet spot: 5-25% comments
        if 0.05 <= ratio <= 0.25:
            return 1.0
        elif 0.02 <= ratio < 0.05 or 0.25 < ratio <= 0.40:
            return 0.6
        elif ratio < 0.02:
            # No comments at all — penalize but don't kill score
            return 0.3
        else:
            # > 40% comments — likely a comment dump or license block
            return 0.2

    def _score_structure(
        self, code: str, lines: list[str], language: str
    ) -> float:
        """Score code structure: functions, classes, imports = organized."""
        lang = language.lower()

        # Count structural elements
        func_count = 0
        class_count = 0
        import_count = 0

        for line in lines:
            stripped = line.strip()
            if lang in ("python", "py", ""):
                if stripped.startswith("def "):
                    func_count += 1
                elif stripped.startswith("class "):
                    class_count += 1
                elif stripped.startswith(("import ", "from ")):
                    import_count += 1
            if lang in ("typescript", "javascript", "ts", "js", "tsx", "jsx", ""):
                if re.match(
                    r"(export\s+)?(async\s+)?function\s+\w+", stripped
                ):
                    func_count += 1
                elif re.match(r"(export\s+)?(default\s+)?class\s+\w+", stripped):
                    class_count += 1
                elif stripped.startswith("import "):
                    import_count += 1
                # Arrow functions assigned to const/let
                if re.match(
                    r"(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?\(", stripped
                ):
                    func_count += 1

        total_lines = len(lines)
        has_structure = func_count > 0 or class_count > 0

        if not has_structure:
            # No functions or classes — could be a script or config
            if total_lines < 20:
                return 0.3  # short scripts are fine
            return 0.2  # long unstructured code is bad

        # Good ratio: 1 function per 10-30 lines
        func_density = func_count / max(total_lines, 1) * 100
        if 3 <= func_density <= 15:
            structure_score = 1.0
        elif 1 <= func_density < 3 or 15 < func_density <= 25:
            structure_score = 0.7
        else:
            structure_score = 0.4

        # Bonus for classes (indicates organized code)
        if class_count > 0:
            structure_score = min(1.0, structure_score + 0.1)

        # Bonus for imports (indicates modular code)
        if import_count > 0:
            structure_score = min(1.0, structure_score + 0.05)

        return structure_score

    def _score_naming(self, code: str, language: str) -> float:
        """Score naming conventions: consistent naming = good."""
        # Extract identifiers (simple heuristic: words after def/function/class/const/let/var)
        identifiers = re.findall(
            r"(?:def|function|class|const|let|var|type|interface)\s+(\w+)",
            code,
        )

        if len(identifiers) < 2:
            return 0.5  # not enough data to judge

        # Check naming convention consistency
        snake_count = sum(1 for name in identifiers if re.match(r"^[a-z][a-z0-9_]*$", name))
        camel_count = sum(1 for name in identifiers if re.match(r"^[a-z][a-zA-Z0-9]*$", name))
        pascal_count = sum(1 for name in identifiers if re.match(r"^[A-Z][a-zA-Z0-9]*$", name))
        upper_count = sum(1 for name in identifiers if re.match(r"^[A-Z][A-Z0-9_]*$", name))
        single_char = sum(1 for name in identifiers if len(name) <= 1)

        total = len(identifiers)

        # Penalize single-character names
        if single_char / total > 0.5:
            return 0.2

        # Check consistency: most names should follow one convention
        # (PascalCase for classes is expected alongside snake_case/camelCase)
        max_convention = max(snake_count, camel_count)
        consistency = (max_convention + pascal_count + upper_count) / total

        if consistency > 0.8:
            return 1.0
        elif consistency > 0.6:
            return 0.7
        elif consistency > 0.4:
            return 0.5
        else:
            return 0.3

    def _score_line_length(self, lines: list[str]) -> float:
        """Score line length distribution.  Very long lines = bad."""
        if not lines:
            return 0.0

        lengths = [len(line) for line in lines]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)

        # Minified code detection
        if max_length > 500:
            return 0.1
        if avg_length > 120:
            return 0.2

        # Sweet spot: avg 30-80 chars, max under 120
        if avg_length < 10:
            return 0.3  # very short lines, probably not real code
        if avg_length <= 80 and max_length <= 120:
            return 1.0
        if avg_length <= 100 and max_length <= 200:
            return 0.7

        return 0.5

    def _score_docstrings(self, code: str, language: str) -> float:
        """Score presence of docstrings/JSDoc."""
        lang = language.lower()

        has_docstrings = False

        if lang in ("python", "py", ""):
            # Triple-quote docstrings
            has_docstrings = '"""' in code or "'''" in code

        if lang in ("typescript", "javascript", "ts", "js", "tsx", "jsx", ""):
            # JSDoc comments
            has_docstrings = has_docstrings or "/**" in code

        # Generic: any multi-line comment block
        if not has_docstrings:
            has_docstrings = "/**" in code or '"""' in code

        if has_docstrings:
            # Count how many docstrings
            doc_count = code.count('"""') // 2 + code.count("/**")
            if doc_count >= 3:
                return 1.0
            elif doc_count >= 1:
                return 0.7
        return 0.3

    def _score_complexity_density(self, code: str, lines: list[str]) -> float:
        """Score code complexity density.

        Too dense (many nested ifs/loops per line) = hard to read.
        Moderate complexity with good structure = good.
        """
        total_lines = len(lines)
        if total_lines == 0:
            return 0.0

        # Count complexity indicators
        complexity_keywords = re.findall(
            r"\b(if|else|elif|for|while|switch|case|try|catch|except)\b", code
        )
        complexity = len(complexity_keywords)

        # Complexity per line
        density = complexity / total_lines

        if density == 0:
            return 0.5  # no control flow — could be declarations only
        if density <= 0.15:
            return 1.0  # good: moderate complexity
        if density <= 0.25:
            return 0.7
        if density <= 0.40:
            return 0.4
        return 0.2  # very dense, hard to read

    def _score_blank_lines(self, lines: list[str]) -> float:
        """Score blank line usage.  Some spacing = readable."""
        total = len(lines)
        if total == 0:
            return 0.0

        blank_count = sum(1 for line in lines if not line.strip())
        ratio = blank_count / total

        if 0.05 <= ratio <= 0.25:
            return 1.0  # good spacing
        elif 0.01 <= ratio < 0.05 or 0.25 < ratio <= 0.35:
            return 0.6
        elif ratio < 0.01:
            return 0.3  # wall of text
        else:
            return 0.2  # too many blank lines

    def _score_length(self, lines: list[str]) -> float:
        """Score file length.  Very short or very long = penalize."""
        n = len(lines)
        if n < 5:
            return 0.1  # trivially short
        if n < 10:
            return 0.4
        if n <= 500:
            return 1.0  # sweet spot
        if n <= 1000:
            return 0.7
        if n <= 2000:
            return 0.5
        return 0.3  # very long file


# ---------------------------------------------------------------------------
# Neural quality classifier (optional, needs transformers + torch)
# ---------------------------------------------------------------------------

class CodeQualityClassifier:
    """Small neural classifier for code quality.

    Uses a fine-tuned DistilBERT/CodeBERTa model to score code quality.
    If the model or dependencies aren't available, falls back to
    HeuristicQualityScorer.

    Think of this like the difference between ESLint (heuristic) and
    a trained code reviewer (classifier) — both check quality, but the
    classifier catches subtler issues.
    """

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._tokenizer = None
        self._heuristic = HeuristicQualityScorer()
        self._model_path = model_path

        if model_path and self.is_available():
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load a pre-trained classifier from disk."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path, local_files_only=True
            )
            self._model.eval()
        except Exception:
            # Fall back to heuristic if loading fails
            self._model = None
            self._tokenizer = None

    def score(self, code: str, language: str = "") -> float:
        """Score code quality 0.0-1.0.

        Uses neural model if available, otherwise falls back to heuristics.
        """
        if self._model is not None and self._tokenizer is not None:
            return self._score_neural(code)
        return self._heuristic.score(code, language)

    def score_batch(self, codes: list[str], language: str = "") -> list[float]:
        """Batch scoring for efficiency.

        When using the neural model, batching is significantly faster
        because of GPU parallelism — like Promise.all() vs. awaiting
        each promise sequentially.
        """
        if self._model is not None and self._tokenizer is not None:
            return self._score_neural_batch(codes)
        return [self._heuristic.score(code, language) for code in codes]

    def _score_neural(self, code: str) -> float:
        """Score a single code sample with the neural model."""
        return self._score_neural_batch([code])[0]

    def _score_neural_batch(self, codes: list[str]) -> list[float]:
        """Score a batch of code samples with the neural model."""
        import torch

        inputs = self._tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Model outputs logits for scores 1-5
            # Convert to 0.0-1.0 range
            logits = outputs.logits.squeeze(-1)
            # Clamp to 1-5 range, then normalize to 0-1
            scores = torch.clamp(logits, 1.0, 5.0)
            normalized = (scores - 1.0) / 4.0

        return normalized.tolist()

    @staticmethod
    def is_available() -> bool:
        """Check if transformers + torch are available for neural scoring."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# LLM annotator (for generating training data)
# ---------------------------------------------------------------------------

class QualityAnnotator:
    """Generate training data for the classifier using an LLM.

    Uses Claude or another LLM API to score a sample of code files,
    creating training data for the classifier.

    This is the "expensive but accurate" step that runs once to create
    labels. Think of it like manually labeling a TypeScript project's
    types before letting the compiler infer the rest.
    """

    PROMPT = (
        "Rate this code on educational quality from 1 to 5:\n"
        "\n"
        "1 = Garbage: broken syntax, autogenerated boilerplate, minified/obfuscated,\n"
        "    data dumps, or meaningless snippets\n"
        "2 = Poor: technically runs but terrible style - no structure, bad naming,\n"
        "    no comments, copy-pasted spaghetti\n"
        "3 = Average: functional code with some structure. Not great, not terrible.\n"
        "    A typical file from a mid-level developer's side project.\n"
        "4 = Good: clean, well-structured code. Meaningful names, some documentation,\n"
        "    good use of language features. You'd accept this in a code review.\n"
        "5 = Excellent: textbook-quality code. Clear structure, good documentation,\n"
        "    idiomatic use of the language, educational value. You'd use this to\n"
        "    teach someone the language.\n"
        "\n"
        "Respond with ONLY the number (1-5).\n"
        "\n"
        "```{language}\n"
        "{code}\n"
        "```"
    )

    def __init__(self, api_key: str | None = None, model: str = "claude-3-haiku-20240307"):
        self._api_key = api_key
        self._model = model

    def annotate_batch(
        self, codes: list[str], language: str = ""
    ) -> list[int]:
        """Score a batch of code files using LLM API.

        Returns list of scores 1-5 for each code sample.
        Raises RuntimeError if API key is not configured.
        """
        if not self._api_key:
            raise RuntimeError(
                "No API key configured for LLM annotation. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key to constructor."
            )

        scores = []
        for code in codes:
            score = self._annotate_single(code, language)
            scores.append(score)
        return scores

    def _annotate_single(self, code: str, language: str) -> int:
        """Score a single code sample.

        Truncates to ~4k chars to stay within token limits and reduce cost.
        """
        # Truncate very long files
        truncated = code[:4000] if len(code) > 4000 else code
        prompt = self.PROMPT.format(language=language, code=truncated)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model=self._model,
                max_tokens=4,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            score = int(text[0])  # Take first digit
            return max(1, min(5, score))
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        except (ValueError, IndexError):
            return 3  # Default to average on parse failure

    def format_prompt(self, code: str, language: str = "") -> str:
        """Return the formatted prompt for a code sample (useful for debugging)."""
        truncated = code[:4000] if len(code) > 4000 else code
        return self.PROMPT.format(language=language, code=truncated)


# ---------------------------------------------------------------------------
# Filter plugin (integrates with data pipeline)
# ---------------------------------------------------------------------------

@register_filter("quality_classifier")
class QualityClassifierFilter(FilterPlugin):
    """Filter plugin that uses the quality classifier.

    Three modes:
        - "heuristic": Always available, no dependencies. Fast but less accurate.
        - "classifier": Needs a trained model file. Best balance of speed/accuracy.
        - "llm": Needs API key. Most accurate but slow and expensive.

    Usage in pipeline YAML:
        filters:
          - type: quality_classifier
            threshold: 0.4
            mode: heuristic
    """

    def __init__(
        self,
        threshold: float = 0.4,
        mode: str = "heuristic",
        model_path: str | None = None,
    ):
        self._threshold = threshold
        self._mode = mode
        self._model_path = model_path
        self._scorer: HeuristicQualityScorer | CodeQualityClassifier | None = None

    def name(self) -> str:
        return f"quality_classifier({self._mode})"

    def setup(self, config: dict[str, Any]) -> None:
        self._threshold = config.get("threshold", self._threshold)
        self._mode = config.get("mode", self._mode)
        self._model_path = config.get("model_path", self._model_path)
        # Reset scorer so it gets re-initialized on next check()
        self._scorer = None

    def _get_scorer(self) -> HeuristicQualityScorer | CodeQualityClassifier:
        """Lazy-initialize the scorer based on mode."""
        if self._scorer is not None:
            return self._scorer

        if self._mode == "classifier":
            classifier = CodeQualityClassifier(model_path=self._model_path)
            if classifier._model is not None:
                self._scorer = classifier
                return self._scorer
            # Fall back to heuristic if model not available
            self._mode = "heuristic"

        # Default: heuristic
        self._scorer = HeuristicQualityScorer()
        return self._scorer

    def check(self, record: DataRecord) -> tuple[bool, str]:
        """Score the record, reject if below threshold."""
        scorer = self._get_scorer()
        language = record.metadata.get("language", "")

        if isinstance(scorer, CodeQualityClassifier):
            quality = scorer.score(record.content, language)
        else:
            quality = scorer.score(record.content, language)

        # Store score in metadata for downstream use
        record.metadata["quality_score"] = quality

        if quality < self._threshold:
            return False, f"quality_score={quality:.2f}<{self._threshold}"
        return True, ""
