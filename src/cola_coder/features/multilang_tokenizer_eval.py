"""Multi-Language Tokenizer Evaluator.

Evaluates tokenizer quality across multiple programming languages by measuring:
  - Fertility: average tokens per character (lower = more efficient)
  - Keyword preservation: fraction of language keywords that are single tokens
  - Cross-language consistency: how similarly different languages are tokenized

Works with any tokenizer that exposes an ``encode(text) -> list[int]`` method,
making it compatible with HuggingFace tokenizers, tiktoken, and custom tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """Minimal interface expected from any tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        ...


# ---------------------------------------------------------------------------
# Language keyword lists
# ---------------------------------------------------------------------------

LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "python": [
        "def", "class", "return", "import", "from", "as", "if", "else",
        "elif", "for", "while", "in", "not", "and", "or", "is", "None",
        "True", "False", "try", "except", "finally", "with", "yield",
        "lambda", "pass", "break", "continue", "raise", "assert", "del",
    ],
    "javascript": [
        "function", "const", "let", "var", "return", "if", "else", "for",
        "while", "class", "new", "this", "import", "export", "default",
        "async", "await", "try", "catch", "finally", "throw", "typeof",
        "instanceof", "null", "undefined", "true", "false",
    ],
    "typescript": [
        "function", "const", "let", "var", "return", "if", "else", "for",
        "while", "class", "new", "this", "import", "export", "default",
        "async", "await", "interface", "type", "enum", "namespace",
        "implements", "extends", "abstract", "readonly", "public", "private",
        "protected", "string", "number", "boolean", "void", "null",
    ],
    "java": [
        "public", "private", "protected", "static", "final", "class",
        "interface", "extends", "implements", "return", "if", "else",
        "for", "while", "do", "new", "this", "super", "null", "true",
        "false", "void", "int", "long", "double", "float", "boolean",
        "try", "catch", "finally", "throw", "throws", "import", "package",
    ],
    "rust": [
        "fn", "let", "mut", "const", "struct", "enum", "impl", "trait",
        "use", "mod", "pub", "return", "if", "else", "for", "while",
        "loop", "match", "in", "where", "type", "self", "Self",
        "true", "false", "Some", "None", "Ok", "Err", "async", "await",
    ],
    "go": [
        "func", "var", "const", "type", "struct", "interface", "map",
        "chan", "go", "return", "if", "else", "for", "range", "switch",
        "case", "default", "break", "continue", "import", "package",
        "defer", "select", "make", "new", "nil", "true", "false",
    ],
}

# Representative code snippets for fertility measurement
LANGUAGE_SAMPLES: dict[str, str] = {
    "python": (
        "def compute_sum(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "class Calculator:\n"
        "    def __init__(self):\n"
        "        self.result = 0\n"
        "    def add(self, x):\n"
        "        self.result += x\n"
        "        return self\n"
    ),
    "javascript": (
        "function computeSum(a, b) {\n"
        "    return a + b;\n"
        "}\n\n"
        "class Calculator {\n"
        "    constructor() {\n"
        "        this.result = 0;\n"
        "    }\n"
        "    add(x) {\n"
        "        this.result += x;\n"
        "        return this;\n"
        "    }\n"
        "}\n"
    ),
    "typescript": (
        "function computeSum(a: number, b: number): number {\n"
        "    return a + b;\n"
        "}\n\n"
        "class Calculator {\n"
        "    private result: number = 0;\n"
        "    add(x: number): Calculator {\n"
        "        this.result += x;\n"
        "        return this;\n"
        "    }\n"
        "}\n"
    ),
    "java": (
        "public class Calculator {\n"
        "    private int result = 0;\n"
        "    public int computeSum(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "    public Calculator add(int x) {\n"
        "        this.result += x;\n"
        "        return this;\n"
        "    }\n"
        "}\n"
    ),
    "rust": (
        "fn compute_sum(a: i32, b: i32) -> i32 {\n"
        "    a + b\n"
        "}\n\n"
        "struct Calculator {\n"
        "    result: i32,\n"
        "}\n\n"
        "impl Calculator {\n"
        "    fn add(&mut self, x: i32) -> &mut Self {\n"
        "        self.result += x;\n"
        "        self\n"
        "    }\n"
        "}\n"
    ),
    "go": (
        "func computeSum(a int, b int) int {\n"
        "    return a + b\n"
        "}\n\n"
        "type Calculator struct {\n"
        "    result int\n"
        "}\n\n"
        "func (c *Calculator) Add(x int) *Calculator {\n"
        "    c.result += x\n"
        "    return c\n"
        "}\n"
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LanguageResult:
    """Tokenizer evaluation for a single language."""

    language: str
    fertility: float          # tokens / character
    token_count: int
    char_count: int
    keyword_preservation: float  # fraction of keywords that are single tokens
    keywords_single_token: list[str]
    keywords_multi_token: list[str]

    def summary(self) -> str:
        return (
            f"{self.language}: fertility={self.fertility:.4f}, "
            f"kw_preservation={self.keyword_preservation:.2%}"
        )


@dataclass
class TokenizerEvalReport:
    """Aggregated multi-language evaluation report."""

    results: dict[str, LanguageResult] = field(default_factory=dict)
    cross_language_consistency: float = 0.0  # std-dev of fertilities (lower = more consistent)
    mean_fertility: float = 0.0
    mean_keyword_preservation: float = 0.0

    def add(self, result: LanguageResult) -> None:
        self.results[result.language] = result
        self._recompute()

    def _recompute(self) -> None:
        if not self.results:
            return
        fertilities = [r.fertility for r in self.results.values()]
        kw_preses = [r.keyword_preservation for r in self.results.values()]
        self.mean_fertility = sum(fertilities) / len(fertilities)
        self.mean_keyword_preservation = sum(kw_preses) / len(kw_preses)
        # Consistency = 1 - normalised std dev
        if len(fertilities) > 1:
            mean_f = self.mean_fertility
            variance = sum((f - mean_f) ** 2 for f in fertilities) / len(fertilities)
            std = variance ** 0.5
            # Normalise by mean to get coefficient of variation
            cv = std / (mean_f + 1e-9)
            self.cross_language_consistency = max(0.0, 1.0 - cv)
        else:
            self.cross_language_consistency = 1.0

    def best_language(self) -> str | None:
        """Language with the lowest fertility (most efficient encoding)."""
        if not self.results:
            return None
        return min(self.results, key=lambda lang: self.results[lang].fertility)

    def worst_language(self) -> str | None:
        """Language with the highest fertility (least efficient encoding)."""
        if not self.results:
            return None
        return max(self.results, key=lambda lang: self.results[lang].fertility)

    def summary(self) -> str:
        lines = [
            f"Languages evaluated: {len(self.results)}",
            f"Mean fertility: {self.mean_fertility:.4f} tok/char",
            f"Mean keyword preservation: {self.mean_keyword_preservation:.2%}",
            f"Cross-language consistency: {self.cross_language_consistency:.3f}",
        ]
        for lang, r in sorted(self.results.items()):
            lines.append(f"  {r.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class MultiLangTokenizerEvaluator:
    """Evaluate a tokenizer across multiple programming languages.

    Parameters
    ----------
    tokenizer:
        Any object with an ``encode(text) -> list[int]`` method.
    languages:
        Which languages to evaluate.  Defaults to all built-in languages.
    custom_samples:
        Override the built-in code samples with your own per-language strings.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        languages: list[str] | None = None,
        custom_samples: dict[str, str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.languages = languages or list(LANGUAGE_SAMPLES)
        self.samples = dict(LANGUAGE_SAMPLES)
        if custom_samples:
            self.samples.update(custom_samples)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> TokenizerEvalReport:
        """Run full evaluation and return a TokenizerEvalReport."""
        report = TokenizerEvalReport()
        for lang in self.languages:
            result = self.evaluate_language(lang)
            if result is not None:
                report.add(result)
        return report

    def evaluate_language(self, language: str) -> LanguageResult | None:
        """Evaluate a single language. Returns None if language is unknown."""
        sample = self.samples.get(language)
        if not sample:
            return None

        tokens = self.tokenizer.encode(sample)
        token_count = len(tokens)
        char_count = len(sample)
        fertility = token_count / max(char_count, 1)

        keywords = LANGUAGE_KEYWORDS.get(language, [])
        single: list[str] = []
        multi: list[str] = []
        for kw in keywords:
            kw_tokens = self.tokenizer.encode(kw)
            if len(kw_tokens) == 1:
                single.append(kw)
            else:
                multi.append(kw)

        preservation = len(single) / max(len(keywords), 1)

        return LanguageResult(
            language=language,
            fertility=round(fertility, 6),
            token_count=token_count,
            char_count=char_count,
            keyword_preservation=round(preservation, 4),
            keywords_single_token=single,
            keywords_multi_token=multi,
        )

    def compare_languages(self, lang_a: str, lang_b: str) -> dict[str, float]:
        """Compare two languages, returning a dict of metric differences."""
        result_a = self.evaluate_language(lang_a)
        result_b = self.evaluate_language(lang_b)
        if result_a is None or result_b is None:
            return {}
        return {
            "fertility_diff": result_a.fertility - result_b.fertility,
            "keyword_preservation_diff": (
                result_a.keyword_preservation - result_b.keyword_preservation
            ),
        }
