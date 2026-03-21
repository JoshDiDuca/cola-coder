"""Tests for multilang_tokenizer_eval.py."""

from __future__ import annotations


from cola_coder.features.multilang_tokenizer_eval import (
    FEATURE_ENABLED,
    LANGUAGE_KEYWORDS,
    LANGUAGE_SAMPLES,
    LanguageResult,
    MultiLangTokenizerEvaluator,
    TokenizerEvalReport,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Tokenize by individual characters (worst case — many tokens)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class WordTokenizer:
    """Tokenize by whitespace-split words — each word is one token."""

    def encode(self, text: str) -> list[int]:
        # Simple hash as token ID
        return [hash(w) & 0xFFFF for w in text.split()]


class PerfectKeywordTokenizer:
    """Every token of 1+ chars is a single token (best case)."""

    def encode(self, text: str) -> list[int]:
        return [hash(c) & 0xFFFF for c in text.split()]


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Language samples / keywords completeness
# ---------------------------------------------------------------------------

class TestData:
    def test_samples_non_empty(self):
        for lang, sample in LANGUAGE_SAMPLES.items():
            assert len(sample) > 0, f"Empty sample for {lang}"

    def test_keywords_non_empty(self):
        for lang, kws in LANGUAGE_KEYWORDS.items():
            assert len(kws) > 0, f"Empty keywords for {lang}"

    def test_python_has_def(self):
        assert "def" in LANGUAGE_KEYWORDS["python"]

    def test_javascript_has_function(self):
        assert "function" in LANGUAGE_KEYWORDS["javascript"]


# ---------------------------------------------------------------------------
# LanguageResult
# ---------------------------------------------------------------------------

class TestEvaluateLanguage:
    def test_returns_language_result(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python"])
        result = evaluator.evaluate_language("python")
        assert isinstance(result, LanguageResult)

    def test_fertility_char_tokenizer_high(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python"])
        result = evaluator.evaluate_language("python")
        # Char tokenizer → ~1 token per char
        assert result is not None
        assert result.fertility >= 0.9

    def test_fertility_word_tokenizer_lower(self):
        word_eval = MultiLangTokenizerEvaluator(WordTokenizer(), languages=["python"])
        char_eval = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python"])
        word_result = word_eval.evaluate_language("python")
        char_result = char_eval.evaluate_language("python")
        assert word_result is not None and char_result is not None
        assert word_result.fertility < char_result.fertility

    def test_unknown_language_returns_none(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer())
        result = evaluator.evaluate_language("cobol")
        assert result is None

    def test_keyword_preservation_range(self):
        evaluator = MultiLangTokenizerEvaluator(WordTokenizer(), languages=["python"])
        result = evaluator.evaluate_language("python")
        assert result is not None
        assert 0.0 <= result.keyword_preservation <= 1.0

    def test_summary_str(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python"])
        result = evaluator.evaluate_language("python")
        assert result is not None
        s = result.summary()
        assert "python" in s
        assert "fertility" in s


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

class TestFullEvaluation:
    def test_evaluate_returns_report(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python", "javascript"])
        report = evaluator.evaluate()
        assert isinstance(report, TokenizerEvalReport)

    def test_all_languages_present(self):
        langs = ["python", "javascript"]
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=langs)
        report = evaluator.evaluate()
        for lang in langs:
            assert lang in report.results

    def test_best_worst_language(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python", "javascript", "rust"])
        report = evaluator.evaluate()
        best = report.best_language()
        worst = report.worst_language()
        assert best is not None
        assert worst is not None
        assert report.results[best].fertility <= report.results[worst].fertility

    def test_consistency_score_range(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python", "javascript", "go"])
        report = evaluator.evaluate()
        assert 0.0 <= report.cross_language_consistency <= 1.0

    def test_compare_languages(self):
        evaluator = MultiLangTokenizerEvaluator(CharTokenizer(), languages=["python", "javascript"])
        diff = evaluator.compare_languages("python", "javascript")
        assert "fertility_diff" in diff

    def test_custom_sample(self):
        custom = {"python": "x = 1"}
        evaluator = MultiLangTokenizerEvaluator(
            CharTokenizer(),
            languages=["python"],
            custom_samples=custom,
        )
        result = evaluator.evaluate_language("python")
        assert result is not None
        # "x = 1" is 5 chars; char tokenizer gives 5 tokens → fertility ~1.0
        assert abs(result.fertility - 1.0) < 0.01
