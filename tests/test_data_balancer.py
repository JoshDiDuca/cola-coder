"""Tests for DataBalancer (features/data_balancer.py)."""

from __future__ import annotations

import pytest

from cola_coder.features.data_balancer import (
    FEATURE_ENABLED,
    BalancerReport,
    DataBalancer,
    SampleMetadata,
    compute_weights,
    detect_language,
    estimate_difficulty,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_py_extension(self):
        assert detect_language("", "foo.py") == "python"

    def test_ts_extension(self):
        assert detect_language("", "app.ts") == "typescript"

    def test_js_extension(self):
        assert detect_language("", "main.js") == "javascript"

    def test_python_heuristic(self):
        src = "import os\ndef foo(): pass"
        assert detect_language(src) == "python"

    def test_unknown_fallback(self):
        assert detect_language("hello world", "") == "unknown"


# ---------------------------------------------------------------------------
# estimate_difficulty
# ---------------------------------------------------------------------------


class TestEstimateDifficulty:
    def test_empty_is_zero(self):
        assert estimate_difficulty("") == pytest.approx(0.0)

    def test_score_in_range(self):
        src = "x = 1\ny = 2\n"
        d = estimate_difficulty(src)
        assert 0.0 <= d <= 1.0

    def test_nested_code_harder_than_flat(self):
        flat = "x = 1\ny = 2\n"
        nested = "def f():\n    if True:\n        for i in range(10):\n            x = i\n"
        assert estimate_difficulty(nested) > estimate_difficulty(flat)


# ---------------------------------------------------------------------------
# DataBalancer.analyze
# ---------------------------------------------------------------------------


class TestDataBalancerAnalyze:
    def _make_samples(self, langs, topics=None):
        topics = topics or ["general"] * len(langs)
        return [
            SampleMetadata(idx=i, language=lang, topic=topic)
            for i, (lang, topic) in enumerate(zip(langs, topics))
        ]

    def test_returns_report(self):
        samples = self._make_samples(["python", "typescript"])
        r = DataBalancer().analyze(samples)
        assert isinstance(r, BalancerReport)

    def test_weights_length_matches_samples(self):
        samples = self._make_samples(["python"] * 5 + ["typescript"] * 3)
        r = DataBalancer().analyze(samples)
        assert len(r.weights) == 8

    def test_weights_in_min_max_range(self):
        bal = DataBalancer(min_weight=0.1, max_weight=3.0)
        samples = self._make_samples(["python"] * 5 + ["js"] * 2)
        r = bal.analyze(samples)
        for w in r.weights:
            assert 0.1 <= w <= 3.0

    def test_empty_samples(self):
        r = DataBalancer().analyze([])
        assert r.num_samples == 0
        assert r.weights == []

    def test_language_counts_correct(self):
        samples = self._make_samples(["python", "python", "typescript"])
        r = DataBalancer().analyze(samples)
        assert r.language_counts["python"] == 2
        assert r.language_counts["typescript"] == 1

    def test_imbalance_issue_reported(self):
        # 90% python → should trigger imbalance warning
        samples = self._make_samples(["python"] * 9 + ["typescript"] * 1)
        r = DataBalancer().analyze(samples)
        assert any("imbalance" in i.lower() for i in r.issues)

    def test_effective_samples_reasonable(self):
        samples = self._make_samples(["python"] * 4 + ["typescript"] * 4)
        r = DataBalancer().analyze(samples)
        assert r.effective_samples > 0


# ---------------------------------------------------------------------------
# compute_weights convenience
# ---------------------------------------------------------------------------


class TestComputeWeights:
    def test_returns_list_of_floats(self):
        class S:
            language = "python"
            topic = "general"

        w = compute_weights([S(), S(), S()])
        assert len(w) == 3
        assert all(isinstance(x, float) for x in w)


# ---------------------------------------------------------------------------
# from_sources
# ---------------------------------------------------------------------------


class TestFromSources:
    def test_from_sources_returns_report(self):
        sources = [
            "import os\ndef foo(): pass",
            "const x = 1;\nfunction bar() {}",
        ]
        r = DataBalancer().from_sources(sources, filenames=["a.py", "b.js"])
        assert r.num_samples == 2
        assert "python" in r.language_counts
