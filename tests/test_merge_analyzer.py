"""Tests for features/merge_analyzer.py — Feature 96.

All tests are CPU-only, no model weights, no I/O.
"""

from __future__ import annotations

import pytest

from cola_coder.features.merge_analyzer import (
    FEATURE_ENABLED,
    MergeAnalysisReport,
    MergeAnalyzer,
    MergeRule,
    MergeStats,
    is_enabled,
)


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def test_feature_enabled():
    assert FEATURE_ENABLED is True


def test_is_enabled():
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    a = MergeAnalyzer()
    # Simple merge rules:  h+e→he,  he+l→hel,  hel+lo→hello,  w+o→wo
    a.load_from_pairs([("h", "e"), ("he", "l"), ("hel", "lo"), ("w", "o")])
    return a


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_from_pairs(analyzer):
    assert analyzer.n_rules == 4


def test_load_from_text():
    a = MergeAnalyzer()
    merges_txt = "# version\nh e\nhe l\nhel lo\n"
    a.load_from_text(merges_txt)
    assert a.n_rules == 3


def test_vocabulary_contains_results(analyzer):
    vocab = analyzer.vocabulary
    assert "he" in vocab
    assert "hello" in vocab


def test_rule_repr():
    r = MergeRule(a="h", b="e", rank=0)
    assert "he" in repr(r)
    assert "rank=0" in repr(r)


# ---------------------------------------------------------------------------
# Corpus analysis
# ---------------------------------------------------------------------------


@pytest.fixture
def report(analyzer):
    corpus = [
        ["h", "e", "l", "lo", "w", "o"],
        ["h", "e", "he", "l"],
        ["w", "o", "wo"],
    ]
    return analyzer.analyse_corpus(corpus, top_n=5)


def test_analyse_returns_report(report):
    assert isinstance(report, MergeAnalysisReport)


def test_total_merges(report, analyzer):
    assert report.total_merges == analyzer.n_rules


def test_total_tokens_counted(report):
    # 6 + 4 + 3 = 13 tokens
    assert report.total_tokens_in_corpus == 13


def test_coverage_fraction(report):
    assert 0.0 <= report.coverage <= 1.0


def test_top_merges_sorted_by_impact(report):
    impacts = [s.impact_score for s in report.top_merges]
    assert impacts == sorted(impacts, reverse=True)


def test_report_as_dict(report):
    d = report.as_dict()
    assert "total_merges" in d
    assert "coverage" in d


# ---------------------------------------------------------------------------
# Merge tree
# ---------------------------------------------------------------------------


def test_build_tree_for_known_token(analyzer):
    node = analyzer.build_merge_tree("hello")
    assert node.token == "hello"
    assert not node.is_leaf


def test_build_tree_leaf_for_unknown(analyzer):
    node = analyzer.build_merge_tree("xyz")
    assert node.is_leaf


def test_tree_leaves_reconstruct_token(analyzer):
    node = analyzer.build_merge_tree("hello")
    leaves = node.leaves()
    assert "".join(leaves) == "hello"


def test_tree_depth_positive(analyzer):
    node = analyzer.build_merge_tree("hello")
    assert node.depth() > 0


def test_tree_depth_zero_for_leaf(analyzer):
    node = analyzer.build_merge_tree("x")
    assert node.depth() == 0


# ---------------------------------------------------------------------------
# Vocabulary suggestions
# ---------------------------------------------------------------------------


def test_suggest_removals_returns_list(report, analyzer):
    suggestions = analyzer.suggest_removals(report)
    assert isinstance(suggestions, list)


def test_suggest_removals_respects_max(report, analyzer):
    suggestions = analyzer.suggest_removals(report, max_suggestions=2)
    assert len(suggestions) <= 2


# ---------------------------------------------------------------------------
# Code-relevant merges
# ---------------------------------------------------------------------------


def test_code_relevant_merges_default_keywords():
    a = MergeAnalyzer()
    a.load_from_pairs([("de", "f"), ("re", "turn"), ("im", "port"), ("xy", "z")])
    # "def" = "de"+"f", etc.
    relevant = a.find_code_relevant_merges()
    result_tokens = [r.result for r in relevant]
    # At least "def" should be found
    assert "def" in result_tokens


def test_code_relevant_merges_custom_keywords():
    a = MergeAnalyzer()
    a.load_from_pairs([("my", "token"), ("other", "token")])
    relevant = a.find_code_relevant_merges(keywords=["mytoken"])
    assert len(relevant) == 1
    assert relevant[0].result == "mytoken"


# ---------------------------------------------------------------------------
# MergeStats
# ---------------------------------------------------------------------------


def test_merge_stats_impact_score():
    rule = MergeRule(a="ab", b="cd", rank=0)
    stats = MergeStats(rule=rule, frequency=5)
    # impact = 5 * len("abcd") = 20
    assert stats.impact_score == pytest.approx(20.0)
