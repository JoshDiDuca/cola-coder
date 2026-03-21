"""Tests for data_leakage_detector.py (feature 46)."""

from __future__ import annotations


from cola_coder.features.data_leakage_detector import (
    FEATURE_ENABLED,
    ContaminationMatch,
    DataLeakageDetector,
    LeakageReport,
    _jaccard_from_minhash,
    _minhash,
    _shingles,
    is_enabled,
)


def test_feature_enabled():
    assert FEATURE_ENABLED is True
    assert is_enabled() is True


# ---------------------------------------------------------------------------
# Shingle / MinHash helpers
# ---------------------------------------------------------------------------


def test_shingles_basic():
    s = _shingles("hello world", n=5)
    assert isinstance(s, set)
    assert len(s) > 0
    # Each shingle should be exactly 5 chars
    for shingle in s:
        assert len(shingle) == 5


def test_shingles_short_text():
    s = _shingles("hi", n=5)
    # Short text → single shingle
    assert len(s) == 1


def test_shingles_empty():
    s = _shingles("", n=5)
    assert s == set()


def test_minhash_length():
    sig = _minhash({"abc", "def", "ghi"}, num_hashes=64)
    assert len(sig) == 64


def test_minhash_same_input_is_deterministic():
    s = {"foo", "bar", "baz"}
    sig1 = _minhash(s, num_hashes=32)
    sig2 = _minhash(s, num_hashes=32)
    assert sig1 == sig2


def test_jaccard_identical():
    sig = _minhash({"a", "b", "c"}, num_hashes=128)
    j = _jaccard_from_minhash(sig, sig)
    assert j == 1.0


def test_jaccard_disjoint():
    sig_a = _minhash({"aaa", "bbb", "ccc"}, num_hashes=128)
    sig_b = _minhash({"xxx", "yyy", "zzz"}, num_hashes=128)
    j = _jaccard_from_minhash(sig_a, sig_b)
    assert j < 0.1  # very unlikely to match on disjoint sets


# ---------------------------------------------------------------------------
# Detector behavior
# ---------------------------------------------------------------------------


def test_exact_duplicate_detected():
    doc = "def foo(): return 42\n" * 10
    detector = DataLeakageDetector(similarity_threshold=0.8, num_hashes=128)
    detector.index_train([doc, "completely different text about dogs and cats"])
    report = detector.check_eval([doc])
    assert report.has_leakage()
    assert report.num_contaminated == 1
    assert report.matches[0].similarity > 0.8


def test_clean_eval_no_leakage():
    train = ["def foo(): return 42\n" * 10]
    eval_docs = ["completely different document about machine learning"]
    detector = DataLeakageDetector(similarity_threshold=0.8, num_hashes=64)
    detector.index_train(train)
    report = detector.check_eval(eval_docs)
    assert not report.has_leakage()
    assert report.num_contaminated == 0


def test_contamination_rate_calculation():
    doc_a = "the quick brown fox jumps over the lazy dog " * 5
    doc_b = "completely unrelated content about science and math"
    detector = DataLeakageDetector(similarity_threshold=0.8, num_hashes=128)
    detector.index_train([doc_a])
    # 1 of 2 eval docs is contaminated
    report = detector.check_eval([doc_a, doc_b])
    assert report.num_eval_docs == 2
    assert report.num_contaminated == 1
    assert abs(report.contamination_rate - 0.5) < 0.01


def test_match_preview_populated():
    doc = "x = 1 + 2\n" * 20
    detector = DataLeakageDetector(similarity_threshold=0.7, num_hashes=64)
    detector.index_train([doc])
    report = detector.check_eval([doc])
    assert report.has_leakage()
    match = report.matches[0]
    assert len(match.eval_preview) <= 100
    assert len(match.train_preview) <= 100


def test_no_train_docs_indexed():
    detector = DataLeakageDetector()
    report = detector.check_eval(["some eval text"])
    assert not report.has_leakage()
    assert report.num_train_docs == 0


def test_num_train_indexed_property():
    detector = DataLeakageDetector()
    assert detector.num_train_indexed == 0
    detector.index_train(["doc1", "doc2", "doc3"])
    assert detector.num_train_indexed == 3


def test_report_summary_format():
    report = LeakageReport(
        num_eval_docs=10,
        num_train_docs=100,
        num_contaminated=2,
        contamination_rate=0.2,
    )
    s = report.summary()
    assert "eval=10" in s
    assert "train=100" in s
    assert "contaminated=2" in s


def test_match_summary_format():
    match = ContaminationMatch(
        eval_doc_id=3,
        train_doc_id=7,
        similarity=0.95,
        eval_preview="eval text",
        train_preview="train text",
    )
    s = match.summary()
    assert "eval[3]" in s
    assert "train[7]" in s
    assert "similarity=0.950" in s
