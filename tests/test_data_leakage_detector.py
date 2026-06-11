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


# ---------------------------------------------------------------------------
# Containment metric (catches an eval problem embedded in a larger train doc)
# ---------------------------------------------------------------------------


class TestContainmentMetric:
    def test_containment_catches_embedded_problem(self):
        problem = "def is_prime(n):\n    return all(n % i for i in range(2, n))\n"
        big = "import os\n" * 200 + problem + "\nclass Foo:\n    pass\n" * 200
        d = DataLeakageDetector(similarity_threshold=0.8)
        d.index_train([big, "totally unrelated code\n" * 50])

        # jaccard misses the embedded problem (|small| / |large| is tiny)...
        assert d.check_eval([problem], metric="jaccard").num_contaminated == 0
        # ...containment catches it.
        report = d.check_eval([problem], metric="containment")
        assert report.num_contaminated == 1
        assert report.matches[0].similarity >= 0.8

    def test_containment_clean_when_absent(self):
        d = DataLeakageDetector(similarity_threshold=0.8)
        d.index_train(["def foo():\n    return 1\n" * 50])
        report = d.check_eval(
            ["def completely_different_xyz():\n    return 999\n"],
            metric="containment",
        )
        assert report.num_contaminated == 0

    def test_invalid_metric_raises(self):
        d = DataLeakageDetector()
        d.index_train(["x"])
        import pytest
        with pytest.raises(ValueError, match="metric must be"):
            d.check_eval(["y"], metric="cosine")

    def test_jaccard_still_default(self):
        # Backward-compat: default metric is jaccard (existing callers unchanged).
        a = "shared text aaaa bbbb cccc dddd"
        d = DataLeakageDetector(similarity_threshold=0.5)
        d.index_train([a])
        assert d.check_eval([a]).num_contaminated == 1  # exact dup, jaccard ~1


# ---------------------------------------------------------------------------
# Script wiring (check_contamination.py)
# ---------------------------------------------------------------------------


class TestContaminationScriptWiring:
    def test_script_exists_and_uses_detector(self):
        from pathlib import Path
        text = (Path(__file__).parent.parent / "scripts"
                / "check_contamination.py").read_text(encoding="utf-8")
        assert "DataLeakageDetector" in text
        assert 'metric="' in text or "args.metric" in text
        assert "--train-jsonl" in text and "--train-npy" in text

    def test_menu_wires_script(self):
        from pathlib import Path
        text = (Path(__file__).parent.parent / "src" / "cola_coder" / "features"
                / "menus" / "eval_menu.py").read_text(encoding="utf-8")
        assert "_contamination_menu" in text
        assert "check_contamination.py" in text

    def test_eval_docs_are_undiluted_prompts(self):
        # Regression: eval docs must be the prompt (and solution) as SEPARATE
        # units, NOT prompt+test_code concatenated — concatenation diluted the
        # containment signal so a leaked prompt scored below threshold.
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "check_contamination",
            Path(__file__).parent.parent / "scripts" / "check_contamination.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from cola_coder.evaluation.problem_loader import ProblemSet

        ps = ProblemSet()
        ps.add_builtin(extended=True)
        args = type("A", (), {"eval": "all", "eval_jsonl": None})()
        docs = mod._load_eval_docs(args)
        assert len(docs) >= len(ps._problems)
        # The first problem's prompt appears verbatim as one of the docs
        # (not concatenated with its hidden tests).
        assert ps._problems[0].prompt in docs
