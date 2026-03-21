"""Tests for data_quality_report.py."""

import json

import pytest

from cola_coder.features.data_quality_report import DataQualityReport, Report, _compute_stats


@pytest.fixture
def dqr():
    return DataQualityReport(sample_limit=1000)


_PY_SAMPLES = [
    "def add(a, b):\n    return a + b\n",
    "import os\nimport sys\n\nclass Foo:\n    pass\n",
    "for i in range(10):\n    print(i)\n",
]

_TS_SAMPLES = [
    "const x: number = 1;\nexport default x;\n",
    "interface Foo { bar: string; }\n",
]


def test_feature_enabled():
    from cola_coder.features.data_quality_report import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_generate_from_text_file(dqr, tmp_path):
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("\n".join(_PY_SAMPLES), encoding="utf-8")
    report = dqr.generate(data_file)
    assert isinstance(report, Report)
    assert report.total_samples > 0


def test_generate_from_jsonl(dqr, tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"text": s, "quality": 0.75})
        for s in _PY_SAMPLES
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")
    report = dqr.generate(jsonl_file)
    assert report.total_samples == len(_PY_SAMPLES)
    assert report.quality_buckets is not None
    assert report.quality_mean is not None


def test_language_mix_python(dqr, tmp_path):
    data_file = tmp_path / "py.txt"
    data_file.write_text("\n".join(_PY_SAMPLES), encoding="utf-8")
    report = dqr.generate(data_file)
    assert report.language_mix.python >= 0.0


def test_to_markdown(dqr, tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("\n".join(_PY_SAMPLES + _TS_SAMPLES), encoding="utf-8")
    report = dqr.generate(data_file)
    md = report.to_markdown()
    assert isinstance(md, str)
    assert "# Training Data Quality Report" in md
    assert "## Overview" in md


def test_to_json(dqr, tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text(_PY_SAMPLES[0], encoding="utf-8")
    report = dqr.generate(data_file)
    j = report.to_json()
    data = json.loads(j)
    assert "total_samples" in data


def test_nonexistent_path(dqr, tmp_path):
    report = dqr.generate(tmp_path / "does_not_exist.txt")
    assert report.total_samples == 0
    assert len(report.notes) > 0


def test_compute_stats_basic():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = _compute_stats(values)
    assert stats.count == 10
    assert abs(stats.mean - 5.5) < 0.01
    assert stats.min_val == 1
    assert stats.max_val == 10


def test_quality_buckets_distribution(dqr, tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    # Mix of quality scores across all buckets
    scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    lines = [
        json.dumps({"text": f"sample {i}", "quality": s})
        for i, s in enumerate(scores)
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")
    report = dqr.generate(jsonl_file)
    qb = report.quality_buckets
    assert qb is not None
    total = qb.excellent + qb.good + qb.average + qb.poor + qb.reject
    assert abs(total - 1.0) < 0.01
