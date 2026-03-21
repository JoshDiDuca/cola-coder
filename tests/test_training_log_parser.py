"""Tests for training_log_parser.py."""

import json

import pytest

from cola_coder.features.training_log_parser import TrainingLogParser


@pytest.fixture
def parser():
    return TrainingLogParser()


_PLAIN_LOG = """\
step=100  loss=5.1234  lr=3.0e-04  grad_norm=1.23  epoch=1
step=200  loss=4.5678  lr=2.9e-04  grad_norm=1.10  epoch=1
step=300  loss=3.9012  lr=2.8e-04  grad_norm=0.98  epoch=2
"""

_JSONL_LOG = "\n".join(
    json.dumps(
        {
            "step": (i + 1) * 100,
            "loss": 5.0 - i * 0.3,
            "lr": 3e-4 - i * 1e-5,
            "grad_norm": 1.0 - i * 0.05,
            "epoch": 1,
        }
    )
    for i in range(5)
)


def test_feature_enabled():
    from cola_coder.features.training_log_parser import FEATURE_ENABLED, is_enabled

    assert FEATURE_ENABLED is True
    assert is_enabled() is True


def test_parse_plain_text(parser):
    log = parser.parse_text(_PLAIN_LOG)
    assert len(log.records) == 3
    assert log.records[0].step == 100
    assert abs(log.records[0].loss - 5.1234) < 1e-4


def test_parse_jsonl(parser):
    log = parser.parse_text(_JSONL_LOG)
    assert len(log.records) == 5
    assert log.records[0].step == 100


def test_aggregates_computed(parser):
    log = parser.parse_text(_PLAIN_LOG)
    assert log.min_loss is not None
    assert log.max_loss is not None
    assert log.final_loss == log.records[-1].loss
    assert log.total_steps == 300


def test_loss_curve(parser):
    log = parser.parse_text(_PLAIN_LOG)
    curve = log.loss_curve()
    assert len(curve) == 3
    assert all(isinstance(s, int) and isinstance(v, float) for s, v in curve)


def test_lr_curve(parser):
    log = parser.parse_text(_PLAIN_LOG)
    curve = log.lr_curve()
    assert len(curve) == 3


def test_to_json(parser):
    log = parser.parse_text(_PLAIN_LOG)
    j = log.to_json()
    data = json.loads(j)
    assert "records" in data
    assert len(data["records"]) == 3


def test_to_csv(parser):
    log = parser.parse_text(_PLAIN_LOG)
    csv_text = log.to_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("step")
    assert len(lines) == 4  # header + 3 records


def test_plot_ascii(parser):
    log = parser.parse_text(_PLAIN_LOG)
    chart = log.plot_ascii("loss")
    assert isinstance(chart, str)
    assert "LOSS" in chart


def test_nonexistent_file(parser, tmp_path):
    log = parser.parse(tmp_path / "does_not_exist.log")
    assert log.records == []


def test_parse_file(parser, tmp_path):
    log_file = tmp_path / "train.log"
    log_file.write_text(_PLAIN_LOG, encoding="utf-8")
    log = parser.parse(log_file)
    assert len(log.records) == 3
    assert log.source_path == str(log_file)
