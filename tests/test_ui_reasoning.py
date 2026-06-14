"""Tests for the read-only reasoning/GRPO config UI helper."""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.reasoning import read_reasoning

_FAKE_YAML = """\
model:
  dim: 768
reasoning:
  group_size: 8
  advantage_norm: "mean"
  clip_epsilon: 0.2
  clip_epsilon_high: 0.28
sft_warmup:
  enabled: true
  epochs: 5
problem_set:
  source: "builtin"
  difficulty: "all"
"""


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "reasoning.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_returns_path_and_full_parsed(tmp_path):
    result = read_reasoning(_write(tmp_path, _FAKE_YAML))
    assert "error" not in result
    assert result["path"].endswith("reasoning.yaml")
    assert isinstance(result["parsed"], dict)
    assert result["parsed"]["model"]["dim"] == 768


def test_sections_extracted(tmp_path):
    result = read_reasoning(_write(tmp_path, _FAKE_YAML))
    assert result["reasoning"]["group_size"] == 8
    assert result["problem_set"]["difficulty"] == "all"
    assert result["sft_warmup"]["epochs"] == 5


def test_summary_extraction(tmp_path):
    result = read_reasoning(_write(tmp_path, _FAKE_YAML))
    summary = result["summary"]
    assert summary["advantage_norm"] == "mean"
    assert summary["clip_epsilon"] == 0.2
    assert summary["clip_epsilon_high"] == 0.28
    assert summary["group_size"] == 8
    assert summary["sft_warmup_enabled"] is True
    assert summary["problem_source"] == "builtin"


def test_missing_sections_default_to_empty(tmp_path):
    result = read_reasoning(_write(tmp_path, "model:\n  dim: 768\n"))
    assert "error" not in result
    assert result["reasoning"] == {}
    assert result["problem_set"] == {}
    assert result["sft_warmup"] == {}


def test_absent_summary_keys_are_none(tmp_path):
    result = read_reasoning(_write(tmp_path, "model:\n  dim: 768\n"))
    summary = result["summary"]
    assert summary["advantage_norm"] is None
    assert summary["clip_epsilon"] is None
    assert summary["clip_epsilon_high"] is None
    assert summary["group_size"] is None
    assert summary["sft_warmup_enabled"] is None
    assert summary["problem_source"] is None


def test_non_dict_section_treated_as_empty(tmp_path):
    text = "reasoning: not_a_mapping\nproblem_set:\n  - a\n  - b\n"
    result = read_reasoning(_write(tmp_path, text))
    assert "error" not in result
    assert result["reasoning"] == {}
    assert result["problem_set"] == {}


def test_missing_file_returns_error(tmp_path):
    result = read_reasoning(str(tmp_path / "does_not_exist.yaml"))
    assert "error" in result
    assert "not found" in result["error"]


def test_garbage_yaml_returns_error(tmp_path):
    result = read_reasoning(_write(tmp_path, "this: : : [unbalanced\n  - {"))
    assert "error" in result


def test_non_mapping_top_level_returns_error(tmp_path):
    result = read_reasoning(_write(tmp_path, "- just\n- a\n- list\n"))
    assert "error" in result
    assert "shape" in result["error"]


def test_empty_file_returns_error(tmp_path):
    result = read_reasoning(_write(tmp_path, ""))
    assert "error" in result


def test_real_reasoning_yaml_parses():
    repo_root = Path(__file__).resolve().parents[1]
    real = repo_root / "configs" / "reasoning.yaml"
    result = read_reasoning(str(real))
    assert "error" not in result
    assert isinstance(result["parsed"], dict)
    assert result["parsed"]
