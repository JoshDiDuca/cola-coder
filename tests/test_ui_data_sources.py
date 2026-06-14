"""Tests for the read-only data-sources UI view."""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.data_sources_view import read_data_sources

_FAKE_YAML = """\
sources:
  code:
    dataset: "bigcode/starcoderdata"
    weight: 0.7
    enabled: true
    languages:
      - typescript
      - javascript
      - python
  text:
    dataset: "HuggingFaceFW/fineweb-edu"
    weight: 0.2
    enabled: true
  math:
    dataset: "open-web-math/open-web-math"
    weight: 0.1
    enabled: true

github:
  min_stars: 50
"""


def _write(tmp_path: Path, text: str, name: str = "data_sources.yaml") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parses_three_sources(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, _FAKE_YAML))
    assert "error" not in result
    names = {s["name"] for s in result["sources"]}
    assert names == {"code", "text", "math"}


def test_weights_datasets_languages(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, _FAKE_YAML))
    by_name = {s["name"]: s for s in result["sources"]}

    assert by_name["code"]["weight"] == 0.7
    assert by_name["code"]["dataset"] == "bigcode/starcoderdata"
    assert by_name["code"]["languages"] == ["typescript", "javascript", "python"]
    assert by_name["code"]["kind"] == "code"

    assert by_name["text"]["weight"] == 0.2
    assert by_name["text"]["languages"] == []  # no languages key


def test_total_weight(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, _FAKE_YAML))
    assert result["total_weight"] is not None
    assert abs(result["total_weight"] - 1.0) < 1e-9


def test_summary_one_line(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, _FAKE_YAML))
    summary = result["summary"]
    assert summary.startswith("3 sources")
    assert "code 70%" in summary
    assert "text 20%" in summary
    assert "math 10%" in summary


def test_parsed_full_yaml_included(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, _FAKE_YAML))
    assert isinstance(result["parsed"], dict)
    # Non-source keys preserved in the raw view.
    assert result["parsed"]["github"]["min_stars"] == 50


def test_missing_file_returns_error() -> None:
    result = read_data_sources("does/not/exist.yaml")
    assert "error" in result
    assert "sources" not in result


def test_garbage_yaml_returns_error(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, "key: [unterminated\n  : :"))
    assert "error" in result


def test_non_mapping_top_level_returns_error(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, "- just\n- a\n- list\n"))
    assert "error" in result


def test_missing_sources_section_returns_error(tmp_path: Path) -> None:
    result = read_data_sources(_write(tmp_path, "github:\n  min_stars: 50\n"))
    assert "error" in result


def test_percentage_weights_handled(tmp_path: Path) -> None:
    text = (
        "sources:\n"
        "  code:\n    dataset: ds\n    weight: 70\n"
        "  text:\n    dataset: ds2\n    weight: 30\n"
    )
    result = read_data_sources(_write(tmp_path, text))
    assert "error" not in result
    assert result["total_weight"] == 100.0
    assert "code 70%" in result["summary"]


def test_sources_as_list_shape(tmp_path: Path) -> None:
    text = (
        "sources:\n"
        "  - name: code\n    dataset: ds\n    weight: 0.7\n"
        "  - name: text\n    dataset: ds2\n    weight: 0.3\n"
    )
    result = read_data_sources(_write(tmp_path, text))
    assert "error" not in result
    names = {s["name"] for s in result["sources"]}
    assert names == {"code", "text"}


def test_real_config_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    real = repo_root / "configs" / "data_sources.yaml"
    result = read_data_sources(str(real))
    assert "error" not in result
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
