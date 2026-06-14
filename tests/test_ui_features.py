"""Tests for the UI feature-toggle browsing helper (``cola_coder.ui.features``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cola_coder.ui.features import list_features

# Repo root: tests/ -> repo root is one level up.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_FEATURES = _REPO_ROOT / "configs" / "features.yaml"


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "features.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_flat_bool_counts(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "features:\n  a: true\n  b: false\n  c: true\n",
    )
    result = list_features(path)
    assert "error" not in result
    assert result["total"] == 3
    assert result["enabled"] == 2
    assert result["path"] == path


def test_flat_bool_single_all_group(tmp_path: Path) -> None:
    path = _write(tmp_path, "features:\n  a: true\n  b: false\n")
    result = list_features(path)
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["category"] == "All"
    keys = [f["key"] for f in group["features"]]
    assert keys == ["a", "b"]


def test_flat_bool_value_equals_enabled(tmp_path: Path) -> None:
    path = _write(tmp_path, "features:\n  a: true\n  b: false\n")
    feats = {f["key"]: f for f in list_features(path)["groups"][0]["features"]}
    assert feats["a"]["enabled"] is True
    assert feats["a"]["value"] is True
    assert feats["b"]["enabled"] is False
    assert feats["b"]["value"] is False


def test_nested_dict_enabled_and_value_passthrough(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "features:\n"
        "  a:\n"
        "    enabled: true\n"
        "    note: hello\n"
        "  b:\n"
        "    enabled: false\n"
        "    threshold: 5\n",
    )
    result = list_features(path)
    assert result["total"] == 2
    assert result["enabled"] == 1
    feats = {f["key"]: f for f in result["groups"][0]["features"]}
    assert feats["a"]["enabled"] is True
    assert feats["a"]["value"] == {"enabled": True, "note": "hello"}
    assert feats["b"]["enabled"] is False
    assert feats["b"]["value"] == {"enabled": False, "threshold": 5}


def test_nested_category_grouping(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "features:\n"
        "  a:\n"
        "    enabled: true\n"
        "    category: Training\n"
        "  b:\n"
        "    enabled: false\n"
        "    category: Generation\n"
        "  c:\n"
        "    enabled: true\n"
        "    category: Training\n",
    )
    result = list_features(path)
    cats = {g["category"]: g for g in result["groups"]}
    assert set(cats) == {"Training", "Generation"}
    assert [f["key"] for f in cats["Training"]["features"]] == ["a", "c"]
    assert [f["key"] for f in cats["Generation"]["features"]] == ["b"]
    assert result["total"] == 3
    assert result["enabled"] == 2


def test_mixed_flat_and_nested(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "features:\n"
        "  flat_on: true\n"
        "  nested:\n"
        "    enabled: true\n"
        "    category: X\n",
    )
    result = list_features(path)
    assert result["total"] == 2
    assert result["enabled"] == 2
    cats = {g["category"]: [f["key"] for f in g["features"]] for g in result["groups"]}
    assert cats["All"] == ["flat_on"]
    assert cats["X"] == ["nested"]


def test_non_bool_scalar_truthiness(tmp_path: Path) -> None:
    path = _write(tmp_path, "features:\n  num: 3\n  zero: 0\n  empty: ''\n")
    feats = {f["key"]: f for f in list_features(path)["groups"][0]["features"]}
    assert feats["num"]["enabled"] is True
    assert feats["num"]["value"] == 3
    assert feats["zero"]["enabled"] is False
    assert feats["empty"]["enabled"] is False


def test_missing_file_returns_error() -> None:
    result = list_features("does/not/exist/features.yaml")
    assert "error" in result
    assert "groups" not in result


def test_garbage_yaml_returns_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "features: [unclosed\n  : : :\n")
    result = list_features(path)
    assert "error" in result


def test_empty_file_returns_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    result = list_features(path)
    assert "error" in result


def test_no_features_key_falls_back_to_top_level(tmp_path: Path) -> None:
    path = _write(tmp_path, "a: true\nb: false\n")
    result = list_features(path)
    assert "error" not in result
    assert result["total"] == 2
    assert result["enabled"] == 1


def test_real_features_yaml_schema_guard() -> None:
    assert _REAL_FEATURES.is_file(), f"missing real config: {_REAL_FEATURES}"
    result = list_features(str(_REAL_FEATURES))
    assert "error" not in result, result
    assert result["total"] > 0
    assert result["enabled"] <= result["total"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
