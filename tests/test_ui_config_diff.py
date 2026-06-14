"""Hermetic tests for ui.config_diff.compare_configs."""

from __future__ import annotations

import textwrap

from cola_coder.ui.config_diff import _flatten, compare_configs


def _write(path, text: str) -> str:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


def _make_pair(tmp_path):
    """Two configs: shared keys (some differing), only-A, only-B, a nested list."""
    a = _write(
        tmp_path / "a.yaml",
        """
        model:
          dim: 512
          n_layers: 8
          only_a_key: 1
        training:
          batch_size: 4
          languages: [python, typescript]
        shared_scalar: keep
        """,
    )
    b = _write(
        tmp_path / "b.yaml",
        """
        model:
          dim: 1280
          n_layers: 8
          only_b_key: 2
        training:
          batch_size: 8
          languages: [typescript]
        shared_scalar: keep
        """,
    )
    return a, b


def test_changed_keys_and_values(tmp_path):
    a, b = _make_pair(tmp_path)
    result = compare_configs(a, b)
    changed = result["changed"]
    keys = [c["key"] for c in changed]
    assert keys == ["model.dim", "training.batch_size", "training.languages"]
    by_key = {c["key"]: c for c in changed}
    assert by_key["model.dim"]["a"] == 512
    assert by_key["model.dim"]["b"] == 1280
    assert by_key["training.batch_size"]["a"] == 4
    assert by_key["training.batch_size"]["b"] == 8


def test_nested_list_compared_as_whole(tmp_path):
    a, b = _make_pair(tmp_path)
    result = compare_configs(a, b)
    by_key = {c["key"]: c for c in result["changed"]}
    assert by_key["training.languages"]["a"] == ["python", "typescript"]
    assert by_key["training.languages"]["b"] == ["typescript"]


def test_only_a_and_only_b(tmp_path):
    a, b = _make_pair(tmp_path)
    result = compare_configs(a, b)
    assert result["only_a"] == ["model.only_a_key"]
    assert result["only_b"] == ["model.only_b_key"]


def test_changed_sorted_by_key(tmp_path):
    a, b = _make_pair(tmp_path)
    result = compare_configs(a, b)
    keys = [c["key"] for c in result["changed"]]
    assert keys == sorted(keys)


def test_only_lists_sorted(tmp_path):
    a = _write(
        tmp_path / "a.yaml",
        """
        z: 1
        a: 1
        m: 1
        """,
    )
    b = _write(tmp_path / "b.yaml", "shared: 1\n")
    result = compare_configs(a, b)
    assert result["only_a"] == ["a", "m", "z"]
    assert result["only_a"] == sorted(result["only_a"])


def test_paths_and_parsed_echoed(tmp_path):
    a, b = _make_pair(tmp_path)
    result = compare_configs(a, b)
    assert result["a"]["path"] == a
    assert result["b"]["path"] == b
    assert result["a"]["parsed"]["model"]["dim"] == 512
    assert result["b"]["parsed"]["model"]["dim"] == 1280


def test_identical_configs_empty_diff(tmp_path):
    a, b = _make_pair(tmp_path)
    # compare A with itself
    result = compare_configs(a, a)
    assert result["changed"] == []
    assert result["only_a"] == []
    assert result["only_b"] == []


def test_error_passthrough_missing_a(tmp_path):
    b = _write(tmp_path / "b.yaml", "x: 1\n")
    missing = str(tmp_path / "nope.yaml")
    result = compare_configs(missing, b)
    assert "error" in result
    assert "a:" in result["error"]
    assert "error" in result["a"]
    assert "error" not in result["b"]
    # diff keys must NOT be present on the error path
    assert "changed" not in result


def test_error_passthrough_missing_b(tmp_path):
    a = _write(tmp_path / "a.yaml", "x: 1\n")
    missing = str(tmp_path / "nope.yaml")
    result = compare_configs(a, missing)
    assert "error" in result
    assert "b:" in result["error"]
    assert "error" in result["b"]
    assert "error" not in result["a"]


def test_error_passthrough_both_missing(tmp_path):
    miss_a = str(tmp_path / "a.yaml")
    miss_b = str(tmp_path / "b.yaml")
    result = compare_configs(miss_a, miss_b)
    assert "error" in result
    assert "a:" in result["error"] and "b:" in result["error"]


def test_flatten_dotted_keys():
    nested = {"model": {"dim": 1, "rope": {"theta": 10000}}, "top": 5, "lst": [1, 2]}
    flat = _flatten(nested)
    assert flat == {"model.dim": 1, "model.rope.theta": 10000, "top": 5, "lst": [1, 2]}


def test_flatten_non_dict_returns_empty():
    assert _flatten(None) == {}
    assert _flatten([1, 2, 3]) == {}
    assert _flatten("scalar") == {}


def test_never_raises_on_non_dict_parsed(tmp_path):
    # YAML that parses to a list, not a dict — flatten yields {} so no crash.
    a = _write(tmp_path / "a.yaml", "- 1\n- 2\n")
    b = _write(tmp_path / "b.yaml", "model:\n  dim: 1\n")
    result = compare_configs(a, b)
    assert result["changed"] == []
    assert result["only_a"] == []
    assert result["only_b"] == ["model.dim"]
