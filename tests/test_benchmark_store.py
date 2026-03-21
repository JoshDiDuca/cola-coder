"""Tests for BenchmarkStore (features/benchmark_store.py)."""

from __future__ import annotations

import time

import pytest

from cola_coder.features.benchmark_store import BenchmarkStore, CompareResult


@pytest.fixture()
def store(tmp_path) -> BenchmarkStore:
    return BenchmarkStore(base_dir=tmp_path / "benchmarks")


class TestIsEnabled:
    def test_feature_enabled(self):
        from cola_coder.features.benchmark_store import FEATURE_ENABLED, is_enabled

        assert FEATURE_ENABLED is True
        assert is_enabled() is True


class TestSaveLoad:
    def test_save_returns_path(self, store):
        path = store.save({"score": 0.5}, "run1")
        assert path.exists()
        assert path.suffix == ".json"

    def test_load_returns_dict(self, store):
        store.save({"score": 0.75}, "run1")
        data = store.load("run1")
        assert data["score"] == pytest.approx(0.75)

    def test_load_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_save_overwrites(self, store):
        store.save({"v": 1}, "run1")
        store.save({"v": 2}, "run1")
        data = store.load("run1")
        assert data["v"] == 2

    def test_meta_added(self, store):
        store.save({"x": 1}, "test_run")
        data = store.load("test_run")
        assert "_meta" in data
        assert data["_meta"]["name"] == "test_run"

    def test_save_creates_dir(self, tmp_path):
        deep_store = BenchmarkStore(base_dir=tmp_path / "a" / "b" / "c")
        deep_store.save({"x": 1}, "test")
        assert (tmp_path / "a" / "b" / "c" / "test.json").exists()


class TestList:
    def test_list_empty(self, store):
        assert store.list() == []

    def test_list_returns_entries(self, store):
        store.save({"a": 1}, "run_a")
        store.save({"b": 2}, "run_b")
        entries = store.list()
        assert len(entries) == 2
        names = {e.name for e in entries}
        assert "run_a" in names
        assert "run_b" in names

    def test_list_sorted_newest_first(self, store):
        store.save({"a": 1}, "first")
        time.sleep(0.01)
        store.save({"b": 2}, "second")
        entries = store.list()
        assert entries[0].name == "second"  # newest first

    def test_entry_has_keys(self, store):
        store.save({"pass_rate": 0.8, "model": "tiny"}, "run1")
        entries = store.list()
        assert "pass_rate" in entries[0].keys
        assert "model" in entries[0].keys


class TestDelete:
    def test_delete_existing(self, store):
        store.save({"x": 1}, "to_delete")
        assert store.delete("to_delete") is True
        assert not store.exists("to_delete")

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_exists(self, store):
        assert store.exists("nonexistent") is False
        store.save({}, "thing")
        assert store.exists("thing") is True


class TestCompare:
    def test_compare_basic(self, store):
        store.save({"pass_rate": 0.70, "model": "tiny-v1"}, "a")
        store.save({"pass_rate": 0.80, "model": "tiny-v2"}, "b")
        result = store.compare("a", "b")
        assert isinstance(result, CompareResult)
        assert result.name_a == "a"
        assert result.name_b == "b"

    def test_compare_numeric_delta(self, store):
        store.save({"score": 0.5}, "v1")
        store.save({"score": 0.7}, "v2")
        result = store.compare("v1", "v2")
        assert "score" in result.differences
        assert result.differences["score"]["delta"] == pytest.approx(0.2)

    def test_compare_no_diff(self, store):
        store.save({"score": 0.5}, "v1")
        store.save({"score": 0.5}, "v2")
        result = store.compare("v1", "v2")
        assert "score" not in result.differences

    def test_compare_keys_a_only(self, store):
        store.save({"extra": 1, "shared": 2}, "a")
        store.save({"shared": 3}, "b")
        result = store.compare("a", "b")
        assert "extra" in result.keys_a_only

    def test_compare_summary_string(self, store):
        store.save({"score": 0.5}, "v1")
        store.save({"score": 0.7}, "v2")
        result = store.compare("v1", "v2")
        s = result.summary()
        assert "v1" in s and "v2" in s

    def test_compare_missing_raises(self, store):
        store.save({}, "existing")
        with pytest.raises(FileNotFoundError):
            store.compare("existing", "missing")


class TestDataclassSave:
    def test_save_dataclass(self, store):
        from dataclasses import dataclass

        @dataclass
        class Result:
            pass_rate: float
            model: str

        r = Result(pass_rate=0.85, model="small")
        store.save(r, "dc_result")
        data = store.load("dc_result")
        assert data["pass_rate"] == pytest.approx(0.85)
        assert data["model"] == "small"
