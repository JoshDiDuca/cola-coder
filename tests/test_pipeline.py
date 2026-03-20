"""Tests for the extensible data pipeline system.

Tests:
- DataRecord creation
- Registry decorators (register + lookup)
- LocalFileSource streaming
- LengthFilter
- Pipeline composition (source + filter + transform)
- MixedSource weighted sampling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cola_coder.data.pipeline import (
    DataPipeline,
    DataRecord,
    DataSource,
    FilterPlugin,
    PipelineStats,
    Transform,
)
from cola_coder.data.registry import (
    _FILTER_REGISTRY,
    _SOURCE_REGISTRY,
    _TRANSFORM_REGISTRY,
    get_filter,
    get_source,
    get_transform,
    register_filter,
    register_source,
    register_transform,
)


# ---------------------------------------------------------------------------
# DataRecord tests
# ---------------------------------------------------------------------------

class TestDataRecord:
    def test_create_basic(self):
        record = DataRecord(content="print('hello')")
        assert record.content == "print('hello')"
        assert record.metadata == {}

    def test_create_with_metadata(self):
        record = DataRecord(
            content="const x = 1;",
            metadata={"source": "local", "language": "typescript"},
        )
        assert record.content == "const x = 1;"
        assert record.metadata["source"] == "local"
        assert record.metadata["language"] == "typescript"

    def test_metadata_default_factory_isolation(self):
        """Each DataRecord should have its own metadata dict."""
        r1 = DataRecord(content="a")
        r2 = DataRecord(content="b")
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_source_decorator(self):
        """Import sources package to trigger registrations, then verify."""
        import cola_coder.data.sources  # noqa: F401

        assert "huggingface" in _SOURCE_REGISTRY
        assert "local" in _SOURCE_REGISTRY
        assert "mixed" in _SOURCE_REGISTRY

    def test_register_filter_decorator(self):
        import cola_coder.data.filters  # noqa: F401

        assert "quality" in _FILTER_REGISTRY
        assert "length" in _FILTER_REGISTRY

    def test_register_transform_decorator(self):
        import cola_coder.data.transforms  # noqa: F401

        assert "normalize_whitespace" in _TRANSFORM_REGISTRY
        assert "add_metadata" in _TRANSFORM_REGISTRY

    def test_get_source_returns_class(self):
        import cola_coder.data.sources  # noqa: F401
        from cola_coder.data.sources.local import LocalFileSource

        cls = get_source("local")
        assert cls is LocalFileSource

    def test_get_filter_returns_class(self):
        import cola_coder.data.filters  # noqa: F401
        from cola_coder.data.filters.length import LengthFilter

        cls = get_filter("length")
        assert cls is LengthFilter

    def test_get_transform_returns_class(self):
        import cola_coder.data.transforms  # noqa: F401
        from cola_coder.data.transforms.whitespace import NormalizeWhitespace

        cls = get_transform("normalize_whitespace")
        assert cls is NormalizeWhitespace

    def test_get_unknown_source_raises(self):
        with pytest.raises(KeyError, match="Unknown data source"):
            get_source("nonexistent_source_xyz")

    def test_get_unknown_filter_raises(self):
        with pytest.raises(KeyError, match="Unknown filter"):
            get_filter("nonexistent_filter_xyz")

    def test_get_unknown_transform_raises(self):
        with pytest.raises(KeyError, match="Unknown transform"):
            get_transform("nonexistent_transform_xyz")


# ---------------------------------------------------------------------------
# LocalFileSource tests
# ---------------------------------------------------------------------------

class TestLocalFileSource:
    def test_stream_from_directory(self, tmp_path: Path):
        """LocalFileSource should yield DataRecords from files on disk."""
        # Create sample files
        (tmp_path / "hello.py").write_text("def hello():\n    print('hello world')\n")
        (tmp_path / "main.ts").write_text("const x: number = 42;\nconsole.log(x);\n")
        (tmp_path / "readme.md").write_text("# Just a readme\n")

        from cola_coder.data.sources.local import LocalFileSource

        source = LocalFileSource(paths=[str(tmp_path)])
        records = list(source.stream())

        assert len(records) == 3
        assert all(isinstance(r, DataRecord) for r in records)
        assert all(r.metadata["source"] == "local" for r in records)

    def test_filter_by_extension(self, tmp_path: Path):
        """Should only yield files matching the specified extensions."""
        (tmp_path / "hello.py").write_text("def hello():\n    print('hello world')\n")
        (tmp_path / "main.ts").write_text("const x: number = 42;\nconsole.log(x);\n")
        (tmp_path / "readme.md").write_text("# Just a readme with enough content here\n")

        from cola_coder.data.sources.local import LocalFileSource

        source = LocalFileSource(
            paths=[str(tmp_path)],
            extensions=[".py"],
        )
        records = list(source.stream())

        assert len(records) == 1
        assert "hello.py" in records[0].metadata["path"]

    def test_skips_tiny_files(self, tmp_path: Path):
        """Files with < 10 chars should be skipped."""
        (tmp_path / "tiny.py").write_text("x = 1")  # 5 chars
        (tmp_path / "ok.py").write_text("def hello():\n    return 42\n")

        from cola_coder.data.sources.local import LocalFileSource

        source = LocalFileSource(paths=[str(tmp_path)])
        records = list(source.stream())

        assert len(records) == 1
        assert "ok.py" in records[0].metadata["path"]

    def test_nonexistent_path_warns(self, tmp_path: Path, capsys):
        """Non-existent paths should warn, not crash."""
        from cola_coder.data.sources.local import LocalFileSource

        source = LocalFileSource(paths=[str(tmp_path / "nope")])
        records = list(source.stream())

        assert len(records) == 0
        captured = capsys.readouterr()
        assert "does not exist" in captured.out


# ---------------------------------------------------------------------------
# LengthFilter tests
# ---------------------------------------------------------------------------

class TestLengthFilter:
    def test_accepts_normal_length(self):
        from cola_coder.data.filters.length import LengthFilter

        f = LengthFilter(min_lines=3, max_lines=100)
        record = DataRecord(content="line1\nline2\nline3\nline4\n")
        keep, reason = f.check(record)
        assert keep is True

    def test_rejects_too_short(self):
        from cola_coder.data.filters.length import LengthFilter

        f = LengthFilter(min_lines=5, max_lines=100)
        record = DataRecord(content="line1\nline2\n")
        keep, reason = f.check(record)
        assert keep is False
        assert "too_short" in reason

    def test_rejects_too_long(self):
        from cola_coder.data.filters.length import LengthFilter

        f = LengthFilter(min_lines=1, max_lines=5)
        content = "\n".join(f"line{i}" for i in range(20))
        record = DataRecord(content=content)
        keep, reason = f.check(record)
        assert keep is False
        assert "too_long" in reason

    def test_setup_from_config(self):
        from cola_coder.data.filters.length import LengthFilter

        f = LengthFilter()
        f.setup({"min_lines": 10, "max_lines": 50})
        assert f._min_lines == 10
        assert f._max_lines == 50


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_converts_tabs(self):
        from cola_coder.data.transforms.whitespace import NormalizeWhitespace

        t = NormalizeWhitespace(tab_width=4)
        record = DataRecord(content="def f():\n\treturn 1\n")
        result = t.apply(record)
        assert "\t" not in result.content
        assert "    return 1" in result.content

    def test_removes_trailing_whitespace(self):
        from cola_coder.data.transforms.whitespace import NormalizeWhitespace

        t = NormalizeWhitespace()
        record = DataRecord(content="hello   \nworld  \n")
        result = t.apply(record)
        assert result.content == "hello\nworld\n"

    def test_collapses_blank_lines(self):
        from cola_coder.data.transforms.whitespace import NormalizeWhitespace

        t = NormalizeWhitespace()
        record = DataRecord(content="a\n\n\n\n\nb\n")
        result = t.apply(record)
        # Should have at most 2 consecutive blank lines (2 blanks = 3 newlines)
        # 4+ consecutive newlines means 3+ blank lines, which should not happen
        assert "\n\n\n\n" not in result.content
        assert "a\n\n\nb" in result.content


class TestAddMetadata:
    def test_adds_line_count(self):
        from cola_coder.data.transforms.metadata import AddMetadata

        t = AddMetadata()
        record = DataRecord(content="line1\nline2\nline3\n")
        result = t.apply(record)
        assert result.metadata["line_count"] == 4  # 3 newlines + 1

    def test_adds_char_count(self):
        from cola_coder.data.transforms.metadata import AddMetadata

        t = AddMetadata()
        record = DataRecord(content="hello")
        result = t.apply(record)
        assert result.metadata["char_count"] == 5

    def test_adds_content_hash(self):
        from cola_coder.data.transforms.metadata import AddMetadata

        t = AddMetadata()
        r1 = t.apply(DataRecord(content="hello"))
        r2 = t.apply(DataRecord(content="hello"))
        r3 = t.apply(DataRecord(content="world"))

        assert r1.metadata["content_hash"] == r2.metadata["content_hash"]
        assert r1.metadata["content_hash"] != r3.metadata["content_hash"]

    def test_detects_python(self):
        from cola_coder.data.transforms.metadata import AddMetadata

        t = AddMetadata()
        code = "def hello():\n    self.x = 1\n    return self.x\n"
        result = t.apply(DataRecord(content=code))
        assert result.metadata["estimated_language"] == "python"

    def test_does_not_overwrite_existing_language(self):
        from cola_coder.data.transforms.metadata import AddMetadata

        t = AddMetadata()
        record = DataRecord(
            content="def hello(): pass",
            metadata={"estimated_language": "custom"},
        )
        result = t.apply(record)
        assert result.metadata["estimated_language"] == "custom"


# ---------------------------------------------------------------------------
# Pipeline composition tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_simple_pipeline(self, tmp_path: Path):
        """Pipeline with LocalFileSource + LengthFilter should work end-to-end."""
        # Create files with varying lengths
        (tmp_path / "short.py").write_text("x = 1\n")  # 1 line — too short
        long_content = "\n".join(f"line_{i} = {i}" for i in range(20))
        (tmp_path / "good.py").write_text(long_content)
        (tmp_path / "also_good.py").write_text(
            "\n".join(f"def func_{i}(): pass" for i in range(15))
        )

        from cola_coder.data.filters.length import LengthFilter
        from cola_coder.data.sources.local import LocalFileSource

        pipeline = DataPipeline(
            sources=[LocalFileSource(paths=[str(tmp_path)], extensions=[".py"])],
            filters=[LengthFilter(min_lines=5, max_lines=1000)],
            transforms=[],
        )

        records = list(pipeline.stream())

        # short.py should be filtered out (1 line < min 5)
        assert len(records) == 2
        contents = [r.content for r in records]
        assert any("line_0 = 0" in c for c in contents)

    def test_pipeline_with_transforms(self, tmp_path: Path):
        """Transforms should be applied to records that pass filters."""
        content = "def hello():\n\treturn 'world'\t  \n"
        (tmp_path / "test.py").write_text(content)

        from cola_coder.data.sources.local import LocalFileSource
        from cola_coder.data.transforms.metadata import AddMetadata
        from cola_coder.data.transforms.whitespace import NormalizeWhitespace

        pipeline = DataPipeline(
            sources=[LocalFileSource(paths=[str(tmp_path)])],
            filters=[],
            transforms=[NormalizeWhitespace(), AddMetadata()],
        )

        records = list(pipeline.stream())
        assert len(records) == 1

        record = records[0]
        # Whitespace should be normalized
        assert "\t" not in record.content
        # Metadata should be added
        assert "line_count" in record.metadata
        assert "content_hash" in record.metadata

    def test_pipeline_stats(self, tmp_path: Path):
        """Pipeline should track stats for kept and rejected records."""
        # a.py needs >= 10 chars to pass LocalFileSource's minimum
        (tmp_path / "a.py").write_text("short_file_content\n")
        (tmp_path / "b.py").write_text("\n".join(f"line{i}" for i in range(20)))

        from cola_coder.data.filters.length import LengthFilter
        from cola_coder.data.sources.local import LocalFileSource

        pipeline = DataPipeline(
            sources=[LocalFileSource(paths=[str(tmp_path)])],
            filters=[LengthFilter(min_lines=5)],
        )

        records = list(pipeline.stream())
        assert pipeline.stats.kept == 1
        assert pipeline.stats.rejected == 1
        assert pipeline.stats.total == 2

    def test_content_stream(self, tmp_path: Path):
        """content_stream() should yield plain strings."""
        (tmp_path / "test.py").write_text("hello world code here!\n")

        from cola_coder.data.sources.local import LocalFileSource

        pipeline = DataPipeline(
            sources=[LocalFileSource(paths=[str(tmp_path)])],
        )

        contents = list(pipeline.content_stream())
        assert len(contents) == 1
        assert isinstance(contents[0], str)
        assert "hello world" in contents[0]


# ---------------------------------------------------------------------------
# MixedSource tests
# ---------------------------------------------------------------------------

class TestMixedSource:
    def _make_list_source(self, items: list[str], source_name: str) -> DataSource:
        """Helper: create a simple in-memory source."""
        from typing import Iterator

        class ListSource(DataSource):
            def __init__(self, items: list[str], name_str: str):
                self._items = items
                self._name = name_str

            def name(self) -> str:
                return self._name

            def stream(self) -> Iterator[DataRecord]:
                for item in self._items:
                    yield DataRecord(
                        content=item,
                        metadata={"source": self._name},
                    )

            def estimate_size(self) -> int | None:
                return len(self._items)

        return ListSource(items, source_name)

    def test_mixed_source_yields_from_both(self):
        """MixedSource should yield records from all sources."""
        from cola_coder.data.sources.mixed import MixedSource

        s1 = self._make_list_source(["a1", "a2", "a3"], "source_a")
        s2 = self._make_list_source(["b1", "b2", "b3"], "source_b")

        mixed = MixedSource(sources=[(s1, 0.5), (s2, 0.5)], seed=42)
        records = list(mixed.stream())

        assert len(records) == 6
        sources_seen = {r.metadata["source"] for r in records}
        assert sources_seen == {"source_a", "source_b"}

    def test_mixed_source_respects_weights(self):
        """With extreme weights, most records should come from the heavy source."""
        from cola_coder.data.sources.mixed import MixedSource

        # 100 items from heavy source, 100 from light source
        heavy_items = [f"heavy_{i}" for i in range(100)]
        light_items = [f"light_{i}" for i in range(100)]

        s1 = self._make_list_source(heavy_items, "heavy")
        s2 = self._make_list_source(light_items, "light")

        mixed = MixedSource(sources=[(s1, 0.9), (s2, 0.1)], seed=42)
        records = list(mixed.stream())

        # All 200 records should be yielded eventually
        assert len(records) == 200

        # Check ordering: among the first 50 records, most should be from heavy
        first_50_heavy = sum(
            1 for r in records[:50] if r.metadata["source"] == "heavy"
        )
        # With 90% weight, we expect ~45 out of 50 from heavy
        # Use a generous threshold to avoid flaky tests
        assert first_50_heavy >= 30, (
            f"Expected at least 30/50 from heavy source with 90% weight, "
            f"got {first_50_heavy}"
        )

    def test_mixed_source_handles_exhausted_source(self):
        """When one source runs out, should continue with remaining sources."""
        from cola_coder.data.sources.mixed import MixedSource

        s1 = self._make_list_source(["a1"], "short")
        s2 = self._make_list_source(["b1", "b2", "b3", "b4", "b5"], "long")

        mixed = MixedSource(sources=[(s1, 0.5), (s2, 0.5)], seed=42)
        records = list(mixed.stream())

        assert len(records) == 6  # 1 from short + 5 from long

    def test_mixed_source_estimate_size(self):
        from cola_coder.data.sources.mixed import MixedSource

        s1 = self._make_list_source(["a", "b"], "s1")
        s2 = self._make_list_source(["c"], "s2")

        mixed = MixedSource(sources=[(s1, 0.5), (s2, 0.5)])
        assert mixed.estimate_size() == 3

    def test_mixed_source_empty_raises(self):
        from cola_coder.data.sources.mixed import MixedSource

        with pytest.raises(ValueError, match="at least one source"):
            MixedSource(sources=[])


# ---------------------------------------------------------------------------
# PipelineStats tests
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_tracks_kept_and_rejected(self):
        stats = PipelineStats()
        stats.record_kept()
        stats.record_kept()
        stats.record_rejection("quality", "too_short")

        assert stats.total == 3
        assert stats.kept == 2
        assert stats.rejected == 1

    def test_summary_format(self):
        stats = PipelineStats()
        stats.record_kept()
        stats.record_rejection("quality", "too_short")

        summary = stats.summary()
        assert "Total:" in summary
        assert "Kept:" in summary
        assert "Rejected:" in summary
        assert "quality" in summary
