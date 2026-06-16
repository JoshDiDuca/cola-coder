"""Tests for the memory-workbench UI helpers in ``cola_coder.ui.memory_ops_view``.

Covers ``memory_export``, ``memory_add``, ``memory_search`` and ``memory_compact``
end-to-end against a REAL :class:`cola_coder.memory.manager.MemoryManager` store.
Every test uses a fresh ``tmp_path`` as ``project_root`` so the markdown store is
created under ``<tmp_path>/.cola/memory/`` and the real repo store is never touched.

No network, GPU, or model is involved — these are pure markdown file I/O + CPU
TF-IDF helpers, exactly as documented in the module docstring.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.memory_ops_view import (
    memory_add,
    memory_compact,
    memory_export,
    memory_search,
)


def _root(tmp_path: Path) -> str:
    """Return the ``tmp_path`` as the ``str`` ``project_root`` the helpers expect."""
    return str(tmp_path)


class TestMemoryExport:
    """``memory_export`` over fresh and populated stores."""

    def test_uninitialised_store_is_empty(self, tmp_path: Path) -> None:
        """A never-initialised tmp_path yields ``initialized=False`` and no files."""
        result = memory_export(_root(tmp_path))

        assert result == {"initialized": False, "files": []}

    def test_export_after_add_reports_files_with_content(self, tmp_path: Path) -> None:
        """After an add, export reports ``initialized=True`` with the entry text."""
        root = _root(tmp_path)
        token = "ExportableTokenABC"
        memory_add(root, "pattern", f"Always use {token} for caching", "example body")

        result = memory_export(root)

        assert result["initialized"] is True
        files = result["files"]
        assert isinstance(files, list) and files, "export should list memory files"
        # At least one file's content must contain the added text.
        assert any(token in file["content"] for file in files)
        # Every file entry has the documented schema keys.
        for file in files:
            assert set(file) == {
                "type",
                "name",
                "content",
                "truncated",
                "entry_count",
            }
            assert isinstance(file["truncated"], bool)
            assert isinstance(file["entry_count"], int)


class TestMemoryAdd:
    """``memory_add`` auto-init, per-kind behaviour, and validation."""

    def test_add_auto_initialises_and_returns_stats(self, tmp_path: Path) -> None:
        """A first add on a fresh store auto-initialises and returns live stats."""
        root = _root(tmp_path)
        stats = memory_add(root, "pattern", "Prefer composition over inheritance", "why")

        assert "error" not in stats
        # Refreshed memory_stats schema keys must all be present.
        assert set(stats) >= {
            "total_entries",
            "pinned",
            "types",
            "size_bytes",
            "oldest_at",
            "newest_at",
            "recent_sample",
        }
        assert isinstance(stats["total_entries"], int)
        assert stats["total_entries"] >= 1
        # The store should now be initialised and visible via export.
        assert memory_export(root)["initialized"] is True

    def test_each_kind_adds_an_entry(self, tmp_path: Path) -> None:
        """pattern/error/decision/domain/session each grow total_entries + types."""
        root = _root(tmp_path)
        kind_to_type = {
            "pattern": "patterns",
            "error": "errors",
            "decision": "decisions",
            "domain": "domain_knowledge",
            "session": "session_log",
        }

        previous_total = 0
        for kind, type_key in kind_to_type.items():
            stats = memory_add(
                root,
                kind,
                f"distinctive {kind} primary content for testing",
                f"distinctive {kind} secondary content for testing",
            )
            assert "error" not in stats, f"kind {kind!r} unexpectedly errored: {stats}"
            assert stats["total_entries"] > previous_total, (
                f"adding kind {kind!r} did not grow total_entries"
            )
            previous_total = stats["total_entries"]
            # The file/type for this kind must now be reflected in the stats.
            assert type_key in stats["types"], (
                f"type {type_key!r} missing from stats.types after adding {kind!r}"
            )

        # Final export must surface a file for every theme that was written to.
        exported_types = {file["type"] for file in memory_export(root)["files"]}
        assert set(kind_to_type.values()) <= exported_types

    def test_unknown_kind_returns_error(self, tmp_path: Path) -> None:
        """An unrecognised ``kind`` yields ``{"error": ...}`` and writes nothing."""
        root = _root(tmp_path)
        result = memory_add(root, "bogus_kind", "some content")

        assert "error" in result
        # Nothing should have been written — the store stays uninitialised.
        assert memory_export(root) == {"initialized": False, "files": []}

    def test_empty_and_whitespace_primary_add_nothing(self, tmp_path: Path) -> None:
        """Empty / whitespace-only primary errors and leaves total_entries unchanged."""
        root = _root(tmp_path)
        # Seed a real entry first so we have a baseline count.
        baseline = memory_add(root, "pattern", "A real seeded pattern entry", "body")
        count = baseline["total_entries"]
        assert count >= 1

        for bad_primary in ("", "   ", "\n\t  "):
            result = memory_add(root, "pattern", bad_primary, "irrelevant")
            assert "error" in result, f"primary {bad_primary!r} should be rejected"

        # The count must be exactly unchanged after the invalid attempts.
        after = memory_add(root, "pattern", "Another valid entry to re-read stats", "")
        # The only successful add since baseline is this one, so +1 exactly.
        assert after["total_entries"] == count + 1


class TestMemorySearch:
    """``memory_search`` TF-IDF retrieval and validation."""

    def test_search_finds_added_entry(self, tmp_path: Path) -> None:
        """Searching a distinctive token returns a hit referencing that token."""
        root = _root(tmp_path)
        token = "ScoreMapperXYZ"
        # Body must clear chunk_min_length (20 chars) — keep it comfortably long.
        memory_add(
            root,
            "pattern",
            f"Use {token} to map error counts to normalised scores",
            f"{token} replaces inline score-mapping loops everywhere in the codebase",
        )

        result = memory_search(root, token)

        assert "error" not in result
        assert result["query"] == token
        results = result["results"]
        assert isinstance(results, list) and results, "expected at least one hit"
        # Every result follows the documented schema.
        for hit in results:
            assert set(hit) == {
                "content",
                "source_file",
                "section",
                "relevance_score",
            }
            assert isinstance(hit["relevance_score"], float)
            assert isinstance(hit["content"], str)
            assert isinstance(hit["source_file"], str)
            assert isinstance(hit["section"], str)
        # At least one hit's content references the distinctive token.
        assert any(token in hit["content"] for hit in results)

    def test_empty_query_returns_error(self, tmp_path: Path) -> None:
        """Empty / whitespace-only queries return ``{"error": ...}``."""
        root = _root(tmp_path)
        memory_add(root, "pattern", "Some content so the store exists", "body")

        for bad_query in ("", "   ", "\n\t"):
            result = memory_search(root, bad_query)
            assert "error" in result, f"query {bad_query!r} should be rejected"

    def test_search_uninitialised_store_is_defensive(self, tmp_path: Path) -> None:
        """Searching a fresh store returns an empty result list, not an error."""
        result = memory_search(_root(tmp_path), "anything")

        assert "error" not in result
        assert result == {"query": "anything", "results": []}


class TestMemoryCompact:
    """``memory_compact`` duplicate removal and shape guarantees."""

    def test_compact_uninitialised_returns_error(self, tmp_path: Path) -> None:
        """Compacting a never-initialised store returns ``{"error": ...}``."""
        result = memory_compact(_root(tmp_path))

        assert "error" in result

    def test_compact_removes_duplicates(self, tmp_path: Path) -> None:
        """Adding the SAME pattern twice then compacting removes >= 1 duplicate."""
        root = _root(tmp_path)
        primary = "Duplicate pattern entry used to exercise compaction"
        secondary = "identical example body for both adds"
        memory_add(root, "pattern", primary, secondary)
        memory_add(root, "pattern", primary, secondary)

        result = memory_compact(root)

        # Shape guarantees (hold regardless of dedup semantics).
        assert "error" not in result
        assert isinstance(result["removed_total"], int)
        assert isinstance(result["removed"], list)
        for item in result["removed"]:
            assert set(item) == {"name", "removed"}
            assert isinstance(item["name"], str)
            assert isinstance(item["removed"], int)

        # The two adds differ only by their _Added timestamp, which compaction
        # normalises away — so at least one duplicate must be removed.
        assert result["removed_total"] >= 1

    def test_compact_clean_store_does_not_raise(self, tmp_path: Path) -> None:
        """Compacting a store with no duplicates is defensive and well-shaped."""
        root = _root(tmp_path)
        memory_add(root, "pattern", "A single unique pattern entry", "body")

        result = memory_compact(root)

        assert "error" not in result
        assert isinstance(result["removed_total"], int)
        assert isinstance(result["removed"], list)
