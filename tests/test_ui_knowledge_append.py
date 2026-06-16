"""Tests for the UI knowledge-base append helpers.

Covers the two append-only writers used by the local UI to file knowledge into
the repo's living markdown logs:

* :func:`cola_coder.ui.research_log_view.research_log_append` — appends a dated
  ``## <date> — <title>`` section to ``<root>/docs/research-log.md``.
* :func:`cola_coder.ui.backlog_view.backlog_append` — appends a
  ``- **ID** [cat, sev] `status` (date) — description`` bullet under a single
  ``## Filed from the UI`` section to ``<root>/ai_backlog.md``.

Every test uses ``tmp_path`` as ``root`` so nothing touches the real repo files.
No network, GPU, or model access is involved.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.backlog_view import backlog, backlog_append
from cola_coder.ui.research_log_view import research_log, research_log_append

_MANUAL_SECTION = "## Filed from the UI"


def _research_log_path(root: Path) -> Path:
    """Return the research-log markdown path under ``root``."""
    return root / "docs" / "research-log.md"


def _backlog_path(root: Path) -> Path:
    """Return the backlog markdown path under ``root``."""
    return root / "ai_backlog.md"


class TestResearchLogAppend:
    """Behaviour of :func:`research_log_append`."""

    def test_fresh_append_creates_file_and_entry(self, tmp_path: Path) -> None:
        """A first append creates the file and yields a single matching entry."""
        result = research_log_append(
            str(tmp_path),
            title="Speculative decoding sweep",
            body="Measured 1.8x acceptance on the TS corpus.",
            date="2026-06-16",
        )

        assert "error" not in result
        assert result["count"] == 1
        entry = result["entries"][0]
        assert entry["title"] == "Speculative decoding sweep"
        assert entry["date"] == "2026-06-16"

        log_path = _research_log_path(tmp_path)
        assert log_path.is_file()

    def test_second_append_is_append_only(self, tmp_path: Path) -> None:
        """A second (older-dated) append keeps both entries without clobbering."""
        first = research_log_append(
            str(tmp_path),
            title="First finding",
            body="Original body text that must survive.",
            date="2026-06-16",
        )
        assert first["count"] == 1

        second = research_log_append(
            str(tmp_path),
            title="Second finding",
            body="A later-added but older-dated entry.",
            date="2026-01-01",
        )

        assert "error" not in second
        assert second["count"] == 2

        titles = {entry["title"] for entry in second["entries"]}
        assert titles == {"First finding", "Second finding"}

        raw = _research_log_path(tmp_path).read_text(encoding="utf-8")
        assert "First finding" in raw
        assert "Second finding" in raw
        # The first entry's body text is still present (no clobber).
        assert "Original body text that must survive." in raw

    def test_empty_title_is_rejected_and_nothing_written(self, tmp_path: Path) -> None:
        """An empty/whitespace title returns an error and writes nothing."""
        result = research_log_append(str(tmp_path), title="   ", body="some body")

        assert "error" in result
        assert not _research_log_path(tmp_path).exists()

    def test_empty_body_is_rejected(self, tmp_path: Path) -> None:
        """An empty/whitespace body returns an error and writes nothing."""
        result = research_log_append(str(tmp_path), title="A title", body="  \n  ")

        assert "error" in result
        assert not _research_log_path(tmp_path).exists()

    def test_non_iso_date_is_rejected(self, tmp_path: Path) -> None:
        """A non-ISO explicit date returns an error and writes nothing."""
        for bad_date in ("2026/06/16", "yesterday"):
            result = research_log_append(
                str(tmp_path),
                title="Title",
                body="Body",
                date=bad_date,
            )
            assert "error" in result
            assert not _research_log_path(tmp_path).exists()

    def test_body_appears_verbatim_in_raw_file(self, tmp_path: Path) -> None:
        """The supplied body text is written verbatim into the log file."""
        body = "Verbatim line one.\nVerbatim line two with `code`."
        result = research_log_append(
            str(tmp_path),
            title="Verbatim check",
            body=body,
            date="2026-06-16",
        )

        assert "error" not in result
        raw = _research_log_path(tmp_path).read_text(encoding="utf-8")
        assert body in raw

    def test_no_explicit_date_defaults_to_today(self, tmp_path: Path) -> None:
        """Omitting ``date`` succeeds and produces an ISO-dated entry."""
        result = research_log_append(
            str(tmp_path),
            title="Default date",
            body="Body for default date.",
        )

        assert "error" not in result
        assert result["count"] == 1
        # research_log re-parsed the file: the surviving entry round-trips.
        reparsed = research_log(str(tmp_path))
        assert reparsed["count"] == 1


class TestBacklogAppend:
    """Behaviour of :func:`backlog_append`."""

    def test_fresh_append_creates_file_and_section(self, tmp_path: Path) -> None:
        """A first append creates the file with the manual section and item."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-100",
            category="ui",
            description="Add a knowledge-filing panel.",
            severity="medium",
            status="open",
            date="2026-06-16",
        )

        assert "error" not in result
        assert result["count"] >= 1
        assert result["open_count"] >= 1

        ids = {item["id"] for item in result["items"]}
        assert "UI-100" in ids

        item = next(item for item in result["items"] if item["id"] == "UI-100")
        assert item["category"] == "ui"
        assert item["status"] == "open"

        raw = _backlog_path(tmp_path).read_text(encoding="utf-8")
        assert _MANUAL_SECTION in raw

    def test_second_append_shares_single_section(self, tmp_path: Path) -> None:
        """Two appends sit under one ``## Filed from the UI`` section."""
        backlog_append(
            str(tmp_path),
            item_id="UI-100",
            category="ui",
            description="First filed item.",
            date="2026-06-16",
        )
        result = backlog_append(
            str(tmp_path),
            item_id="OPS-200",
            category="ops",
            description="Second filed item.",
            date="2026-06-16",
        )

        assert "error" not in result
        assert result["count"] == 2

        raw = _backlog_path(tmp_path).read_text(encoding="utf-8")
        assert raw.count(_MANUAL_SECTION) == 1

        ids = {item["id"] for item in result["items"]}
        assert {"UI-100", "OPS-200"} <= ids

    def test_empty_id_is_rejected(self, tmp_path: Path) -> None:
        """An empty id returns an error and appends nothing."""
        result = backlog_append(
            str(tmp_path),
            item_id="  ",
            category="ui",
            description="desc",
        )
        assert "error" in result
        assert not _backlog_path(tmp_path).exists()

    def test_empty_category_is_rejected(self, tmp_path: Path) -> None:
        """An empty category returns an error and appends nothing."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-1",
            category="   ",
            description="desc",
        )
        assert "error" in result
        assert not _backlog_path(tmp_path).exists()

    def test_empty_description_is_rejected(self, tmp_path: Path) -> None:
        """An empty description returns an error and appends nothing."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-1",
            category="ui",
            description="   ",
        )
        assert "error" in result
        assert not _backlog_path(tmp_path).exists()

    def test_unknown_status_is_rejected(self, tmp_path: Path) -> None:
        """A status outside the known set returns an error and appends nothing."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-1",
            category="ui",
            description="desc",
            status="wip",
        )
        assert "error" in result
        assert not _backlog_path(tmp_path).exists()

    def test_non_iso_date_is_rejected(self, tmp_path: Path) -> None:
        """A non-ISO explicit date returns an error and appends nothing."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-1",
            category="ui",
            description="desc",
            date="2026/06/16",
        )
        assert "error" in result
        assert not _backlog_path(tmp_path).exists()

    def test_done_status_increments_done_count(self, tmp_path: Path) -> None:
        """A ``done`` item is reflected in ``done_count`` of the refreshed view."""
        result = backlog_append(
            str(tmp_path),
            item_id="UI-DONE",
            category="ui",
            description="Already shipped.",
            status="done",
            date="2026-06-16",
        )

        assert "error" not in result
        assert result["done_count"] >= 1
        item = next(item for item in result["items"] if item["id"] == "UI-DONE")
        assert item["status"] == "done"

    def test_preexisting_content_is_preserved(self, tmp_path: Path) -> None:
        """Pre-existing backlog items survive a subsequent append."""
        existing = (
            "# AI Backlog\n\n"
            "## Tracked\n\n"
            "- **BUG-001** [bug, high] `open` (2026-05-01) — Existing tracked item.\n"
        )
        path = _backlog_path(tmp_path)
        path.write_text(existing, encoding="utf-8")

        # Sanity: the original item parses before any append.
        before = backlog(str(tmp_path))
        before_ids = {item["id"] for item in before["items"]}
        assert "BUG-001" in before_ids

        result = backlog_append(
            str(tmp_path),
            item_id="UI-NEW",
            category="ui",
            description="Newly filed item.",
            date="2026-06-16",
        )

        assert "error" not in result
        ids = {item["id"] for item in result["items"]}
        assert "BUG-001" in ids
        assert "UI-NEW" in ids

        raw = path.read_text(encoding="utf-8")
        assert "Existing tracked item." in raw
        assert _MANUAL_SECTION in raw
