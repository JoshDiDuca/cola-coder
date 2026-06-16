"""Tests for the checkpoint-notes UI helpers (UI-100).

Covers ``checkpoint_notes`` / ``set_checkpoint_note`` / ``delete_checkpoint_note``
in :mod:`cola_coder.ui.checkpoint_notes_view`. These are pure JSON-sidecar
helpers: no model, no GPU, no network. Every test uses a fresh ``tmp_path`` as
``root`` so the sidecar lands at ``tmp_path/.cola/checkpoint_notes.json``.

A central invariant under test: the sidecar lives under ``.cola/`` and NOTHING
is ever written under a ``checkpoints/`` directory — the helpers must never
touch live checkpoint dirs.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.checkpoint_notes_view import (
    checkpoint_notes,
    delete_checkpoint_note,
    set_checkpoint_note,
)

_FIXED_NOW = "2026-06-16T07:10:00"


def _sidecar(root: Path) -> Path:
    """Return the expected sidecar path for ``root``."""
    return root / ".cola" / "checkpoint_notes.json"


def _assert_under_cola_not_checkpoints(root: Path) -> None:
    """Assert the sidecar exists under .cola/ and nothing is under checkpoints/."""
    assert _sidecar(root).is_file()
    assert not (root / "checkpoints").exists()


class TestCheckpointNotesRead:
    """Reading notes (the GET-style view)."""

    def test_fresh_root_returns_empty_notes(self, tmp_path: Path) -> None:
        """A fresh root with no sidecar yields an empty-but-valid view."""
        result = checkpoint_notes(str(tmp_path))
        assert result == {"notes": []}

    def test_fresh_root_never_creates_anything(self, tmp_path: Path) -> None:
        """Reading must not create the sidecar or any checkpoints/ dir."""
        checkpoint_notes(str(tmp_path))
        assert not _sidecar(tmp_path).exists()
        assert not (tmp_path / "checkpoints").exists()


class TestSetCheckpointNote:
    """Upserting notes."""

    def test_set_with_label_and_note(self, tmp_path: Path) -> None:
        """A valid set returns a one-entry view with the passed fields/stamp."""
        result = set_checkpoint_note(
            str(tmp_path),
            key="checkpoints/small/step_1000",
            label="best so far",
            note="before reasoning warmup",
            now=_FIXED_NOW,
        )
        assert "error" not in result
        notes = result["notes"]
        assert len(notes) == 1
        entry = notes[0]
        assert entry["key"] == "checkpoints/small/step_1000"
        assert entry["label"] == "best so far"
        assert entry["note"] == "before reasoning warmup"
        assert entry["updated_at"] == _FIXED_NOW
        _assert_under_cola_not_checkpoints(tmp_path)

    def test_set_label_only_is_valid(self, tmp_path: Path) -> None:
        """A label with an empty note is still a valid note."""
        result = set_checkpoint_note(str(tmp_path), key="k1", label="tagged", now=_FIXED_NOW)
        assert "error" not in result
        assert result["notes"][0]["label"] == "tagged"
        assert result["notes"][0]["note"] == ""

    def test_set_note_only_is_valid(self, tmp_path: Path) -> None:
        """A note with an empty label is still a valid note."""
        result = set_checkpoint_note(str(tmp_path), key="k1", note="some text", now=_FIXED_NOW)
        assert "error" not in result
        assert result["notes"][0]["note"] == "some text"
        assert result["notes"][0]["label"] == ""

    def test_second_key_preserves_first_and_sorts(self, tmp_path: Path) -> None:
        """Setting a second key keeps the first; view is sorted by key."""
        set_checkpoint_note(str(tmp_path), key="zeta", label="z", now=_FIXED_NOW)
        result = set_checkpoint_note(str(tmp_path), key="alpha", label="a", now=_FIXED_NOW)
        notes = result["notes"]
        assert len(notes) == 2
        assert [n["key"] for n in notes] == ["alpha", "zeta"]

    def test_set_existing_key_updates_in_place(self, tmp_path: Path) -> None:
        """Re-setting a key updates it (count stays 1, fields/stamp change)."""
        set_checkpoint_note(str(tmp_path), key="k1", label="old", now=_FIXED_NOW)
        new_now = "2026-06-17T09:00:00"
        result = set_checkpoint_note(str(tmp_path), key="k1", label="new", now=new_now)
        notes = result["notes"]
        assert len(notes) == 1
        assert notes[0]["label"] == "new"
        assert notes[0]["updated_at"] == new_now


class TestSetValidation:
    """Input validation and trimming/truncation on set."""

    def test_blank_key_errors_and_writes_nothing(self, tmp_path: Path) -> None:
        """A blank/whitespace key is rejected and no sidecar is created."""
        result = set_checkpoint_note(str(tmp_path), key="   ", label="x", now=_FIXED_NOW)
        assert "error" in result
        assert not _sidecar(tmp_path).exists()

    def test_empty_label_and_note_errors_and_writes_nothing(self, tmp_path: Path) -> None:
        """A key with both label and note blank/whitespace is rejected."""
        result = set_checkpoint_note(str(tmp_path), key="k1", label="  ", note="\t", now=_FIXED_NOW)
        assert "error" in result
        assert not _sidecar(tmp_path).exists()

    def test_label_and_note_are_stripped(self, tmp_path: Path) -> None:
        """Surrounding whitespace is trimmed from label and note."""
        result = set_checkpoint_note(
            str(tmp_path), key="k1", label="  spaced  ", note="  text  ", now=_FIXED_NOW
        )
        entry = result["notes"][0]
        assert entry["label"] == "spaced"
        assert entry["note"] == "text"

    def test_overlong_label_and_note_are_truncated(self, tmp_path: Path) -> None:
        """Label capped at 80 chars, note capped at 2000 chars."""
        result = set_checkpoint_note(
            str(tmp_path),
            key="k1",
            label="L" * 200,
            note="N" * 5000,
            now=_FIXED_NOW,
        )
        entry = result["notes"][0]
        assert len(entry["label"]) == 80
        assert len(entry["note"]) == 2000


class TestDeleteCheckpointNote:
    """Deleting notes."""

    def test_delete_existing_key(self, tmp_path: Path) -> None:
        """Deleting an existing key removes it and shrinks the view."""
        set_checkpoint_note(str(tmp_path), key="k1", label="a", now=_FIXED_NOW)
        set_checkpoint_note(str(tmp_path), key="k2", label="b", now=_FIXED_NOW)
        result = delete_checkpoint_note(str(tmp_path), key="k1")
        assert "error" not in result
        keys = [n["key"] for n in result["notes"]]
        assert keys == ["k2"]

    def test_delete_missing_key_errors(self, tmp_path: Path) -> None:
        """Deleting a key that does not exist returns an error."""
        set_checkpoint_note(str(tmp_path), key="k1", label="a", now=_FIXED_NOW)
        result = delete_checkpoint_note(str(tmp_path), key="nope")
        assert "error" in result

    def test_delete_blank_key_errors(self, tmp_path: Path) -> None:
        """Deleting a blank/whitespace key returns an error."""
        result = delete_checkpoint_note(str(tmp_path), key="   ")
        assert "error" in result


class TestPersistenceAndRobustness:
    """Round-trip persistence and tolerance of malformed sidecars."""

    def test_round_trip_persists_to_disk(self, tmp_path: Path) -> None:
        """A fresh read re-loads the file and returns the same entry."""
        set_checkpoint_note(
            str(tmp_path), key="k1", label="best", note="keep", now=_FIXED_NOW
        )
        reread = checkpoint_notes(str(tmp_path))
        assert reread["notes"] == [
            {
                "key": "k1",
                "label": "best",
                "note": "keep",
                "updated_at": _FIXED_NOW,
            }
        ]

    def test_malformed_json_object_returns_error(self, tmp_path: Path) -> None:
        """A sidecar that is a JSON array (not object) yields an error, not a raise."""
        sidecar = _sidecar(tmp_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("[]", encoding="utf-8")
        result = checkpoint_notes(str(tmp_path))
        assert "error" in result

    def test_invalid_json_returns_error(self, tmp_path: Path) -> None:
        """A sidecar containing invalid JSON yields an error, not a raise."""
        sidecar = _sidecar(tmp_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("{bad json", encoding="utf-8")
        result = checkpoint_notes(str(tmp_path))
        assert "error" in result
