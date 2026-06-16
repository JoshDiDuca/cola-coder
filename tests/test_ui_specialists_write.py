"""Tests for the specialist-registry WRITE helpers in ``specialists_view``.

Covers ``save_specialist`` (upsert) and ``remove_specialist`` (delete) against a
temporary ``configs/specialists.yaml`` (via the ``tmp_path`` fixture). These are
pure filesystem/YAML helpers: no network, no GPU, no model loading. The
read-only ``specialists_view`` is used to assert the post-write state.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cola_coder.ui.specialists_view import (
    remove_specialist,
    save_specialist,
    specialists_view,
)

_SPECIALISTS_REL = Path("configs") / "specialists.yaml"


def _registry_path(root: Path) -> Path:
    """Return the ``configs/specialists.yaml`` path under ``root``."""
    return root / _SPECIALISTS_REL


def _find(view: dict, domain: str) -> dict | None:
    """Return the specialist entry for ``domain`` in a view dict, or ``None``."""
    for entry in view["specialists"]:
        if entry["domain"] == domain:
            return entry
    return None


class TestSaveSpecialistCreate:
    """Saving into a non-existent registry."""

    def test_creates_file_and_dir(self, tmp_path: Path) -> None:
        """First save creates ``configs/specialists.yaml`` (and parent dir)."""
        path = _registry_path(tmp_path)
        assert not path.exists()

        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react", "jsx"],
        )

        assert "error" not in view
        assert path.is_file()
        assert path.parent.name == "configs"

    def test_entry_appears_in_refreshed_view(self, tmp_path: Path) -> None:
        """The saved entry is present with correct fields and count == 1."""
        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react", "jsx"],
            config="configs/tiny.yaml",
            confidence_threshold=0.6,
            description="React specialist",
        )

        assert view["exists"] is True
        assert view["count"] == 1
        entry = _find(view, "react")
        assert entry is not None
        assert entry["checkpoint"] == "checkpoints/react/latest"
        assert entry["keywords"] == ["react", "jsx"]
        assert entry["config"] == "configs/tiny.yaml"
        assert entry["confidence_threshold"] == 0.6
        assert entry["description"] == "React specialist"


class TestSaveSpecialistSiblings:
    """Saving must not clobber other domains."""

    def test_second_domain_preserves_first(self, tmp_path: Path) -> None:
        """Saving a second domain keeps the first (count == 2, both present)."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
        )
        view = save_specialist(
            root=str(tmp_path),
            domain="graphql",
            checkpoint="checkpoints/graphql/latest",
            keywords=["graphql", "schema"],
        )

        assert "error" not in view
        assert view["count"] == 2
        assert _find(view, "react") is not None
        assert _find(view, "graphql") is not None


class TestSaveSpecialistUpdate:
    """Saving an existing domain updates in place."""

    def test_update_in_place(self, tmp_path: Path) -> None:
        """Re-saving a domain updates it without changing the count."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/v1",
            keywords=["react"],
        )
        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/v2",
            keywords=["react", "hooks"],
        )

        assert "error" not in view
        assert view["count"] == 1
        entry = _find(view, "react")
        assert entry is not None
        assert entry["checkpoint"] == "checkpoints/react/v2"
        assert entry["keywords"] == ["react", "hooks"]


class TestSaveSpecialistValidation:
    """Validation failures return ``{"error": ...}`` and never write."""

    def test_empty_domain(self, tmp_path: Path) -> None:
        """An empty domain is rejected and no file is created."""
        result = save_specialist(
            root=str(tmp_path),
            domain="",
            checkpoint="checkpoints/x/latest",
            keywords=[],
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_whitespace_domain(self, tmp_path: Path) -> None:
        """A whitespace-only domain is rejected and no file is created."""
        result = save_specialist(
            root=str(tmp_path),
            domain="   ",
            checkpoint="checkpoints/x/latest",
            keywords=[],
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_domain_with_forward_slash(self, tmp_path: Path) -> None:
        """A domain containing ``/`` is rejected and no file is created."""
        result = save_specialist(
            root=str(tmp_path),
            domain="react/native",
            checkpoint="checkpoints/x/latest",
            keywords=[],
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_domain_with_backslash(self, tmp_path: Path) -> None:
        """A domain containing ``\\`` is rejected and no file is created."""
        result = save_specialist(
            root=str(tmp_path),
            domain="react\\native",
            checkpoint="checkpoints/x/latest",
            keywords=[],
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_empty_checkpoint(self, tmp_path: Path) -> None:
        """An empty/whitespace checkpoint is rejected and no file is created."""
        result = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="   ",
            keywords=[],
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_confidence_threshold_too_high(self, tmp_path: Path) -> None:
        """confidence_threshold == 1.5 is out of [0, 1] and rejected."""
        result = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=[],
            confidence_threshold=1.5,
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()

    def test_confidence_threshold_too_low(self, tmp_path: Path) -> None:
        """confidence_threshold == -0.1 is out of [0, 1] and rejected."""
        result = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=[],
            confidence_threshold=-0.1,
        )
        assert "error" in result
        assert not _registry_path(tmp_path).exists()


class TestSaveSpecialistKeywordCleaning:
    """Keywords are stripped and empties dropped."""

    def test_keywords_cleaned(self, tmp_path: Path) -> None:
        """Whitespace-only keywords are dropped and surrounding spaces stripped."""
        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["  react  ", "   ", "jsx", ""],
        )

        entry = _find(view, "react")
        assert entry is not None
        assert entry["keywords"] == ["react", "jsx"]


class TestSaveSpecialistOptionalFields:
    """Optional fields are omitted when None/empty."""

    def test_optional_fields_omitted(self, tmp_path: Path) -> None:
        """config/description not stored when None/empty → come back as None."""
        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
            config=None,
            confidence_threshold=None,
            description="   ",
        )

        # View coercion surfaces missing optional fields as None.
        entry = _find(view, "react")
        assert entry is not None
        assert entry["config"] is None
        assert entry["confidence_threshold"] is None
        assert entry["description"] is None

        # The stored entry must literally omit the empty/None keys.
        raw = yaml.safe_load(_registry_path(tmp_path).read_text(encoding="utf-8"))
        stored = raw["specialists"]["react"]
        assert "config" not in stored
        assert "confidence_threshold" not in stored
        assert "description" not in stored


class TestRemoveSpecialist:
    """Removing entries by domain."""

    def test_remove_existing(self, tmp_path: Path) -> None:
        """Removing an existing domain drops the count and the entry."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
        )
        save_specialist(
            root=str(tmp_path),
            domain="graphql",
            checkpoint="checkpoints/graphql/latest",
            keywords=["graphql"],
        )

        view = remove_specialist(root=str(tmp_path), domain="react")

        assert "error" not in view
        assert view["count"] == 1
        assert _find(view, "react") is None
        assert _find(view, "graphql") is not None

    def test_remove_missing_domain(self, tmp_path: Path) -> None:
        """Removing a domain absent from the registry returns an error."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
        )

        result = remove_specialist(root=str(tmp_path), domain="vue")
        assert "error" in result
        # The existing entry is untouched.
        assert specialists_view(str(tmp_path))["count"] == 1

    def test_remove_when_file_missing(self, tmp_path: Path) -> None:
        """Removing from a non-existent registry returns an error."""
        assert not _registry_path(tmp_path).exists()
        result = remove_specialist(root=str(tmp_path), domain="react")
        assert "error" in result

    def test_remove_empty_domain(self, tmp_path: Path) -> None:
        """An empty domain is rejected on removal."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
        )
        result = remove_specialist(root=str(tmp_path), domain="   ")
        assert "error" in result


class TestSiblingTopLevelKeyPreserved:
    """A save must preserve unrelated top-level YAML keys."""

    def test_sibling_key_preserved(self, tmp_path: Path) -> None:
        """A sibling ``other`` top-level key survives a ``save_specialist`` call."""
        path = _registry_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({"other": {"foo": 1}, "specialists": {}}),
            encoding="utf-8",
        )

        view = save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react"],
        )
        assert "error" not in view

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["other"] == {"foo": 1}
        assert "react" in raw["specialists"]


class TestRoundTrip:
    """A saved entry round-trips through the view with correct types."""

    def test_round_trip_types(self, tmp_path: Path) -> None:
        """confidence_threshold comes back as float, keywords as list[str]."""
        save_specialist(
            root=str(tmp_path),
            domain="react",
            checkpoint="checkpoints/react/latest",
            keywords=["react", "jsx"],
            confidence_threshold=0.75,
        )

        view = specialists_view(str(tmp_path))
        entry = _find(view, "react")
        assert entry is not None
        assert isinstance(entry["confidence_threshold"], float)
        assert entry["confidence_threshold"] == 0.75
        assert isinstance(entry["keywords"], list)
        assert all(isinstance(k, str) for k in entry["keywords"])
        assert entry["keywords"] == ["react", "jsx"]
