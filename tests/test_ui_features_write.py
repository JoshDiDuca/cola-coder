"""Hermetic tests for the SAFE feature-toggle write-path.

Every test writes its own fake ``features.yaml`` (with category comments and a
few flags) into ``tmp_path`` — the REAL ``configs/features.yaml`` is never
touched. Core guarantee under test: a toggle changes exactly one line, leaving
all comments, other keys, ordering, and formatting byte-for-byte intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cola_coder.ui.features_write import set_feature

FIXTURE = """\
# Feature toggles for Cola-Coder optional features.
# Set to true/false to enable/disable each feature.

features:

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------
  crash_recovery: true
  moe_layer: false        # Experimental — disabled by default
  perplexity_tracker: true

  # ---------------------------------------------------------------------------
  # Generation
  # ---------------------------------------------------------------------------
  beam_search: true
  best_of_n_verification: true  # Generate N candidates
  speculative_decoding: false
"""


@pytest.fixture()
def features_file(tmp_path: Path) -> Path:
    path = tmp_path / "features.yaml"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_toggle_true_to_false_changes_value(features_file: Path) -> None:
    result = set_feature("crash_recovery", False, str(features_file))
    assert result == {
        "ok": True,
        "key": "crash_recovery",
        "enabled": False,
        "path": str(features_file),
    }

    import yaml

    parsed = yaml.safe_load(features_file.read_text(encoding="utf-8"))
    assert parsed["features"]["crash_recovery"] is False


def test_toggle_false_to_true_changes_value(features_file: Path) -> None:
    result = set_feature("speculative_decoding", True, str(features_file))
    assert result["ok"] is True

    import yaml

    parsed = yaml.safe_load(features_file.read_text(encoding="utf-8"))
    assert parsed["features"]["speculative_decoding"] is True


def test_only_one_line_changes_rest_byte_preserved(features_file: Path) -> None:
    before = FIXTURE.splitlines(keepends=True)
    set_feature("crash_recovery", False, str(features_file))
    after = features_file.read_text(encoding="utf-8").splitlines(keepends=True)

    assert len(before) == len(after)
    diffs = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diffs) == 1
    b, a = diffs[0]
    assert b == "  crash_recovery: true\n"
    assert a == "  crash_recovery: false\n"


def test_comments_preserved_verbatim(features_file: Path) -> None:
    set_feature("crash_recovery", False, str(features_file))
    text = features_file.read_text(encoding="utf-8")
    assert "# Feature toggles for Cola-Coder optional features." in text
    assert "# Training" in text
    assert "# Generation" in text
    assert (
        "  # ---------------------------------------------------------------------------"
        in text
    )


def test_inline_comment_preserved(features_file: Path) -> None:
    set_feature("moe_layer", True, str(features_file))
    text = features_file.read_text(encoding="utf-8")
    assert "  moe_layer: true        # Experimental — disabled by default\n" in text


def test_inline_comment_preserved_on_other_key(features_file: Path) -> None:
    set_feature("best_of_n_verification", False, str(features_file))
    text = features_file.read_text(encoding="utf-8")
    assert "  best_of_n_verification: false  # Generate N candidates\n" in text


def test_key_order_preserved(features_file: Path) -> None:
    set_feature("perplexity_tracker", False, str(features_file))

    import yaml

    parsed = yaml.safe_load(features_file.read_text(encoding="utf-8"))
    keys = list(parsed["features"].keys())
    assert keys == [
        "crash_recovery",
        "moe_layer",
        "perplexity_tracker",
        "beam_search",
        "best_of_n_verification",
        "speculative_decoding",
    ]


def test_round_trip_toggle_back_restores_file(features_file: Path) -> None:
    original = features_file.read_text(encoding="utf-8")
    set_feature("crash_recovery", False, str(features_file))
    assert features_file.read_text(encoding="utf-8") != original
    set_feature("crash_recovery", True, str(features_file))
    assert features_file.read_text(encoding="utf-8") == original


def test_unknown_key_returns_error(features_file: Path) -> None:
    result = set_feature("does_not_exist", True, str(features_file))
    assert result == {"error": "unknown feature: does_not_exist"}


def test_unknown_key_leaves_file_unchanged(features_file: Path) -> None:
    original = features_file.read_text(encoding="utf-8")
    set_feature("does_not_exist", True, str(features_file))
    assert features_file.read_text(encoding="utf-8") == original


def test_missing_file_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    result = set_feature("crash_recovery", True, str(missing))
    assert "error" in result
    assert "path not found" in result["error"]


def test_idempotent_set_same_value(features_file: Path) -> None:
    # crash_recovery is already true; setting true again must succeed and be a no-op.
    original = features_file.read_text(encoding="utf-8")
    result = set_feature("crash_recovery", True, str(features_file))
    assert result["ok"] is True
    assert features_file.read_text(encoding="utf-8") == original


def test_partial_substring_key_not_matched(tmp_path: Path) -> None:
    # A key that is a prefix of another must not accidentally match.
    path = tmp_path / "features.yaml"
    path.write_text(
        "features:\n  beam: false\n  beam_search: true\n", encoding="utf-8"
    )
    result = set_feature("beam", True, str(path))
    assert result["ok"] is True

    import yaml

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["features"]["beam"] is True
    assert parsed["features"]["beam_search"] is True  # untouched


def test_no_trailing_newline_added(features_file: Path) -> None:
    set_feature("crash_recovery", False, str(features_file))
    text = features_file.read_text(encoding="utf-8")
    # Fixture ends with exactly one trailing newline; preserve it.
    assert text.endswith("speculative_decoding: false\n")
    assert not text.endswith("\n\n\n")
