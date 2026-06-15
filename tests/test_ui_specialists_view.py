"""specialists_view: read configs/specialists.yaml into the SpecialistsView shape.

Read-only, never raises: missing/empty/malformed → a valid result or {"error"}.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui import specialists_view as spv


def _write(root: Path, body: str) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "specialists.yaml").write_text(body, encoding="utf-8")


def test_missing_file_is_not_built(tmp_path: Path) -> None:
    out = spv.specialists_view(str(tmp_path))
    assert out["exists"] is False
    assert out["count"] == 0
    assert out["specialists"] == []


def test_empty_registry(tmp_path: Path) -> None:
    _write(tmp_path, "specialists: {}\n")
    out = spv.specialists_view(str(tmp_path))
    assert out["exists"] is True
    assert out["count"] == 0


def test_populated_registry_parsed_and_sorted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "specialists:\n"
        "  react:\n"
        "    checkpoint: checkpoints/react/latest\n"
        "    config: configs/tiny.yaml\n"
        "    keywords: [react, jsx, hook]\n"
        "    confidence_threshold: 0.6\n"
        "    description: React specialist\n"
        "  graphql:\n"
        "    checkpoint: checkpoints/graphql/latest\n",
    )
    out = spv.specialists_view(str(tmp_path))
    assert out["count"] == 2
    domains = [s["domain"] for s in out["specialists"]]
    assert domains == ["graphql", "react"]  # sorted
    react = next(s for s in out["specialists"] if s["domain"] == "react")
    assert react["checkpoint"] == "checkpoints/react/latest"
    assert react["keywords"] == ["react", "jsx", "hook"]
    assert react["confidence_threshold"] == 0.6
    # Missing optionals default cleanly.
    graphql = next(s for s in out["specialists"] if s["domain"] == "graphql")
    assert graphql["config"] is None
    assert graphql["keywords"] == []
    assert graphql["confidence_threshold"] is None


def test_malformed_yaml_returns_error(tmp_path: Path) -> None:
    _write(tmp_path, "specialists: : : [unbalanced\n")
    out = spv.specialists_view(str(tmp_path))
    assert "error" in out


def test_non_dict_entries_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "specialists:\n  bogus: 42\n  ok:\n    checkpoint: c/latest\n")
    out = spv.specialists_view(str(tmp_path))
    assert out["count"] == 1
    assert out["specialists"][0]["domain"] == "ok"
