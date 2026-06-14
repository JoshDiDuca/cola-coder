"""Tests for the CLI scripts catalog UI helper.

Hermetic: each test builds its own fake ``.claude/rules/scripts-reference.md`` and
``scripts/`` directory under ``tmp_path``. One test runs against the real repo.
"""

from __future__ import annotations

from pathlib import Path

from cola_coder.ui.scripts_catalog import list_scripts

_DOC = """# Scripts Reference (fake)

## Alpha Category
| Script | Purpose |
|--------|---------|
| `alpha_one.py` | First alpha script |
| `alpha_two.py` | Second alpha script — flags: --foo |

## Beta Category
| Script | Purpose |
|--------|---------|
| `beta_one.py` | The beta script |
"""


def _build_repo(tmp_path: Path, *, doc: str | None = _DOC, disk: list[str] | None = None) -> Path:
    """Create a fake root with an optional reference doc and a scripts/ dir."""
    if doc is not None:
        rules_dir = tmp_path / ".claude" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        (rules_dir / "scripts-reference.md").write_text(doc, encoding="utf-8")

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if disk:
        for name in disk:
            (scripts_dir / name).write_text("# stub\n", encoding="utf-8")
    return tmp_path


def test_parses_categories_and_purposes(tmp_path: Path) -> None:
    root = _build_repo(tmp_path, disk=["alpha_one.py", "beta_one.py"])
    result = list_scripts(str(root))

    assert "error" not in result
    by_name = {s["name"]: s for s in result["scripts"]}
    assert by_name["alpha_one.py"]["category"] == "Alpha Category"
    assert by_name["alpha_one.py"]["purpose"] == "First alpha script"
    assert by_name["alpha_two.py"]["purpose"] == "Second alpha script — flags: --foo"
    assert by_name["beta_one.py"]["category"] == "Beta Category"


def test_exists_reflects_disk(tmp_path: Path) -> None:
    # alpha_one + beta_one on disk; alpha_two documented but missing.
    root = _build_repo(tmp_path, disk=["alpha_one.py", "beta_one.py"])
    result = list_scripts(str(root))
    by_name = {s["name"]: s for s in result["scripts"]}

    assert by_name["alpha_one.py"]["exists"] is True
    assert by_name["beta_one.py"]["exists"] is True
    assert by_name["alpha_two.py"]["exists"] is False


def test_extra_disk_script_is_uncategorized(tmp_path: Path) -> None:
    root = _build_repo(
        tmp_path,
        disk=["alpha_one.py", "alpha_two.py", "beta_one.py", "mystery.py"],
    )
    result = list_scripts(str(root))
    by_name = {s["name"]: s for s in result["scripts"]}

    assert by_name["mystery.py"]["category"] == "Uncategorized"
    assert by_name["mystery.py"]["purpose"] == ""
    assert by_name["mystery.py"]["exists"] is True
    assert "Uncategorized" in result["categories"]


def test_categories_in_document_order(tmp_path: Path) -> None:
    root = _build_repo(tmp_path, disk=["mystery.py"])
    result = list_scripts(str(root))

    # Alpha before Beta (doc order), Uncategorized appended last for the extra.
    assert result["categories"] == ["Alpha Category", "Beta Category", "Uncategorized"]


def test_count_and_on_disk(tmp_path: Path) -> None:
    root = _build_repo(
        tmp_path,
        disk=["alpha_one.py", "beta_one.py", "mystery.py"],
    )
    result = list_scripts(str(root))

    # 3 documented + 1 extra disk script = 4 catalog entries.
    assert result["count"] == 4
    # 3 .py files on disk.
    assert result["on_disk"] == 3
    assert len(result["scripts"]) == 4


def test_scripts_sorted_by_category_then_name(tmp_path: Path) -> None:
    root = _build_repo(tmp_path, disk=["mystery.py"])
    result = list_scripts(str(root))
    keys = [(s["category"], s["name"]) for s in result["scripts"]]
    assert keys == sorted(keys)


def test_no_uncategorized_when_no_extras(tmp_path: Path) -> None:
    root = _build_repo(tmp_path, disk=["alpha_one.py", "alpha_two.py", "beta_one.py"])
    result = list_scripts(str(root))
    assert "Uncategorized" not in result["categories"]
    assert result["categories"] == ["Alpha Category", "Beta Category"]


def test_disk_only_fallback_when_doc_absent(tmp_path: Path) -> None:
    root = _build_repo(tmp_path, doc=None, disk=["foo.py", "bar.py"])
    result = list_scripts(str(root))

    assert "error" not in result
    assert result["categories"] == ["Uncategorized"]
    assert result["on_disk"] == 2
    assert result["count"] == 2
    assert all(s["category"] == "Uncategorized" for s in result["scripts"])
    assert all(s["purpose"] == "" for s in result["scripts"])
    assert all(s["exists"] is True for s in result["scripts"])


def test_empty_root_no_doc_no_scripts(tmp_path: Path) -> None:
    # No doc, no scripts dir at all -> empty disk-only catalog, no error.
    result = list_scripts(str(tmp_path))
    assert "error" not in result
    assert result["count"] == 0
    assert result["on_disk"] == 0
    assert result["categories"] == []
    assert result["scripts"] == []


def test_tolerates_extra_columns_and_whitespace(tmp_path: Path) -> None:
    doc = """## Weird
|  `spaced.py`   |   has padding   |  extra col  |
"""
    root = _build_repo(tmp_path, doc=doc, disk=["spaced.py"])
    result = list_scripts(str(root))
    by_name = {s["name"]: s for s in result["scripts"]}
    assert by_name["spaced.py"]["category"] == "Weird"
    assert by_name["spaced.py"]["purpose"] == "has padding"


def test_error_on_unreadable_doc(tmp_path: Path, monkeypatch) -> None:
    # Force a genuinely broken read to exercise the error path.
    root = _build_repo(tmp_path, disk=["alpha_one.py"])

    import cola_coder.ui.scripts_catalog as mod

    def _boom(self, *args, **kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr(mod.Path, "read_text", _boom)
    result = list_scripts(str(root))
    assert "error" in result


def test_real_repo_catalog() -> None:
    # tests/ -> repo root is parent of this file's parent.
    repo_root = Path(__file__).resolve().parent.parent
    result = list_scripts(str(repo_root))

    assert "error" not in result
    assert result["count"] > 30
    assert "train.py" in {s["name"] for s in result["scripts"]}
    assert len(result["categories"]) > 1
