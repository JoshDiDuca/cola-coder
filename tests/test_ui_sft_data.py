"""Tests for the UI instruction/SFT + reasoning-problem JSONL browser.

Hermetic: every test writes its own JSONL fixtures under ``tmp_path`` and never
touches the real project data dirs or any training process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cola_coder.ui.sft_data import list_sft_files, preview_sft


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):  # raw line (e.g. malformed JSON)
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row) + "\n")


def _make_instructions(path: Path, n: int = 3) -> None:
    rows = [{"instruction": f"do task {i}", "output": f"result {i}"} for i in range(n)]
    _write_jsonl(path, rows)


def _make_reasoning(path: Path, n: int = 2) -> None:
    rows = [
        {
            "task_id": f"prob_{i}",
            "prompt": f"def f{i}(): ...",
            "test_code": f"assert f{i}() == {i}",
            "entry_point": f"f{i}",
            "difficulty": "easy",
        }
        for i in range(n)
    ]
    _write_jsonl(path, rows)


# ── list_sft_files ────────────────────────────────────────────────────────────


def test_list_finds_instructions_and_reasoning(tmp_path: Path) -> None:
    _make_instructions(tmp_path / "data" / "sft" / "instructions.jsonl", n=3)
    _make_reasoning(tmp_path / "data" / "reasoning" / "problems.jsonl", n=2)

    results = list_sft_files(str(tmp_path))
    names = {entry["name"] for entry in results}
    assert names == {"instructions.jsonl", "problems.jsonl"}


def test_list_classifies_kinds(tmp_path: Path) -> None:
    _make_instructions(tmp_path / "data" / "sft" / "instructions.jsonl")
    _make_reasoning(tmp_path / "data" / "reasoning" / "problems.jsonl")

    by_name = {e["name"]: e for e in list_sft_files(str(tmp_path))}
    assert by_name["instructions.jsonl"]["kind"] == "instructions"
    assert by_name["problems.jsonl"]["kind"] == "reasoning_problems"


def test_list_num_records_counts_nonempty_lines(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sft" / "instructions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 3 records with a blank line interleaved (blank must not be counted).
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"instruction": "a", "output": "b"}) + "\n")
        handle.write("\n")
        handle.write(json.dumps({"instruction": "c", "output": "d"}) + "\n")
        handle.write(json.dumps({"instruction": "e", "output": "f"}) + "\n")

    entry = list_sft_files(str(tmp_path))[0]
    assert entry["num_records"] == 3


def test_list_newest_first_by_mtime(tmp_path: Path) -> None:
    older = tmp_path / "data" / "sft" / "old.jsonl"
    newer = tmp_path / "data" / "reasoning" / "new.jsonl"
    _make_instructions(older)
    _make_reasoning(newer)

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    results = list_sft_files(str(tmp_path))
    assert [e["name"] for e in results] == ["new.jsonl", "old.jsonl"]


def test_list_entry_has_full_schema(tmp_path: Path) -> None:
    _make_instructions(tmp_path / "data" / "sft" / "instructions.jsonl")
    entry = list_sft_files(str(tmp_path))[0]
    assert set(entry) == {"name", "path", "kind", "num_records", "size_bytes", "mtime"}
    assert entry["size_bytes"] > 0
    assert isinstance(entry["mtime"], float)


def test_list_missing_root_returns_empty(tmp_path: Path) -> None:
    assert list_sft_files(str(tmp_path / "does_not_exist")) == []


def test_list_empty_when_no_jsonl(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True)
    assert list_sft_files(str(tmp_path)) == []


def test_list_sniffs_chatml_messages_as_instructions(tmp_path: Path) -> None:
    # ChatML rows in a neutrally-named file under data/ should sniff to instructions.
    path = tmp_path / "data" / "pairs.jsonl"
    _write_jsonl(
        path,
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}],
    )
    entry = next(e for e in list_sft_files(str(tmp_path)) if e["name"] == "pairs.jsonl")
    assert entry["kind"] == "instructions"


# ── preview_sft ────────────────────────────────────────────────────────────────


def test_preview_returns_rows_and_fields(tmp_path: Path) -> None:
    path = tmp_path / "data" / "reasoning" / "problems.jsonl"
    _make_reasoning(path, n=2)

    result = preview_sft(str(path), n=10)
    assert result["count"] == 2
    assert result["truncated"] is False
    assert len(result["records"]) == 2
    assert "task_id" in result["fields"]
    assert "difficulty" in result["fields"]


def test_preview_fields_are_union_across_rows(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sft" / "mixed.jsonl"
    _write_jsonl(
        path,
        [
            {"instruction": "a", "output": "b"},
            {"instruction": "c", "output": "d", "input": "ctx"},
        ],
    )
    result = preview_sft(str(path), n=10)
    assert set(result["fields"]) == {"instruction", "output", "input"}


def test_preview_truncated_when_count_exceeds_n(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sft" / "instructions.jsonl"
    _make_instructions(path, n=5)

    result = preview_sft(str(path), n=2)
    assert result["count"] == 5
    assert result["truncated"] is True
    assert len(result["records"]) == 2


def test_preview_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sft" / "instructions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"instruction": "a", "output": "b"}) + "\n")
        handle.write("{not valid json\n")  # malformed — skipped, still counted
        handle.write(json.dumps({"instruction": "c", "output": "d"}) + "\n")

    result = preview_sft(str(path), n=10)
    # All 3 non-empty lines counted, but only 2 parse into records.
    assert result["count"] == 3
    assert len(result["records"]) == 2


def test_preview_caps_huge_field(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sft" / "big.jsonl"
    _write_jsonl(path, [{"instruction": "x" * 5000, "output": "ok"}])

    result = preview_sft(str(path), n=10)
    assert len(result["records"][0]["instruction"]) == 2000
    assert result["records"][0]["output"] == "ok"


def test_preview_missing_path_returns_error(tmp_path: Path) -> None:
    result = preview_sft(str(tmp_path / "nope.jsonl"))
    assert "error" in result
    assert "records" not in result
