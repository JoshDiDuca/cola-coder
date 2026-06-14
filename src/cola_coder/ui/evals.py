"""Evaluation/quality artifact browsing helpers for the local UI/dashboard.

Lightweight, read-only discovery of the evaluation, quality-report, regression,
and benchmark artifacts the project writes to disk (JSON / JSONL / Markdown /
text). All functions are robust to missing or malformed inputs and never raise on
bad data — they return empty results or an ``{"error": ...}`` dict instead.

Artifacts are produced by several scripts and land in different spots, so kind
detection is best-effort (filename hints + cheap content sniffing) and the
``summary`` string is extracted only when it is cheap to do so:

- ``scripts/quality_report.py`` -> ``reports/quality_report_*.md`` + ``.json``
  (``QualityReport.to_dict``: ``humaneval_pass_at_1``, ``smoke_test_passed``, ...)
- ``scripts/evaluate.py`` / ``scripts/run_eval_suite.py`` -> HumanEval-shaped
  JSON carrying ``pass@k`` metrics (``--output eval_results.json``)
- ``scripts/regression_test.py --save`` -> ``{checkpoint, total, passed, failed,
  pass_rate, details}`` JSON
- ``scripts/completion_benchmark.py`` -> classified by filename when present
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_MAX_CONTENT_CHARS = 40000

# Directories that conventionally hold eval/quality artifacts.
_SCAN_DIRS = ("reports", "eval_results", "evaluations", "evals")

# Filename hints for kind classification (lowercased substring match).
_HUMANEVAL_HINTS = ("humaneval", "human_eval", "eval_result", "eval_suite", "pass_at")
_QUALITY_HINTS = ("quality_report", "quality-report", "qualityreport")
_REGRESSION_HINTS = ("regression",)
_COMPLETION_HINTS = ("completion_benchmark", "completion-benchmark", "completion_bench")

# File extensions we consider artifacts.
_ARTIFACT_SUFFIXES = (".json", ".jsonl", ".md", ".txt")


def list_eval_results(root: str = ".") -> list[dict]:
    """Discover eval/quality/regression artifacts under ``root``.

    Scans the conventional artifact directories (``reports``, ``eval_results``,
    ``evaluations``, ``evals``) plus the top level of ``root`` for ``*.json``,
    ``*.jsonl``, ``*.md`` and ``*.txt`` files, classifies each, and extracts a
    cheap one-line ``summary`` when possible.

    Each entry is a dict with keys:
      - ``name``  — the filename.
      - ``path``  — root-relative path (consistent across entries).
      - ``kind``  — one of ``"humaneval"``, ``"quality_report"``, ``"regression"``,
        ``"completion_benchmark"``, ``"other"``.
      - ``mtime`` — modification time (float epoch seconds).
      - ``summary`` — short human string (e.g. ``"pass@1 0.34, pass@10 0.51"``),
        or ``""`` when nothing cheap can be extracted.

    Results are sorted newest-first by ``mtime``. Unreadable files are skipped.
    Missing directories are ignored. Returns ``[]`` when nothing is found. Never
    raises.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    seen: set[str] = set()
    results: list[dict] = []

    for candidate in _candidate_paths(root_path):
        try:
            resolved = str(candidate.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        try:
            stat = candidate.stat()
        except OSError:
            continue

        try:
            rel = os.path.relpath(str(candidate), str(root_path))
        except (OSError, ValueError):
            rel = str(candidate)

        kind = _classify(candidate)
        results.append(
            {
                "name": candidate.name,
                "path": rel,
                "kind": kind,
                "mtime": stat.st_mtime,
                "summary": _extract_summary(candidate, kind),
            }
        )

    results.sort(key=lambda entry: entry["mtime"], reverse=True)
    return results


def read_eval_result(path: str) -> dict:
    """Read and parse one artifact.

    Returns ``{"path", "kind", "parsed", "content", "truncated"}``:

      - JSON files parse into ``parsed`` (``content`` is ``None``).
      - JSONL files parse into ``parsed`` as a list of per-line objects;
        unparseable lines are skipped (``content`` is ``None``).
      - ``.md`` / ``.txt`` files put text into ``content`` (capped at 40000
        chars, ``truncated`` set accordingly; ``parsed`` is ``None``).

    On any failure (missing path, unreadable, undecodable) returns
    ``{"error": str}``. Never raises.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {"error": f"path not found: {path}"}

    kind = _classify(file_path)
    suffix = file_path.suffix.lower()

    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": str(exc)}

    if suffix == ".json":
        try:
            parsed: object | None = json.loads(raw)
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "path": path,
            "kind": kind,
            "parsed": parsed,
            "content": None,
            "truncated": False,
        }

    if suffix == ".jsonl":
        parsed_list: list = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                parsed_list.append(json.loads(line))
            except ValueError:
                continue
        return {
            "path": path,
            "kind": kind,
            "parsed": parsed_list,
            "content": None,
            "truncated": False,
        }

    # .md / .txt (and any other text artifact) -> raw content, capped.
    truncated = len(raw) > _MAX_CONTENT_CHARS
    content = raw[:_MAX_CONTENT_CHARS]
    return {
        "path": path,
        "kind": kind,
        "parsed": None,
        "content": content,
        "truncated": truncated,
    }


# ── Internals ───────────────────────────────────────────────────────────────


def _candidate_paths(root_path: Path) -> list[Path]:
    """Collect artifact-shaped files from the conventional dirs + the top level."""
    candidates: list[Path] = []

    # Recurse the conventional artifact directories.
    for dirname in _SCAN_DIRS:
        scan_dir = root_path / dirname
        if not scan_dir.is_dir():
            continue
        try:
            for found in scan_dir.rglob("*"):
                if found.is_file() and found.suffix.lower() in _ARTIFACT_SUFFIXES:
                    candidates.append(found)
        except OSError:
            continue

    # Also pick up loose artifacts at the top level (e.g. eval_results.json,
    # results_v1.json) whose name hints at an eval/quality/regression artifact.
    try:
        for found in root_path.iterdir():
            if not found.is_file():
                continue
            if found.suffix.lower() not in _ARTIFACT_SUFFIXES:
                continue
            if _classify(found) != "other":
                candidates.append(found)
    except OSError:
        pass

    return candidates


def _classify(path: Path) -> str:
    """Best-effort kind from the filename (and parent dir) hints."""
    name = path.name.lower()

    if any(hint in name for hint in _QUALITY_HINTS):
        return "quality_report"
    if any(hint in name for hint in _COMPLETION_HINTS):
        return "completion_benchmark"
    if any(hint in name for hint in _REGRESSION_HINTS):
        return "regression"
    if any(hint in name for hint in _HUMANEVAL_HINTS):
        return "humaneval"
    return "other"


def _extract_summary(path: Path, kind: str) -> str:
    """Extract a cheap one-line summary from an artifact, or ``""``.

    Only JSON artifacts are sniffed (parsing one small JSON file is cheap);
    Markdown/text/JSONL get an empty summary to stay fast and avoid surprises.
    """
    if path.suffix.lower() != ".json":
        return ""

    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return ""

    if not isinstance(data, dict):
        return ""

    # pass@k metrics live either at the top level or under a "metrics" dict.
    pass_summary = _pass_at_k_summary(data)
    if pass_summary:
        return pass_summary

    if kind == "quality_report" or "humaneval_pass_at_1" in data:
        return _quality_summary(data)

    if kind == "regression" or {"passed", "failed"} <= set(data):
        return _regression_summary(data)

    # run_eval_suite style report.
    if {"n_passed", "n_failed"} <= set(data):
        return _suite_summary(data)

    return ""


def _pass_at_k_summary(data: dict) -> str:
    """``pass@1 0.34, pass@10 0.51`` from top-level or nested ``metrics`` keys."""
    sources: list[dict] = [data]
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        sources.append(metrics)

    parts: list[str] = []
    for source in sources:
        for key, value in source.items():
            if isinstance(key, str) and key.lower().startswith("pass@") and _is_number(value):
                parts.append(f"{key} {float(value):.2f}")
        if parts:
            break
    return ", ".join(parts)


def _quality_summary(data: dict) -> str:
    """Summary line for a quality report dict."""
    parts: list[str] = []
    pass1 = data.get("humaneval_pass_at_1")
    if _is_number(pass1):
        parts.append(f"pass@1 {float(pass1):.2f}")
    smoke = data.get("smoke_test_passed")
    if isinstance(smoke, bool):
        parts.append(f"smoke {'pass' if smoke else 'fail'}")
    step = data.get("training_step")
    if isinstance(step, int):
        parts.append(f"step {step}")
    loss = data.get("training_loss")
    if _is_number(loss):
        parts.append(f"loss {float(loss):.3f}")
    return ", ".join(parts)


def _regression_summary(data: dict) -> str:
    """Summary line for a regression-result dict."""
    passed = data.get("passed")
    total = data.get("total")
    rate = data.get("pass_rate")
    if isinstance(passed, int) and isinstance(total, int):
        head = f"{passed}/{total} passed"
    elif isinstance(passed, int):
        head = f"{passed} passed"
    else:
        head = ""
    if _is_number(rate):
        suffix = f"rate {float(rate):.2f}"
        return f"{head}, {suffix}" if head else suffix
    return head


def _suite_summary(data: dict) -> str:
    """Summary line for a run_eval_suite report dict."""
    n_passed = data.get("n_passed")
    n_failed = data.get("n_failed")
    n_skipped = data.get("n_skipped")
    parts: list[str] = []
    if isinstance(n_passed, int):
        parts.append(f"{n_passed} passed")
    if isinstance(n_failed, int):
        parts.append(f"{n_failed} failed")
    if isinstance(n_skipped, int) and n_skipped:
        parts.append(f"{n_skipped} skipped")
    return ", ".join(parts)


def _is_number(value: object) -> bool:
    """True for a real int/float (excludes bool, which is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
