"""Tokenizer inspection helpers for the local UI/dashboard.

Lightweight, read-only inspection of a HuggingFace byte-level BPE
``tokenizer.json`` file. The file is plain JSON, so this module parses it
directly with the stdlib ``json`` module — it never loads the heavy
``tokenizers``/``transformers`` libraries. All functions are robust to missing
or malformed inputs and never raise — they return an ``{"error": ...}`` dict
instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _candidate_paths() -> list[Path]:
    """Default tokenizer.json locations to probe, in priority order.

    0. ``checkpoints/<run>/tokenizer.json`` — a tokenizer saved NEXT TO a checkpoint
       output (newest run first). Highest priority so a run is self-describing about
       its tokenizer and resume/inference/eval use the exact one it was trained with.
    1. ``storage.tokenizer_path`` from ``configs/storage.yaml`` (if readable).
    2. ``data/<dataset>/tokenizer.json`` — the canonical per-dataset location.
    3. Common project-relative locations (``tokenizer.json``, ``tokenizer/``,
       ``tokenizers/`` dirs).
    """
    candidates: list[Path] = []

    # 0. tokenizer.json saved next to a checkpoint output (newest run first).
    ckpt_root = Path("checkpoints")
    if ckpt_root.is_dir():
        try:
            ckpt_toks = [
                d / "tokenizer.json"
                for d in ckpt_root.iterdir()
                if d.is_dir() and (d / "tokenizer.json").is_file()
            ]
            ckpt_toks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(ckpt_toks)
        except OSError:
            pass

    # 1. configs/storage.yaml -> storage.tokenizer_path
    storage_yaml = Path("configs/storage.yaml")
    if storage_yaml.is_file():
        try:
            with open(storage_yaml, encoding="utf-8") as handle:
                parsed = yaml.safe_load(handle) or {}
            storage = parsed.get("storage", {}) if isinstance(parsed, dict) else {}
            tok_path = storage.get("tokenizer_path") if isinstance(storage, dict) else None
            if isinstance(tok_path, str) and tok_path:
                candidates.append(Path(tok_path))
        except (OSError, yaml.YAMLError):
            pass

    # 2. data/<dataset>/tokenizer.json — canonical per-dataset location.
    data_root = Path("data")
    if data_root.is_dir():
        try:
            for child in sorted(data_root.iterdir()):
                if child.is_dir():
                    candidates.append(child / "tokenizer.json")
        except OSError:
            pass

    # 3. Common project-relative fallbacks.
    candidates.append(Path("tokenizer.json"))
    candidates.append(Path("tokenizer") / "tokenizer.json")
    candidates.append(Path("tokenizers") / "tokenizer.json")

    return candidates


def _resolve_tokenizer_file(path: str | None) -> Path | None:
    """Resolve ``path`` (file OR containing dir, or None) to a tokenizer.json.

    Returns the resolved ``Path`` to an existing tokenizer.json, or ``None`` if
    nothing could be found.
    """
    if path is not None:
        candidate = Path(path)
        if candidate.is_dir():
            nested = candidate / "tokenizer.json"
            return nested if nested.is_file() else None
        return candidate if candidate.is_file() else None

    for candidate in _candidate_paths():
        if candidate.is_dir():
            nested = candidate / "tokenizer.json"
            if nested.is_file():
                return nested
        elif candidate.is_file():
            return candidate
    return None


def _has_digit_splitting(node: object) -> bool:
    """Recursively search a pre_tokenizer node for a Digits step.

    Returns True if any node is ``{"type": "Digits", "individual_digits": true}``.
    A pre_tokenizer may be a single step, a ``{"type": "Sequence",
    "pretokenizers": [...]}`` wrapper, or arbitrarily nested.
    """
    if isinstance(node, dict):
        if node.get("type") == "Digits" and node.get("individual_digits") is True:
            return True
        return any(_has_digit_splitting(value) for value in node.values())
    if isinstance(node, list):
        return any(_has_digit_splitting(item) for item in node)
    return False


def tokenizer_info(path: str | None = None) -> dict:
    """Inspect a tokenizer.json. If ``path`` is None, DISCOVER the default: try the path from
    configs/storage.yaml, then common locations (e.g. a ``tokenizer/`` or ``tokenizers/`` dir, or
    a *.json next to it). ``path`` may point at the tokenizer.json file OR its containing dir.

    Returns:
      {"path": str,                       # resolved tokenizer.json path
       "vocab_size": int,
       "n_merges": int,                   # len(model.merges) if present else 0
       "special_tokens": list[str],       # added_tokens content strings (e.g. <|fim_prefix|>, <think>)
       "has_fim_tokens": bool,            # any added token matching <|fim_*|>
       "digit_splitting": bool,           # pre_tokenizer contains a Digits step with individual_digits true
       "model_type": str}                 # tokenizer "model"."type", e.g. "BPE"
    On any failure (not found / bad JSON) return {"error": "..."}. Never raise.
    """
    resolved = _resolve_tokenizer_file(path)
    if resolved is None:
        target = path if path is not None else "<default locations>"
        return {"error": f"tokenizer.json not found: {target}"}

    try:
        with open(resolved, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    if not isinstance(data, dict):
        return {"error": f"invalid tokenizer.json (not an object): {resolved}"}

    model = data.get("model")
    if not isinstance(model, dict):
        model = {}

    # vocab_size: len(model.vocab) when it's a dict.
    vocab = model.get("vocab")
    vocab_size = len(vocab) if isinstance(vocab, dict) else 0

    # n_merges: len(model.merges) if present.
    merges = model.get("merges")
    n_merges = len(merges) if isinstance(merges, list) else 0

    # model_type, e.g. "BPE".
    model_type = model.get("type")
    if not isinstance(model_type, str):
        model_type = ""

    # added_tokens: list of {"content": str, ...}.
    added = data.get("added_tokens")
    special_tokens: list[str] = []
    if isinstance(added, list):
        for entry in added:
            if isinstance(entry, dict):
                content = entry.get("content")
                if isinstance(content, str):
                    special_tokens.append(content)

    has_fim_tokens = any(
        tok.startswith("<|fim_") and tok.endswith("|>") for tok in special_tokens
    )

    digit_splitting = _has_digit_splitting(data.get("pre_tokenizer"))

    return {
        "path": str(resolved),
        "vocab_size": vocab_size,
        "n_merges": n_merges,
        "special_tokens": special_tokens,
        "has_fim_tokens": has_fim_tokens,
        "digit_splitting": digit_splitting,
        "model_type": model_type,
    }
