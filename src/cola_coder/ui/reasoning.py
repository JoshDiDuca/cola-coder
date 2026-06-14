"""Reasoning/GRPO config browsing helpers for the local UI/dashboard.

Lightweight, read-only inspection of ``configs/reasoning.yaml`` — the config that
drives SFT warmup + GRPO reasoning fine-tuning. All functions are robust to
missing or malformed inputs and never raise — they return an {"error": ...}
dict instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def read_reasoning(path: str = "configs/reasoning.yaml") -> dict:
    """Parse reasoning.yaml into a UI-friendly summary. Returns:
      {"path": str,
       "parsed": dict,                 # the full safe_loaded yaml
       "reasoning": dict,              # the `reasoning` section (or {})
       "problem_set": dict,            # the `problem_set` section (or {})
       "sft_warmup": dict,             # the `sft_warmup` section (or {})
       "summary": {                    # a few headline values pulled out for the UI
           "advantage_norm": <any|None>, "clip_epsilon": <any|None>,
           "clip_epsilon_high": <any|None>, "group_size": <any|None>,
           "sft_warmup_enabled": <bool|None>, "problem_source": <any|None>
       }}
    On any failure (missing file / bad YAML / non-mapping) return {"error": "..."}.
    Never raises.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {"error": f"path not found: {path}"}

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": str(exc)}

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return {"error": f"invalid YAML: {exc}"}

    if not isinstance(parsed, dict):
        type_name = type(parsed).__name__
        return {"error": f"unexpected reasoning config shape: {type_name}"}

    reasoning = parsed.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}

    problem_set = parsed.get("problem_set")
    problem_set = problem_set if isinstance(problem_set, dict) else {}

    sft_warmup = parsed.get("sft_warmup")
    sft_warmup = sft_warmup if isinstance(sft_warmup, dict) else {}

    summary = {
        "advantage_norm": reasoning.get("advantage_norm"),
        "clip_epsilon": reasoning.get("clip_epsilon"),
        "clip_epsilon_high": reasoning.get("clip_epsilon_high"),
        "group_size": reasoning.get("group_size"),
        "sft_warmup_enabled": sft_warmup.get("enabled"),
        "problem_source": problem_set.get("source"),
    }

    return {
        "path": path,
        "parsed": parsed,
        "reasoning": reasoning,
        "problem_set": problem_set,
        "sft_warmup": sft_warmup,
        "summary": summary,
    }
