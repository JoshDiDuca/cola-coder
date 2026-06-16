"""Config "at a glance" summary endpoint helper for the local UI (UI-103).

Parses a training YAML config into a grouped, human-readable hyperparameter
summary — Model / Training / Data / Checkpoint groups plus a Derived group
(effective batch = batch_size × gradient_accumulation). Complements the raw
config editor (UI-018) and the VRAM estimate (model dims) with a scannable knob
overview, so a user can see "what is this run configured to do" without reading
YAML.

Values are coerced to ``str`` at this boundary (the schema-first rule: the TS
type is a concrete ``{label, value}`` — no open JSON crosses the wire). Pure file
read; MAIN-SAFE; never raises (returns ``{"error": str}`` on a malformed file).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# (yaml_key, display_label) per section, in display order. Only keys PRESENT in
# the config produce an item — missing keys are silently skipped.
_MODEL_KEYS: list[tuple[str, str]] = [
    ("dim", "dim"),
    ("n_layers", "layers"),
    ("n_heads", "query heads"),
    ("n_kv_heads", "KV heads"),
    ("ffn_dim_multiplier", "FFN mult"),
    ("max_seq_len", "seq len"),
    ("vocab_size", "vocab"),
    ("rope_theta", "RoPE theta"),
    ("dropout", "dropout"),
    ("qk_norm", "QK-norm"),
]
_TRAINING_KEYS: list[tuple[str, str]] = [
    ("batch_size", "batch size"),
    ("gradient_accumulation", "grad accum"),
    ("learning_rate", "learning rate"),
    ("min_lr", "min LR"),
    ("warmup_steps", "warmup steps"),
    ("max_steps", "max steps"),
    ("weight_decay", "weight decay"),
    ("grad_clip", "grad clip"),
    ("precision", "precision"),
    ("optimizer", "optimizer"),
    ("lr_schedule", "LR schedule"),
    ("z_loss", "z-loss"),
    ("gradient_checkpointing", "grad checkpointing"),
]
_DATA_KEYS: list[tuple[str, str]] = [
    ("dataset", "dataset"),
    ("languages", "languages"),
    ("max_tokens_per_file", "max tokens/file"),
    ("num_workers", "workers"),
    ("fim_rate", "FIM rate"),
]
_CHECKPOINT_KEYS: list[tuple[str, str]] = [
    ("save_every", "save every"),
    ("output_dir", "output dir"),
    ("max_checkpoints", "max checkpoints"),
]


def _fmt(value: object) -> str:
    """Coerce a YAML scalar/list to a compact display string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _group(title: str, section: object, keys: list[tuple[str, str]]) -> dict | None:
    """Build one ``{title, items}`` group from a config section, or None if empty."""
    if not isinstance(section, dict):
        return None
    items = [
        {"label": label, "value": _fmt(section[key])}
        for key, label in keys
        if key in section and section[key] is not None
    ]
    return {"title": title, "items": items} if items else None


def config_summary(config_path: str) -> dict:
    """Return a grouped hyperparameter summary of the YAML config at ``config_path``.

    ``{"path", "name", "exists", "groups": [{"title", "items": [{"label","value"}]}]}``.
    A missing file yields ``exists=False`` with empty groups (not an error); a
    malformed YAML returns ``{"error": str}``. Never raises.
    """
    path = Path(config_path)
    name = path.name
    if not path.is_file():
        return {"path": str(path), "name": name, "exists": False, "groups": []}

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {"error": f"could not read {path}: {exc}"}
    if not isinstance(parsed, dict):
        return {"error": f"{path} is not a YAML mapping"}

    groups: list[dict] = []
    for title, section_key, keys in (
        ("Model", "model", _MODEL_KEYS),
        ("Training", "training", _TRAINING_KEYS),
        ("Data", "data", _DATA_KEYS),
        ("Checkpoint", "checkpoint", _CHECKPOINT_KEYS),
    ):
        group = _group(title, parsed.get(section_key), keys)
        if group is not None:
            groups.append(group)

    # Derived: effective batch = batch_size × gradient_accumulation (both present).
    training = parsed.get("training")
    if isinstance(training, dict):
        bs = training.get("batch_size")
        ga = training.get("gradient_accumulation")
        if isinstance(bs, int) and isinstance(ga, int):
            groups.append(
                {
                    "title": "Derived",
                    "items": [{"label": "effective batch", "value": str(bs * ga)}],
                }
            )

    return {"path": str(path), "name": name, "exists": True, "groups": groups}
