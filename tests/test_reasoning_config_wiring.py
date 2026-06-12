"""Regression tests: every reasoning.yaml knob must reach train_reasoning.py.

History: the reasoning/problem_set sections of configs/reasoning.yaml were
once read by NOTHING — parallel_generation: true ran serially, kl_coeff was
a knob with no implementation behind it, max_thinking_tokens: 512 silently
became 256. A config value no code reads is a silent no-op; these tests make
that class of bug fail loudly.
"""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "train_reasoning.py"
CONFIG = ROOT / "configs" / "reasoning.yaml"

# Sections whose every key must be referenced by name in the script.
# (model/training/checkpoint/data are consumed by the generic Config loader
# and trainer machinery, not this script.)
_WIRED_SECTIONS = ("reasoning", "problem_set", "sft_warmup")

# Keyword arguments the GRPOTrainer(...) call in the script must pass so the
# resolved config values actually reach the trainer.
_REQUIRED_TRAINER_KWARGS = {
    "learning_rate",
    "group_size",
    "clip_epsilon",
    "clip_epsilon_high",
    "advantage_norm",
    "ppo_epochs",
    "max_thinking_tokens",
    "reward_fn",
    "parallel_generation",
    "parallel_rewards",
    "reward_workers",
}


def test_every_config_knob_is_read_by_the_script():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    script_text = SCRIPT.read_text(encoding="utf-8")

    unread = []
    for section in _WIRED_SECTIONS:
        for key in (cfg.get(section) or {}):
            if f'"{key}"' not in script_text and f"'{key}'" not in script_text:
                unread.append(f"{section}.{key}")

    assert not unread, (
        f"configs/reasoning.yaml keys read nowhere in train_reasoning.py: {unread} — "
        "either wire them (CLI > config > default) or delete them from the config."
    )


def test_grpo_trainer_call_passes_all_wireable_kwargs():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    trainer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GRPOTrainer"
    ]
    assert trainer_calls, "no GRPOTrainer(...) call found in train_reasoning.py"

    passed = {kw.arg for kw in trainer_calls[0].keywords if kw.arg}
    missing = _REQUIRED_TRAINER_KWARGS - passed
    assert not missing, (
        f"GRPOTrainer call in train_reasoning.py does not pass: {missing} — "
        "config values resolved but not passed are still silent no-ops."
    )


def test_config_has_no_phantom_grpo_knobs():
    # kl_coeff/reward_baseline were knobs with no implementation behind them.
    # If a KL term is ever implemented, wire it end-to-end and update this.
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    reasoning = cfg.get("reasoning") or {}
    assert "kl_coeff" not in reasoning
    assert "reward_baseline" not in reasoning
