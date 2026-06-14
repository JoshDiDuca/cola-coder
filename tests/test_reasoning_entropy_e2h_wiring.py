"""Regression: the entropy clip controller (IDEA-013/020) and E2H scheduler
(MODEL-042) must actually reach the GRPO trainer from train_reasoning.py — not be
implemented-but-unreachable. Guards against the features going phantom.
"""

import ast
import inspect
from pathlib import Path

from cola_coder.reasoning.curriculum_scheduler import VerifierEffortCurriculum
from cola_coder.reasoning.entropy_controller import EntropyClipController
from cola_coder.reasoning.grpo import GRPOTrainer

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_reasoning.py"


def test_trainer_accepts_the_features():
    init_params = inspect.signature(GRPOTrainer.__init__).parameters
    assert "entropy_clip_controller" in init_params
    train_params = inspect.signature(GRPOTrainer.train).parameters
    assert "e2h_scheduler" in train_params


def test_script_defines_the_flags():
    text = SCRIPT.read_text(encoding="utf-8")
    for flag in ("--entropy-control", "--entropy-target", "--e2h"):
        assert flag in text, f"{flag} not defined in train_reasoning.py"


def test_script_passes_features_to_trainer():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    trainer_call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "GRPOTrainer"
    )
    assert "entropy_clip_controller" in {kw.arg for kw in trainer_call.keywords}

    # The .train(...) call must forward e2h_scheduler.
    train_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "train"
    ]
    assert any(
        "e2h_scheduler" in {kw.arg for kw in c.keywords} for c in train_calls
    ), "train_reasoning.py does not forward e2h_scheduler to .train()"


def test_curriculum_floor_preset_is_valid():
    # The preset the script uses with --entropy-control + --curriculum must build.
    c = EntropyClipController(
        target_entropy=0.7, clip_low=0.2, clip_high=0.28, max_clip_high=0.48,
        difficulty_floors={"easy": 0.3, "medium": 0.6, "hard": 1.0},
    )
    assert c.floor_for("hard") == 1.0
    assert c.floor_for("easy") == 0.3
    # And the E2H scheduler builds with defaults.
    assert VerifierEffortCurriculum().min_active >= 1
