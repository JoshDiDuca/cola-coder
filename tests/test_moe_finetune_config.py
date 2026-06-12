"""MODEL-003: derive_moe_finetune_config — short, low-LR recipe for upcycled MoE.

Upcycling copies the dense FFN into every expert (identical at first); a short
low-LR fine-tune differentiates them. This helper rescales only the training
section so `train.py --resume <moe_dir> --config <derived>` runs that recipe.
"""

import pytest

from cola_coder.model.config import derive_moe_finetune_config


def _cfg(**training):
    base = {"learning_rate": 3.0e-4, "min_lr": 3.0e-5, "max_steps": 20000, "warmup_steps": 500}
    base.update(training)
    return {"model": {"dim": 256, "moe": {"enabled": True, "num_experts": 8}}, "training": base}


class TestScaling:
    def test_lr_and_steps_scaled(self):
        out = derive_moe_finetune_config(_cfg(), lr_fraction=0.1, step_fraction=0.15)
        assert out["training"]["learning_rate"] == pytest.approx(3.0e-5)
        assert out["training"]["max_steps"] == 3000  # round(20000 * 0.15)

    def test_min_lr_stays_below_peak(self):
        # Base min_lr (3e-5) exceeds the new peak (3e-5*... ) in aggressive cases;
        # must be clamped below the lowered learning rate.
        out = derive_moe_finetune_config(_cfg(), lr_fraction=0.1, step_fraction=0.15)
        assert out["training"]["min_lr"] <= out["training"]["learning_rate"]
        assert out["training"]["min_lr"] == pytest.approx(3.0e-6)  # new_lr * 0.1

    def test_warmup_short_relative_to_schedule(self):
        out = derive_moe_finetune_config(_cfg(), step_fraction=0.15)
        # min(base_warmup=500, round(3000*0.05)=150) = 150
        assert out["training"]["warmup_steps"] == 150

    def test_defaults_when_training_keys_missing(self):
        out = derive_moe_finetune_config({"model": {}}, lr_fraction=0.1, step_fraction=0.1)
        # Falls back to documented defaults (3e-4 lr, 20000 steps).
        assert out["training"]["learning_rate"] == pytest.approx(3.0e-5)
        assert out["training"]["max_steps"] == 2000


class TestNonMutationAndPreservation:
    def test_input_not_mutated(self):
        cfg = _cfg()
        before_lr = cfg["training"]["learning_rate"]
        derive_moe_finetune_config(cfg)
        assert cfg["training"]["learning_rate"] == before_lr  # deep-copied

    def test_model_and_moe_block_preserved(self):
        out = derive_moe_finetune_config(_cfg())
        assert out["model"]["moe"] == {"enabled": True, "num_experts": 8}
        assert out["model"]["dim"] == 256


class TestOutputDirIsolation:
    """The fine-tune resumes from the upcycled MoE dir but must NOT save over the
    dense base checkpoint that the base config's output_dir still points at."""

    def _cfg_with_ckpt(self, output_dir):
        c = _cfg()
        c["checkpoint"] = {"output_dir": output_dir}
        return c

    def test_output_dir_redirected_to_moe_ft(self):
        out = derive_moe_finetune_config(self._cfg_with_ckpt("./checkpoints/4080_max"))
        assert out["checkpoint"]["output_dir"] == "./checkpoints/4080_max_moe_ft"

    def test_output_dir_differs_from_base(self):
        base = "./checkpoints/4080_max"
        out = derive_moe_finetune_config(self._cfg_with_ckpt(base))
        assert out["checkpoint"]["output_dir"] != base  # never clobbers the dense base

    def test_trailing_slash_handled(self):
        out = derive_moe_finetune_config(self._cfg_with_ckpt("./checkpoints/small/"))
        assert out["checkpoint"]["output_dir"] == "./checkpoints/small_moe_ft"

    def test_missing_checkpoint_section_gets_default(self):
        out = derive_moe_finetune_config(_cfg())  # no checkpoint key
        assert out["checkpoint"]["output_dir"] == "./checkpoints/model_moe_ft"

    def test_input_checkpoint_not_mutated(self):
        cfg = self._cfg_with_ckpt("./checkpoints/4080_max")
        derive_moe_finetune_config(cfg)
        assert cfg["checkpoint"]["output_dir"] == "./checkpoints/4080_max"


class TestValidation:
    def test_bad_lr_fraction_raises(self):
        with pytest.raises(ValueError, match="lr_fraction"):
            derive_moe_finetune_config(_cfg(), lr_fraction=0.0)
        with pytest.raises(ValueError, match="lr_fraction"):
            derive_moe_finetune_config(_cfg(), lr_fraction=1.5)

    def test_bad_step_fraction_raises(self):
        with pytest.raises(ValueError, match="step_fraction"):
            derive_moe_finetune_config(_cfg(), step_fraction=0.0)

    def test_max_steps_never_zero(self):
        # Tiny base + tiny fraction must still yield at least 1 step.
        out = derive_moe_finetune_config(_cfg(max_steps=3), step_fraction=0.1)
        assert out["training"]["max_steps"] >= 1
