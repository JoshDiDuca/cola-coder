"""BUG-114: a non-finite (NaN/Inf) loss must NEVER update the weights.

bf16 — the RTX 4080 primary precision — runs with GradScaler(enabled=False), so
unlike fp16 there is no scaler to skip the optimizer step on inf/NaN gradients.
Both training loops (trainer.py pretraining, train_sft.py SFT) now guard the
backward with `torch.isfinite(loss)`.

Two layers of coverage:
1. Behavioral — the skip pattern (don't backward, don't step on a non-finite
   loss) provably leaves weights unchanged, while a finite loss updates them.
2. Static — both train loops actually contain the guard BEFORE their backward
   call, so the protection can't silently regress.
"""

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


class TestSkipPatternLeavesWeightsUnchanged:
    """Validates the safety property the guard relies on."""

    def _tiny(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(8, 4)
        return model, opt, x

    def test_finite_loss_updates_weights(self):
        model, opt, x = self._tiny()
        before = model.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()  # finite
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        assert not torch.equal(model.weight.detach(), before)  # changed

    def test_nan_loss_leaves_weights_unchanged(self):
        model, opt, x = self._tiny()
        before = model.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        loss = model(x).mean() * float("nan")  # non-finite
        # The guard: skip backward + step entirely.
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
        assert torch.equal(model.weight.detach(), before)  # untouched

    def test_stepping_on_nan_grad_WOULD_corrupt(self):
        # Control: WITHOUT the guard, a NaN loss poisons every weight — this is
        # exactly what the guard prevents.
        model, opt, x = self._tiny()
        opt.zero_grad(set_to_none=True)
        loss = model(x).mean() * float("nan")
        loss.backward()
        opt.step()
        assert torch.isnan(model.weight.detach()).any()


def _guards_backward(source: str) -> bool:
    """True if `torch.isfinite(loss)` appears before a `.backward()` call."""
    finite_idx = source.find("torch.isfinite(loss)")
    backward_idx = source.find(".backward()")
    return finite_idx != -1 and backward_idx != -1 and finite_idx < backward_idx


class TestGuardWiredInBothLoops:
    def test_trainer_guards_backward(self):
        src = (ROOT / "src" / "cola_coder" / "training" / "trainer.py").read_text(
            encoding="utf-8"
        )
        assert _guards_backward(src)
        # The all-non-finite step skip is present too.
        assert "nonfinite_micro" in src

    def test_train_sft_guards_backward(self):
        src = (ROOT / "scripts" / "train_sft.py").read_text(encoding="utf-8")
        assert _guards_backward(src)
        assert "nonfinite_batches" in src

    def test_both_modules_still_parse(self):
        # Guard insertions must not break syntax.
        for rel in ("src/cola_coder/training/trainer.py", "scripts/train_sft.py"):
            ast.parse((ROOT / rel).read_text(encoding="utf-8"))
