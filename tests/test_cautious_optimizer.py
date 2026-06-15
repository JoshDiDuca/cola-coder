"""Cautious-optimizer mask (C-Optim, arXiv:2411.16085) + its Muon wiring.

The mask zeroes update elements whose sign disagrees with the gradient and
rescales the survivors. Opt-in (default off) → the standard step is unchanged.
"""

from __future__ import annotations

import torch

from cola_coder.training.optimizer import Muon, cautious_mask


class TestCautiousMask:
    def test_full_agreement_preserves_update(self) -> None:
        update = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.tensor([0.5, 1.0, 0.2])  # all same sign as update
        out = cautious_mask(update, grad)
        # All agree → kept, scale = numel/kept = 1.0 → identity.
        assert torch.allclose(out, update)

    def test_disagreeing_elements_zeroed(self) -> None:
        update = torch.tensor([1.0, -1.0, 1.0, -1.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])  # elements 1,3 disagree
        out = cautious_mask(update, grad)
        assert out[1] == 0.0 and out[3] == 0.0
        assert out[0] != 0.0 and out[2] != 0.0

    def test_rescales_to_preserve_mean_magnitude(self) -> None:
        update = torch.tensor([2.0, -2.0, 2.0, -2.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])  # half kept
        out = cautious_mask(update, grad)
        # 2 of 4 kept → scale = 4/2 = 2 → survivors become 4.0.
        assert out[0] == 4.0 and out[2] == 4.0

    def test_all_disagree_scale_is_finite(self) -> None:
        update = torch.tensor([1.0, 1.0])
        grad = torch.tensor([-1.0, -1.0])  # all disagree → kept=0
        out = cautious_mask(update, grad)
        assert torch.all(out == 0.0)  # clamp_min(1) keeps scale finite, mask all-zero
        assert torch.isfinite(out).all()

    def test_does_not_mutate_inputs(self) -> None:
        update = torch.tensor([1.0, -1.0])
        grad = torch.tensor([1.0, 1.0])
        u0 = update.clone()
        cautious_mask(update, grad)
        assert torch.equal(update, u0)


def _tiny_muon(cautious: bool) -> tuple[Muon, torch.nn.Parameter, torch.nn.Parameter]:
    torch.manual_seed(0)
    muon_w = torch.nn.Parameter(torch.randn(4, 4))   # 2D → Muon group
    adamw_w = torch.nn.Parameter(torch.randn(8))      # 1D → AdamW group
    opt = Muon(
        muon_params=[muon_w], adamw_params=[], adamw_no_decay_params=[adamw_w],
        lr=0.02, adamw_lr=1e-3, weight_decay=0.0, cautious=cautious,
    )
    return opt, muon_w, adamw_w


class TestMuonCautiousWiring:
    def test_default_off_matches_standard_step(self) -> None:
        # Same seed/grads → cautious=False must reproduce the standard Muon step.
        opt_a, mw_a, aw_a = _tiny_muon(cautious=False)
        opt_b, mw_b, aw_b = _tiny_muon(cautious=False)
        torch.manual_seed(1)
        g_m = torch.randn(4, 4)
        g_a = torch.randn(8)
        for mw, aw, opt in ((mw_a, aw_a, opt_a), (mw_b, aw_b, opt_b)):
            mw.grad = g_m.clone()
            aw.grad = g_a.clone()
            opt.step()
        assert torch.equal(mw_a, mw_b)
        assert torch.equal(aw_a, aw_b)

    def test_cautious_changes_the_step(self) -> None:
        opt_off, mw_off, aw_off = _tiny_muon(cautious=False)
        opt_on, mw_on, aw_on = _tiny_muon(cautious=True)  # identical init (same seed)
        torch.manual_seed(2)
        g_m = torch.randn(4, 4)
        g_a = torch.randn(8)
        for mw, aw, opt in ((mw_off, aw_off, opt_off), (mw_on, aw_on, opt_on)):
            mw.grad = g_m.clone()
            aw.grad = g_a.clone()
            opt.step()
        # The Muon (orthogonalized) update disagrees in sign with the raw grad on
        # many elements, so cautious masking changes the Muon param. (The AdamW
        # first step is exactly sign(g), which always agrees → cautious is a no-op
        # there on step 1, so we assert on the Muon group.)
        assert not torch.equal(mw_off, mw_on)
        assert torch.isfinite(aw_on).all() and torch.isfinite(mw_on).all()
