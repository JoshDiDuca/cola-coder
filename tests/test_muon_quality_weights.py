"""IDEA-019: quality-weights x Muon interaction.

Muon orthogonalizes the update (Newton-Schulz), discarding gradient MAGNITUDE for
the 2D weight matrices. The correctness question for a quality-weighted run: do
per-sample quality weights still influence the Muon update, or are they neutered?

Validated here: RELATIVE in-batch per-sample weights reshape the averaged-gradient
DIRECTION, so they DO change the Muon update (not neutered) — provided they are
applied as a weighted MEAN over per-sample losses (which `language_modeling_loss`
does), not as a global loss multiplier.

NUANCE (documented, not asserted): in exact arithmetic a GLOBAL loss scale would be
fully washed out by orthogonalization. In practice Newton-Schulz runs in bf16, so
a large global scale (e.g. 1000x) shifts the rounding enough to perturb the update
by an amount comparable to a real reweight at this tiny test scale. The practical
guidance is unchanged: keep quality weights RELATIVE (weighted mean), never a global
factor — the relative-weighting path is the one validated to survive Muon below.

These guard the assumption before defaulting to Muon (MODEL-025). Tests-only — they
do not modify optimizer.py or any train-path file.
"""

from __future__ import annotations

import copy

import torch

from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer, language_modeling_loss
from cola_coder.training.optimizer import create_optimizer

# Two DISTINCT sequences so the per-sample gradients (and thus a reweighting)
# actually differ.
_X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                   [40, 39, 38, 37, 36, 35, 34, 33]])


def _base_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
        max_seq_len=32, dropout=0.0,
    ))


def _muon_weight(model: Transformer) -> torch.Tensor:
    """The first 2D in-block weight — exactly what create_optimizer routes to Muon."""
    return next(p for n, p in model.named_parameters()
                if p.dim() == 2 and "blocks." in n)


def _muon_update(base: Transformer, weights: torch.Tensor) -> torch.Tensor:
    """Run one Muon step from a copy of `base` with the given sample weights; return Δweight."""
    model = copy.deepcopy(base)
    opt = create_optimizer(model, optimizer="muon", muon_lr=0.02)
    p0 = _muon_weight(model).detach().clone()
    opt.zero_grad()
    language_modeling_loss(model(_X), _X, sample_weights=weights).backward()
    opt.step()
    return _muon_weight(model).detach() - p0


class TestQualityWeightsUnderMuon:
    def test_relative_weights_change_muon_update(self):
        # The key claim: skewed in-batch weights vs uniform produce a DIFFERENT
        # Muon update. Both runs are at loss_scale 1.0, so the entire difference
        # is the reweighting (no global-scale rounding confound) — proving quality
        # weighting survives orthogonalization.
        base = _base_model()
        d_uniform = _muon_update(base, torch.tensor([1.0, 1.0]))
        d_skewed = _muon_update(base, torch.tensor([1.0, 9.0]))
        assert (d_skewed - d_uniform).norm() > 1e-3, (
            "quality weights had ~no effect on the Muon update — neutered by "
            "orthogonalization"
        )

    def test_stronger_skew_moves_update_more(self):
        # Monotonicity check: a heavier skew departs further from uniform than a
        # light one — the effect tracks the weighting, it isn't noise.
        base = _base_model()
        d_uniform = _muon_update(base, torch.tensor([1.0, 1.0]))
        d_light = _muon_update(base, torch.tensor([1.0, 2.0]))
        d_heavy = _muon_update(base, torch.tensor([1.0, 50.0]))
        assert (d_heavy - d_uniform).norm() > (d_light - d_uniform).norm()

    def test_adamw_baseline_relative_weights_also_matter(self):
        # Sanity: the same skew also moves an AdamW run, so the effect isn't a
        # Muon artifact — quality weighting is a real, optimizer-agnostic signal.
        model_u = _base_model()
        model_s = copy.deepcopy(model_u)
        opt_u = create_optimizer(model_u, optimizer="adamw", learning_rate=1e-3)
        opt_s = create_optimizer(model_s, optimizer="adamw", learning_rate=1e-3)
        for model, opt, w in ((model_u, opt_u, [1.0, 1.0]), (model_s, opt_s, [1.0, 9.0])):
            opt.zero_grad()
            language_modeling_loss(model(_X), _X,
                                   sample_weights=torch.tensor(w)).backward()
            opt.step()
        assert not torch.allclose(_muon_weight(model_u), _muon_weight(model_s), atol=1e-6)
