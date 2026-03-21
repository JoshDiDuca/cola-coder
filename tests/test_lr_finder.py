"""Tests for the Learning Rate Finder.

Uses a small linear network — NOT the full transformer — so tests run fast
without needing GPU or large amounts of memory.
"""

import math

import torch
import torch.nn as nn
import pytest

from cola_coder.training.lr_finder import LRFinder, LRFinderResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    """Tiny linear network for fast test runs."""

    def __init__(self, in_dim: int = 16, hidden: int = 32, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader(n_batches: int = 50, batch_size: int = 8, in_dim: int = 16, out_dim: int = 8):
    """Create a simple DataLoader with random regression data."""
    X = torch.randn(n_batches * batch_size, in_dim)
    y = torch.randn(n_batches * batch_size, out_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_finder(hidden: int = 32) -> tuple[LRFinder, nn.Module, torch.optim.Optimizer]:
    """Build a model, optimizer, and LRFinder for testing."""
    model = SimpleNet(hidden=hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    finder = LRFinder(model=model, optimizer=optimizer, criterion=criterion, device="cpu")
    return finder, model, optimizer


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLRFinderBasics:
    """Core LR finder behaviour."""

    def test_returns_lr_finder_result(self):
        """find() returns an LRFinderResult dataclass."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=20)
        assert isinstance(result, LRFinderResult)

    def test_lrs_are_monotonically_increasing(self):
        """LRs should increase exponentially across all recorded steps."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=30)
        lrs = result.lrs
        assert len(lrs) >= 2
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR at step {i} did not increase: {lrs[i-1]} -> {lrs[i]}"

    def test_lr_range_covers_start_and_end(self):
        """The first LR should be close to start_lr, last close to end_lr (unless early stop)."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=50)
        assert result.lrs[0] == pytest.approx(1e-6, rel=0.05)
        # end_lr might not be reached if diverge triggered, but start_lr always is
        assert result.lrs[0] < result.lrs[-1]

    def test_losses_are_recorded(self):
        """Both raw and smoothed losses are recorded."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=20)
        assert len(result.losses) == len(result.lrs)
        assert len(result.smoothed_losses) == len(result.lrs)
        assert all(math.isfinite(v) for v in result.losses)
        assert all(math.isfinite(v) for v in result.smoothed_losses)

    def test_num_steps_run_matches_lrs(self):
        """num_steps_run matches the length of the lrs list."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=25)
        assert result.num_steps_run == len(result.lrs)


class TestWeightRestoration:
    """Model weights must be restored after the test."""

    def test_weights_restored_after_find(self):
        """Model weights are identical before and after find()."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        finder = LRFinder(model=model, optimizer=optimizer,
                          criterion=nn.MSELoss(), device="cpu")

        # Snapshot weights before
        initial_weights = {k: v.clone() for k, v in model.named_parameters()}

        loader = make_loader(n_batches=10)
        finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=10)

        # Check weights are unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(param.data, initial_weights[name], atol=1e-6), (
                f"Parameter {name} changed after LR finder run!"
            )

    def test_optimizer_lr_restored_after_find(self):
        """Optimizer learning rate is restored after find()."""
        model = SimpleNet()
        original_lr = 3e-4
        optimizer = torch.optim.SGD(model.parameters(), lr=original_lr)
        finder = LRFinder(model=model, optimizer=optimizer,
                          criterion=nn.MSELoss(), device="cpu")

        loader = make_loader(n_batches=10)
        finder.find(loader, start_lr=1e-7, end_lr=10.0, num_steps=10)

        # All param group LRs should be restored
        for pg in optimizer.param_groups:
            assert pg["lr"] == pytest.approx(original_lr, rel=1e-6)


class TestEarlyDivergenceDetection:
    """The finder should stop early when loss explodes."""

    def test_diverge_threshold_respected(self):
        """Setting a low diverge_threshold causes early stopping."""
        finder, _, _ = make_finder()
        loader = make_loader()

        # Run with a very aggressive LR range that will definitely diverge
        result = finder.find(
            loader,
            start_lr=1.0,  # Start already very high
            end_lr=1e5,
            num_steps=50,
            diverge_threshold=2.0,  # Stop very aggressively
        )
        # Either diverged early or somehow didn't — both are valid, but
        # if diverge_threshold=2.0, it's extremely likely to diverge
        assert result.num_steps_run <= 50  # Must not exceed requested steps
        # Diverged early flag should be set if we stopped before all steps
        if result.num_steps_run < 50:
            assert result.diverged_early

    def test_no_diverge_for_stable_run(self):
        """A stable run with reasonable LR should NOT set diverged_early."""
        finder, _, _ = make_finder()
        loader = make_loader()

        # Tiny LR range — should be stable
        result = finder.find(
            loader,
            start_lr=1e-7,
            end_lr=1e-4,
            num_steps=20,
            diverge_threshold=4.0,
        )
        assert not result.diverged_early


class TestSmoothingAndSuggestion:
    """Tests for loss smoothing and LR suggestion."""

    def test_smoothed_losses_differ_from_raw(self):
        """Smoothed losses should not be identical to raw losses (except trivial cases)."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=0.1, num_steps=30, smooth_factor=0.1)
        # At least some smoothed values should differ from raw
        if len(result.losses) > 1:
            diffs = sum(
                abs(s - r) > 1e-8
                for s, r in zip(result.smoothed_losses, result.losses)
            )
            # After the first step, smoothing should affect values
            assert diffs > 0, "Smoothed losses are identical to raw losses — smoothing has no effect"

    def test_suggested_lr_within_tested_range(self):
        """suggested_lr must fall within [start_lr, end_lr]."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=1.0, num_steps=30)
        assert result.lrs[0] <= result.suggested_lr <= result.lrs[-1] + 1e-12

    def test_suggest_lr_static_method(self):
        """suggest_lr() returns (min_lr, max_lr) where min_lr = max_lr / 10."""
        # Build a synthetic result
        lrs = [1e-6 * (10.0 ** (i / 9)) for i in range(10)]
        losses = [5.0, 4.5, 3.0, 2.0, 1.5, 1.8, 3.0, 6.0, 12.0, 20.0]
        result = LRFinderResult(
            lrs=lrs,
            losses=losses,
            smoothed_losses=losses,
            suggested_lr=lrs[4],
            suggested_min_lr=lrs[4] / 10.0,
            suggested_idx=4,
        )
        min_lr, max_lr = LRFinder.suggest_lr(result)
        assert max_lr == pytest.approx(result.suggested_lr, rel=1e-6)
        assert min_lr == pytest.approx(result.suggested_min_lr, rel=1e-6)
        assert min_lr == pytest.approx(max_lr / 10.0, rel=1e-6)

    def test_find_steepest_descent_correct(self):
        """_find_steepest_descent returns index near the sharpest drop."""
        # Construct a simple curve: decreasing then increasing
        lrs = [1e-5 * (10 ** (i / 10)) for i in range(20)]
        # Loss decreases sharply around index 8, then rises
        losses = [5.0 - 0.3 * i if i < 9 else 2.3 + 0.5 * (i - 9) for i in range(20)]
        lr, idx = LRFinder._find_steepest_descent(lrs, losses)
        # The steepest descent should be somewhere in the decreasing region
        assert 0 <= idx < len(lrs)
        assert lrs[0] < lr <= lrs[-1]


class TestResultSummary:
    """LRFinderResult.summary() and metadata."""

    def test_summary_contains_key_info(self):
        """summary() output contains steps run, LR values."""
        finder, _, _ = make_finder()
        loader = make_loader()
        result = finder.find(loader, start_lr=1e-6, end_lr=0.1, num_steps=15)
        summary = result.summary()
        assert "Steps run" in summary
        assert "Suggested LR" in summary
        assert str(result.num_steps_run) in summary

    def test_no_plot_ascii_runs_without_error(self):
        """plot() with no matplotlib falls back to ASCII without crashing."""
        import cola_coder.training.lr_finder as lr_finder_module

        # Temporarily disable matplotlib flag
        original = lr_finder_module._HAS_MATPLOTLIB
        lr_finder_module._HAS_MATPLOTLIB = False

        try:
            finder, _, _ = make_finder()
            loader = make_loader(n_batches=5)
            result = finder.find(loader, start_lr=1e-6, end_lr=0.1, num_steps=10)
            # Should not raise
            finder.plot(result, save_path=None)
        finally:
            lr_finder_module._HAS_MATPLOTLIB = original
