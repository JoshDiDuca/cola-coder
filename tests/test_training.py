"""Tests for the training pipeline components."""

import math
import pytest
import torch

from cola_coder.training.optimizer import create_optimizer, create_scheduler
from cola_coder.training.metrics import TrainingMetrics
from cola_coder.model.config import ModelConfig
from cola_coder.model.transformer import Transformer


def make_test_model():
    config = ModelConfig(
        vocab_size=256, dim=64, n_layers=2,
        n_heads=4, n_kv_heads=2, max_seq_len=64,
    )
    return Transformer(config)


class TestOptimizer:
    """Tests for optimizer and scheduler creation."""

    def test_create_optimizer(self):
        model = make_test_model()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        assert len(optimizer.param_groups) == 2  # decay and no-decay groups

    def test_weight_decay_groups(self):
        """Weight decay is only applied to appropriate parameters."""
        model = make_test_model()
        optimizer = create_optimizer(model, weight_decay=0.1)

        # First group: with decay
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        # Second group: without decay
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_optimizer_step(self):
        """Optimizer can take a step without errors."""
        model = make_test_model()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        token_ids = torch.randint(0, 256, (2, 16))
        loss = model.compute_loss(token_ids)
        loss.backward()
        optimizer.step()  # Should not raise


class TestScheduler:
    """Tests for learning rate scheduler."""

    def test_warmup(self):
        """LR increases linearly during warmup."""
        model = make_test_model()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(optimizer, warmup_steps=100, max_steps=1000)

        lrs = []
        for step in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # LR should be increasing during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decay(self):
        """LR decreases after warmup."""
        model = make_test_model()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer, warmup_steps=10, max_steps=100, min_lr_ratio=0.1
        )

        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        lr_at_warmup_end = scheduler.get_last_lr()[0]

        # Continue training
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        lr_after_decay = scheduler.get_last_lr()[0]
        assert lr_after_decay < lr_at_warmup_end

    def test_min_lr_floor(self):
        """LR doesn't go below the minimum."""
        model = make_test_model()
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer, warmup_steps=10, max_steps=100, min_lr_ratio=0.1
        )

        for _ in range(200):  # Go past max_steps
            optimizer.step()
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr >= 1e-3 * 0.1 * 0.99  # Allow small floating point error


class TestMetrics:
    """Tests for training metrics tracking."""

    def test_update_and_log(self):
        metrics = TrainingMetrics()
        metrics.update(loss=2.5, num_tokens=1000)
        metrics.update(loss=2.0, num_tokens=1000)

        # Should return None for non-logging steps
        assert metrics.log(step=1, lr=1e-3, log_interval=100) is None

        # Should return a string at logging interval
        msg = metrics.log(step=100, lr=1e-3, log_interval=100)
        assert msg is not None
        assert "loss" in msg
        assert "lr" in msg

    def test_history_tracking(self):
        metrics = TrainingMetrics()
        for i in range(100):
            metrics.update(loss=2.0 - i * 0.01, num_tokens=1000)

        metrics.log(step=100, lr=1e-3, log_interval=100)
        assert len(metrics.loss_history) == 1
        assert len(metrics.lr_history) == 1
