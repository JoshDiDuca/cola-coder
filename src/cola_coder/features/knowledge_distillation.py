"""Knowledge Distillation: train a smaller student model to match a larger teacher model.

The core idea: instead of only learning from hard labels (correct/incorrect), the student
also learns from the teacher's *soft* probability distribution over all tokens. The teacher's
softmax output contains rich relational information — e.g. "this token is 40% likely, that
similar token is 35% likely" — that hard labels throw away.

For a TS dev: think of it like type inference. Hard labels are `string | number`, but the
teacher's soft distribution is the full generic type `T extends ...` with all the nuance.

The loss combines:
  - CE loss:  student vs ground-truth labels (standard next-token prediction)
  - KD loss:  KL divergence between soft teacher and soft student at high temperature

Temperature T > 1 "softens" the distributions, making small probabilities more visible and
giving the student a richer gradient signal.

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether knowledge distillation is enabled."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training.

    Args:
        temperature: Softening temperature applied to logits before KL divergence.
            Higher values (e.g. 4-10) produce softer distributions and expose more
            inter-class similarity signal to the student.
        alpha: Weight of the KD loss relative to CE loss.
            total = alpha * kd_loss + (1 - alpha) * ce_loss
            alpha=0.5 balances both equally; alpha=1.0 uses only KD loss.
        teacher_frozen: Whether to freeze teacher parameters on trainer init.
            Almost always True — we never want to update the teacher.
    """

    temperature: float = 4.0
    alpha: float = 0.5
    teacher_frozen: bool = True


# ---------------------------------------------------------------------------
# Core loss functions
# ---------------------------------------------------------------------------


def soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL-divergence-based soft cross-entropy between student and teacher distributions.

    Applies temperature scaling to both logits, then computes KL(teacher || student).
    The T^2 scaling factor compensates for the magnitude reduction caused by dividing
    logits by T, ensuring gradient magnitudes stay comparable across temperatures.

    Args:
        student_logits: Shape (N, vocab_size) — raw student outputs.
        teacher_logits: Shape (N, vocab_size) — raw teacher outputs.
        temperature:    Softening temperature > 1.

    Returns:
        Scalar KD loss (mean over batch).
    """
    T = temperature
    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    log_probs = F.log_softmax(student_logits / T, dim=-1)
    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * logP) - sum(P * logQ)
    # F.kl_div expects log-probabilities for input, probabilities for target.
    kd_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")
    # Rescale by T^2 so the gradient magnitude is temperature-invariant.
    return kd_loss * (T ** 2)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: DistillationConfig,
) -> torch.Tensor:
    """Combined CE + KD loss for knowledge distillation.

    Args:
        student_logits: Shape (batch, seq_len, vocab_size).
        teacher_logits: Shape (batch, seq_len, vocab_size).
        labels:         Shape (batch, seq_len) — integer token ids.
        config:         DistillationConfig controlling temperature and alpha.

    Returns:
        Scalar combined loss: alpha * kd_loss + (1 - alpha) * ce_loss.
    """
    batch, seq_len, vocab_size = student_logits.shape

    # Flatten to (batch*seq_len, vocab_size) for loss functions
    s_flat = student_logits.reshape(-1, vocab_size)
    t_flat = teacher_logits.reshape(-1, vocab_size)
    l_flat = labels.reshape(-1)

    # Standard cross-entropy against ground-truth labels
    ce_loss = F.cross_entropy(s_flat, l_flat)

    # Soft KD loss against teacher soft targets
    kd_loss = soft_cross_entropy(s_flat, t_flat, config.temperature)

    # Weighted combination
    total = config.alpha * kd_loss + (1.0 - config.alpha) * ce_loss
    return total


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DistillationTrainer:
    """Wrapper that pairs a teacher and student model for distillation training.

    The trainer does *not* own an optimizer — that belongs to the outer training
    loop. It only handles the forward pass and loss computation so it can slot
    into any existing training harness.

    Usage (TS analogy):
        // Like a proxy that intercepts student.forward(), adds the teacher signal,
        // and returns an enriched loss object instead of a plain scalar.

    Args:
        teacher: Frozen (or to-be-frozen) larger model.
        student: Smaller model being trained.
        config:  DistillationConfig.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()

        if self.config.teacher_frozen:
            self.freeze_teacher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeze_teacher(self) -> None:
        """Set all teacher parameters to requires_grad=False."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass for both models and compute distillation losses.

        The teacher runs inside torch.no_grad() regardless of freeze state,
        since we never need teacher gradients.

        Args:
            input_ids: Shape (batch, seq_len) — token ids fed to both models.
            labels:    Shape (batch, seq_len) — ground-truth token ids for CE loss.

        Returns:
            Dict with keys:
                'total'   — combined alpha*KD + (1-alpha)*CE loss (scalar)
                'ce_loss' — cross-entropy component (scalar)
                'kd_loss' — KL divergence component (scalar)
        """
        # Teacher forward — no gradients needed
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            teacher_logits = self._extract_logits(teacher_out)

        # Student forward — gradients flow normally
        student_out = self.student(input_ids)
        student_logits = self._extract_logits(student_out)

        vocab_size = student_logits.shape[-1]
        s_flat = student_logits.reshape(-1, vocab_size)
        t_flat = teacher_logits.reshape(-1, vocab_size)
        l_flat = labels.reshape(-1)

        ce_loss = F.cross_entropy(s_flat, l_flat)
        kd_loss = soft_cross_entropy(s_flat, t_flat, self.config.temperature)
        total = self.config.alpha * kd_loss + (1.0 - self.config.alpha) * ce_loss

        return {"total": total, "ce_loss": ce_loss, "kd_loss": kd_loss}

    def summary(self) -> dict:
        """Return a human-readable summary of the distillation setup.

        Returns:
            Dict with parameter counts, config values, and freeze status.
        """
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_frozen = all(not p.requires_grad for p in self.teacher.parameters())

        return {
            "teacher_param_count": teacher_params,
            "student_param_count": student_params,
            "compression_ratio": round(teacher_params / max(student_params, 1), 2),
            "teacher_frozen": teacher_frozen,
            "temperature": self.config.temperature,
            "alpha": self.config.alpha,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_logits(model_output) -> torch.Tensor:
        """Extract a logit tensor from whatever the model returns.

        Handles:
          - Plain tensor (e.g. nn.Sequential ending in Linear)
          - Tuple/list — takes the first element
          - Object with .logits attribute (HuggingFace-style)
        """
        if isinstance(model_output, torch.Tensor):
            return model_output
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        if hasattr(model_output, "logits"):
            return model_output.logits
        raise TypeError(
            f"Cannot extract logits from model output of type {type(model_output)}. "
            "Expected Tensor, tuple/list, or object with .logits attribute."
        )
