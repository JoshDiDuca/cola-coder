"""Model Distillation Helper.

Utilities for knowledge distillation:
  - Temperature scaling to soften teacher logits
  - Soft label generation from teacher probabilities
  - KL divergence computation between teacher and student distributions
  - Teacher-student logit alignment metrics
  - Combined hard+soft loss computation

For a TS dev: this is like having a senior dev (teacher model) guide a
junior dev (student model) by sharing not just the right answer but also
how confident they are in each possible answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple


FEATURE_ENABLED = True


def is_enabled() -> bool:  # noqa: D401
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Core math utilities
# ---------------------------------------------------------------------------


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Compute softmax with optional temperature scaling.

    Parameters
    ----------
    logits:
        Raw unnormalised logits.
    temperature:
        T > 1 makes the distribution softer (more uniform).
        T < 1 makes it sharper (more peaked).
        T = 1 is standard softmax.
    """
    if not logits:
        return []
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    scaled = [v / temperature for v in logits]
    max_val = max(scaled)
    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Compute log-softmax with temperature scaling."""
    probs = softmax(logits, temperature)
    return [math.log(p + 1e-12) for p in probs]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Compute KL divergence KL(P || Q) = sum_i P_i log(P_i / Q_i).

    P and Q must be valid probability distributions (non-negative, sum to 1).
    """
    if len(p) != len(q):
        raise ValueError(f"Length mismatch: p={len(p)}, q={len(q)}")
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-12:
            kl += pi * math.log(pi / (qi + 1e-12))
    return max(0.0, kl)


def js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence, a symmetric version of KL.

    Range: [0, log(2)] in nats.
    """
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def cross_entropy(targets: list[float], log_probs: list[float]) -> float:
    """Compute cross-entropy H(targets, log_probs) = -sum_i targets_i * log_probs_i."""
    if len(targets) != len(log_probs):
        raise ValueError(f"Length mismatch: targets={len(targets)}, log_probs={len(log_probs)}")
    return -sum(t * lp for t, lp in zip(targets, log_probs) if t > 1e-12)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def scale_logits(logits: list[float], temperature: float) -> list[float]:
    """Scale logits by temperature (divide)."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return [v / temperature for v in logits]


def find_optimal_temperature(
    logits: list[list[float]],
    labels: list[int],
    temperatures: list[float] | None = None,
) -> float:
    """Find the temperature that minimises NLL on a small validation set.

    Parameters
    ----------
    logits:
        List of logit vectors, one per example.
    labels:
        Ground-truth class indices, one per example.
    temperatures:
        Candidate temperatures to try.  Defaults to a grid from 0.1 to 5.0.

    Returns
    -------
    The temperature with the lowest mean NLL.
    """
    if temperatures is None:
        temperatures = [0.1 * i for i in range(1, 51)]  # 0.1 to 5.0

    best_temp = 1.0
    best_nll = float("inf")

    for temp in temperatures:
        nll = 0.0
        for lgt, label in zip(logits, labels):
            log_p = log_softmax(lgt, temperature=temp)
            if 0 <= label < len(log_p):
                nll -= log_p[label]
            else:
                nll += 10.0  # penalty for invalid label
        mean_nll = nll / max(len(logits), 1)
        if mean_nll < best_nll:
            best_nll = mean_nll
            best_temp = temp

    return best_temp


# ---------------------------------------------------------------------------
# Soft labels
# ---------------------------------------------------------------------------


def generate_soft_labels(
    teacher_logits: list[list[float]],
    temperature: float = 2.0,
) -> list[list[float]]:
    """Convert teacher logits to soft probability distributions.

    Parameters
    ----------
    teacher_logits:
        Shape [batch_size, vocab_size].
    temperature:
        Softening temperature.  Higher values produce softer distributions.
    """
    return [softmax(logits, temperature=temperature) for logits in teacher_logits]


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """Metrics comparing teacher and student logit distributions."""

    mean_kl_divergence: float   # KL(teacher || student)
    mean_js_divergence: float
    top1_agreement: float       # fraction of examples where top-1 token matches
    top5_agreement: float       # fraction where true top-1 is in student's top-5
    mean_rank_diff: float       # average difference in top-1 rank between teacher/student

    def summary(self) -> str:
        return (
            f"AlignmentReport: KL={self.mean_kl_divergence:.4f}, "
            f"JS={self.mean_js_divergence:.4f}, "
            f"top1_agree={self.top1_agreement:.2%}, "
            f"top5_agree={self.top5_agreement:.2%}"
        )


def compute_alignment(
    teacher_logits: list[list[float]],
    student_logits: list[list[float]],
    temperature: float = 1.0,
) -> AlignmentReport:
    """Compute alignment between teacher and student distributions.

    Parameters
    ----------
    teacher_logits, student_logits:
        Shape [batch_size, vocab_size].
    temperature:
        Applied to both sets of logits before comparison.
    """
    if len(teacher_logits) != len(student_logits):
        raise ValueError("Batch sizes must match")
    if not teacher_logits:
        return AlignmentReport(0.0, 0.0, 0.0, 0.0, 0.0)

    kl_vals: list[float] = []
    js_vals: list[float] = []
    top1_matches = 0
    top5_matches = 0
    rank_diffs: list[float] = []

    for t_lgt, s_lgt in zip(teacher_logits, student_logits):
        t_prob = softmax(t_lgt, temperature)
        s_prob = softmax(s_lgt, temperature)

        kl_vals.append(kl_divergence(t_prob, s_prob))
        js_vals.append(js_divergence(t_prob, s_prob))

        t_top1 = max(range(len(t_prob)), key=lambda i: t_prob[i])
        s_top1 = max(range(len(s_prob)), key=lambda i: s_prob[i])
        top1_matches += int(t_top1 == s_top1)

        s_top5 = sorted(range(len(s_prob)), key=lambda i: -s_prob[i])[:5]
        top5_matches += int(t_top1 in s_top5)

        # Rank of teacher top-1 in student
        s_ranked = sorted(range(len(s_prob)), key=lambda i: -s_prob[i])
        try:
            s_rank = s_ranked.index(t_top1)
        except ValueError:
            s_rank = len(s_prob)
        rank_diffs.append(float(s_rank))

    n = len(teacher_logits)
    return AlignmentReport(
        mean_kl_divergence=round(sum(kl_vals) / n, 6),
        mean_js_divergence=round(sum(js_vals) / n, 6),
        top1_agreement=round(top1_matches / n, 4),
        top5_agreement=round(top5_matches / n, 4),
        mean_rank_diff=round(sum(rank_diffs) / n, 4),
    )


# ---------------------------------------------------------------------------
# Combined distillation loss
# ---------------------------------------------------------------------------


class DistillationLoss(NamedTuple):
    """Combined distillation loss components."""

    hard_loss: float       # standard cross-entropy with ground truth labels
    soft_loss: float       # KL divergence with teacher soft labels
    total_loss: float      # alpha * hard + (1 - alpha) * soft


def compute_distillation_loss(
    student_logits: list[float],
    teacher_logits: list[float],
    true_label: int,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> DistillationLoss:
    """Compute the combined distillation loss for a single example.

    Parameters
    ----------
    student_logits:
        Student model output logits (vocab_size,).
    teacher_logits:
        Teacher model output logits (vocab_size,).
    true_label:
        Ground-truth token index.
    alpha:
        Weight for hard loss (1 - alpha for soft loss).
    temperature:
        Temperature for softening both distributions.
    """
    # Hard loss: standard cross-entropy
    student_log_probs = log_softmax(student_logits, temperature=1.0)
    hard = -student_log_probs[true_label] if 0 <= true_label < len(student_log_probs) else 10.0

    # Soft loss: KL(teacher || student) with temperature scaling
    teacher_probs = softmax(teacher_logits, temperature=temperature)
    student_log_probs_soft = log_softmax(student_logits, temperature=temperature)
    # KL(teacher || student) = sum(teacher * (log_teacher - log_student))
    soft = cross_entropy(teacher_probs, student_log_probs_soft)
    # Scale by T^2 (Hinton et al. convention)
    soft *= temperature ** 2

    total = alpha * hard + (1 - alpha) * soft
    return DistillationLoss(
        hard_loss=round(hard, 6),
        soft_loss=round(soft, 6),
        total_loss=round(total, 6),
    )
