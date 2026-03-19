"""Optimizer and learning rate scheduler.

The optimizer is the algorithm that actually updates the model's weights
based on the computed gradients. Think of it like this:

  gradient = "which direction should each weight move to reduce error"
  optimizer = "HOW to move each weight (how big of a step, with momentum, etc.)"

AdamW is the standard optimizer for transformer training. It's an improved
version of SGD (Stochastic Gradient Descent) that:
1. Keeps a running average of gradients (momentum — don't change direction too fast)
2. Keeps a running average of squared gradients (adapt step size per-weight)
3. Applies weight decay correctly (penalize large weights to prevent overfitting)

The learning rate schedule controls how big of a step the optimizer takes:
- Warmup: start small, gradually increase (prevents early instability)
- Cosine decay: gradually decrease from peak to min_lr (fine-tune as we converge)

This is the exact same recipe used by GPT-2, LLaMA, Mistral, etc.
"""

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Create AdamW optimizer with weight decay only on appropriate parameters.

    Key insight: weight decay should NOT be applied to:
    - Bias parameters (they don't benefit from regularization)
    - Normalization weights (RMSNorm scale factors)
    - 1D parameters in general (biases, norms)

    This is a common pattern in transformer training that prevents
    regularization from interfering with normalization layers.

    Args:
        model: The transformer model.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization strength (penalizes large weights).
        betas: Adam momentum parameters.
               0.9 = fast momentum (recent gradients matter more)
               0.95 = slow second moment (more stable step sizes)

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into two groups: with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay 1D params (biases, norm weights) or embedding
        if param.dim() <= 1 or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log parameter counts for each group
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay:,} params with weight decay, {n_no_decay:,} without")

    return AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,  # Small constant for numerical stability
        fused=torch.cuda.is_available(),  # Use faster fused kernel on GPU
    )


def create_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine learning rate schedule with linear warmup.

    The LR follows this pattern:
    1. Linear warmup: 0 → peak_lr over warmup_steps
    2. Cosine decay: peak_lr → min_lr over remaining steps

    Visual (ASCII):

        LR
        ^
    peak|     /‾‾‾‾‾‾\
        |    /          \
        |   /             ‾‾‾‾‾‾  min_lr
        |  /
        | /
     0  +--------------------------> step
        0  warmup        max_steps

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: How many steps to linearly increase LR.
        max_steps: Total training steps.
        min_lr_ratio: min_lr / peak_lr. Default 0.1 means min_lr = peak_lr * 0.1.

    Returns:
        LambdaLR scheduler. Call scheduler.step() after each optimizer.step().
    """

    def lr_lambda(step: int) -> float:
        """Compute the LR multiplier for a given step."""
        # Phase 1: Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Phase 2: Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        # Cosine goes from 1 to -1 over [0, pi], we map to [1, min_lr_ratio]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
