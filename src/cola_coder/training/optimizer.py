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


# ── Muon (MomentUm Orthogonalized by Newton-Schulz) ─────────────────────────
#
# 2025-26 state of the art for pretraining hidden layers (Keller Jordan's
# nanoGPT speedruns; validated at scale by Moonshot's Moonlight/Kimi K2 and
# GLM-4.5): instead of Adam's per-coordinate scaling, orthogonalize the
# momentum matrix so the update has a flat singular-value spectrum —
# steepest descent under a spectral-norm trust region. Public results show
# ~1.5-2x token efficiency over tuned AdamW for the 2D weight matrices.
# Embeddings, norms, and other non-2D params keep AdamW (Muon's geometry
# only makes sense for matrices).


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal polar factor of G via Newton-Schulz.

    Quintic iteration with coefficients tuned for fast convergence (Keller
    Jordan's formulation). Runs in bfloat16 — the result doesn't need to be
    exact, just to flatten the singular values to ~1.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    # Normalize so the spectral norm is <= 1 (required for convergence)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights, embedded AdamW for everything else.

    A single optimizer object (rather than two separate ones) so the
    existing LR scheduler and checkpoint save/load paths work unchanged:
    LambdaLR scales every param group's lr multiplicatively, and
    state_dict() serializes both the Muon momentum buffers and the AdamW
    moments.

    Param groups carry a ``use_muon`` flag:
    - use_muon=True  → momentum + Newton-Schulz orthogonalization, scaled by
      sqrt(max(1, rows/cols)) per Moonlight's update-RMS matching
    - use_muon=False → standard decoupled AdamW
    """

    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter],
        adamw_no_decay_params: list[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, ns_steps=ns_steps,
            betas=betas, eps=eps, weight_decay=weight_decay,
        )
        param_groups = [
            dict(params=muon_params, use_muon=True, lr=lr,
                 weight_decay=weight_decay),
            dict(params=adamw_params, use_muon=False, lr=adamw_lr,
                 weight_decay=weight_decay),
            dict(params=adamw_no_decay_params, use_muon=False, lr=adamw_lr,
                 weight_decay=0.0),
        ]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        momentum = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            # Nesterov-style lookahead
            update = g.add(buf, alpha=momentum)
            update = _zeropower_via_newtonschulz5(
                update.reshape(update.size(0), -1), steps=group["ns_steps"],
            ).reshape(p.shape)
            # Match update RMS across differently-shaped matrices (Moonlight)
            rows, cols = p.size(0), p.numel() // p.size(0)
            scale = max(1.0, rows / cols) ** 0.5
            # Decoupled weight decay (same convention as AdamW below)
            if group["weight_decay"] > 0.0:
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"] * scale)

    def _adamw_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
                state["step"] = 0
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            step = state["step"]
            bias1 = 1 - beta1 ** step
            bias2 = 1 - beta2 ** step
            denom = (exp_avg_sq / bias2).sqrt_().add_(group["eps"])
            if group["weight_decay"] > 0.0:
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg / bias1, denom, value=-group["lr"])


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    optimizer: str = "adamw",
    muon_lr: float = 0.02,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay only on appropriate parameters.

    Key insight: weight decay should NOT be applied to:
    - Bias parameters (they don't benefit from regularization)
    - Normalization weights (RMSNorm scale factors)
    - 1D parameters in general (biases, norms)

    This is a common pattern in transformer training that prevents
    regularization from interfering with normalization layers.

    The token embedding (tok_emb.weight, tied to the output head) IS decayed,
    following the GPT-2/nanoGPT convention of decaying all 2D matmul weights.

    Args:
        model: The transformer model.
        learning_rate: Peak learning rate.
        weight_decay: L2 regularization strength (penalizes large weights).
        betas: Adam momentum parameters.
               0.9 = fast momentum (recent gradients matter more)
               0.95 = slow second moment (more stable step sizes)
        optimizer: "adamw" (default) or "muon" — Muon orthogonalizes updates
            for the 2D hidden weights (attention/FFN projections) and keeps
            AdamW for embeddings + norms. Fresh runs only: the saved
            optimizer state of one type can't be loaded into the other.
        muon_lr: Update scale for the Muon groups (only used with "muon").

    Returns:
        Configured optimizer (AdamW or hybrid Muon).
    """
    # Separate parameters into two groups: with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay 1D params (biases, norm weights)
        if param.dim() <= 1 or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if optimizer == "muon":
        # Muon applies only to 2D hidden matrices INSIDE the blocks.
        # The (tied) embedding/output matrix is semantically a lookup
        # table, not a hidden transform — it stays on AdamW, as in the
        # nanoGPT/Moonlight recipes.
        muon_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.dim() == 2 and "blocks." in n
        ]
        muon_ids = {id(p) for p in muon_params}
        adamw_decay = [p for p in decay_params if id(p) not in muon_ids]

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_decay + no_decay_params)
        print(f"Optimizer: Muon on {n_muon:,} params, AdamW on {n_adamw:,} params")

        return Muon(
            muon_params=muon_params,
            adamw_params=adamw_decay,
            adamw_no_decay_params=no_decay_params,
            lr=muon_lr,
            adamw_lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

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
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
    wsd_decay_fraction: float = 0.2,
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
        schedule: "cosine" (default) or "wsd" (warmup-stable-decay, the
            MiniCPM/DeepSeek recipe): hold the peak LR for most of the run,
            decay linearly only in the final ``wsd_decay_fraction``. The
            practical win is continual pretraining — extending max_steps
            doesn't reshape the whole curve like cosine does, you just keep
            training in the stable phase and decay when you actually stop.
        wsd_decay_fraction: Fraction of max_steps spent decaying (wsd only).

    Returns:
        LambdaLR scheduler. Call scheduler.step() after each optimizer.step().
    """

    def cosine_lambda(step: int) -> float:
        """Compute the LR multiplier for a given step."""
        # Phase 1: Linear warmup.
        # (step + 1) so the very first optimizer step uses a small nonzero
        # LR — step/warmup would return 0 at step 0, wasting the first
        # update entirely.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        # Phase 2: Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        # Cosine goes from 1 to -1 over [0, pi], we map to [1, min_lr_ratio]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    def wsd_lambda(step: int) -> float:
        """Warmup → stable plateau → linear decay."""
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        decay_start = int(max_steps * (1.0 - wsd_decay_fraction))
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return 1.0 - (1.0 - min_lr_ratio) * min(progress, 1.0)

    return LambdaLR(optimizer, wsd_lambda if schedule == "wsd" else cosine_lambda)
