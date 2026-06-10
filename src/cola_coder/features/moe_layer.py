"""Mixture of Experts (MoE) Layer: optional drop-in replacement for standard FFN.

Replaces the single SwiGLU FFN with multiple "expert" FFNs and a learned router
that decides which expert(s) process each token. This gives the model more
capacity (more parameters) without proportionally increasing compute, since only
top-k experts are active per token.

Key concepts:
- Each expert is a full SwiGLU FFN
- A small router network assigns tokens to experts
- Only top-k experts are activated per token (sparse computation)
- Load balancing loss encourages even expert utilization
- Capacity factor limits how many tokens each expert processes

For a TS dev: imagine a microservices architecture where a router sends each
request to the most relevant service. MoE does the same thing with neural
network "sub-modules" — each expert specializes in different patterns.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


def resolve_moe_layers(moe_layers: str, n_layers: int) -> set[int]:
    """Resolve the ``moe_layers`` config string to a set of layer indices.

    Args:
        moe_layers: "all" (every block), "alternate" (every other block,
            starting at index 1), or a comma-separated list of indices
            (e.g. "0,2,4").
        n_layers: Total number of transformer blocks.

    Returns:
        Set of block indices that should use a MoE FFN.

    Note: ``scripts/upcycle_to_moe.py`` converts EVERY FFN block, so an
    upcycled checkpoint round-trips correctly only with moe_layers="all".
    """
    spec = (moe_layers or "all").strip().lower()
    if spec == "all":
        return set(range(n_layers))
    if spec == "alternate":
        return {i for i in range(n_layers) if i % 2 == 1}
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < n_layers:
                indices.add(idx)
    return indices


def detect_moe_checkpoint(checkpoint_dir: str | Path) -> dict | None:
    """Detect whether a checkpoint directory holds an upcycled MoE model.

    Reads the ``moe_config.json`` sidecar written by upcycle_to_moe.py when
    present; otherwise inspects the safetensors keys for ``.ffn.experts.``
    and infers the expert/shared counts from them.

    Returns:
        A dict with num_experts / num_shared_experts / top_k when the
        checkpoint is MoE, else None.
    """
    ckpt = Path(checkpoint_dir)

    sidecar = ckpt / "moe_config.json"
    if sidecar.exists():
        try:
            cfg = json.loads(sidecar.read_text(encoding="utf-8"))
            return {
                "num_experts": int(cfg.get("num_experts", 8)),
                "num_shared_experts": int(cfg.get("num_shared_experts", 1)),
                "top_k": int(cfg.get("top_k", 2)),
            }
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

    # Fallback: infer from the safetensors weight keys.
    model_file = ckpt / "model.safetensors"
    if not model_file.exists():
        return None
    try:
        from safetensors import safe_open

        with safe_open(str(model_file), framework="pt") as f:
            keys = list(f.keys())
    except Exception:
        return None

    if not any(".ffn.experts." in k for k in keys):
        return None

    expert_ids, shared_ids = set(), set()
    for k in keys:
        if ".ffn.experts." in k:
            expert_ids.add(k.split(".ffn.experts.")[1].split(".")[0])
        elif ".ffn.shared_experts." in k:
            shared_ids.add(k.split(".ffn.shared_experts.")[1].split(".")[0])
    return {
        "num_experts": len(expert_ids),
        "num_shared_experts": len(shared_ids),
        "top_k": 2,
    }


class ExpertRouter(nn.Module):
    """Learned router that assigns tokens to experts.

    Produces a probability distribution over experts for each token,
    then selects the top-k experts.
    """

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: (batch * seq_len, dim) — flattened token representations

        Returns:
            router_logits: (batch * seq_len, num_experts) — raw scores
            router_probs: (batch * seq_len, num_experts) — softmax probabilities
            router_targets: same as router_probs (used for load balancing)
        """
        logits = self.gate(x)  # (tokens, num_experts)
        probs = F.softmax(logits, dim=-1)
        return logits, probs, probs


class MoEFFN(nn.Module):
    """Mixture of Experts FFN — drop-in replacement for SwiGLUFFN.

    Contains multiple SwiGLU expert FFNs and a router. For each token,
    only the top-k experts are activated, keeping compute manageable
    while increasing total model capacity.

    Supports the DeepSeek-MoE shared expert approach: one or more experts
    always run for every token (handling common cross-domain patterns),
    while the remaining routed experts specialize. The final output is the
    sum of the shared expert output and the weighted routed expert output.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        capacity_factor: float = 1.25,
        num_shared_experts: int = 1,
        aux_loss_weight: float = 0.01,
    ):
        """
        Args:
            dim: Model dimension (input/output size)
            hidden_dim: Hidden dimension for each expert's SwiGLU FFN
            num_experts: Total number of routed expert FFNs
            top_k: Number of routed experts activated per token
            dropout: Dropout rate
            capacity_factor: Max fraction of tokens each expert handles
                             (1.0 = even split, >1.0 = allow some imbalance)
            num_shared_experts: Number of always-active shared experts
                                (DeepSeek-MoE style). 0 = disabled.
            aux_loss_weight: Weight for the load-balancing auxiliary loss.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_shared_experts = num_shared_experts

        # Router decides which experts process each token
        self.router = ExpertRouter(dim, num_experts)

        # Each expert is a full SwiGLU FFN
        # Using nn.ModuleList so PyTorch tracks parameters
        self.experts = nn.ModuleList([
            _SwiGLUExpert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        # Shared experts: always-active, not subject to routing (DeepSeek-MoE)
        # These handle common cross-domain patterns; routed experts specialize.
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                _SwiGLUExpert(dim, hidden_dim, dropout)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()

        # Auxiliary loss weight for load balancing
        self.aux_loss_weight = aux_loss_weight
        self._aux_loss = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        """Load balancing auxiliary loss — add to main loss during training."""
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tokens through top-k experts.

        Args:
            x: (batch_size, seq_len, dim)

        Returns:
            (batch_size, seq_len, dim) — same shape as input
        """
        batch_size, seq_len, dim = x.shape

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, dim)  # (B*S, dim)
        num_tokens = x_flat.shape[0]

        # Route tokens to experts
        router_logits, router_probs, _ = self.router(x_flat)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize so weights sum to 1
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute load balancing loss
        self._aux_loss = self._load_balancing_loss(router_logits, top_k_indices, num_tokens)

        # Process tokens through selected experts
        output = torch.zeros_like(x_flat)  # (B*S, dim)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*S,) — which expert for each token
            expert_weights = top_k_weights[:, k]   # (B*S,) — weight for this expert

            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                # Apply capacity factor: limit tokens per expert
                token_indices = mask.nonzero(as_tuple=True)[0]
                capacity = int(self.capacity_factor * num_tokens / self.num_experts)
                if len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]

                # Process selected tokens through this expert
                expert_input = x_flat[token_indices]
                expert_output = self.experts[expert_id](expert_input)

                # Weighted addition to output
                weights = expert_weights[token_indices].unsqueeze(-1)
                output[token_indices] += weights * expert_output

        # Shared experts: always-active for every token (DeepSeek-MoE approach).
        # Each shared expert runs on the full flattened sequence, then averaged
        # and added to the routed output. This lets shared experts capture
        # cross-domain patterns while routed experts specialize.
        if self.shared_experts:
            shared_out = torch.zeros_like(x_flat)
            for shared_expert in self.shared_experts:
                shared_out = shared_out + shared_expert(x_flat)
            shared_out = shared_out / max(len(self.shared_experts), 1)
            output = output + shared_out

        return output.view(batch_size, seq_len, dim)

    def _load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.

        Encourages the router to distribute tokens evenly across experts.
        Without this, the model tends to collapse to using only 1-2 experts.

        Loss = num_experts * sum_i(f_i * P_i) where:
        - f_i = fraction of tokens assigned to expert i
        - P_i = mean routing probability for expert i
        """
        if not self.training:
            return torch.tensor(0.0, device=router_logits.device)

        # f_i: fraction of tokens routed to each expert
        # Count how many times each expert appears in top-k selections
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for k in range(self.top_k):
            for expert_id in range(self.num_experts):
                expert_counts[expert_id] += (top_k_indices[:, k] == expert_id).float().sum()
        fraction_per_expert = expert_counts / (num_tokens * self.top_k)

        # P_i: mean routing probability per expert
        router_probs = F.softmax(router_logits, dim=-1)
        mean_prob_per_expert = router_probs.mean(dim=0)

        # Load balancing loss
        aux_loss = self.num_experts * (fraction_per_expert * mean_prob_per_expert).sum()
        return self.aux_loss_weight * aux_loss


class _SwiGLUExpert(nn.Module):
    """Single SwiGLU expert — identical architecture to SwiGLUFFN."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.dropout(self.down_proj(gate * value))


class MoEConfig:
    """Configuration for MoE layers."""

    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
        num_shared_experts: int = 1,
        moe_layers: list[int] | None = None,
    ):
        """
        Args:
            num_experts: Number of routed expert FFNs per MoE layer
            top_k: Experts activated per token
            capacity_factor: Capacity multiplier per expert
            aux_loss_weight: Weight for load balancing auxiliary loss
            num_shared_experts: Number of always-active shared experts (DeepSeek-MoE). 0 = disabled.
            moe_layers: Which transformer layers get MoE (None = all)
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.num_shared_experts = num_shared_experts
        self.moe_layers = moe_layers

    @property
    def total_expert_params_multiplier(self) -> float:
        """How much the FFN parameters are multiplied by MoE.

        With 8 routed experts + 1 shared expert, there are 9x the FFN params,
        but only top_k routed + num_shared_experts are active per token.
        """
        return self.num_experts + self.num_shared_experts

    @property
    def active_params_fraction(self) -> float:
        """Fraction of expert params active per token.

        Includes shared experts (always active) plus top_k routed experts.
        """
        total = self.num_experts + self.num_shared_experts
        if total == 0:
            return 0.0
        return (self.top_k + self.num_shared_experts) / total
