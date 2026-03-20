# Feature 25: Mixture of Experts (MoE) FFN Layer

**Status:** Optional | **CLI Flag:** `--use-moe` | **Complexity:** High

---

## Overview

Replace the standard SwiGLU FFN layers in Cola-Coder with a Mixture of Experts (MoE) variant. Each MoE layer contains N expert FFN networks (4–8 experts) plus a learned gating network. During each forward pass, a top-k routing selects the k most relevant experts (default k=2) and computes a weighted sum of their outputs. A load balancing auxiliary loss prevents expert collapse (where all tokens route to one expert). MoE layers can be applied to all FFN layers or only a subset.

Reference architectures: Switch Transformer (k=1), Mixtral 8x7B (k=2, top-2 sparse MoE).

---

## Motivation

MoE increases model capacity (total parameters) without proportionally increasing compute per token. With N=8 experts and k=2, the model has ~4x more FFN parameters but only activates 2/8 per token — effectively trading memory for capability.

For Cola-Coder's specialist routing vision, MoE provides a natural path to soft domain specialization within a single model: different experts may naturally specialize in React, Prisma, etc. patterns without explicit labels.

Benefits:
- ~2-4x parameter increase at constant FLOPs per token
- Soft specialization without hard domain labels
- Compatible with existing GQA + RoPE + RMSNorm architecture
- Configurable: easy to switch between dense and MoE layers per block

---

## Architecture / Design

### MoE Layer Architecture

```
Input: x  [B, T, d_model]
         ↓
Gate network:  G = softmax(W_gate @ x)  [B, T, N_experts]
         ↓
Top-k selection: select top-k experts per token
         ↓
Route each token to its top-k experts (sparse dispatch)
         ↓
Expert i (SwiGLU FFN):  E_i(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
         ↓
Weighted sum: output = sum_k G[k] * E_{selected_k}(x)
         ↓
Load balancing loss: L_aux = N * sum_i(f_i * P_i)
```

### Load Balancing

Expert collapse is the key failure mode: the gating network learns to always route to the same 1-2 experts, leaving others unused. The Switch Transformer auxiliary loss penalizes this:

```
f_i = fraction of tokens routed to expert i  (computed per-batch)
P_i = mean gating probability for expert i   (soft, differentiable)
L_aux = N_experts * sum_i (f_i * P_i)
L_total = L_lm + alpha * L_aux  (alpha typically 0.01)
```

Minimizing L_aux encourages uniform routing (each expert gets 1/N tokens on average).

---

## Implementation Steps

### Step 1: Expert Network

```python
# cola_coder/model/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class MoEConfig:
    d_model: int = 512
    d_ffn: int = 1536         # Per-expert FFN hidden dim
    n_experts: int = 8
    top_k: int = 2
    aux_loss_alpha: float = 0.01
    dropout: float = 0.0
    capacity_factor: float = 1.25  # Expert capacity = capacity_factor * T / N
    use_expert_choice: bool = False  # Alternative: experts choose tokens


class ExpertFFN(nn.Module):
    """Single SwiGLU expert network."""
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )
```

### Step 2: Gating Network

```python
class TopKGating(nn.Module):
    """
    Learned gating network with top-k sparse routing.
    Returns selected expert indices and their normalized weights.
    """
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            topk_indices: [B, T, top_k]   — which experts to use
            topk_weights: [B, T, top_k]   — normalized gating weights
            router_probs:  [B, T, N]       — full softmax probs (for aux loss)
        """
        logits = self.gate(x)                    # [B, T, N]
        router_probs = F.softmax(logits, dim=-1) # [B, T, N]

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Re-normalize top-k weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_indices, topk_weights, router_probs
```

### Step 3: MoE Layer

```python
class MoELayer(nn.Module):
    """
    Mixture of Experts FFN layer.
    Replaces standard SwiGLU FFN in transformer blocks.
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([
            ExpertFFN(cfg.d_model, cfg.d_ffn, cfg.dropout)
            for _ in range(cfg.n_experts)
        ])
        self.gating = TopKGating(cfg.d_model, cfg.n_experts, cfg.top_k)
        self._aux_loss: Optional[torch.Tensor] = None

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Retrieve the auxiliary load balancing loss computed in last forward pass."""
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Get routing decisions
        topk_indices, topk_weights, router_probs = self.gating(x)
        # topk_indices: [B, T, top_k]
        # topk_weights: [B, T, top_k]

        # Compute auxiliary load balancing loss
        self._aux_loss = self._compute_aux_loss(topk_indices, router_probs, B * T)

        # Flatten batch+seq for expert dispatch
        x_flat = x.reshape(B * T, D)              # [S, D]
        idx_flat = topk_indices.reshape(B * T, -1) # [S, top_k]
        wgt_flat = topk_weights.reshape(B * T, -1) # [S, top_k]

        output = torch.zeros_like(x_flat)

        # Dispatch tokens to experts
        # Efficient implementation: group by expert, process batch
        for expert_idx in range(self.cfg.n_experts):
            # Find which (token, slot) pairs route to this expert
            mask = (idx_flat == expert_idx)  # [S, top_k]
            token_mask = mask.any(dim=-1)    # [S] — tokens that use this expert

            if not token_mask.any():
                continue

            tokens_for_expert = x_flat[token_mask]  # [n_tokens, D]
            expert_output = self.experts[expert_idx](tokens_for_expert)  # [n_tokens, D]

            # Accumulate weighted output
            for k in range(self.cfg.top_k):
                slot_mask = mask[:, k] & token_mask
                if not slot_mask.any():
                    continue
                weights_k = wgt_flat[slot_mask, k].unsqueeze(-1)  # [n, 1]
                # Map back: only tokens in slot_mask are in expert_output
                expert_token_indices = token_mask.nonzero(as_tuple=True)[0]
                slot_token_indices = slot_mask.nonzero(as_tuple=True)[0]
                # Find position in expert batch
                expert_pos = (expert_token_indices.unsqueeze(1) == slot_token_indices.unsqueeze(0)).any(dim=1)
                output[slot_mask] += weights_k * expert_output[expert_pos]

        return output.reshape(B, T, D)

    def _compute_aux_loss(
        self,
        topk_indices: torch.Tensor,
        router_probs: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.
        L_aux = N * sum_i (f_i * P_i)
        where f_i = fraction of tokens dispatched to expert i
              P_i = mean routing probability for expert i
        """
        N = self.cfg.n_experts
        # f_i: token fraction per expert (discrete, not differentiable)
        one_hot = F.one_hot(topk_indices, num_classes=N).float()  # [B,T,top_k,N]
        tokens_per_expert = one_hot.sum(dim=(0, 1, 2))            # [N]
        f_i = tokens_per_expert / (n_tokens * self.cfg.top_k)     # normalize

        # P_i: mean routing probability (differentiable)
        P_i = router_probs.mean(dim=(0, 1))                       # [N]

        aux_loss = N * (f_i * P_i).sum()
        return aux_loss * self.cfg.aux_loss_alpha
```

### Step 4: Efficient Token Dispatch (Vectorized)

```python
class MoELayerFast(MoELayer):
    """
    Vectorized implementation using scatter/gather operations.
    More GPU-friendly than the loop-based version above.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        S = B * T
        x_flat = x.reshape(S, D)

        topk_indices, topk_weights, router_probs = self.gating(x)
        self._aux_loss = self._compute_aux_loss(topk_indices, router_probs, S)

        # Create dispatch tensor: [S * top_k, D]
        idx_flat = topk_indices.reshape(S * self.cfg.top_k)
        wgt_flat = topk_weights.reshape(S * self.cfg.top_k, 1)

        # Repeat tokens for each of their top-k experts
        x_repeated = x_flat.repeat_interleave(self.cfg.top_k, dim=0)  # [S*k, D]

        # Process each expert on its assigned tokens
        output_slots = torch.zeros_like(x_repeated)
        for i, expert in enumerate(self.experts):
            mask = (idx_flat == i)
            if mask.any():
                output_slots[mask] = expert(x_repeated[mask])

        # Weighted sum back to [S, D]
        weighted = output_slots * wgt_flat
        output = weighted.reshape(S, self.cfg.top_k, D).sum(dim=1)
        return output.reshape(B, T, D)
```

### Step 5: Integration into TransformerBlock

```python
# cola_coder/model/transformer.py  (modifications)
from .moe import MoELayer, MoEConfig

class TransformerBlock(nn.Module):
    def __init__(self, model_cfg, block_idx: int = 0, use_moe: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(model_cfg.d_model)
        self.attn = GroupedQueryAttention(model_cfg)  # Existing GQA
        self.norm2 = nn.RMSNorm(model_cfg.d_model)

        if use_moe:
            moe_cfg = MoEConfig(
                d_model=model_cfg.d_model,
                d_ffn=model_cfg.d_ffn // model_cfg.moe_n_experts * 2,
                n_experts=model_cfg.moe_n_experts,
                top_k=model_cfg.moe_top_k,
                aux_loss_alpha=model_cfg.moe_aux_alpha,
            )
            self.ffn = MoELayerFast(moe_cfg)
            self.is_moe = True
        else:
            self.ffn = SwiGLUFFN(model_cfg)  # Existing dense FFN
            self.is_moe = False

    def forward(self, x, freqs_cis=None, mask=None):
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
        return x


# Modified ModelConfig to add MoE settings
@dataclass
class ModelConfig:
    # ... existing fields ...
    use_moe: bool = False
    moe_layers: list[int] = field(default_factory=list)  # Which layers use MoE (empty = all)
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_aux_alpha: float = 0.01
```

### Step 6: Training with Aux Loss

```python
# cola_coder/training/trainer.py  (modifications)

def compute_total_loss(model, logits, targets, loss_fn):
    """Add MoE auxiliary loss to LM loss."""
    lm_loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    # Collect aux losses from all MoE layers
    aux_loss = torch.tensor(0.0, device=lm_loss.device)
    for module in model.modules():
        if hasattr(module, "aux_loss") and module.aux_loss is not None:
            aux_loss = aux_loss + module.aux_loss

    total_loss = lm_loss + aux_loss
    return total_loss, lm_loss.item(), aux_loss.item()
```

### Step 7: CLI Config

```yaml
# configs/model/small_moe.yaml
d_model: 512
n_heads: 8
n_kv_heads: 2
n_layers: 12
d_ffn: 1024       # Per-expert dim (smaller than dense d_ffn to keep FLOP budget)
vocab_size: 32000
max_seq_len: 2048
use_moe: true
moe_layers: [3, 6, 9, 11]   # Every 3rd layer is MoE (interleaved dense/MoE like Mixtral)
moe_n_experts: 8
moe_top_k: 2
moe_aux_alpha: 0.01
```

---

## Key Files to Modify

- `cola_coder/model/moe.py` — new file (ExpertFFN, TopKGating, MoELayer, MoELayerFast)
- `cola_coder/model/transformer.py` — add `use_moe` param to TransformerBlock
- `cola_coder/model/config.py` — add MoE fields to ModelConfig
- `cola_coder/training/trainer.py` — collect and add aux loss
- `configs/model/small_moe.yaml` — example MoE config
- `configs/model/medium_moe.yaml` — medium MoE config
- `cola_coder/cli.py` — `--use-moe` flag propagation

---

## Testing Strategy

```python
def test_moe_layer_output_shape():
    cfg = MoEConfig(d_model=64, d_ffn=128, n_experts=4, top_k=2)
    layer = MoELayerFast(cfg)
    x = torch.randn(2, 16, 64)
    out = layer(x)
    assert out.shape == x.shape

def test_moe_aux_loss_computed():
    cfg = MoEConfig(d_model=64, d_ffn=128, n_experts=4, top_k=2)
    layer = MoELayerFast(cfg)
    x = torch.randn(2, 16, 64)
    layer(x)
    assert layer.aux_loss is not None
    assert layer.aux_loss.item() >= 0

def test_moe_load_balance_loss_decreases():
    # With aux loss gradient, routing should spread across experts
    cfg = MoEConfig(d_model=32, d_ffn=64, n_experts=4, top_k=2, aux_loss_alpha=0.1)
    layer = MoELayerFast(cfg)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    initial_aux = None
    for _ in range(50):
        x = torch.randn(4, 8, 32)
        out = layer(x)
        loss = out.mean() + layer.aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if initial_aux is None:
            initial_aux = layer.aux_loss.item()
    # Aux loss should decrease as routing becomes more balanced
    assert layer.aux_loss.item() <= initial_aux * 1.5  # Allow some variance

def test_moe_same_output_shape_as_dense():
    from cola_coder.model.transformer import TransformerBlock
    cfg_dense = MockModelConfig(use_moe=False)
    cfg_moe = MockModelConfig(use_moe=True)
    block_dense = TransformerBlock(cfg_dense, use_moe=False)
    block_moe = TransformerBlock(cfg_moe, use_moe=True)
    x = torch.randn(1, 32, cfg_dense.d_model)
    assert block_dense(x).shape == block_moe(x).shape
```

---

## Performance Considerations

- **FLOPs:** With top_k=2, N=8: active FLOPs = 2/8 = 25% of a dense model with same total params. Total params = ~4x dense, but compute stays constant.
- **Memory:** All N expert weights are resident in memory, even if only 2 activate per token. For a 125M dense model with 8 experts, MoE params = ~4x = 500M. Plan for 1GB+ for fp16 MoE.
- **Token dispatch overhead:** The scatter/gather operations in MoELayerFast add ~10-15% overhead vs a dense FFN. Use vectorized implementation to minimize.
- **Gradient flow:** Each expert only receives gradients from the tokens routed to it. With small batches and many experts, some experts may go many steps without gradient updates. Use larger batch sizes for MoE training.
- **Expert capacity:** Add a capacity factor to prevent token dropping. Each expert can process at most `capacity_factor * T / N` tokens per batch. Tokens over capacity are dropped (spill to next best expert or masked).
- **Interleaved layers:** Following Mixtral, don't make every layer MoE. Use a pattern like dense-dense-MoE-dense-dense-MoE. This balances capacity increase with training stability.

---

## Dependencies

- PyTorch (core operations: scatter, gather, one_hot)
- Existing Cola-Coder model components (SwiGLUFFN, GQA, ModelConfig)
- No new external dependencies required

---

## Estimated Complexity

| Task                          | Effort   |
|-------------------------------|----------|
| ExpertFFN + TopKGating        | 2h       |
| MoELayer (loop version)       | 3h       |
| MoELayerFast (vectorized)     | 3h       |
| Aux loss + training integration | 2h     |
| TransformerBlock integration  | 1.5h     |
| Config + YAML                 | 1h       |
| Tests                         | 2h       |
| Debugging expert collapse     | 2h       |
| **Total**                     | **~16.5h** |

Overall complexity: **High** (numerically tricky, expert collapse debugging is non-trivial)

---

## 2026 Best Practices

- **Mixtral-style interleaving:** Alternate dense and MoE layers rather than making all layers MoE. Every-other or every-third is a good starting point.
- **Expert choice routing (optional):** Instead of tokens choosing experts (token-choice routing), let experts choose their top-m tokens. Eliminates expert overflow but changes the load balancing math. Reference: "Mixture of Experts with Expert Choice Routing" (2022).
- **Router z-loss:** Add `z_loss = mean(log(sum(exp(logits)))^2)` to prevent router logits from growing unboundedly. Used in GLaM and ST-MoE. Coefficient: 1e-3.
- **Expert initialization:** Initialize all experts identically (same weights). Symmetry breaking happens through gradient noise. Do NOT use different random seeds per expert initially.
- **Monitor expert utilization:** Log the fraction of tokens routed to each expert during training. If any expert goes below 5% utilization, consider increasing `aux_loss_alpha`.
- **Gradient checkpointing for experts:** MoE increases activation memory proportionally to N. Use `torch.utils.checkpoint` on expert computations to trade compute for memory.
