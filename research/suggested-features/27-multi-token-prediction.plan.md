# Feature 27: Multi-Token Prediction

**Status:** Optional | **CLI Flag:** `--multi-token-pred` | **Complexity:** Medium

---

## Overview

Add N additional prediction heads to the transformer, each predicting tokens 1, 2, ..., N steps ahead simultaneously from the same hidden states. Training uses a weighted sum of losses across all heads with exponential decay (head_1: 1.0, head_2: 0.5, head_3: 0.25). At inference time, the extra heads can be used for speculative decoding (Feature 26) or discarded. The technique improves sample efficiency and forces the model to learn longer-horizon representations.

Reference: "Better & Faster Large Language Models via Multi-Token Prediction" (Gloeckle et al., Meta, 2024).

---

## Motivation

Standard next-token prediction treats each token prediction independently. Multi-token prediction (MTP) forces the model to reason about future context while predicting the current token:

- **Better representations:** Predicting k tokens ahead requires richer hidden states that encode future structure, not just immediate context
- **Sample efficiency:** MTP is effectively K× data augmentation — each training example generates K separate prediction targets
- **Speculative decoding synergy:** The extra heads can directly serve as a draft for Feature 26, eliminating the need for a separate draft model
- **Observed gains:** Meta's paper reports 10-20% perplexity improvement on code tasks at no extra inference cost

The inference-time cost of the extra heads is zero — they are simply not called during standard generation.

---

## Architecture / Design

### MTP Head Architecture

```
Transformer trunk produces hidden states: h[t]  [B, T, d_model]
         ↓
Standard LM head:   logits_+1 = W_lm @ h[t]      → predict token at t+1
                                                    (standard LM objective)
MTP head 2:         logits_+2 = W_head2 @ h[t]    → predict token at t+2
MTP head 3:         logits_+3 = W_head3 @ h[t]    → predict token at t+3
MTP head N:         logits_+N = W_headN @ h[t]    → predict token at t+N
```

Each additional head is a simple linear projection from `d_model` to `vocab_size`. Optionally, a small 1-layer transformer block can be inserted before each head for more expressive n-step prediction.

### Loss Weighting

```
L_total = sum_{k=1}^{N} weight_k * CE(logits_+k, target_+k)
where:
  weight_1 = 1.0   (standard next-token prediction, unchanged)
  weight_2 = 0.5
  weight_3 = 0.25
  weight_k = 2^(-(k-1))   (exponential decay)
```

The weights ensure the primary (1-step) loss dominates and the model doesn't sacrifice standard quality for multi-step accuracy.

### Meta's Architecture Detail

Meta's paper uses a slightly different architecture: each MTP head has its own small transformer block (1 layer) operating on the hidden states. This allows the head to specialize for N-step prediction rather than just using a linear head. The plan supports both variants.

---

## Implementation Steps

### Step 1: MTP Head Module

```python
# cola_coder/model/mtp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class MTPConfig:
    num_heads: int = 3              # Number of extra prediction heads (not counting main)
    head_weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    use_transformer_head: bool = False  # True = 1-layer transformer per head, False = linear
    d_model: int = 512
    vocab_size: int = 32000
    # Only used if use_transformer_head=True:
    head_n_heads: int = 4
    head_d_ffn: int = 512


class LinearMTPHead(nn.Module):
    """Simple linear head for predicting token +k ahead."""
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(hidden))


class TransformerMTPHead(nn.Module):
    """1-layer transformer + linear head for k-step prediction (Meta style)."""
    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.head_n_heads, batch_first=True, bias=False
        )
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_d_ffn, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.head_d_ffn, cfg.d_model, bias=False),
        )
        self.out_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Causal mask
        T = hidden.size(1)
        mask = torch.triu(torch.ones(T, T, device=hidden.device), diagonal=1).bool()
        x = hidden
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                 attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return self.lm_head(self.out_norm(x))


class MultiTokenPredictionHeads(nn.Module):
    """
    Collection of N additional prediction heads for tokens +2, +3, ..., +N.
    The +1 head is the standard LM head already in the model.
    """
    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        # We need `num_heads` extra heads (for predicting +2, +3, ..., +N+1)
        # Index 0 = predicts +2, index 1 = predicts +3, etc.
        if cfg.use_transformer_head:
            self.heads = nn.ModuleList([
                TransformerMTPHead(cfg) for _ in range(cfg.num_heads)
            ])
        else:
            self.heads = nn.ModuleList([
                LinearMTPHead(cfg.d_model, cfg.vocab_size)
                for _ in range(cfg.num_heads)
            ])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Return list of logits tensors, one per extra head."""
        return [head(hidden) for head in self.heads]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

### Step 2: MTP Loss Computation

```python
# cola_coder/training/mtp_loss.py
import torch
import torch.nn.functional as F
from cola_coder.model.mtp import MTPConfig

def compute_mtp_loss(
    main_logits: torch.Tensor,     # [B, T, V] — standard LM head output
    mtp_logits: list[torch.Tensor], # list of [B, T, V] — extra head outputs
    input_ids: torch.Tensor,       # [B, T] — full token sequence
    cfg: MTPConfig,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute weighted multi-token prediction loss.

    Standard target for main logits: input_ids[:, 1:] (next token)
    Target for head k (0-indexed): input_ids[:, k+2:]  (token k+2 steps ahead)

    Returns:
        total_loss: weighted sum of all head losses
        loss_breakdown: dict with loss per head for logging
    """
    B, T, V = main_logits.shape
    loss_breakdown = {}

    # Standard next-token loss (weight = 1.0)
    main_targets = input_ids[:, 1:]                          # [B, T-1]
    main_logits_clipped = main_logits[:, :-1, :]             # [B, T-1, V]
    main_loss = F.cross_entropy(
        main_logits_clipped.reshape(-1, V),
        main_targets.reshape(-1),
        ignore_index=ignore_index,
    )
    loss_breakdown["loss_+1"] = main_loss.item()

    total_loss = cfg.head_weights[0] * main_loss

    # Extra prediction head losses
    for k, (logits_k, weight) in enumerate(
        zip(mtp_logits, cfg.head_weights[1:]), start=1
    ):
        offset = k + 1  # Head k predicts token at position t + offset
        if offset >= T:
            break  # Sequence too short for this head

        # Targets: tokens at positions offset, offset+1, ..., T-1
        targets_k = input_ids[:, offset:]                    # [B, T-offset]
        logits_k_clipped = logits_k[:, :T - offset, :]      # [B, T-offset, V]

        head_loss = F.cross_entropy(
            logits_k_clipped.reshape(-1, V),
            targets_k.reshape(-1),
            ignore_index=ignore_index,
        )
        loss_breakdown[f"loss_+{offset}"] = head_loss.item()
        total_loss = total_loss + weight * head_loss

    return total_loss, loss_breakdown
```

### Step 3: Integration into Main Model

```python
# cola_coder/model/transformer.py  (modifications)
from .mtp import MultiTokenPredictionHeads, MTPConfig

class CodaCoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # ... existing components ...
        self.use_mtp = cfg.use_mtp
        if cfg.use_mtp:
            mtp_cfg = MTPConfig(
                num_heads=cfg.mtp_num_heads,
                head_weights=cfg.mtp_head_weights,
                use_transformer_head=cfg.mtp_use_transformer_head,
                d_model=cfg.d_model,
                vocab_size=cfg.vocab_size,
            )
            self.mtp_heads = MultiTokenPredictionHeads(mtp_cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_mtp_logits: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        # ... existing forward pass ...
        hidden = self._trunk_forward(input_ids)  # [B, T, d_model]
        main_logits = self.lm_head(self.norm(hidden))

        mtp_logits = None
        if self.use_mtp and return_mtp_logits:
            mtp_logits = self.mtp_heads(hidden)

        return main_logits, mtp_logits
```

### Step 4: Using MTP Heads for Speculative Decoding

```python
# cola_coder/speculative/mtp_draft.py
"""
Use the model's own MTP heads as draft tokens for speculative decoding.
This eliminates the need for a separate draft model (Feature 26).
When --multi-token-pred and --speculative are both enabled, prefer this approach.
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def mtp_speculative_step(
    model,
    context: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[list[int], list[torch.Tensor]]:
    """
    Get main + MTP head predictions for speculative decoding.

    Returns:
        draft_tokens: tokens proposed by heads 1, 2, ..., N
        head_probs:   probability distributions from each head
    """
    main_logits, mtp_logits = model(context, return_mtp_logits=True)

    # Main head: token at position -1 (next token)
    main_probs = F.softmax(main_logits[0, -1, :] / temperature, dim=-1)
    main_token = torch.multinomial(main_probs, 1).item()

    draft_tokens = [main_token]
    draft_probs = [main_probs]

    # MTP heads: tokens at positions +2, +3, ...
    for head_logits in mtp_logits:
        head_probs = F.softmax(head_logits[0, -1, :] / temperature, dim=-1)
        head_token = torch.multinomial(head_probs, 1).item()
        draft_tokens.append(head_token)
        draft_probs.append(head_probs)

    return draft_tokens, draft_probs
```

### Step 5: Training Integration and CLI

```python
# Modified training loop
def training_step(model, batch, cfg):
    input_ids = batch["input_ids"]
    main_logits, mtp_logits = model(input_ids, return_mtp_logits=model.use_mtp)

    if model.use_mtp:
        from cola_coder.training.mtp_loss import compute_mtp_loss
        total_loss, breakdown = compute_mtp_loss(
            main_logits, mtp_logits or [], input_ids, model.mtp_heads.cfg
        )
        return total_loss, breakdown
    else:
        loss = F.cross_entropy(
            main_logits[:, :-1].reshape(-1, main_logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        return loss, {"loss_+1": loss.item()}


# CLI flag additions to configs/model/small.yaml
"""
use_mtp: true
mtp_num_heads: 2          # Predict +2 and +3 ahead
mtp_head_weights: [1.0, 0.5, 0.25]
mtp_use_transformer_head: false  # Linear heads for speed
"""
```

---

## Key Files to Modify

- `cola_coder/model/mtp.py` — new file (head modules)
- `cola_coder/training/mtp_loss.py` — new file (loss computation)
- `cola_coder/model/transformer.py` — add `mtp_heads`, modify forward
- `cola_coder/model/config.py` — add MTP fields to ModelConfig
- `cola_coder/training/trainer.py` — call `compute_mtp_loss` when MTP enabled
- `cola_coder/speculative/mtp_draft.py` — speculative decoding via MTP
- `configs/model/*.yaml` — add MTP config fields

---

## Testing Strategy

```python
def test_mtp_heads_output_shapes():
    cfg = MTPConfig(num_heads=2, d_model=64, vocab_size=100)
    heads = MultiTokenPredictionHeads(cfg)
    hidden = torch.randn(2, 16, 64)
    outputs = heads(hidden)
    assert len(outputs) == 2
    assert all(o.shape == (2, 16, 100) for o in outputs)

def test_mtp_loss_weights():
    # Loss with weight 1.0 for +1, 0.5 for +2
    main_logits = torch.randn(1, 10, 50)
    mtp_logits = [torch.randn(1, 10, 50), torch.randn(1, 10, 50)]
    input_ids = torch.randint(0, 50, (1, 10))
    cfg = MTPConfig(num_heads=2, head_weights=[1.0, 0.5, 0.25], d_model=64, vocab_size=50)
    total_loss, breakdown = compute_mtp_loss(main_logits, mtp_logits, input_ids, cfg)
    assert "loss_+1" in breakdown
    assert "loss_+2" in breakdown
    assert total_loss.item() > 0

def test_mtp_does_not_change_inference_output():
    # With return_mtp_logits=False, output should be same as without MTP
    cfg = ModelConfig(use_mtp=True, mtp_num_heads=2)
    model = CodaCoderModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    out_with, _ = model(x, return_mtp_logits=True)
    out_without, _ = model(x, return_mtp_logits=False)
    assert torch.allclose(out_with, out_without)

def test_mtp_head_count():
    cfg = MTPConfig(num_heads=3, d_model=64, vocab_size=100)
    heads = MultiTokenPredictionHeads(cfg)
    assert len(heads.heads) == 3
```

---

## Performance Considerations

- **Training cost:** MTP adds N linear projections to each training step. With N=2 and linear heads, overhead is ~5-10% extra compute. With transformer heads, ~20-30%.
- **No inference overhead:** MTP heads are not called during standard generation (`return_mtp_logits=False`). Zero cost at inference.
- **Memory during training:** Extra heads add small parameter count (~2 × d_model × vocab_size per head = ~33M params for d_model=512, vocab_size=32k). Use weight tying with the main LM head if vocab_size is large.
- **Loss balance:** Monitor `loss_+2` and `loss_+3` separately. If they plateau while `loss_+1` still decreases, the extra heads are not contributing. Consider increasing head weights.
- **Gradient flow:** The extra heads backpropagate through the same trunk hidden states, providing additional gradient signal. This is the core benefit — not the MTP logits themselves.

---

## Dependencies

- Existing Cola-Coder model (trunk forward pass, RMSNorm, LM head)
- Feature 26 (speculative decoding) — optional synergy via `mtp_draft.py`
- No new external packages required

---

## Estimated Complexity

| Task                          | Effort   |
|-------------------------------|----------|
| LinearMTPHead + TransformerMTPHead | 2h  |
| MultiTokenPredictionHeads     | 1h       |
| MTP loss computation          | 2h       |
| Model forward integration     | 2h       |
| Training loop integration     | 1h       |
| MTP-based speculative draft   | 2h       |
| Tests                         | 2h       |
| **Total**                     | **~12h** |

Overall complexity: **Medium** (conceptually clean, main work is integration and loss bookkeeping)

---

## 2026 Best Practices

- **Start with linear heads:** TransformerMTPHead adds complexity without guaranteed gains on small models. Use linear heads first; upgrade to transformer heads only if perplexity gap exists.
- **Weight decay on extra heads:** Apply standard weight decay (0.1) to MTP head parameters. Without it, they can overfit to short-range patterns and hurt generalization.
- **Disable during fine-tuning:** When fine-tuning a pretrained model on domain-specific data, disable MTP heads (set `return_mtp_logits=False` in training) to avoid catastrophic forgetting. Optionally fine-tune the heads separately afterward.
- **K=2 or K=3 is sufficient:** Meta's paper shows diminishing returns beyond K=4. K=2 captures the majority of the benefit with minimal overhead.
- **Head weight scheduling:** Consider annealing head weights over training: start with equal weights, then gradually upweight the +1 (main) loss toward the end of training for final fine-tuning of standard quality.
- **MTP + MoE synergy:** MTP heads placed after MoE layers (Feature 25) receive richer representations from the mixture of experts, potentially amplifying both effects. Experiment with this combination.
