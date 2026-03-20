# 74 - Contrastive Code Learning

## Overview

Train Cola-Coder on (good code, bad code) pairs using contrastive learning. Positive samples are working TypeScript code; negative samples are derived by introducing bugs, style violations, or type errors. Loss functions: SimCLR-style contrastive loss, DPO (Direct Preference Optimization), or triplet loss. Can be used as an auxiliary loss during pretraining or as a standalone fine-tuning stage.

**Feature flag:** `config.contrastive_learning.enabled` (default: `false`; experimental)

---

## Motivation

Standard language model pretraining treats all tokens equally. It doesn't distinguish between high-quality idiomatic TypeScript and code with subtle bugs. Contrastive learning injects a quality signal:

- The model's representations of "correct" code should be different from "incorrect" code
- DPO (used in Anthropic's RLHF work) is a particularly elegant approach: given a preference pair (good, bad), update the model to prefer the good one without a separate reward model
- This is especially powerful for code because we can automatically generate bug-introduced negatives at scale, without human labeling

**Expected benefit**: model generates fewer type errors and antipatterns, even before explicit RLHF.

---

## Architecture / Design

### Methods Comparison

| Method | Pros | Cons | Best for |
|--------|------|------|----------|
| SimCLR contrastive | Simple, well-understood | Needs large batch for negatives | Embedding quality |
| DPO | No reward model needed, elegant | Requires reference model | Preference fine-tuning |
| Triplet loss | Flexible margin | Sensitive to negative mining quality | Ranking tasks |

**Recommended default: DPO.** It directly optimizes the generation policy and has been shown to work well for code quality. SimCLR is a good auxiliary during pretraining.

### Negative Generation Methods

```python
NegativeType = Literal[
    "introduce_type_error",      # assign wrong type to variable
    "remove_return_type",        # remove : ReturnType annotation
    "swap_operators",            # > → <, + → -, === → !==
    "corrupt_variable_name",     # rename variable to typo
    "remove_null_check",         # remove if (x !== null) guards
    "add_unused_variable",       # add dead code
    "wrong_generic_param",       # T → any
    "break_async_await",         # remove await keyword
]
```

---

## Implementation Steps

### Step 1: Negative Sample Generator (`data/negative_generator.py`)

```python
import re
import random
from pathlib import Path

class NegativeSampleGenerator:
    """Generate 'bad' versions of TypeScript code for contrastive training."""

    def __init__(self, methods: list[str] = None, seed: int = 42):
        self.methods = methods or [
            "introduce_type_error",
            "swap_operators",
            "remove_null_check",
            "add_unused_variable",
        ]
        self.rng = random.Random(seed)

    def generate(self, positive_code: str) -> tuple[str, str]:
        """Return (positive, negative) pair."""
        method = self.rng.choice(self.methods)
        negative = getattr(self, f"_method_{method}")(positive_code)
        if negative is None or negative == positive_code:
            # Fallback: try another method
            negative = self._method_add_unused_variable(positive_code)
        return positive_code, negative or positive_code

    def _method_introduce_type_error(self, code: str) -> str | None:
        """Change a number literal assignment to a string."""
        pattern = r'(:\s*number\s*=\s*)(\d+)'
        match = re.search(pattern, code)
        if not match:
            return None
        # Replace number with string
        return code[:match.start(2)] + '"not_a_number"' + code[match.end(2):]

    def _method_swap_operators(self, code: str) -> str | None:
        """Swap a comparison operator."""
        swaps = [("===", "!=="), (">", "<"), (">=", "<="), ("&&", "||")]
        for original, replacement in swaps:
            if original in code:
                idx = code.index(original)
                return code[:idx] + replacement + code[idx + len(original):]
        return None

    def _method_remove_null_check(self, code: str) -> str | None:
        """Remove a null/undefined guard."""
        patterns = [
            r'if\s*\([^)]*(?:!==?\s*null|!==?\s*undefined|!=\s*null)[^)]*\)\s*\{[^}]*\}',
            r'if\s*\([^)]+\?[^)]+\)',
        ]
        for pat in patterns:
            match = re.search(pat, code, re.DOTALL)
            if match:
                # Remove the entire if block
                return code[:match.start()] + code[match.end():]
        return None

    def _method_add_unused_variable(self, code: str) -> str:
        """Add a dead variable assignment at the top of a function."""
        # Find function body opening brace
        match = re.search(r'\{', code)
        if not match:
            return code + '\nconst _unused = "dead_code";\n'
        insertion_point = match.end()
        return (
            code[:insertion_point]
            + '\n  const _unused_var: string = "this variable is never used";\n'
            + code[insertion_point:]
        )

    def _method_wrong_generic_param(self, code: str) -> str | None:
        """Replace a concrete generic type with any."""
        match = re.search(r'<([A-Z][a-zA-Z]*)>', code)
        if match:
            return code[:match.start(1)] + "any" + code[match.end(1):]
        return None

    def _method_break_async_await(self, code: str) -> str | None:
        """Remove an await keyword."""
        if "await " not in code:
            return None
        return code.replace("await ", "", 1)
```

### Step 2: DPO Loss (`training/dpo_loss.py`)

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    model,
    ref_model,
    chosen_ids: torch.Tensor,     # (B, T) - good code tokens
    rejected_ids: torch.Tensor,   # (B, T) - bad code tokens
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Direct Preference Optimization loss.
    ref_model: frozen reference model (same architecture, initial weights)
    beta: temperature for DPO (lower = stronger preference signal)
    """
    def get_log_probs(m, input_ids):
        # Compute sum of log probs for the sequence
        logits = m(input_ids[:, :-1])   # (B, T-1, V)
        targets = input_ids[:, 1:]      # (B, T-1)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, V)
        token_log_probs = log_probs.gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        return token_log_probs.sum(dim=-1)  # (B,)

    with torch.no_grad():
        ref_chosen_lp = get_log_probs(ref_model, chosen_ids)
        ref_rejected_lp = get_log_probs(ref_model, rejected_ids)

    model_chosen_lp = get_log_probs(model, chosen_ids)
    model_rejected_lp = get_log_probs(model, rejected_ids)

    # DPO objective: maximize log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
    chosen_rewards = beta * (model_chosen_lp - ref_chosen_lp)
    rejected_rewards = beta * (model_rejected_lp - ref_rejected_lp)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    metrics = {
        "dpo_loss": loss.item(),
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "rejected_reward_mean": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
    }
    return loss, metrics
```

### Step 3: SimCLR Contrastive Loss (auxiliary) (`training/contrastive_loss.py`)

```python
import torch
import torch.nn.functional as F

def simclr_code_loss(
    embeddings_pos: torch.Tensor,   # (B, D) - good code embeddings
    embeddings_neg: torch.Tensor,   # (B, D) - bad code embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.
    Treats each (pos[i], neg[i]) pair; within the batch, other neg samples
    serve as additional negatives.
    """
    B = embeddings_pos.shape[0]

    # Normalize
    z_pos = F.normalize(embeddings_pos, dim=-1)
    z_neg = F.normalize(embeddings_neg, dim=-1)

    # Concatenate: [pos_0, pos_1, ..., neg_0, neg_1, ...]
    z_all = torch.cat([z_pos, z_neg], dim=0)  # (2B, D)

    # Similarity matrix
    sim = torch.mm(z_all, z_all.T) / temperature  # (2B, 2B)
    sim.fill_diagonal_(-1e9)  # mask self-similarity

    # Positive pairs: (pos_i, pos_i) ... but we only have one positive per anchor
    # Instead: treat pos[i] as anchor, pos[i] as positive, all neg[j] as negatives
    # Labels: for anchor pos[i], positive is at position i (the same index in pos half)
    labels = torch.arange(B).to(embeddings_pos.device)

    # Cross-entropy over negatives
    loss = F.cross_entropy(sim[:B, B:], labels)   # pos anchors vs all negatives
    return loss

def get_sequence_embedding(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract mean-pooled hidden state as sequence embedding."""
    with torch.no_grad():
        # For a decoder-only model, take last-layer hidden states
        hidden = model.get_hidden_states(input_ids)  # (B, T, D)
        # Mean pool (excluding padding if any)
        return hidden.mean(dim=1)  # (B, D)
```

### Step 4: Triplet Loss (`training/triplet_loss.py`)

```python
import torch
import torch.nn.functional as F

def triplet_code_loss(
    anchor_emb: torch.Tensor,     # (B, D) - context/prompt embedding
    positive_emb: torch.Tensor,   # (B, D) - good completion embedding
    negative_emb: torch.Tensor,   # (B, D) - bad completion embedding
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Triplet loss: anchor+positive should be closer than anchor+negative by margin.
    """
    d_pos = F.pairwise_distance(anchor_emb, positive_emb)
    d_neg = F.pairwise_distance(anchor_emb, negative_emb)
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss
```

### Step 5: Contrastive Dataset Builder (`data/contrastive_dataset.py`)

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ContrastiveSample:
    positive_ids: list[int]
    negative_ids: list[int]
    negative_method: str

class ContrastiveDatasetBuilder:
    def __init__(self, base_tokens: np.ndarray, tokenizer, config: dict):
        self.base_tokens = base_tokens
        self.tokenizer = tokenizer
        self.neg_gen = NegativeSampleGenerator(
            methods=config.get("negative_methods", None)
        )

    def build_sample(self, positive_tokens: list[int]) -> ContrastiveSample | None:
        positive_code = self.tokenizer.decode(positive_tokens)
        pos_code, neg_code = self.neg_gen.generate(positive_code)
        if pos_code == neg_code:
            return None

        negative_tokens = self.tokenizer.encode(neg_code)
        return ContrastiveSample(
            positive_ids=positive_tokens,
            negative_ids=negative_tokens,
            negative_method=self.neg_gen.methods[0],  # last used method
        )
```

### Step 6: Trainer Integration

```python
# In training/trainer.py, contrastive auxiliary loss:

if config.contrastive_learning.enabled and step % config.contrastive_learning.interval == 0:
    pos_batch, neg_batch = contrastive_data_loader.get_batch()

    if config.contrastive_learning.method == "dpo":
        c_loss, c_metrics = dpo_loss(
            model, ref_model,
            chosen_ids=pos_batch,
            rejected_ids=neg_batch,
            beta=config.contrastive_learning.beta,
        )
        total_loss = lm_loss + config.contrastive_learning.weight * c_loss

    elif config.contrastive_learning.method == "simclr":
        pos_emb = get_sequence_embedding(model, pos_batch)
        neg_emb = get_sequence_embedding(model, neg_batch)
        c_loss = simclr_code_loss(pos_emb, neg_emb)
        total_loss = lm_loss + config.contrastive_learning.weight * c_loss
```

### Step 7: Config

```yaml
contrastive_learning:
  enabled: false
  method: dpo              # dpo | simclr | triplet
  weight: 0.1              # multiplier for contrastive loss
  interval: 10             # run contrastive step every N training steps
  beta: 0.1                # DPO temperature
  negative_methods:
    - introduce_type_error
    - swap_operators
    - add_unused_variable
  data_fraction: 0.2       # use 20% of training data for contrastive
```

---

## Key Files to Modify

- `data/negative_generator.py` - New file: negative sample creation
- `data/contrastive_dataset.py` - New file: dataset builder
- `training/dpo_loss.py` - New file: DPO loss
- `training/contrastive_loss.py` - New file: SimCLR loss
- `training/triplet_loss.py` - New file: triplet loss
- `training/trainer.py` - Integrate contrastive auxiliary loss
- `config/training.yaml` - Add `contrastive_learning` section

---

## Testing Strategy

1. **Negative generator unit tests**: for each method, assert the output differs from input and is a valid string.
2. **DPO loss gradient test**: run `dpo_loss` with synthetic tensors, assert loss is differentiable (backward pass succeeds).
3. **SimCLR loss test**: identical pos/neg embeddings → loss near 0; orthogonal embeddings → high loss.
4. **Triplet margin test**: when d_pos < d_neg - margin, assert loss == 0.
5. **Contrastive dataset test**: build 100 contrastive samples from a fixture corpus, assert no sample has identical pos and neg tokens.
6. **Integration test**: run 100 training steps with DPO enabled, assert total_loss decreases and `reward_margin` metric is logged.

---

## Performance Considerations

- DPO requires a forward pass through both the training model and the reference model. This doubles compute for contrastive steps. Mitigate by running contrastive only every 10 steps (`interval: 10`).
- Reference model weights: load once at training start, kept frozen. This adds one model's worth of VRAM (same size as training model). On RTX 3080 (10GB), this may be the limiting factor for larger models.
- SimCLR requires computing embeddings via a separate forward pass. For decoder-only models, "embedding" means mean-pooling the final hidden states—this is not a standard forward pass and requires modifying the model to expose hidden states.
- Negative generation is pure Python string manipulation, ~1ms per sample. Not a bottleneck.

---

## Dependencies

No new Python dependencies. DPO and SimCLR are implemented from scratch using PyTorch primitives.

---

## Estimated Complexity

**High.** DPO implementation requires careful numerical stability (log-sum-exp tricks), reference model management, and integration with the existing training loop. SimCLR requires hidden state extraction which is not standard in a causal LM. Triplet loss is simpler but requires good negative mining. Estimated implementation time: 5-7 days for DPO + SimCLR.

---

## 2026 Best Practices

- **DPO over RLHF for code quality**: DPO (Rafailov et al. 2023) is the 2024-2026 standard for preference optimization without a separate reward model. For code quality where "good vs bad" pairs can be automatically generated, DPO is ideal.
- **Automatic negative generation is approximate**: regex-based negative generation creates plausible but noisy negatives. Some negatives may still compile (e.g., adding an unused variable doesn't break tsc). Consider running tsc to verify the negative actually fails before including it in training.
- **Weight the auxiliary loss conservatively**: start with `weight: 0.05`. Too high a contrastive weight can cause the model to collapse (all representations become similar to push bad code away, good code representations also shift).
- **Reference model management**: in DPO, the reference model should be the model checkpoint from before the contrastive fine-tuning stage starts. Freeze it completely (no gradients, eval mode). Consider offloading it to CPU and moving to GPU only during DPO steps to save VRAM.
- **Monitor reward margin**: the metric `chosen_reward - rejected_reward` (called "reward margin" or "implicit reward") should be positive and growing. If it's negative or not growing, the DPO training is ineffective.
