# Feature 32: Knowledge Distillation

## Overview

Knowledge distillation trains a smaller "student" model to mimic the output distribution
of a larger "teacher" model. Instead of training only on hard labels (correct next token),
the student also learns from the teacher's full probability distribution over the
vocabulary — the "soft targets." These soft targets carry more information: the teacher
expresses nuanced uncertainty across semantically similar tokens.

For Cola-Coder, the teacher can be Claude via the Anthropic API, a local larger
checkpoint, or a pretrained open model (CodeLlama, DeepSeek-Coder). This lets a small
model punch above its weight on code generation.

Status: OPTIONAL — enable via `--feature distillation` or CLI menu toggle.

---

## Motivation

- Hard-label training only tells the model "token X is correct." Soft targets tell it
  "token X is very likely, token Y is somewhat plausible, token Z is not plausible at all."
- This richer signal accelerates learning and improves generalization.
- Practical: you have access to Claude API (strong teacher) and a small local model
  (cheap inference). Distillation is the principled way to transfer knowledge.
- Offline distillation pre-computes teacher logits once, amortizing API cost over many
  training epochs.

---

## Architecture / Design

### Loss Function

The combined distillation loss is:

```
L_total = (1 - alpha) * L_CE(student_logits, hard_labels)
        + alpha       * L_KL(student_logits/T, teacher_logits/T) * T^2
```

Where:
- `T` = temperature (2–4 typical; softens both distributions for better gradient signal)
- `alpha` = distillation weight (0.5–0.9; higher = rely more on teacher)
- `T^2` corrects for the temperature scaling on gradients

```python
# cola_coder/distillation/loss.py

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,  # (B, T, V)
    teacher_logits: torch.Tensor,  # (B, T, V)
    hard_labels: torch.Tensor,     # (B, T) — integer token ids
    temperature: float = 2.0,
    alpha: float = 0.7,
    ignore_index: int = -100,
) -> torch.Tensor:
    B, T, V = student_logits.shape

    # Hard-label cross-entropy
    ce_loss = F.cross_entropy(
        student_logits.view(B * T, V),
        hard_labels.view(B * T),
        ignore_index=ignore_index,
    )

    # Soft-target KL divergence
    # Only compute over non-padding positions
    mask = hard_labels.view(B * T) != ignore_index
    s_log_soft = F.log_softmax(student_logits.view(B * T, V)[mask] / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits.view(B * T, V)[mask] / temperature, dim=-1)
    kl_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (temperature ** 2)

    return (1.0 - alpha) * ce_loss + alpha * kl_loss
```

### Offline Teacher Data Collection

Pre-compute and cache teacher logits/completions so they are not recomputed every epoch:

```python
# cola_coder/distillation/collect_teacher.py

import json
import numpy as np
from pathlib import Path
from anthropic import Anthropic


def collect_teacher_logprobs_claude(
    prompts: list[str],
    output_dir: Path,
    model: str = "claude-3-5-haiku-20241022",  # cheapest capable model
    max_tokens: int = 256,
) -> None:
    """
    For each prompt, get Claude's completion and top-5 token logprobs.
    Saves one .jsonl file per batch.
    NOTE: Claude API returns logprobs at the character/token level.
    """
    client = Anthropic()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, prompt in enumerate(prompts):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        completion = response.content[0].text
        records.append({"prompt": prompt, "completion": completion})
        if (i + 1) % 100 == 0:
            print(f"Collected {i+1}/{len(prompts)}")

    with open(output_dir / "teacher_completions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def collect_teacher_logits_local(
    prompts: list[str],
    teacher_model,          # loaded nn.Module
    tokenizer,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    """
    For a local teacher checkpoint, save full vocabulary logits.
    Much richer signal than API-based collection.
    """
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_model.eval().to(device)

    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = teacher_model(tokens).logits  # (1, T, V)
        # Save as float16 to halve disk usage
        np.save(
            output_dir / f"logits_{i:06d}.npy",
            logits.cpu().to(torch.float16).numpy(),
        )
```

### Training Pipeline

```python
# cola_coder/distillation/trainer.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from .loss import distillation_loss


class DistillationTrainer:
    def __init__(
        self,
        student_model,
        teacher_logits_dir: Path | None,   # None = teacher-free (CE only)
        tokenizer,
        config: "DistillationConfig",
    ):
        self.student = student_model
        self.teacher_dir = teacher_logits_dir
        self.tokenizer = tokenizer
        self.cfg = config

    def load_teacher_logits(self, idx: int) -> torch.Tensor | None:
        if self.teacher_dir is None:
            return None
        path = self.teacher_dir / f"logits_{idx:06d}.npy"
        if not path.exists():
            return None
        import numpy as np
        return torch.from_numpy(np.load(path)).float()

    def train_step(self, batch: dict, step: int) -> dict:
        input_ids = batch["input_ids"].to(self.cfg.device)
        labels = batch["labels"].to(self.cfg.device)
        teacher_logits = batch.get("teacher_logits")
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(self.cfg.device)

        student_logits = self.student(input_ids)  # (B, T, V)

        if teacher_logits is not None:
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=self.cfg.temperature,
                alpha=self.cfg.alpha,
            )
        else:
            import torch.nn.functional as F
            B, T, V = student_logits.shape
            loss = F.cross_entropy(
                student_logits.view(B * T, V),
                labels.view(B * T),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": student_logits}
```

### Config Dataclass

```python
# in config.py

@dataclass
class DistillationConfig:
    enabled: bool = False
    temperature: float = 2.0
    alpha: float = 0.7            # weight for KL loss vs CE loss
    teacher_type: str = "claude"  # "claude" | "local_checkpoint" | "none"
    teacher_model_path: str = ""  # path to local teacher checkpoint
    teacher_logits_dir: str = "data/teacher_logits"
    collect_on_start: bool = False  # run collection pass before training
    claude_model: str = "claude-3-5-haiku-20241022"
    max_teacher_tokens: int = 256
```

---

## Implementation Steps

1. **Create `cola_coder/distillation/` package**: `__init__.py`, `loss.py`,
   `collect_teacher.py`, `trainer.py`, `dataset.py`.

2. **Add `DistillationConfig`** to `config.py`.

3. **Implement `DistillationDataset`** that reads tokenized prompts alongside
   pre-computed teacher logits (when available):
   ```python
   class DistillationDataset(torch.utils.data.Dataset):
       def __getitem__(self, idx):
           tokens = self.tokenized[idx]
           teacher = self.load_teacher_logits(idx)  # None if not available
           return {"input_ids": tokens[:-1], "labels": tokens[1:],
                   "teacher_logits": teacher}
   ```

4. **Add CLI command** "Collect teacher data" — prompts for teacher type, data path,
   then runs collection. Expensive step done once.

5. **Integrate into `train.py`**: if `cfg.distillation.enabled`, wrap training loop
   with `DistillationTrainer`.

6. **Evaluation**: after distillation, run HumanEval and compare pass@k vs baseline
   to quantify knowledge transfer.

7. **Handle vocabulary mismatch**: if teacher and student use different tokenizers,
   align logits by mapping teacher vocab to student vocab (lossy but workable).

---

## Key Files to Modify

| File | Change |
|---|---|
| `config.py` | Add `DistillationConfig` |
| `train.py` | Accept `--distill` flag, use `DistillationTrainer` |
| `cli/menu.py` | Add "Collect teacher data" and "Train with distillation" options |
| `cola_coder/distillation/` | New package (all new files) |
| `requirements.txt` | Add `anthropic` (already likely present) |

---

## Testing Strategy

```python
# tests/test_distillation.py

def test_distillation_loss_is_lower_with_matching_teacher():
    """When student matches teacher exactly, KL component should be near zero."""
    V = 100
    logits = torch.randn(2, 10, V)
    labels = torch.randint(0, V, (2, 10))
    # Teacher == student
    loss_matched = distillation_loss(logits, logits.clone(), labels, alpha=0.9)
    # Teacher is random
    teacher_random = torch.randn(2, 10, V)
    loss_random = distillation_loss(logits, teacher_random, labels, alpha=0.9)
    assert loss_matched < loss_random

def test_distillation_loss_alpha_zero_equals_ce():
    """alpha=0 should give pure cross-entropy."""
    import torch.nn.functional as F
    V = 100
    logits = torch.randn(2, 10, V)
    labels = torch.randint(0, V, (2, 10))
    teacher = torch.randn(2, 10, V)
    dist_loss = distillation_loss(logits, teacher, labels, alpha=0.0)
    ce_loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    assert torch.allclose(dist_loss, ce_loss, atol=1e-5)

def test_teacher_logits_saved_and_loaded():
    import numpy as np, tempfile
    tmp = Path(tempfile.mkdtemp())
    logits = torch.randn(1, 32, 512).to(torch.float16)
    np.save(tmp / "logits_000000.npy", logits.numpy())
    loaded = torch.from_numpy(np.load(tmp / "logits_000000.npy")).float()
    assert loaded.shape == logits.shape
```

---

## Performance Considerations

- **Offline vs online distillation**: online = teacher runs every step (expensive if API).
  Offline = pre-compute once (recommended). Storage: ~10 MB per 1 000 prompts at
  fp16 with 4K vocab, more with larger vocab.
- **Temperature**: higher T (3–5) smooths teacher distribution further, helping the
  student learn from tails. Too high collapses everything toward uniform.
- **Alpha scheduling**: start with `alpha=0.9` (lean heavily on teacher) and decay
  toward `alpha=0.3` in final epochs (learn from data distribution).
- **Sequence length**: teacher logit files scale as `O(T * V)`. For T=512, V=32768,
  fp16: 512 * 32768 * 2 bytes = 32 MB per sample. Use small vocab (4096-16384) or
  save only top-K logits (top-100 usually sufficient for KL accuracy).
- **Top-K teacher logits trick**:
  ```python
  top_k_vals, top_k_idx = logits.topk(100, dim=-1)
  # Store sparse: indices + values instead of full V
  ```

---

## Dependencies

```
anthropic>=0.30.0    # Claude API teacher (optional, only if teacher_type=claude)
numpy>=1.26.0        # logit storage (already required)
torch>=2.2.0         # base requirement
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Loss function | 1 hour |
| Teacher collection (Claude API) | 2 hours |
| Teacher collection (local model) | 2 hours |
| DistillationDataset | 2 hours |
| DistillationTrainer integration | 3 hours |
| CLI integration | 1 hour |
| Tests + evaluation | 2 hours |
| **Total** | **~13 hours** |

Complexity rating: **Medium** — loss function is straightforward; main complexity is
data pipeline for offline teacher logits.

---

## 2026 Best Practices

- **Speculative decoding synergy**: distillation from a large teacher + speculative
  decoding (small model drafts, large model verifies) is the standard inference stack
  in 2026. Distillation improves draft quality.
- **RLHF data augmentation**: use distillation on human-preference data by having the
  teacher generate multiple completions ranked by quality; train student to match
  teacher's chosen completion distribution.
- **Generalized KD (GKD)**: train student on-policy (use student's own generated
  completions as training data) to reduce train/inference distribution mismatch.
  Shown to outperform standard offline KD for code generation.
- **Layer-level distillation**: beyond final logits, distill intermediate hidden states
  and attention patterns (hint-based KD). Adds complexity but helps smaller models
  develop similar internal representations.
- **Vocabulary-free distillation**: when teacher and student have different tokenizers,
  use embedding-space alignment instead of logit matching to avoid vocabulary mapping.
