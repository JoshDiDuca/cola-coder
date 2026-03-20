# Feature 29: Adaptive Computation (Early Exit)

**Status:** Optional | **CLI Flag:** `--adaptive-exit` | **Complexity:** High

---

## Overview

Allow "easy" tokens to exit the transformer after fewer layers, skipping the remaining layers entirely. A lightweight confidence classifier is added after each transformer block. If the classifier's confidence exceeds a per-layer threshold, the current hidden state is projected directly to the output vocabulary and the token is emitted without processing through the remaining layers. Training uses auxiliary cross-entropy losses at each exit point. Expected speedup: 30–50% on code with repetitive or predictable patterns.

Reference: CALM (Confident Adaptive Language Modeling, Schuster et al., 2022).

---

## Motivation

Not every token requires the full depth of the transformer. Consider:
- Common keywords: `const`, `return`, `function` — easily predicted after 2-3 layers
- Closing brackets `}`, `;` — syntactically determined, minimal computation needed
- Repeated identifiers in a long function body

Adaptive computation exploits this:
- Easy tokens (high confidence early) exit at layer 3 of 12 → 75% layer savings
- Hard tokens (novel identifiers, type signatures) use all 12 layers as normal
- Net speedup depends on the fraction of "easy" tokens in typical code

---

## Architecture / Design

### Per-Layer Exit Points

```
Layer 0 → hidden_0
Layer 1 → hidden_1 → [exit_classifier_1] → confidence > threshold_1? → EARLY EXIT
Layer 2 → hidden_2 → [exit_classifier_2] → confidence > threshold_2? → EARLY EXIT
...
Layer N-1 → hidden_{N-1} → [standard LM head] → output
```

Each exit classifier is a small 2-layer MLP:
```
exit_classifier_k:
  input: hidden_k  [d_model]
  → Linear(d_model, exit_dim)  (exit_dim = 128-256)
  → GELU
  → Linear(exit_dim, vocab_size)
  → softmax
  → max confidence
```

### Exit Decision

```python
probs = softmax(exit_classifier_k(hidden_k))
confidence = probs.max(dim=-1)
if confidence > threshold[k]:
    emit probs.argmax()
    skip layers k+1...N-1
else:
    continue to next layer
```

### Training: Auxiliary Losses

```
L_total = L_main +  sum_{k=1}^{N-1} weight_k * L_auxiliary_k
where:
  L_main = CE at final layer (standard LM objective)
  L_auxiliary_k = CE at exit point k with hard labels
  weight_k = 0.3 * (k / N)  (earlier exits weighted less)
```

The auxiliary losses train each exit point to be independently predictive.

---

## Implementation Steps

### Step 1: Exit Classifier Module

```python
# cola_coder/model/adaptive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AdaptiveConfig:
    n_layers: int = 12
    d_model: int = 512
    vocab_size: int = 32000
    exit_dim: int = 256           # Hidden dim of exit classifier
    # Exit thresholds per layer (higher = less likely to exit)
    # Calibrated offline; later layers have lower thresholds
    thresholds: list[float] = field(default_factory=lambda: [0.9] * 12)
    # Which layers have exit classifiers (e.g., every other layer)
    exit_layers: list[int] = field(default_factory=list)  # Empty = all layers
    aux_loss_weight: float = 0.3
    min_layer: int = 2  # Never exit before this layer (avoid too-early exits)


class ExitClassifier(nn.Module):
    """Lightweight classifier for early exit decision."""
    def __init__(self, d_model: int, exit_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj1 = nn.Linear(d_model, exit_dim, bias=False)
        self.proj2 = nn.Linear(exit_dim, vocab_size, bias=False)
        nn.init.normal_(self.proj1.weight, std=0.02)
        nn.init.normal_(self.proj2.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns logits over vocab. [B, T, vocab_size] or [B, vocab_size]"""
        x = self.norm(hidden)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)

    def confidence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns max softmax confidence for each position. [B, T]"""
        logits = self.forward(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values
```

### Step 2: AdaptiveTransformer

```python
# cola_coder/model/adaptive_transformer.py
import torch
import torch.nn as nn
from .adaptive import ExitClassifier, AdaptiveConfig
from .transformer import TransformerBlock  # Existing block

class AdaptiveTransformer(nn.Module):
    """
    Transformer with per-layer early exit classifiers.
    During training: all layers run, aux losses are collected.
    During inference: exit when confidence > threshold.
    """
    def __init__(self, model_cfg, adaptive_cfg: AdaptiveConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.adaptive_cfg = adaptive_cfg

        # Main transformer components
        self.embedding = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        # Exit classifiers
        exit_layer_set = set(adaptive_cfg.exit_layers or range(model_cfg.n_layers - 1))
        self.exit_classifiers = nn.ModuleDict({
            str(k): ExitClassifier(
                model_cfg.d_model,
                adaptive_cfg.exit_dim,
                model_cfg.vocab_size,
            )
            for k in range(model_cfg.n_layers - 1)
            if k in exit_layer_set and k >= adaptive_cfg.min_layer
        })

    def forward_train(
        self,
        input_ids: torch.Tensor,
        freqs_cis,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Training forward: run ALL layers, collect auxiliary losses.
        Returns (main_logits, {exit_logits_dict}).
        """
        x = self.embedding(input_ids)
        exit_logits = {}

        for k, block in enumerate(self.blocks):
            x = block(x, freqs_cis, mask)
            key = str(k)
            if key in self.exit_classifiers:
                exit_logits[k] = self.exit_classifiers[key](x)

        x = self.norm(x)
        main_logits = self.lm_head(x)
        return main_logits, exit_logits

    @torch.no_grad()
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        freqs_cis,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Inference forward: exit early if confidence exceeds threshold.
        Returns (logits, metadata).
        """
        x = self.embedding(input_ids)
        thresholds = self.adaptive_cfg.thresholds
        meta = {"exit_layer": None, "early_exit": False}

        for k, block in enumerate(self.blocks):
            x = block(x, freqs_cis, mask)
            key = str(k)
            if key in self.exit_classifiers:
                exit_logits = self.exit_classifiers[key](x)
                # Only check last token position (generation step)
                last_probs = F.softmax(exit_logits[:, -1, :], dim=-1)
                confidence = last_probs.max(dim=-1).values  # [B]
                threshold = thresholds[k] if k < len(thresholds) else 0.9

                if confidence.min().item() >= threshold:
                    meta["exit_layer"] = k
                    meta["early_exit"] = True
                    return exit_logits, meta

        x = self.norm(x)
        main_logits = self.lm_head(x)
        meta["exit_layer"] = len(self.blocks) - 1
        return main_logits, meta
```

### Step 3: Auxiliary Loss Computation

```python
# cola_coder/training/adaptive_loss.py
import torch
import torch.nn.functional as F
from cola_coder.model.adaptive import AdaptiveConfig

def compute_adaptive_loss(
    main_logits: torch.Tensor,
    exit_logits: dict[int, torch.Tensor],
    input_ids: torch.Tensor,
    adaptive_cfg: AdaptiveConfig,
    n_layers: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total loss = main LM loss + weighted auxiliary exit losses.
    """
    B, T, V = main_logits.shape
    targets = input_ids[:, 1:]  # Next token targets [B, T-1]
    breakdown = {}

    # Main LM loss
    main_loss = F.cross_entropy(
        main_logits[:, :-1, :].reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    breakdown["loss_main"] = main_loss.item()
    total_loss = main_loss

    # Auxiliary losses at each exit point
    for layer_k, logits_k in exit_logits.items():
        # Weight: earlier exits get less weight (they have harder job)
        weight = adaptive_cfg.aux_loss_weight * ((layer_k + 1) / n_layers)
        aux_loss = F.cross_entropy(
            logits_k[:, :-1, :].reshape(-1, V),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )
        breakdown[f"loss_exit_{layer_k}"] = aux_loss.item()
        total_loss = total_loss + weight * aux_loss

    return total_loss, breakdown
```

### Step 4: Threshold Calibration

```python
# cola_coder/model/threshold_calibration.py
import torch
import numpy as np
from typing import Iterator

def calibrate_exit_thresholds(
    model,
    val_loader,
    device: str = "cuda",
    target_speedup: float = 1.5,  # Desired speedup ratio
    min_accuracy: float = 0.99,   # Min accuracy relative to full model
) -> list[float]:
    """
    Find per-layer thresholds that achieve target_speedup while
    keeping accuracy above min_accuracy.

    Strategy: binary search on threshold values.
    """
    model.eval().to(device)
    # Collect (layer_k, confidence, correct) tuples from val set
    layer_stats: dict[int, list[tuple[float, bool]]] = {}

    exit_layers = [int(k) for k in model.exit_classifiers.keys()]

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            # Get full model logits (reference)
            main_logits, exit_logits_dict = model.forward_train(
                input_ids, model.freqs_cis[:input_ids.shape[1]], None
            )
            main_preds = main_logits[:, :-1, :].argmax(dim=-1)  # [B, T-1]
            targets = input_ids[:, 1:]

            for k, exit_logits in exit_logits_dict.items():
                exit_preds = exit_logits[:, :-1, :].argmax(dim=-1)
                exit_probs = F.softmax(exit_logits[:, :-1, :], dim=-1)
                confidence = exit_probs.max(dim=-1).values  # [B, T-1]

                match = (exit_preds == targets).reshape(-1)
                conf_flat = confidence.reshape(-1)

                if k not in layer_stats:
                    layer_stats[k] = []
                for conf_val, is_match in zip(conf_flat.cpu().tolist(), match.cpu().tolist()):
                    layer_stats[k].append((conf_val, is_match))

    # For each layer, find threshold that gives acceptable accuracy
    thresholds = [1.0] * (max(exit_layers) + 2)
    for k in sorted(layer_stats.keys()):
        stats = sorted(layer_stats[k], key=lambda x: -x[0])  # Sort by confidence desc
        # Find threshold where accuracy >= min_accuracy
        cumulative_correct = 0
        cumulative_total = 0
        for conf, correct in stats:
            cumulative_correct += int(correct)
            cumulative_total += 1
            if cumulative_total > 0:
                accuracy = cumulative_correct / cumulative_total
                if accuracy >= min_accuracy:
                    thresholds[k] = conf
                    break

    print(f"Calibrated thresholds: {[f'{t:.3f}' for t in thresholds[:len(exit_layers)]]}")
    return thresholds
```

### Step 5: Speedup Tracker + CLI

```python
# cola_coder/model/adaptive_stats.py
from dataclasses import dataclass

@dataclass
class AdaptiveStats:
    exit_layer_counts: dict[int, int]
    total_tokens: int

    @property
    def mean_exit_layer(self) -> float:
        total = sum(self.exit_layer_counts.values())
        weighted = sum(k * c for k, c in self.exit_layer_counts.items())
        return weighted / total if total > 0 else 0.0

    @property
    def estimated_speedup(self) -> float:
        n_layers = max(self.exit_layer_counts.keys()) + 1
        return n_layers / (self.mean_exit_layer + 1)


@app.command()
def calibrate_thresholds(
    checkpoint: str = typer.Argument(...),
    val_data: str = typer.Option("data/val.jsonl"),
    target_speedup: float = typer.Option(1.5),
    min_accuracy: float = typer.Option(0.99),
    output: str = typer.Option("configs/adaptive_thresholds.yaml"),
):
    """Calibrate adaptive exit thresholds on validation data."""
    import yaml
    model = load_model(checkpoint)
    thresholds = calibrate_exit_thresholds(
        model, make_val_loader(val_data),
        target_speedup=target_speedup,
        min_accuracy=min_accuracy,
    )
    yaml.dump({"thresholds": thresholds}, open(output, "w"))
    console.print(f"[green]Saved thresholds to {output}[/green]")
    console.print(f"Expected speedup: {sum(thresholds)/len(thresholds):.2f}x")
```

---

## Key Files to Modify

- `cola_coder/model/adaptive.py` — ExitClassifier, AdaptiveConfig
- `cola_coder/model/adaptive_transformer.py` — AdaptiveTransformer
- `cola_coder/training/adaptive_loss.py` — auxiliary loss computation
- `cola_coder/model/threshold_calibration.py` — offline calibration
- `cola_coder/model/adaptive_stats.py` — speedup tracking
- `cola_coder/training/trainer.py` — use adaptive loss when enabled
- `cola_coder/generate.py` — switch to `forward_inference` when `--adaptive-exit`
- `cola_coder/cli.py` — `calibrate-thresholds` command
- `configs/adaptive.yaml` — exit_layers, thresholds, aux_loss_weight

---

## Testing Strategy

```python
def test_exit_classifier_output_shape():
    clf = ExitClassifier(d_model=64, exit_dim=32, vocab_size=100)
    hidden = torch.randn(2, 16, 64)
    logits = clf(hidden)
    assert logits.shape == (2, 16, 100)

def test_adaptive_transformer_train_mode():
    # All layers should run in training mode
    model = AdaptiveTransformer(small_cfg, adaptive_cfg)
    x = torch.randint(0, 32000, (1, 32))
    main_logits, exit_logits = model.forward_train(x, freqs, mask)
    assert len(exit_logits) > 0
    assert main_logits.shape == (1, 32, 32000)

def test_early_exit_at_high_confidence():
    # Mock model where exit classifier outputs very high confidence
    # Verify that forward_inference returns before last layer
    pass

def test_adaptive_loss_decreases():
    # Train for 10 steps, verify total loss decreases
    pass

def test_no_quality_degradation_at_threshold_1():
    # With threshold=1.0 (never exit early), output should match full model
    model.adaptive_cfg.thresholds = [1.0] * 12
    # ... outputs should be identical to standard forward pass
```

---

## Performance Considerations

- **Exit classifier overhead:** Each ExitClassifier is a 2-layer MLP with exit_dim=256. At d_model=512, cost is ~2×512×256 MACs per token position — small compared to attention layers.
- **Calibration data size:** Calibrate on 10,000+ tokens for reliable threshold estimates. Too few tokens leads to overfitted thresholds.
- **Token-level vs sequence-level:** This implementation uses token-level adaptive computation. Sequence-level (exit entire sequence early) is simpler but less granular.
- **KV cache complexity:** When a token exits early, its KV pairs for the skipped layers don't need to be computed or stored. This saves both compute and memory in the KV cache.
- **Hardware efficiency:** On GPU, conditional execution (if-branch per token) is difficult to batch efficiently. Consider halting-exit (process in batches, use masking instead of true early exit) for better GPU utilization.

---

## Dependencies

- Existing Cola-Coder transformer (TransformerBlock)
- Feature 27 (MTP) — optional synergy: MTP heads and exit classifiers share structure
- `scipy` for threshold binary search optimization

---

## Estimated Complexity

| Task                              | Effort   |
|-----------------------------------|----------|
| ExitClassifier + AdaptiveConfig   | 2h       |
| AdaptiveTransformer (train+infer) | 5h       |
| Auxiliary loss computation        | 2h       |
| Threshold calibration             | 3h       |
| Speedup stats tracking            | 1h       |
| CLI + config                      | 1.5h     |
| Tests                             | 2h       |
| Debugging (confidence calibration can be tricky) | 3h |
| **Total**                         | **~19.5h** |

Overall complexity: **High** (conditional execution is complex, calibration is non-trivial, GPU batching requires careful design)

---

## 2026 Best Practices

- **CALM reference implementation:** The CALM paper (Schuster et al., 2022) provides the theoretical grounding and empirical validation for token-level adaptive computation in transformers. Study it before implementing.
- **Calibrate on target distribution:** Thresholds calibrated on general text will be wrong for code. Always calibrate on a held-out TypeScript/JavaScript corpus representative of the actual use case.
- **Halting criterion alternatives:** Instead of max softmax confidence, try: (a) entropy of the distribution (lower = more certain), (b) difference between top-2 probabilities, (c) trained binary classifier. Max softmax is simple but known to be overconfident.
- **Layer stacking for skipped layers:** Some papers concatenate the hidden state with a "skip embedding" before the exit LM head to give the head information about how many layers were skipped. This can improve exit quality at early layers.
- **Soft exit for training stability:** During training, instead of hard early exit, use soft mixing: output = alpha * exit_logits + (1-alpha) * continue_through_remaining_layers. This avoids gradient disconnection. Use hard exit only at inference.
- **Profile before deploying:** Measure actual wall-clock speedup on your specific GPU. On A100/H100 with tensor parallelism, the speedup may be smaller than expected due to communication overhead. On single-GPU setups like RTX 3080/4080, speedup is more predictable.
