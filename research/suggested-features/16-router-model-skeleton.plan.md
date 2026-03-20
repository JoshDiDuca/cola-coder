# Feature 16: Router Model Skeleton

**Status:** Optional | **CLI Flag:** `--use-router` | **Complexity:** Medium

---

## Overview

A lightweight classifier model that reads a code prompt and outputs a specialist domain ID. The router is a small transformer (or MLP) with shared embeddings that performs a fast forward pass, producing softmax probabilities over N specialist domains. It routes each prompt to the most appropriate specialist model (React, Next.js, GraphQL, Prisma, Zod, Testing, General TS) before generation begins.

The router must be fast enough to add negligible latency — target <5M parameters and <10ms inference on CPU.

---

## Motivation

Cola-Coder will eventually host multiple specialist models fine-tuned on domain-specific TypeScript/JavaScript code. Without a router, the user must manually select a specialist or the system falls back to the general model always. A learned router:

- Automatically selects the best specialist for a given prompt
- Improves generation quality by matching prompt domain to expert knowledge
- Keeps the system simple for users who don't know which specialist to pick
- Enables future ensemble and cascade routing strategies (features 19, 20, 23)

A <5M param model adds essentially zero overhead compared to the specialist forward pass.

---

## Architecture / Design

### Model Architecture

```
Input: token sequence (prompt, truncated to 128 tokens)
         ↓
Embedding Layer  [vocab_size × embed_dim=128]
         ↓
Positional Encoding (learned or RoPE-lite)
         ↓
2-3 Transformer Encoder Layers
  - 4 attention heads
  - FFN dim = 256
  - RMSNorm
  - No causal mask (bidirectional for classification)
         ↓
CLS pooling (mean over sequence or [CLS] token)
         ↓
Linear Classification Head [embed_dim → num_domains]
         ↓
Softmax → domain probabilities
```

### Domain Labels

```python
DOMAINS = {
    0: "react",
    1: "nextjs",
    2: "graphql",
    3: "prisma",
    4: "zod",
    5: "testing",
    6: "general_ts",
}
NUM_DOMAINS = 7
```

### Parameter Budget

| Component         | Params (approx) |
|-------------------|----------------|
| Embedding (32k vocab, 128 dim) | 4.1M |
| 2x Transformer layers          | 0.5M |
| Classification head            | 0.9K |
| **Total**                      | **~4.6M** |

This fits comfortably under 5M. If vocabulary is shared with the main model's tokenizer, embedding weights can be frozen/copied from a pretrained checkpoint.

---

## Implementation Steps

### Step 1: Define RouterConfig

```python
# cola_coder/router/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RouterConfig:
    vocab_size: int = 32000
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    max_seq_len: int = 128
    num_domains: int = 7
    dropout: float = 0.1
    use_cls_token: bool = True
    pooling: str = "mean"  # "mean" | "cls" | "max"
    checkpoint_path: Optional[str] = None
```

### Step 2: Implement RouterModel

```python
# cola_coder/router/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import RouterConfig

class RouterAttention(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # Bidirectional (no causal mask) — full self-attention
        attn = torch.einsum("bthd,bshd->bhts", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, C)
        return self.proj(out)


class RouterBlock(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.embed_dim)
        self.attn = RouterAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RouterModel(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Embedding(cfg.max_seq_len + 1, cfg.embed_dim)
        self.blocks = nn.ModuleList([RouterBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_domains)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            T = T + 1
        positions = torch.arange(T, device=input_ids.device)
        x = x + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Pooling
        if self.cfg.pooling == "cls" and self.cfg.use_cls_token:
            pooled = x[:, 0, :]
        elif self.cfg.pooling == "mean":
            pooled = x.mean(dim=1)
        else:  # max
            pooled = x.max(dim=1).values
        logits = self.head(pooled)
        return logits

    def predict(self, input_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Returns (domain_id, probabilities)."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits, dim=-1)
            domain_id = probs.argmax(dim=-1).item()
        return domain_id, probs

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

### Step 3: Router Trainer

```python
# cola_coder/router/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import RouterModel
from .config import RouterConfig

def train_router(
    model: RouterModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = "cuda",
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        val_acc = evaluate_router(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.3f}")
```

### Step 4: CLI Integration

```python
# cola_coder/cli.py  (additions)
@app.command()
def route(
    prompt: str = typer.Argument(..., help="Code prompt to classify"),
    router_checkpoint: str = typer.Option(None, "--router-ckpt"),
    show_probs: bool = typer.Option(False, "--show-probs"),
):
    """Route a prompt to the appropriate specialist domain."""
    router = load_router(router_checkpoint)
    domain_id, probs = router.predict(tokenize(prompt))
    domain_name = DOMAINS[domain_id]
    console.print(f"[bold green]Routed to:[/bold green] {domain_name}")
    if show_probs:
        for i, (name, p) in enumerate(zip(DOMAINS.values(), probs[0].tolist())):
            bar = "█" * int(p * 20)
            console.print(f"  {name:15s} {bar:20s} {p:.3f}")
```

---

## Key Files to Modify

- `cola_coder/router/__init__.py` — new package
- `cola_coder/router/config.py` — RouterConfig dataclass
- `cola_coder/router/model.py` — RouterModel nn.Module
- `cola_coder/router/trainer.py` — training loop
- `cola_coder/router/inference.py` — load + predict helpers
- `cola_coder/cli.py` — add `route` and `train-router` subcommands
- `configs/router.yaml` — default router config
- `cola_coder/specialist_registry.py` — integration point (feature 18)

---

## Testing Strategy

```python
# tests/test_router_model.py
def test_router_forward_pass():
    cfg = RouterConfig(vocab_size=1000, embed_dim=64, num_heads=4, num_layers=2)
    model = RouterModel(cfg)
    input_ids = torch.randint(0, 1000, (2, 64))
    logits = model(input_ids)
    assert logits.shape == (2, cfg.num_domains)

def test_router_param_count():
    cfg = RouterConfig()
    model = RouterModel(cfg)
    assert model.param_count() < 5_000_000

def test_router_predict():
    cfg = RouterConfig(vocab_size=1000, embed_dim=64)
    model = RouterModel(cfg)
    input_ids = torch.randint(0, 1000, (1, 32))
    domain_id, probs = model.predict(input_ids)
    assert 0 <= domain_id < cfg.num_domains
    assert abs(probs.sum().item() - 1.0) < 1e-5

def test_router_inference_speed():
    import time
    cfg = RouterConfig()
    model = RouterModel(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 128))
    # Warm up
    for _ in range(3):
        model.predict(input_ids)
    start = time.perf_counter()
    for _ in range(100):
        model.predict(input_ids)
    elapsed = (time.perf_counter() - start) / 100
    assert elapsed < 0.010, f"Router inference too slow: {elapsed*1000:.1f}ms"
```

---

## Performance Considerations

- **Model size:** Keep embed_dim=128, 2 layers. Larger models have diminishing routing accuracy returns vs latency cost.
- **Tokenization:** Truncate prompts to 128 tokens for router — sufficient for import detection; full prompt not needed.
- **Batching:** Router supports batch inference; if generating in batch mode, route all prompts simultaneously.
- **Device:** CPU inference is acceptable at <5M params. Move to GPU only if batching many prompts.
- **Caching:** Cache routing decisions for identical prompts using an LRU cache on the token hash.
- **Quantization:** INT8 quantize the router (torch.ao.quantization) for further speedup on CPU.

---

## Dependencies

- PyTorch >= 2.2 (for `nn.RMSNorm`)
- Existing Cola-Coder tokenizer (shared vocab)
- Feature 17 (training data generator) for labeled data
- Feature 18 (specialist registry) for dispatch after routing
- Feature 19 (confidence-based routing) for fallback logic

---

## Estimated Complexity

| Task                        | Effort   |
|-----------------------------|----------|
| RouterConfig + RouterModel  | 2h       |
| Training loop               | 2h       |
| CLI integration             | 1h       |
| Tests                       | 1h       |
| Integration with registry   | 1h       |
| **Total**                   | **~7h**  |

Overall complexity: **Medium** (self-contained, small model, straightforward classification)

---

## 2026 Best Practices

- **RMSNorm over LayerNorm:** Already standard in modern transformers; lower compute cost.
- **Label smoothing (0.1):** Prevents overconfident routing on ambiguous prompts.
- **Mean pooling over CLS:** Mean pooling consistently matches or beats CLS token for sentence classification in 2024-2026 research.
- **Bidirectional attention for classification:** Unlike the generative model, the router should see the full prompt context.
- **Shared tokenizer:** Reuse the main model's tokenizer/vocabulary; avoids a separate tokenization pipeline.
- **Distillation option:** If a larger domain classifier exists, distill its soft labels into the router for better calibration.
- **torch.compile():** Apply `torch.compile(model)` for ~20% speedup on repeated CPU inference (PyTorch 2.x).
- **Safetensors checkpoint:** Save router weights in safetensors format, consistent with the rest of Cola-Coder.
