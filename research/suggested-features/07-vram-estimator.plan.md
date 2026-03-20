# 07 - VRAM Estimator

## Overview

Given a model config YAML, estimate peak VRAM usage before training begins. The estimator accounts for model parameters, optimizer states, activations, gradient accumulation, data type (bf16/fp16/fp32), gradient checkpointing savings, and KV-cache for inference. Outputs a detailed breakdown table and warns if the estimated usage would exceed the detected GPU capacity.

---

## Motivation

One of the most frustrating training failures is an out-of-memory (OOM) crash 10 minutes into a run. This is entirely preventable with a pre-flight VRAM check. New users frequently misconfigure batch size or model dimensions and only discover the problem after the GPU crashes.

The estimator serves multiple purposes:
- **Pre-flight check**: Run before `cola-train.ps1` to verify feasibility
- **Config tuning**: Experiment with different batch sizes and see instant VRAM impact
- **Architecture sizing**: Decide between small/medium/large model configs
- **Documentation**: The breakdown table explains what VRAM actually goes where

---

## Architecture / Design

### VRAM Components

```
Total VRAM = Model Parameters
           + Optimizer States
           + Activations (forward pass)
           + Gradients
           + Framework Overhead
           + Safety Buffer
```

#### 1. Model Parameters

```
model_params * bytes_per_param

bytes_per_param:
  fp32 = 4 bytes
  fp16 = 2 bytes
  bf16 = 2 bytes
```

#### 2. Optimizer States (Adam/AdamW)

Adam stores 3 copies of the parameters: gradients (1x), momentum (1x), variance (1x).
In mixed precision training, optimizer states are typically kept in fp32:

```
optimizer_states = model_params * 4 bytes * 3  (fp32, always)
```

For 8-bit Adam (bitsandbytes), the factor drops to approximately 1.5x instead of 3x.

#### 3. Activations

Activations are proportional to batch size, sequence length, model dimension, and number of layers. The formula for a decoder-only transformer:

```
activations_per_layer = batch_size * seq_len * (
    4 * dim           (attention QKV + output)
  + 8 * dim           (FFN intermediate, SwiGLU has 2 gate projections)
)

total_activations = activations_per_layer * n_layers * bytes_per_activation
```

With gradient checkpointing: recompute activations on backward pass, storing only 1 layer at a time:
```
activations_with_checkpointing ≈ total_activations * sqrt(n_layers) / n_layers
                                ≈ total_activations * 0.3  (rule of thumb: ~50-70% reduction)
```

#### 4. Gradients

Same size as model parameters, stored in fp32 for mixed precision:
```
gradients = model_params * 4 bytes
```

#### 5. Framework Overhead

PyTorch allocates CUDA memory in blocks with fragmentation. Empirical overhead is ~500MB-1GB:
```
overhead = 512 MB (conservative)
```

#### 6. KV-Cache (Inference Only)

For inference with kv-cache:
```
kv_cache = 2 * n_kv_heads * head_dim * n_layers * seq_len * batch_size * bytes_per_param
```
(Factor of 2 for keys and values.)

---

## Implementation Steps

### Step 1: Config reader

```python
# src/tools/vram_estimator.py
import yaml
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    ffn_multiplier: float      # e.g. 2.67 for SwiGLU
    max_seq_len: int

@dataclass
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    dtype: str                 # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool
    optimizer: str             # "adamw", "adam8bit"

def load_configs_from_yaml(config_path: Path) -> tuple[ModelConfig, TrainingConfig]:
    data = yaml.safe_load(config_path.read_text())

    model = data.get("model", {})
    training = data.get("training", {})

    mc = ModelConfig(
        vocab_size=model.get("vocab_size", 32000),
        dim=model.get("dim", 512),
        n_heads=model.get("n_heads", 8),
        n_kv_heads=model.get("n_kv_heads", model.get("n_heads", 8)),
        n_layers=model.get("n_layers", 6),
        ffn_multiplier=model.get("ffn_multiplier", 2.67),
        max_seq_len=model.get("max_seq_len", 512),
    )

    tc = TrainingConfig(
        batch_size=training.get("batch_size", 8),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        dtype=training.get("dtype", "bf16"),
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        optimizer=training.get("optimizer", "adamw"),
    )

    return mc, tc
```

### Step 2: Parameter count

```python
# src/tools/vram_estimator.py (continued)

def count_model_parameters(mc: ModelConfig) -> dict[str, int]:
    """
    Count parameters per component.
    Assumes a standard decoder-only transformer architecture (GQA-aware).
    """
    head_dim = mc.dim // mc.n_heads

    # Embedding
    embedding = mc.vocab_size * mc.dim

    # Per-layer attention: Q, K, V, O projections
    q_proj = mc.dim * mc.dim                         # n_heads * head_dim
    k_proj = mc.dim * (mc.n_kv_heads * head_dim)    # GQA: fewer KV heads
    v_proj = mc.dim * (mc.n_kv_heads * head_dim)
    o_proj = mc.dim * mc.dim
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # Per-layer FFN (SwiGLU has 3 weight matrices: gate, up, down)
    ffn_hidden = int(mc.dim * mc.ffn_multiplier)
    ffn_per_layer = (
        mc.dim * ffn_hidden   # gate
        + mc.dim * ffn_hidden  # up
        + ffn_hidden * mc.dim  # down
    )

    # RMSNorm: one per layer (2 norms: pre-attn, pre-ffn), one final
    norm_per_layer = mc.dim * 2
    final_norm = mc.dim

    # LM head (often tied with embedding)
    lm_head = mc.vocab_size * mc.dim

    total_per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
    total_layers = total_per_layer * mc.n_layers

    total = embedding + total_layers + final_norm + lm_head

    return {
        "embedding": embedding,
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "norm_per_layer": norm_per_layer,
        "n_layers": mc.n_layers,
        "lm_head": lm_head,
        "total": total,
    }
```

### Step 3: VRAM breakdown calculator

```python
# src/tools/vram_estimator.py (continued)
from dataclasses import dataclass

BYTES_PER_DTYPE = {"fp32": 4, "fp16": 2, "bf16": 2}
OPTIMIZER_MULTIPLIERS = {"adamw": 3.0, "adam8bit": 1.5, "sgd": 1.0}

@dataclass
class VRAMBreakdown:
    model_params_mb: float
    optimizer_states_mb: float
    activations_mb: float
    gradients_mb: float
    overhead_mb: float
    kv_cache_mb: float = 0.0

    @property
    def training_total_mb(self) -> float:
        return (self.model_params_mb + self.optimizer_states_mb
                + self.activations_mb + self.gradients_mb + self.overhead_mb)

    @property
    def inference_total_mb(self) -> float:
        return self.model_params_mb + self.kv_cache_mb + 256  # 256MB inference overhead

def estimate_vram(mc: ModelConfig, tc: TrainingConfig) -> VRAMBreakdown:
    param_counts = count_model_parameters(mc)
    total_params = param_counts["total"]
    bytes_param = BYTES_PER_DTYPE[tc.dtype]

    # Model parameters
    model_mb = total_params * bytes_param / 1024**2

    # Gradients: fp32 for mixed precision, same dtype for full precision
    grad_bytes = 4 if tc.dtype in ("fp16", "bf16") else bytes_param
    grad_mb = total_params * grad_bytes / 1024**2

    # Optimizer states (always fp32 for adamw stability)
    opt_multiplier = OPTIMIZER_MULTIPLIERS.get(tc.optimizer, 3.0)
    optimizer_mb = total_params * 4 * opt_multiplier / 1024**2

    # Activations
    head_dim = mc.dim // mc.n_heads
    ffn_hidden = int(mc.dim * mc.ffn_multiplier)
    act_per_layer = tc.batch_size * mc.max_seq_len * (
        4 * mc.dim        # Q, K, V, out proj activations
        + 3 * ffn_hidden  # SwiGLU: gate + up + intermediate
    ) * bytes_param

    if tc.gradient_checkpointing:
        # Store sqrt(n_layers) layers worth of activations
        effective_layers = max(1, int(math.sqrt(mc.n_layers)))
        act_mb = act_per_layer * effective_layers / 1024**2
    else:
        act_mb = act_per_layer * mc.n_layers / 1024**2

    # Gradient accumulation doesn't change peak VRAM much
    # (gradients accumulate in-place)

    # Framework overhead
    overhead_mb = 512.0

    # KV-cache for inference (max_seq_len)
    kv_elements = (2 * mc.n_kv_heads * head_dim
                   * mc.n_layers * mc.max_seq_len * 1)  # batch=1 for inference
    kv_mb = kv_elements * bytes_param / 1024**2

    return VRAMBreakdown(
        model_params_mb=model_mb,
        optimizer_states_mb=optimizer_mb,
        activations_mb=act_mb,
        gradients_mb=grad_mb,
        overhead_mb=overhead_mb,
        kv_cache_mb=kv_mb,
    )
```

### Step 4: GPU detection

```python
# src/tools/vram_estimator.py (continued)
import torch
import subprocess

def get_gpu_capacity_mb() -> Optional[float]:
    """Return GPU VRAM in MB using torch or nvidia-smi."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / 1024**2
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None
```

### Step 5: Rich breakdown table

```python
# src/tools/vram_estimator.py (continued)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Optional

console = Console()

def render_vram_estimate(
    breakdown: VRAMBreakdown,
    gpu_capacity_mb: Optional[float],
    mc: ModelConfig,
    tc: TrainingConfig,
) -> None:
    param_info = count_model_parameters(mc)
    total_params_m = param_info["total"] / 1e6

    table = Table(title="VRAM Estimate", box=box.ROUNDED, show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Size (MB)", justify="right", style="bold")
    table.add_column("Notes", style="dim")

    def add_row(label, mb, note=""):
        color = "red" if mb > 3000 else "yellow" if mb > 1000 else "green"
        table.add_row(label, f"[{color}]{mb:.0f}[/{color}]", note)

    add_row("Model Parameters",
            breakdown.model_params_mb,
            f"{total_params_m:.1f}M params × {BYTES_PER_DTYPE[tc.dtype]}B ({tc.dtype})")
    add_row("Gradients",
            breakdown.gradients_mb,
            f"fp32 for mixed-precision stability")
    add_row("Optimizer States (Adam)",
            breakdown.optimizer_states_mb,
            f"3× params in fp32 (m, v, grad)")
    add_row("Activations",
            breakdown.activations_mb,
            f"batch={tc.batch_size}, seq={mc.max_seq_len}" +
            (" [grad ckpt]" if tc.gradient_checkpointing else ""))
    add_row("Framework Overhead",
            breakdown.overhead_mb,
            "CUDA allocator, fragments, workspace")

    table.add_section()
    total = breakdown.training_total_mb
    fit_color = "green" if (gpu_capacity_mb and total < gpu_capacity_mb * 0.9) else "red"
    table.add_row(
        "[bold]TRAINING TOTAL[/bold]",
        f"[bold {fit_color}]{total:.0f}[/bold {fit_color}]",
        f"({total/1024:.1f} GB)"
    )
    table.add_row(
        "[bold]INFERENCE (KV-cache)[/bold]",
        f"[bold]{breakdown.inference_total_mb:.0f}[/bold]",
        f"batch=1, seq={mc.max_seq_len}"
    )

    console.print(table)

    if gpu_capacity_mb:
        pct = total / gpu_capacity_mb * 100
        if total > gpu_capacity_mb:
            console.print(Panel(
                f"[bold red]EXCEEDS GPU CAPACITY[/bold red]\n"
                f"Estimated: {total:.0f} MB  |  GPU: {gpu_capacity_mb:.0f} MB\n"
                f"Overrun: {total - gpu_capacity_mb:.0f} MB ({pct:.0f}% of capacity)\n\n"
                f"Suggestions:\n"
                f"  - Reduce batch_size (currently {tc.batch_size})\n"
                f"  - Enable gradient_checkpointing: true\n"
                f"  - Reduce n_layers or dim\n"
                f"  - Use 8-bit optimizer (adam8bit)",
                style="red",
            ))
        elif total > gpu_capacity_mb * 0.85:
            console.print(Panel(
                f"[yellow]WARNING: High VRAM usage ({pct:.0f}%)[/yellow]\n"
                f"May OOM on some operations. Consider increasing gradient_accumulation_steps.",
                style="yellow",
            ))
        else:
            console.print(Panel(
                f"[green]OK: {pct:.0f}% of {gpu_capacity_mb:.0f}MB GPU[/green]  "
                f"({total:.0f} MB estimated, {gpu_capacity_mb - total:.0f} MB headroom)",
                style="green",
            ))
```

### Step 6: CLI entry point

```python
# src/tools/vram_estimator.py (continued)
import typer

app = typer.Typer()

@app.command()
def main(
    config: Path = typer.Argument(..., help="Path to model config YAML"),
    mode: str = typer.Option("train", help="Mode: train or inference"),
):
    mc, tc = load_configs_from_yaml(config)
    breakdown = estimate_vram(mc, tc)
    gpu_mb = get_gpu_capacity_mb()
    render_vram_estimate(breakdown, gpu_mb, mc, tc)

if __name__ == "__main__":
    app()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/tools/vram_estimator.py` | New: full estimator |
| `src/menu/menus/train_menu.py` | Add "Estimate VRAM" item |
| `src/train.py` | Call estimator as pre-flight check before training starts |
| `configs/*.yaml` | Ensure `model.*` and `training.*` sections are complete |

Pre-flight hook in train.py:
```python
# Before training begins
from src.tools.vram_estimator import load_configs_from_yaml, estimate_vram, get_gpu_capacity_mb, render_vram_estimate
mc, tc = load_configs_from_yaml(config_path)
breakdown = estimate_vram(mc, tc)
gpu_mb = get_gpu_capacity_mb()
if gpu_mb and breakdown.training_total_mb > gpu_mb:
    render_vram_estimate(breakdown, gpu_mb, mc, tc)
    if not confirm("VRAM estimate exceeds GPU capacity. Continue anyway?", default=False):
        raise SystemExit(1)
```

---

## Testing Strategy

- **Parameter count test**: For a known architecture (e.g., GPT-2 small: dim=768, n_heads=12, n_layers=12), verify param count matches published 117M
- **GQA test**: With n_heads=8, n_kv_heads=2, verify K/V projections are 4x smaller than Q projection
- **Gradient checkpointing test**: Verify activation estimate is lower with `gradient_checkpointing=True`
- **Adam8bit test**: Verify optimizer_states_mb is ~50% lower with `optimizer=adam8bit` vs `adamw`
- **GPU capacity warning test**: Mock a 10GB GPU and a config that estimates 12GB — verify red warning panel shown
- **Feasibility test**: `dim=512, n_layers=6, batch=8` should estimate < 4GB (fits on 4GB GPU)

---

## Performance Considerations

- The estimator is a pure arithmetic calculation — instantaneous
- GPU detection via torch takes ~100ms (CUDA initialization)
- The estimator is run at most once per training job — performance is irrelevant

---

## Dependencies

| Package | Use | Install |
|---------|-----|---------|
| `pyyaml` | Config loading | Already installed |
| `torch` | GPU detection | Already installed |
| `rich` | Table display | Already installed |
| `typer` | CLI | Add to requirements.txt |

---

## Estimated Complexity

**Low** — 1 day.

- Parameter count formulas: 2 hours
- VRAM breakdown calculator: 2 hours
- GPU detection: 1 hour
- Rich display: 1.5 hours
- CLI + integration: 1.5 hours
- Testing: 2 hours

Total: ~10 hours

---

## 2026 Best Practices

- **Empirical validation**: Compare estimates against actual `torch.cuda.memory_allocated()` during training. The first time you run a new config, check real vs. estimated. Calibrate the overhead constant if needed.
- **Conservative estimates**: It's better to predict 9GB and use 8GB than to predict 7GB and OOM. Add at least 10-15% safety margin.
- **Account for gradient checkpointing correctly**: Many estimators assume gradient checkpointing saves exactly 50% of activations. The actual savings depend on the recomputation schedule. The `sqrt(n_layers)` formula is a better approximation.
- **bf16 vs fp16 activations**: Both are 2 bytes per element, but bf16 is numerically more stable for training. The VRAM calculation is the same, but prefer bf16 in the default config.
- **Communicate the breakdown**: Don't just output "Estimated: 7.3 GB". Show the user where VRAM is going. Optimizer states being 3x model size often surprises new users.
