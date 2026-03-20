# 08 - Config Validation

## Overview

Validate YAML config files before any training or inference command executes. Checks include type correctness, value ranges, cross-field compatibility (n_heads must divide dim, n_kv_heads must divide n_heads), VRAM feasibility, and path existence. Uses Pydantic v2 models for declarative validation with clear, actionable error messages. Runs automatically before `cola-train.ps1` and from the menu.

---

## Motivation

Config mistakes are a constant source of frustration. Common failures include:
- `n_heads: 7` — doesn't divide `dim: 512` → model crashes on matrix shape mismatch
- `learning_rate: 3e4` (missing the minus sign) → wildly high LR, immediate loss explosion
- `batch_size: 0` → zero-division error deep in the training loop
- `val_data_path: data/val.npy` — file doesn't exist → crash 30 minutes into training
- `n_kv_heads: 10` when `n_heads: 8` — GQA constraint violated

These errors are not caught until code is already running. Config validation catches them in milliseconds, before a single training step.

---

## Architecture / Design

### Validation Layers

```
Layer 1: Type validation (Pydantic)
  - int vs float vs string
  - Positive values where required
  - Path fields resolve to existing files

Layer 2: Range validation (custom validators)
  - learning_rate in [1e-6, 1.0]
  - batch_size >= 1
  - 0.0 < val_ratio < 1.0

Layer 3: Cross-field compatibility (root validators)
  - dim % n_heads == 0
  - n_heads % n_kv_heads == 0
  - max_seq_len is a power of 2 (optional but recommended)
  - ffn_hidden > dim (for standard architectures)

Layer 4: VRAM feasibility (integration with feature 07)
  - Estimate VRAM, warn if exceeds GPU
```

### Error Display Format

```
Config Validation Failed: configs/small.yaml

  ERROR   [model.n_heads]
          n_heads=7 does not divide dim=512.
          Fix: use a divisor of 512, e.g. n_heads=8

  ERROR   [training.learning_rate]
          learning_rate=30000.0 is outside valid range [1e-6, 1.0].
          Did you mean 3e-4?

  WARNING [training.val_ratio]
          val_ratio=0.0 disables validation split.
          Recommended: val_ratio=0.05

  WARNING [data.val_data_path]
          File not found: data/val.npy
          Run prepare_data first, or set val_ratio > 0.
```

---

## Implementation Steps

### Step 1: Pydantic models

```python
# src/config/schema.py
from __future__ import annotations
from typing import Optional, Literal
from pathlib import Path
import math

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_V2 = False


class ModelConfig(BaseModel):
    vocab_size: int = Field(ge=1, le=1_000_000, default=32000)
    dim: int = Field(ge=64, le=16384, default=512)
    n_heads: int = Field(ge=1, le=256, default=8)
    n_kv_heads: int = Field(ge=1, le=256, default=8)
    n_layers: int = Field(ge=1, le=128, default=6)
    max_seq_len: int = Field(ge=64, le=65536, default=512)
    ffn_multiplier: float = Field(gt=1.0, le=8.0, default=2.6875)
    dropout: float = Field(ge=0.0, lt=1.0, default=0.0)
    tie_embeddings: bool = True
    rope_theta: float = Field(gt=0.0, default=10000.0)

    @field_validator("n_heads")
    @classmethod
    def heads_divide_dim(cls, v, info):
        dim = info.data.get("dim")
        if dim and dim % v != 0:
            raise ValueError(
                f"n_heads={v} does not divide dim={dim}. "
                f"Valid options: {[i for i in range(1, dim+1) if dim % i == 0 and i <= 64]}"
            )
        return v

    @field_validator("n_kv_heads")
    @classmethod
    def kv_heads_divide_n_heads(cls, v, info):
        n_heads = info.data.get("n_heads")
        if n_heads and n_heads % v != 0:
            raise ValueError(
                f"n_kv_heads={v} does not divide n_heads={n_heads}. "
                f"n_kv_heads must be a divisor of n_heads for GQA."
            )
        return v

    @field_validator("max_seq_len")
    @classmethod
    def seq_len_power_of_two(cls, v):
        if v & (v - 1) != 0:
            next_pow2 = 2 ** math.ceil(math.log2(v))
            # Warning, not error
            import warnings
            warnings.warn(
                f"max_seq_len={v} is not a power of 2. "
                f"Performance is best with powers of 2 (e.g. {next_pow2})."
            )
        return v


class TrainingConfig(BaseModel):
    batch_size: int = Field(ge=1, le=4096, default=8)
    learning_rate: float = Field(gt=0.0, default=3e-4)
    min_learning_rate: float = Field(ge=0.0, default=3e-5)
    weight_decay: float = Field(ge=0.0, le=1.0, default=0.1)
    gradient_clip: float = Field(ge=0.0, default=1.0)
    warmup_steps: int = Field(ge=0, default=100)
    max_steps: int = Field(ge=1, default=10000)
    gradient_accumulation_steps: int = Field(ge=1, default=1)
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    gradient_checkpointing: bool = False
    optimizer: Literal["adamw", "adam8bit", "sgd"] = "adamw"
    val_interval: int = Field(ge=1, default=500)
    save_interval: int = Field(ge=1, default=1000)

    @field_validator("learning_rate")
    @classmethod
    def lr_in_sane_range(cls, v):
        if v > 0.1:
            raise ValueError(
                f"learning_rate={v} seems too high (> 0.1). "
                f"Typical range is 1e-4 to 1e-3. Did you mean {v/1e4:.2e}?"
            )
        if v < 1e-7:
            raise ValueError(
                f"learning_rate={v} is extremely small (< 1e-7). "
                f"Training will effectively not learn. Typical range: 1e-4 to 1e-3."
            )
        return v

    @field_validator("min_learning_rate")
    @classmethod
    def min_lr_less_than_max_lr(cls, v, info):
        lr = info.data.get("learning_rate")
        if lr and v >= lr:
            raise ValueError(
                f"min_learning_rate={v} must be less than learning_rate={lr}."
            )
        return v


class DataConfig(BaseModel):
    train_data_path: Optional[Path] = None
    val_data_path: Optional[Path] = None
    val_ratio: float = Field(ge=0.0, le=0.5, default=0.05)
    split_seed: int = 42
    chunk_size: int = Field(ge=64, le=65536, default=512)

    @field_validator("train_data_path", "val_data_path")
    @classmethod
    def path_must_exist_if_provided(cls, v):
        if v is not None and not v.exists():
            raise ValueError(
                f"File not found: {v}. "
                f"Run prepare_data first."
            )
        return v

    @field_validator("chunk_size")
    @classmethod
    def chunk_matches_seq_len_warning(cls, v):
        # Not an error, just a note — warn if mismatch (can't cross-validate here easily)
        return v


class ColaCoderConfig(BaseModel):
    """Top-level config schema."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    run_name: Optional[str] = None
    output_dir: Path = Path("runs")
    seed: int = 42
```

### Step 2: Validation runner with rich output

```python
# src/config/validator.py
from pathlib import Path
from typing import Optional
import yaml
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.config.schema import ColaCoderConfig

console = Console()

class ValidationError(Exception):
    pass

def validate_config(config_path: Path, check_vram: bool = True) -> ColaCoderConfig:
    """
    Load and validate a config YAML. Returns the parsed config if valid.
    Raises ValidationError with formatted message if invalid.
    """
    if not config_path.exists():
        raise ValidationError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as e:
        raise ValidationError(f"YAML parse error in {config_path}:\n{e}")

    # Collect pydantic errors
    from pydantic import ValidationError as PydanticError
    captured_warnings = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            config = ColaCoderConfig.model_validate(raw)
        except PydanticError as e:
            _render_validation_errors(config_path, e.errors(), [])
            raise ValidationError(f"Config validation failed: {len(e.errors())} error(s)")
        captured_warnings = list(w)

    errors = []
    warn_messages = []

    for warning in captured_warnings:
        warn_messages.append(str(warning.message))

    # VRAM check
    if check_vram:
        vram_warnings = _check_vram_feasibility(config)
        warn_messages.extend(vram_warnings)

    if warn_messages:
        _render_warnings(config_path, warn_messages)

    return config


def _render_validation_errors(path: Path, errors: list, warnings: list) -> None:
    console.print(Panel(
        f"[bold red]Config Validation Failed[/bold red]: {path}",
        style="red"
    ))
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Level", width=9)
    table.add_column("Field", style="cyan", width=30)
    table.add_column("Message")

    for err in errors:
        field = ".".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        table.add_row("[bold red]ERROR[/bold red]", f"[{field}]", msg)

    for w in warnings:
        table.add_row("[bold yellow]WARNING[/bold yellow]", "", w)

    console.print(table)


def _render_warnings(path: Path, warnings_list: list[str]) -> None:
    if not warnings_list:
        return
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Level", width=9)
    table.add_column("Message")
    for w in warnings_list:
        table.add_row("[bold yellow]WARNING[/bold yellow]", w)
    console.print(Panel(table, title=f"Config Warnings: {path}", style="yellow"))


def _check_vram_feasibility(config: ColaCoderConfig) -> list[str]:
    """Run VRAM estimator, return warnings if over capacity."""
    try:
        from src.tools.vram_estimator import estimate_vram, get_gpu_capacity_mb, ModelConfig, TrainingConfig
        mc = ModelConfig(
            vocab_size=config.model.vocab_size,
            dim=config.model.dim,
            n_heads=config.model.n_heads,
            n_kv_heads=config.model.n_kv_heads,
            n_layers=config.model.n_layers,
            ffn_multiplier=config.model.ffn_multiplier,
            max_seq_len=config.model.max_seq_len,
        )
        tc = TrainingConfig(
            batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            dtype=config.training.dtype,
            gradient_checkpointing=config.training.gradient_checkpointing,
            optimizer=config.training.optimizer,
        )
        breakdown = estimate_vram(mc, tc)
        gpu_mb = get_gpu_capacity_mb()
        if gpu_mb and breakdown.training_total_mb > gpu_mb * 0.95:
            return [
                f"Estimated VRAM {breakdown.training_total_mb:.0f}MB may exceed "
                f"GPU capacity {gpu_mb:.0f}MB. Consider reducing batch_size or enabling gradient_checkpointing."
            ]
    except Exception:
        pass
    return []
```

### Step 3: Pre-training hook

```python
# src/train.py (modification)
from src.config.validator import validate_config, ValidationError
from src.cli import confirm
from rich.console import Console

console = Console()

def run_training(config_path: Path, **kwargs):
    # Validate config before starting
    try:
        config = validate_config(config_path)
        console.print("[green]Config validation passed.[/green]")
    except ValidationError as e:
        console.print(f"\n[red]{e}[/red]")
        raise SystemExit(1)

    # Continue with training...
```

### Step 4: Standalone validation CLI

```python
# src/config/validator_cli.py
import typer
from pathlib import Path
from src.config.validator import validate_config, ValidationError
from rich.console import Console

console = Console()
app = typer.Typer()

@app.command()
def main(
    config: Path = typer.Argument(..., help="Path to config YAML"),
    no_vram_check: bool = typer.Option(False, "--no-vram", help="Skip VRAM feasibility check"),
):
    """Validate a Cola-Coder config YAML file."""
    try:
        cfg = validate_config(config, check_vram=not no_vram_check)
        console.print(f"[green]Config valid[/green]: {config}")
        console.print(f"  Model: dim={cfg.model.dim}, heads={cfg.model.n_heads}, layers={cfg.model.n_layers}")
        console.print(f"  Training: batch={cfg.training.batch_size}, lr={cfg.training.learning_rate:.2e}")
    except ValidationError as e:
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/config/__init__.py` | New package |
| `src/config/schema.py` | Pydantic models for all config sections |
| `src/config/validator.py` | Validation runner, error renderer |
| `src/config/validator_cli.py` | Standalone `cola validate` command |
| `src/train.py` | Add `validate_config()` call before training starts |
| `src/generate.py` | Add `validate_config()` call |
| `configs/*.yaml` | Ensure all keys match schema field names |
| `requirements.txt` | Add `pydantic>=2.0` |

---

## Testing Strategy

- **Valid config test**: Load `configs/small.yaml` through validator, expect no errors
- **n_heads divisibility**: Config with `dim=512, n_heads=7` → expect ValidationError with helpful message
- **LR range test**: `learning_rate=50000` → error mentioning "too high"
- **min_lr > max_lr test**: `learning_rate=1e-5, min_learning_rate=1e-4` → error
- **Missing file test**: `train_data_path: nonexistent.npy` → error mentioning path
- **YAML parse error test**: Feed invalid YAML, verify clean error message
- **Warning display test**: `max_seq_len=100` (not power of 2) → warning emitted, not error
- **VRAM check integration test**: Config with huge `batch_size=512` on small GPU → VRAM warning

---

## Performance Considerations

- Pydantic v2 validation is extremely fast (microseconds per config)
- VRAM estimation adds ~100ms (torch CUDA init for GPU detection)
- Total validation time: well under 500ms
- No I/O except reading the YAML file and (optionally) checking path existence

---

## Dependencies

| Package | Use | Install |
|---------|-----|---------|
| `pydantic>=2.0` | Schema validation | `pip install pydantic` |
| `pyyaml` | Config loading | Already installed |
| `rich` | Error display | Already installed |
| `typer` | CLI | Already in requirements |

---

## Estimated Complexity

**Low** — 1 day.

- Pydantic schema design: 3 hours
- Custom validators (cross-field): 2 hours
- Error rendering: 1.5 hours
- Pre-training hook + CLI: 1.5 hours
- Testing: 2 hours

Total: ~10 hours

---

## 2026 Best Practices

- **Pydantic v2 over v1**: Pydantic v2 (released 2023) is significantly faster and has cleaner validator syntax with `@field_validator` and `@model_validator`. Use v2 unless there's a dependency conflict.
- **Fail fast with context**: Error messages must say what the problem is AND how to fix it. "n_heads=7 is invalid" is unhelpful. "n_heads=7 does not divide dim=512. Valid options: [8, 16, 32, 64]" is actionable.
- **Warnings vs errors**: Not everything invalid should be an error. max_seq_len not being a power of 2 is a performance concern, not a correctness problem. Use Python `warnings` for soft issues.
- **Schema as documentation**: The Pydantic schema is also the authoritative documentation of what config fields exist and what values are valid. Keep it in sync with actual usage.
- **Validate in CI**: Add a CI step that validates all configs in `configs/`. This catches schema drift when new fields are added to the trainer but not the schema.
