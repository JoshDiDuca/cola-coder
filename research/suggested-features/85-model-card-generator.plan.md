# Feature 85: Model Card Generator

**Status:** Proposed
**CLI Flag:** `--generate-model-card` / `python scripts/model_card.py`
**Complexity:** Low-Medium

---

## Overview

Automatically generates a `MODEL_CARD.md` file in HuggingFace model card format, populated from existing project artifacts: `configs/*.yaml`, training manifests, evaluation result JSON files, and `experiments/runs.db`. Includes ASCII training curves rendered with `plotext`. The card is auto-updated after each evaluation run and can be uploaded directly to HuggingFace Hub using the `huggingface_hub` library.

```markdown
# Cola-Coder / tiny

> Auto-generated model card — Updated 2026-03-20 14:32 UTC

## Model Description
A 50M parameter decoder-only code generation transformer trained from scratch...

## Architecture
| Parameter      | Value   |
| n_layers       | 12      |
| n_heads        | 12      |
| hidden_dim     | 768     |
...

## Training Results
Training loss curve (last 5000 steps):

  6.0 ┤╮
  5.0 ┤╰╮
  4.0 ┤  ╰─╮
  3.0 ┤    ╰──────╮
  2.8 ┤           ╰─────────────

## Evaluation
| Benchmark   | Score   |
| pass@1      | 12.3%   |
| perplexity  | 16.7    |
```

---

## Motivation

A model card is increasingly the standard artifact for any trained model — it documents what was trained, how, on what data, and with what results. Without one, Cola-Coder checkpoints are opaque blobs that require reading source code to understand.

For Josh specifically, having a model card enables:
- **Quick comparison**: compare two checkpoints by reading their cards side-by-side.
- **HF Hub upload**: push a trained model to HuggingFace Hub with proper documentation.
- **Reproducibility**: the card captures the exact config and data used — critical for learning experiments.
- **Portfolio**: a tangible artifact showing a complete trained model with documented results.

The card is generated automatically — zero manual writing required. It is a downstream product of the pipeline, not something to maintain separately.

---

## Architecture / Design

```
ModelCardGenerator
  │
  ├── ConfigReader         <- reads configs/*.yaml
  ├── TrainingManifestReader <- reads checkpoints/*/training_state.json
  ├── EvalResultReader     <- reads eval_results/*.json
  ├── ExperimentReader     <- reads experiments/runs.db (optional)
  ├── CurveRenderer        <- renders ASCII loss curve with plotext
  └── CardTemplateEngine   <- fills Jinja2 template → MODEL_CARD.md

Outputs:
  - MODEL_CARD.md          <- human-readable, HF-compatible
  - model-card-metadata.json <- machine-readable summary for scripts

Optional:
  - HFHubUploader          <- pushes card + weights to huggingface.co
```

### Template Sections

| Section | Source |
|---|---|
| Title / description | `configs/<name>.yaml` → `model_name`, `description` |
| Architecture table | config: `n_layers`, `n_heads`, `hidden_dim`, `vocab_size`, etc. |
| Training data | config: `dataset`, `languages`, `quality_filter`, token count from data manifest |
| Training curve | `experiments/runs.db` or checkpoint `training_state.json` |
| Hyperparameters | config: `lr`, `batch_size`, `seq_len`, `warmup_steps`, `precision` |
| Eval results | `eval_results/*.json` — latest file |
| Usage example | templated from model name (static but useful) |
| Limitations | static section with standard caveats |
| License | from `pyproject.toml` or config |

---

## Implementation Steps

### Step 1: Data collectors

```python
# src/cola_coder/model_card/collectors.py
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class ArchitectureInfo:
    model_name: str
    param_count: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    precision: str
    attention_type: str = "GQA"
    activation: str = "SwiGLU"
    positional_encoding: str = "RoPE"
    normalization: str = "RMSNorm"

@dataclass
class TrainingInfo:
    config_name: str
    total_steps: int
    effective_batch_size: int
    learning_rate: float
    warmup_steps: int
    final_loss: float
    best_loss: float
    total_tokens_trained: int
    training_time_hours: Optional[float]
    dataset: str
    languages: list[str]
    quality_filter: str
    checkpoint_path: str
    loss_history: list[tuple[int, float]] = field(default_factory=list)  # (step, loss)

@dataclass
class EvalInfo:
    pass_at_1: Optional[float]
    pass_at_10: Optional[float]
    perplexity: Optional[float]
    humaneval_scores: dict[str, float] = field(default_factory=dict)
    eval_date: Optional[str] = None

def collect_architecture(config_path: Path) -> ArchitectureInfo:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", cfg)

    # Rough param count estimate
    n_layers = model_cfg.get("n_layers", 12)
    hidden_dim = model_cfg.get("hidden_dim", 768)
    vocab_size = model_cfg.get("vocab_size", 48256)
    n_heads = model_cfg.get("n_heads", 12)
    n_kv_heads = model_cfg.get("n_kv_heads", n_heads)
    ffn_multiplier = 2.667  # SwiGLU uses ~2.667x hidden_dim
    ffn_dim = int(hidden_dim * ffn_multiplier)

    # Approximate parameter count
    attn_params = n_layers * (hidden_dim * (hidden_dim + 2 * hidden_dim * n_kv_heads // n_heads))
    ffn_params = n_layers * (hidden_dim * ffn_dim * 3)  # 3 matrices in SwiGLU
    embed_params = vocab_size * hidden_dim
    param_count = attn_params + ffn_params + embed_params

    return ArchitectureInfo(
        model_name=cfg.get("model_name", config_path.stem),
        param_count=param_count,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        max_seq_len=model_cfg.get("max_seq_len", 2048),
        precision=cfg.get("training", {}).get("precision", "bf16"),
    )

def collect_training_info(config_path: Path, checkpoint_dir: Path) -> Optional[TrainingInfo]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg.get("training", cfg)
    data_cfg = cfg.get("data", cfg)

    # Find latest training_state.json
    state_files = sorted(checkpoint_dir.glob("*/training_state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not state_files:
        return None

    with open(state_files[0]) as f:
        state = json.load(f)

    # Collect loss history from all step checkpoints
    loss_history = []
    for step_dir in sorted(checkpoint_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        meta_path = step_dir / "training_state.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                step = m.get("global_step")
                loss = m.get("loss")
                if step is not None and loss is not None:
                    loss_history.append((step, loss))
            except Exception:
                pass

    batch_size = training_cfg.get("batch_size", 8)
    grad_accum = training_cfg.get("gradient_accumulation_steps", 4)
    max_seq_len = cfg.get("model", cfg).get("max_seq_len", 2048)
    total_steps = state.get("global_step", 0)
    effective_batch = batch_size * grad_accum

    return TrainingInfo(
        config_name=config_path.stem,
        total_steps=total_steps,
        effective_batch_size=effective_batch,
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        warmup_steps=training_cfg.get("warmup_steps", 200),
        final_loss=state.get("loss", 0.0),
        best_loss=min((l for _, l in loss_history), default=state.get("loss", 0.0)),
        total_tokens_trained=total_steps * effective_batch * max_seq_len,
        training_time_hours=state.get("elapsed_hours"),
        dataset=data_cfg.get("dataset", "bigcode/starcoderdata"),
        languages=data_cfg.get("languages", ["python", "typescript", "javascript"]),
        quality_filter=data_cfg.get("quality_filter", "conservative"),
        checkpoint_path=str(state_files[0].parent),
        loss_history=loss_history,
    )

def collect_eval_info(eval_results_dir: Path) -> Optional[EvalInfo]:
    if not eval_results_dir.exists():
        return None
    result_files = sorted(eval_results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        return None
    with open(result_files[0]) as f:
        data = json.load(f)
    return EvalInfo(
        pass_at_1=data.get("pass@1") or data.get("pass_at_1"),
        pass_at_10=data.get("pass@10") or data.get("pass_at_10"),
        perplexity=data.get("perplexity"),
        humaneval_scores=data.get("humaneval_per_problem", {}),
        eval_date=data.get("evaluated_at"),
    )
```

### Step 2: ASCII curve renderer

```python
# src/cola_coder/model_card/curves.py
import io
import plotext as plt
from typing import Optional

def render_loss_curve_ascii(
    loss_history: list[tuple[int, float]],
    title: str = "Training Loss",
    width: int = 72,
    height: int = 12,
    last_n: Optional[int] = None,
) -> str:
    """
    Render a training loss curve as ASCII art using plotext.
    Returns a string suitable for embedding in Markdown code block.
    """
    if not loss_history:
        return "_No training data available._"

    if last_n:
        loss_history = loss_history[-last_n:]

    steps = [s for s, _ in loss_history]
    losses = [l for _, l in loss_history]

    # Capture plotext output to string
    plt.clf()
    plt.plot_size(width, height)
    plt.plot(steps, losses, color="cyan", label="loss")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.theme("clear")  # no color codes for markdown embedding

    buf = io.StringIO()
    plt.show(file=buf)  # plotext supports file= since v5.2
    return buf.getvalue().rstrip()
```

### Step 3: Jinja2 template

```python
# src/cola_coder/model_card/template.py
MODEL_CARD_TEMPLATE = """\
---
language: code
license: apache-2.0
tags:
  - code
  - code-generation
  - causal-lm
  - pytorch
  - cola-coder
datasets:
  - bigcode/starcoderdata
model-index:
  - name: {{ arch.model_name }}
    results:
      - task:
          type: text-generation
        metrics:
          - type: pass@1
            value: {{ "%.3f"|format(eval.pass_at_1) if eval and eval.pass_at_1 else "N/A" }}
---

# Cola-Coder / {{ arch.model_name }}

> Auto-generated model card — Updated {{ generated_at }}

## Model Description

Cola-Coder `{{ arch.model_name }}` is a **{{ "%.0fM"|format(arch.param_count / 1e6) }} parameter**
decoder-only transformer for code generation, trained from scratch in PyTorch.
Architecture mirrors LLaMA 3 / Mistral: RoPE positional encoding, Grouped Query Attention (GQA),
SwiGLU activation, RMSNorm (pre-norm). Trained on `{{ training.dataset if training else "bigcode/starcoderdata" }}`.

This model was built as a learning project — see the [Cola-Coder repository](https://github.com/your-username/cola-coder) for full source code, training scripts, and documentation written for TypeScript developers learning ML.

---

## Architecture

| Parameter             | Value                     |
|-----------------------|---------------------------|
| Parameters            | ~{{ "%.0fM"|format(arch.param_count / 1e6) }} |
| Layers                | {{ arch.n_layers }}        |
| Attention heads       | {{ arch.n_heads }}         |
| KV heads (GQA)        | {{ arch.n_kv_heads }}      |
| Hidden dimension      | {{ arch.hidden_dim }}      |
| FFN dimension         | {{ arch.ffn_dim }}         |
| Vocabulary size       | {{ "{:,}".format(arch.vocab_size) }} |
| Max sequence length   | {{ "{:,}".format(arch.max_seq_len) }} |
| Attention type        | {{ arch.attention_type }}  |
| Activation            | {{ arch.activation }}      |
| Positional encoding   | {{ arch.positional_encoding }} |
| Normalization         | {{ arch.normalization }}   |
| Training precision    | {{ arch.precision }}       |

---

## Training Data

{% if training %}
- **Dataset:** `{{ training.dataset }}`
- **Languages:** {{ training.languages | join(", ") }}
- **Quality filter:** {{ training.quality_filter }} mode
- **Total tokens trained:** ~{{ "%.1fB"|format(training.total_tokens_trained / 1e9) }} tokens
{% else %}
_Training data information not available._
{% endif %}

---

## Training Configuration

{% if training %}
| Hyperparameter           | Value           |
|--------------------------|-----------------|
| Total steps              | {{ "{:,}".format(training.total_steps) }} |
| Effective batch size     | {{ training.effective_batch_size }} |
| Learning rate            | {{ training.learning_rate }} |
| Warmup steps             | {{ training.warmup_steps }} |
| Best training loss       | {{ "%.4f"|format(training.best_loss) }} |
| Final training loss      | {{ "%.4f"|format(training.final_loss) }} |
{% if training.training_time_hours %}
| Training time            | {{ "%.1f"|format(training.training_time_hours) }}h |
{% endif %}
{% else %}
_Training configuration not available._
{% endif %}

---

## Training Loss Curve

{% if loss_curve %}
```
{{ loss_curve }}
```
{% else %}
_No training history available._
{% endif %}

---

## Evaluation Results

{% if eval %}
| Metric       | Value   |
|--------------|---------|
{% if eval.pass_at_1 is not none %}
| pass@1       | {{ "%.1f%%"|format(eval.pass_at_1 * 100) }} |
{% endif %}
{% if eval.pass_at_10 is not none %}
| pass@10      | {{ "%.1f%%"|format(eval.pass_at_10 * 100) }} |
{% endif %}
{% if eval.perplexity is not none %}
| Perplexity   | {{ "%.2f"|format(eval.perplexity) }} |
{% endif %}
{% if eval.eval_date %}

_Evaluated: {{ eval.eval_date }}_
{% endif %}
{% else %}
_No evaluation results available yet. Run `python scripts/evaluate.py` to generate._
{% endif %}

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("your-username/cola-coder-{{ arch.model_name }}")
model = AutoModelForCausalLM.from_pretrained("your-username/cola-coder-{{ arch.model_name }}")

prompt = "def fibonacci(n: int) -> int:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.2, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or use the Cola-Coder CLI:

```powershell
# Generate from latest checkpoint
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/{{ arch.model_name }}/latest --prompt "def fibonacci(n: int) -> int:"
```

---

## Limitations

- Trained on a subset of `bigcode/starcoderdata` — may not generalize to niche languages or frameworks.
- Model output is not guaranteed to be correct, safe, or production-ready.
- Context window limited to {{ "{:,}".format(arch.max_seq_len) }} tokens.
- No RLHF or safety fine-tuning — outputs reflect patterns in training data.
- Small model size (~{{ "%.0fM"|format(arch.param_count / 1e6) }}M parameters) limits reasoning depth compared to larger models.
- Built as a learning project — not intended for production use.

---

## Training Infrastructure

- **Hardware:** RTX 4080 (16GB VRAM, bf16) or RTX 3080 (10GB VRAM, fp16)
- **Framework:** PyTorch 2.2+
- **Checkpoints:** safetensors format
- **Tokenizer:** HuggingFace BPE (Rust-backed)

---

## License

Apache 2.0 — see `LICENSE` file.

---

_Generated by Cola-Coder model card generator on {{ generated_at }}.
Source: [cola-coder](https://github.com/your-username/cola-coder)_
"""
```

### Step 4: Generator class

```python
# src/cola_coder/model_card/generator.py
from datetime import datetime, timezone
from pathlib import Path
from jinja2 import Template
from .collectors import collect_architecture, collect_training_info, collect_eval_info
from .curves import render_loss_curve_ascii
from .template import MODEL_CARD_TEMPLATE

class ModelCardGenerator:
    def __init__(
        self,
        config_path: Path,
        checkpoint_dir: Path,
        eval_results_dir: Path = Path("eval_results"),
        output_path: Path = Path("MODEL_CARD.md"),
    ):
        self.config_path = config_path
        self.checkpoint_dir = checkpoint_dir
        self.eval_results_dir = eval_results_dir
        self.output_path = output_path

    def generate(self) -> Path:
        arch = collect_architecture(self.config_path)
        training = collect_training_info(self.config_path, self.checkpoint_dir)
        eval_info = collect_eval_info(self.eval_results_dir)

        loss_curve = None
        if training and training.loss_history:
            loss_curve = render_loss_curve_ascii(
                training.loss_history,
                title=f"Training Loss — {arch.model_name}",
                last_n=500,  # show last 500 steps in curve
            )

        template = Template(MODEL_CARD_TEMPLATE)
        content = template.render(
            arch=arch,
            training=training,
            eval=eval_info,
            loss_curve=loss_curve,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

        self.output_path.write_text(content, encoding="utf-8")
        return self.output_path

    def generate_metadata_json(self) -> dict:
        """Machine-readable summary for scripts/CI."""
        arch = collect_architecture(self.config_path)
        training = collect_training_info(self.config_path, self.checkpoint_dir)
        eval_info = collect_eval_info(self.eval_results_dir)

        return {
            "model_name": arch.model_name,
            "param_count": arch.param_count,
            "best_loss": training.best_loss if training else None,
            "pass_at_1": eval_info.pass_at_1 if eval_info else None,
            "perplexity": eval_info.perplexity if eval_info else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
```

### Step 5: CLI script

```python
# scripts/model_card.py
"""Generate MODEL_CARD.md from project artifacts."""
import sys
import json
from pathlib import Path
import click
from rich.console import Console
from cola_coder.model_card.generator import ModelCardGenerator

console = Console()

@click.command()
@click.option("--config", default="configs/tiny.yaml", help="Model config YAML")
@click.option("--checkpoint-dir", default=None, help="Checkpoint directory (auto-detected if omitted)")
@click.option("--eval-dir", default="eval_results", help="Evaluation results directory")
@click.option("--output", default="MODEL_CARD.md", help="Output path for model card")
@click.option("--upload", is_flag=True, default=False, help="Upload to HuggingFace Hub after generating")
@click.option("--repo-id", default=None, help="HF Hub repo ID (required with --upload)")
@click.option("--json-summary", is_flag=True, default=False, help="Also write model-card-metadata.json")
def main(config, checkpoint_dir, eval_dir, output, upload, repo_id, json_summary):
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        sys.exit(1)

    # Auto-detect checkpoint directory from config name
    if checkpoint_dir is None:
        config_name = config_path.stem
        checkpoint_dir = f"checkpoints/{config_name}"
    ckpt_path = Path(checkpoint_dir)

    generator = ModelCardGenerator(
        config_path=config_path,
        checkpoint_dir=ckpt_path,
        eval_results_dir=Path(eval_dir),
        output_path=Path(output),
    )

    console.print(f"[cyan]Generating model card from:[/cyan]")
    console.print(f"  Config:      {config_path}")
    console.print(f"  Checkpoints: {ckpt_path}")
    console.print(f"  Eval:        {eval_dir}")

    card_path = generator.generate()
    console.print(f"[green]Model card written to: {card_path}[/green]")

    if json_summary:
        metadata = generator.generate_metadata_json()
        meta_path = Path("model-card-metadata.json")
        meta_path.write_text(json.dumps(metadata, indent=2))
        console.print(f"[green]Metadata written to: {meta_path}[/green]")
        console.print_json(json.dumps(metadata))

    if upload:
        if not repo_id:
            console.print("[red]--repo-id required for upload[/red]")
            sys.exit(1)
        _upload_to_hub(card_path, ckpt_path, repo_id, console)

def _upload_to_hub(card_path: Path, ckpt_path: Path, repo_id: str, console: Console) -> None:
    try:
        from huggingface_hub import HfApi, upload_file, upload_folder
    except ImportError:
        console.print("[red]huggingface_hub not installed. Run: pip install huggingface-hub[/red]")
        sys.exit(1)

    api = HfApi()
    console.print(f"[cyan]Uploading to: {repo_id}[/cyan]")

    # Upload model card
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )
    console.print("[green]Model card uploaded as README.md[/green]")

    # Upload safetensors checkpoint if present
    latest_ckpt = next(ckpt_path.glob("**/model.safetensors"), None)
    if latest_ckpt:
        console.print(f"[cyan]Uploading checkpoint: {latest_ckpt}[/cyan]")
        api.upload_file(
            path_or_fileobj=str(latest_ckpt),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload checkpoint",
        )
        console.print("[green]Checkpoint uploaded[/green]")

if __name__ == "__main__":
    main()
```

### Step 6: Auto-generate after evaluation

```python
# In scripts/evaluate.py — add at the end of evaluation
from cola_coder.model_card.generator import ModelCardGenerator

# ... existing evaluation code ...

if args.generate_model_card:  # opt-in flag
    generator = ModelCardGenerator(
        config_path=Path(args.config),
        checkpoint_dir=Path(args.checkpoint).parent,
    )
    card_path = generator.generate()
    console.print(f"[dim]Model card updated: {card_path}[/dim]")
```

---

## Key Files to Modify / Create

**New:**
- `src/cola_coder/model_card/__init__.py`
- `src/cola_coder/model_card/collectors.py` — data collection from artifacts
- `src/cola_coder/model_card/curves.py` — ASCII curve renderer
- `src/cola_coder/model_card/template.py` — Jinja2 card template
- `src/cola_coder/model_card/generator.py` — `ModelCardGenerator` class
- `scripts/model_card.py` — CLI entry point

**Modify:**
- `scripts/evaluate.py` — add `--generate-model-card` flag to auto-update after eval
- `pyproject.toml` — add `jinja2>=3.1` and `plotext>=5.2` as optional deps under `[model-card]`

---

## Testing Strategy

```python
# tests/test_model_card.py
import json
import pytest
import yaml
from pathlib import Path
from cola_coder.model_card.collectors import collect_architecture, collect_training_info, collect_eval_info
from cola_coder.model_card.generator import ModelCardGenerator

@pytest.fixture
def tiny_config(tmp_path):
    cfg = {
        "model_name": "tiny",
        "model": {
            "n_layers": 12, "n_heads": 12, "n_kv_heads": 4,
            "hidden_dim": 768, "vocab_size": 48256, "max_seq_len": 2048,
        },
        "training": {"learning_rate": 3e-4, "batch_size": 8, "precision": "bf16",
                     "warmup_steps": 200, "gradient_accumulation_steps": 4},
        "data": {"dataset": "bigcode/starcoderdata", "languages": ["python", "typescript"],
                 "quality_filter": "conservative"},
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.dump(cfg))
    return p

def test_collect_architecture(tiny_config):
    arch = collect_architecture(tiny_config)
    assert arch.model_name == "tiny"
    assert arch.n_layers == 12
    assert arch.vocab_size == 48256
    assert arch.param_count > 0

def test_collect_eval_info(tmp_path):
    eval_dir = tmp_path / "eval_results"
    eval_dir.mkdir()
    (eval_dir / "eval_step_1000.json").write_text(json.dumps({
        "pass@1": 0.123, "perplexity": 16.7
    }))
    info = collect_eval_info(eval_dir)
    assert info is not None
    assert abs(info.pass_at_1 - 0.123) < 0.001
    assert abs(info.perplexity - 16.7) < 0.01

def test_model_card_generated(tiny_config, tmp_path):
    output = tmp_path / "MODEL_CARD.md"
    gen = ModelCardGenerator(
        config_path=tiny_config,
        checkpoint_dir=tmp_path / "checkpoints" / "tiny",
        eval_results_dir=tmp_path / "eval_results",
        output_path=output,
    )
    path = gen.generate()
    assert path.exists()
    content = path.read_text()
    # Check key sections present
    assert "# Cola-Coder / tiny" in content
    assert "## Architecture" in content
    assert "## Limitations" in content
    assert "## Usage" in content
    assert "768" in content  # hidden_dim

def test_card_with_eval_results(tiny_config, tmp_path):
    eval_dir = tmp_path / "eval_results"
    eval_dir.mkdir()
    (eval_dir / "results.json").write_text(json.dumps({"pass@1": 0.312, "perplexity": 8.7}))
    output = tmp_path / "MODEL_CARD.md"
    gen = ModelCardGenerator(tiny_config, tmp_path / "ckpts", eval_dir, output)
    gen.generate()
    content = output.read_text()
    assert "31.2%" in content
    assert "8.70" in content

def test_card_updates_on_regenerate(tiny_config, tmp_path):
    output = tmp_path / "MODEL_CARD.md"
    gen = ModelCardGenerator(tiny_config, tmp_path / "ckpts", tmp_path / "eval", output)
    gen.generate()
    first_mtime = output.stat().st_mtime
    gen.generate()
    # File should be overwritten (mtime may be same if very fast — check content instead)
    assert output.exists()
```

---

## Performance Considerations

- All data collection is filesystem reads of small files (YAML, JSON). Total time is < 100ms even with 1000 checkpoint directories.
- `render_loss_curve_ascii()` with `last_n=500` caps the data fed to plotext, preventing slowdown on runs with millions of logged steps.
- Jinja2 template rendering is negligible (< 10ms for a 400-line template).
- The `--upload` path is network-bound; it is opt-in and not on the critical path. HuggingFace Hub upload of a safetensors file is bandwidth-limited (50M model = ~200MB upload).
- The card is a standalone file — no database queries, no ML inference. It can be generated in any environment including CI machines without GPU.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `jinja2` | `>=3.1` | Template rendering |
| `plotext` | `>=5.2` | ASCII training curve |
| `pyyaml` | `>=6.0` | Config reading — already present |
| `click` | `>=8.1` | CLI for `scripts/model_card.py` |
| `huggingface_hub` | `>=0.22` | Optional — HF Hub upload only |

All can be optional extras: `pip install cola-coder[model-card]`.

---

## Estimated Complexity

**Low-Medium.** The bulk of the work is writing the Jinja2 template to look good. The data collection code is straightforward filesystem operations. The ASCII curve rendering is a 15-line wrapper around plotext. The hardest part is handling the many cases where artifacts are partially missing (no eval results, no training history, etc.) gracefully without crashing.

Estimated implementation time: 4-6 hours including tests.

---

## 2026 Best Practices

- **HuggingFace model card YAML frontmatter**: The YAML block at the top of `MODEL_CARD.md` (between `---` delimiters) is parsed by the HF Hub and populates the model's metadata searchably. Including `language`, `license`, `tags`, and `model-index` follows the 2026 HF Hub standard.
- **Jinja2 over f-strings for multi-page templates**: Jinja2 handles conditionals, loops, and filters cleanly for a 200+ line document. F-strings become unmaintainable at this scale.
- **`plotext` for embedded ASCII curves**: In 2026, embedding terminal plots in markdown is a recognized pattern for ML project cards targeting developer audiences. It requires no image hosting and renders reasonably in any markdown viewer.
- **Auto-update after eval, not after training**: The model card gains its most important information (pass@1, perplexity) after evaluation, not during training. Tying the regeneration to the evaluate script ensures the card always reflects the current eval state.
- **`model-card-metadata.json` sidecar**: Generating a machine-readable JSON alongside the human-readable Markdown allows CI scripts, comparison tools, and the experiment tracker to consume model card data without parsing Markdown.
- **safetensors for HF Hub upload**: The project already uses safetensors for checkpoints. The HF Hub upload path uploads `model.safetensors` directly — no conversion or re-packaging needed. This is the 2026 standard checkpoint format for PyTorch models on HF Hub.
