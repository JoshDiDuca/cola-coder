# 66 - Side-by-Side Checkpoint Comparison

## Overview

A terminal UI that displays outputs from multiple checkpoints simultaneously, using Rich Columns layout. Each column shows one checkpoint's generation for a given prompt. Supports interactive checkpoint and prompt selection, metadata display (step number, training loss, timestamp), and export to HTML or markdown table.

**Feature flag:** `--enable-comparison-ui` (standalone CLI tool, always optional)

---

## Motivation

When you have 10 checkpoints from a training run, comparing them by running them sequentially and mentally tracking the differences is tedious and error-prone. Side-by-side comparison lets you:

- Spot immediately when the model starts generating plausible TypeScript vs gibberish
- Compare the stylistic evolution of generations across training
- Share comparison results with others (HTML export)
- Select the best checkpoint for a specific use case

This is a developer productivity feature that pays off during the exploratory phase of training.

---

## Architecture / Design

### Terminal Layout (Rich)

```
╔══════════════════╦══════════════════╦══════════════════╗
║ step-1000        ║ step-3000        ║ step-5000        ║
║ loss: 3.21       ║ loss: 2.44       ║ loss: 1.89       ║
║ 2026-03-10       ║ 2026-03-12       ║ 2026-03-14       ║
╠══════════════════╬══════════════════╬══════════════════╣
║ function add(    ║ function add(    ║ function add(    ║
║   a,             ║   a: number,     ║   a: number,     ║
║   b              ║   b: number      ║   b: number,     ║
║ ) {              ║ ): number {      ║ ): number {      ║
║   returna + b    ║   return a + b;  ║   return a + b;  ║
║ }                ║ }                ║ }                ║
║                  ║                  ║                  ║
╠══════════════════╬══════════════════╬══════════════════╣
║ [SYNTAX ERROR]   ║ [TYPE OK]        ║ [TYPE OK]        ║
╚══════════════════╩══════════════════╩══════════════════╝
```

### Interaction Flow

```
1. cola-coder compare
2. [Interactive]: Select checkpoints from list (multi-select)
3. [Interactive]: Select prompt from library or type custom
4. Display side-by-side generation
5. Options: [n]ext prompt, [p]rev prompt, [e]xport, [q]uit
```

---

## Implementation Steps

### Step 1: Checkpoint Loader (`comparison/checkpoint_loader.py`)

```python
from pathlib import Path
from safetensors.torch import load_file
import json
import yaml
from dataclasses import dataclass

@dataclass
class CheckpointMeta:
    path: Path
    step: int
    training_loss: float | None
    timestamp: str | None
    model_config: dict | None
    label: str        # display label

def load_checkpoint_meta(checkpoint_path: Path) -> CheckpointMeta:
    """Load metadata without loading model weights."""
    stem = checkpoint_path.stem  # e.g., "step-5000"

    # Look for sidecar manifest
    manifest_path = checkpoint_path.parent / f"{stem}.manifest.yaml"
    meta = CheckpointMeta(
        path=checkpoint_path,
        step=0,
        training_loss=None,
        timestamp=None,
        model_config=None,
        label=stem,
    )

    # Parse step from filename
    import re
    m = re.search(r'step[-_]?(\d+)', stem)
    if m:
        meta.step = int(m.group(1))

    if manifest_path.exists():
        data = yaml.safe_load(manifest_path.read_text())
        meta.training_loss = data.get("training_loss")
        meta.timestamp = data.get("timestamp", "")[:10]
        meta.model_config = data.get("model_config")

    return meta

def list_checkpoints(checkpoint_dir: Path) -> list[CheckpointMeta]:
    checkpoints = sorted(
        checkpoint_dir.glob("*.safetensors"),
        key=lambda p: int(re.search(r'\d+', p.stem).group() or 0)
    )
    return [load_checkpoint_meta(cp) for cp in checkpoints]
```

### Step 2: Interactive Selector (`comparison/selector.py`)

```python
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

def select_checkpoints(
    all_checkpoints: list[CheckpointMeta],
    max_columns: int = 3,
) -> list[CheckpointMeta]:
    """Interactive multi-select for checkpoints."""
    console = Console()
    table = Table(title="Available Checkpoints")
    table.add_column("#", style="dim", width=4)
    table.add_column("Checkpoint")
    table.add_column("Step", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Date")

    for i, cp in enumerate(all_checkpoints):
        table.add_row(
            str(i),
            cp.label,
            str(cp.step),
            f"{cp.training_loss:.4f}" if cp.training_loss else "—",
            cp.timestamp or "—",
        )

    console.print(table)
    selection = Prompt.ask(
        f"Select up to {max_columns} checkpoints (comma-separated indices, e.g. 0,3,7)"
    )
    indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
    return [all_checkpoints[i] for i in indices[:max_columns]]

BUILTIN_PROMPTS = [
    {
        "name": "simple-function",
        "prompt": "// Write a TypeScript function that adds two numbers\nfunction add(",
    },
    {
        "name": "async-fetch",
        "prompt": "// Fetch user data from an API\nasync function fetchUser(id: string) {",
    },
    {
        "name": "interface-impl",
        "prompt": "interface Repository<T> {\n  findById(id: string): Promise<T>;\n}\n\nclass UserRepository implements Repository<User> {",
    },
    {
        "name": "error-handling",
        "prompt": "// Parse JSON safely with error handling\nfunction safeParseJson<T>(input: string): T | null {",
    },
]

def select_prompt(custom: str = None) -> dict:
    if custom:
        return {"name": "custom", "prompt": custom}

    console = Console()
    table = Table(title="Prompts")
    table.add_column("#")
    table.add_column("Name")
    table.add_column("Preview", style="dim")
    for i, p in enumerate(BUILTIN_PROMPTS):
        table.add_row(str(i), p["name"], p["prompt"][:50] + "...")
    table.add_row(str(len(BUILTIN_PROMPTS)), "custom", "[enter your own]")
    console.print(table)

    idx_str = Prompt.ask("Select prompt")
    idx = int(idx_str)
    if idx == len(BUILTIN_PROMPTS):
        custom = Prompt.ask("Enter prompt")
        return {"name": "custom", "prompt": custom}
    return BUILTIN_PROMPTS[idx]
```

### Step 3: Comparison Display (`comparison/display.py`)

```python
from rich.columns import Columns
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console
from rich.text import Text

class ComparisonDisplay:
    def __init__(self, checkpoints: list[CheckpointMeta], models: list, tokenizer):
        self.checkpoints = checkpoints
        self.models = models
        self.tokenizer = tokenizer
        self.console = Console()

    def generate_all(self, prompt: str, max_tokens: int = 200) -> list[str]:
        """Generate output from each checkpoint for the given prompt."""
        outputs = []
        for model in self.models:
            tokens = self.tokenizer.encode(prompt)
            import torch
            input_ids = torch.tensor([tokens]).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.1,  # Low temp for deterministic comparison
                    do_sample=False,
                )
            generated = out[0][len(tokens):]
            outputs.append(self.tokenizer.decode(generated.tolist()))
        return outputs

    def validate_all(self, outputs: list[str], prompt: str) -> list[str]:
        """Quick syntax+type check each output, return status strings."""
        statuses = []
        for out in outputs:
            full_code = prompt + out
            # Quick syntax check via tree-sitter (fast)
            try:
                import tree_sitter_typescript as ts_ts
                from tree_sitter import Language, Parser
                lang = Language(ts_ts.language())
                parser = Parser(lang)
                tree = parser.parse(full_code.encode())
                has_error = tree.root_node.has_error
                statuses.append("[red][SYNTAX ERROR][/]" if has_error else "[green][SYNTAX OK][/]")
            except ImportError:
                statuses.append("[dim][no validator][/]")
        return statuses

    def render(self, prompt: str, outputs: list[str], statuses: list[str]):
        panels = []
        for cp, output, status in zip(self.checkpoints, outputs, statuses):
            # Header with metadata
            header = Text()
            header.append(f"Step: {cp.step}", style="bold cyan")
            if cp.training_loss:
                header.append(f"  loss: {cp.training_loss:.3f}", style="dim")
            if cp.timestamp:
                header.append(f"  {cp.timestamp}", style="dim")

            # Syntax-highlighted code
            full_code = prompt + output
            code_display = Syntax(
                full_code[:600],  # truncate for display
                "typescript",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )

            panel_content = Text()
            panel_content.append_text(header)
            panel_content.append("\n\n")

            from rich.console import Group
            panel = Panel(
                Group(header, code_display, Text.from_markup(f"\n{status}")),
                title=cp.label,
                border_style="blue",
            )
            panels.append(panel)

        self.console.print(Columns(panels, equal=True, expand=True))

    def export_markdown(self, prompt: str, outputs: list[str], path: str):
        lines = [f"# Checkpoint Comparison\n\n**Prompt:**\n```typescript\n{prompt}\n```\n\n"]
        lines.append("| " + " | ".join(cp.label for cp in self.checkpoints) + " |")
        lines.append("|" + "---|" * len(self.checkpoints))

        # Split outputs into lines and interleave
        output_lines = [o.split("\n") for o in outputs]
        max_lines = max(len(l) for l in output_lines)
        for i in range(max_lines):
            row = []
            for ol in output_lines:
                cell = ol[i] if i < len(ol) else ""
                row.append(cell.replace("|", "\\|"))
            lines.append("| " + " | ".join(row) + " |")

        Path(path).write_text("\n".join(lines))
        print(f"Exported to {path}")

    def export_html(self, prompt: str, outputs: list[str], path: str):
        console = Console(record=True, width=200)
        self.render(prompt, outputs, self.validate_all(outputs, prompt))
        html = console.export_html()
        Path(path).write_text(html)
        print(f"Exported HTML to {path}")
```

### Step 4: Main CLI Command (`cli/compare_cmd.py`)

```python
# cola-coder compare \
#   --checkpoint-dir checkpoints/ \
#   --prompt "function add(" \
#   --export comparison.html

import click
from comparison.checkpoint_loader import list_checkpoints
from comparison.selector import select_checkpoints, select_prompt
from comparison.display import ComparisonDisplay

@click.command()
@click.option("--checkpoint-dir", default="checkpoints/", type=click.Path())
@click.option("--prompt", default=None, help="Prompt text (skip interactive selection)")
@click.option("--checkpoints", default=None, help="Comma-separated checkpoint names")
@click.option("--export", default=None, help="Export path (.html or .md)")
@click.option("--max-tokens", default=200)
def compare_cmd(checkpoint_dir, prompt, checkpoints, export, max_tokens):
    all_cps = list_checkpoints(Path(checkpoint_dir))

    if checkpoints:
        names = checkpoints.split(",")
        selected = [cp for cp in all_cps if cp.label in names]
    else:
        selected = select_checkpoints(all_cps, max_columns=3)

    selected_prompt = select_prompt(custom=prompt)

    # Load models
    models = [load_model_from_checkpoint(cp.path) for cp in selected]
    tokenizer = load_tokenizer()

    display = ComparisonDisplay(selected, models, tokenizer)
    outputs = display.generate_all(selected_prompt["prompt"], max_tokens=max_tokens)
    statuses = display.validate_all(outputs, selected_prompt["prompt"])
    display.render(selected_prompt["prompt"], outputs, statuses)

    if export:
        if export.endswith(".html"):
            display.export_html(selected_prompt["prompt"], outputs, export)
        else:
            display.export_markdown(selected_prompt["prompt"], outputs, export)
```

---

## Key Files to Modify

- `comparison/checkpoint_loader.py` - New file
- `comparison/selector.py` - New file
- `comparison/display.py` - New file
- `cli/compare_cmd.py` - New file
- `cli/main.py` - Register `compare` command
- `config/compare.yaml` - Optional config for default prompt library paths

---

## Testing Strategy

1. **Metadata parsing test**: create checkpoint filenames with various step naming conventions, assert `load_checkpoint_meta` extracts correct step numbers.
2. **Display render test**: instantiate `ComparisonDisplay` with mock outputs (no real model), call `render()`, assert no exceptions and output is non-empty.
3. **Export markdown test**: call `export_markdown` on 2 outputs, assert output file contains the prompt and both generations.
4. **Export HTML test**: assert exported HTML file is valid (contains `<html>` tag).
5. **Multi-checkpoint generation test**: with 3 mock models, assert `generate_all` returns list of length 3.

---

## Performance Considerations

- Loading 3 large checkpoints simultaneously requires 3x model VRAM. On RTX 3080 (10GB) with a 117M parameter model (≈450MB), 3 checkpoints fit comfortably. For larger models, load and generate sequentially, keeping only one model in VRAM at a time.
- Low temperature (`temperature=0.1`) makes comparisons deterministic and repeatable.
- Syntax validation via tree-sitter is <1ms per output, negligible.
- Export to HTML uses Rich's `export_html()` which re-renders the full console; this is slow for very long outputs. Truncate displayed output to 600 chars but export full output.

---

## Dependencies

```
rich>=13.0      # Columns, Syntax, Panel (already required)
click>=8.0      # CLI framework (already required)
```

Optional: `tree-sitter-typescript` for inline syntax validation in the comparison view.

---

## Estimated Complexity

**Medium.** The Rich layout is straightforward. The complex part is loading multiple checkpoint models into VRAM and managing their memory lifecycle. Sequential loading (one at a time) avoids VRAM issues but is slower. Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Deterministic generation for comparison**: use `temperature=0.1` or `do_sample=False` when comparing checkpoints. High-temperature outputs vary between runs and make comparison meaningless.
- **Normalize column widths**: Rich's `Columns(equal=True)` ensures each checkpoint gets the same display width regardless of output length. Prevents visual dominance by one column.
- **Export as artifact**: treat comparison exports as training artifacts. Store them in `comparisons/` alongside checkpoints with step numbers in the filename.
- **Prompt library versioning**: store comparison prompts in a versioned YAML file so that comparisons across experiments use the same prompts. Don't allow prompts to change between comparison runs.
- **Lazy model loading**: load models only when needed for generation, then immediately unload to free VRAM. Use a simple context manager pattern: `with loaded_model(cp) as model: outputs.append(generate(model, prompt))`.
