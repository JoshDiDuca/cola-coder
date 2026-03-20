# 03 - Quick Smoke Test

## Overview

After any training run completes, automatically generate completions for 5 canonical TypeScript prompts and print them to the terminal with syntax highlighting. The smoke test is a rapid sanity check — it takes 10-30 seconds and immediately reveals whether the model is producing coherent output, stuck in repetition loops, or outputting garbage tokens. It integrates as an optional post-training hook that the user can accept or decline.

---

## Motivation

Training a transformer can silently fail in many ways:
- Loss decreases but the model outputs repetitive boilerplate
- Tokenizer mismatch causes garbage byte outputs
- Learning rate too high causes incoherent outputs despite numeric "convergence"
- Early checkpoints look fine at step 1000 but degrade by step 5000

Currently, the only way to check is to manually run generate.py after training. The smoke test automates this check and presents outputs in a readable format so problems are immediately visible.

**Key insight**: A model that produces valid-looking TypeScript stubs after 30 minutes of training is almost certainly learning correctly. A model that outputs `<unk><unk><unk>` or `{{{{{{{{` is broken and no amount of further training will fix it.

---

## Architecture / Design

### The 5 Canonical Prompts

These prompts are chosen to cover the major TypeScript patterns in the training corpus:

| # | Prompt | Tests |
|---|--------|-------|
| 1 | `function add(a: number, b: number)` | Basic function with typed params |
| 2 | `interface User {` | Interface/type definition |
| 3 | `const fetchData = async` | Async/await pattern |
| 4 | `class Logger {` | Class definition |
| 5 | `export function validateEmail` | Module export + function |

### Output Display Format

```
╔══════════════════════════════════════════════════════════════╗
║  Quick Smoke Test — Checkpoint: runs/run_001/ckpt_8400.pt    ║
╚══════════════════════════════════════════════════════════════╝

[1/5] function add(a: number, b: number)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROMPT:     function add(a: number, b: number)
  GENERATED:  : number {
                return a + b;
              }

  STATUS:  PASS  (syntax valid, no repetition detected)

[2/5] interface User {
...

Summary: 5/5 outputs generated  |  4/5 syntax valid  |  0 failure patterns detected
```

### Failure Detection

The smoke test flags three failure modes:

1. **Empty output**: Generated 0 tokens or only whitespace
2. **Repetition loop**: Any 4+ gram repeated 3+ times in a row (e.g., `} } } } } } } }`)
3. **Garbage tokens**: High proportion of non-ASCII characters, `<unk>` tokens, or token IDs above vocabulary size

---

## Implementation Steps

### Step 1: Define the 5 canonical prompts

```python
# src/evaluation/smoke_test/prompts.py
from dataclasses import dataclass

@dataclass
class SmokePrompt:
    id: str
    label: str
    text: str
    min_expected_tokens: int = 5    # Fail if fewer than this generated
    description: str = ""

SMOKE_PROMPTS: list[SmokePrompt] = [
    SmokePrompt(
        id="smoke_01_fn",
        label="Basic function",
        text="function add(a: number, b: number)",
        min_expected_tokens=5,
        description="Simple arithmetic function with typed parameters",
    ),
    SmokePrompt(
        id="smoke_02_interface",
        label="Interface definition",
        text="interface User {",
        min_expected_tokens=8,
        description="TypeScript interface with at least one property",
    ),
    SmokePrompt(
        id="smoke_03_async",
        label="Async function",
        text="const fetchData = async",
        min_expected_tokens=6,
        description="Async arrow function with fetch pattern",
    ),
    SmokePrompt(
        id="smoke_04_class",
        label="Class definition",
        text="class Logger {",
        min_expected_tokens=10,
        description="Class with constructor and at least one method",
    ),
    SmokePrompt(
        id="smoke_05_export",
        label="Exported function",
        text="export function validateEmail",
        min_expected_tokens=8,
        description="Exported function with a common utility pattern",
    ),
]
```

### Step 2: Failure detectors

```python
# src/evaluation/smoke_test/detectors.py
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class FailureInfo:
    failure_type: str      # "empty", "repetition", "garbage"
    description: str
    severity: str          # "warning" or "critical"

def detect_empty(text: str) -> Optional[FailureInfo]:
    if not text or not text.strip():
        return FailureInfo("empty", "Model generated no output", "critical")
    if len(text.split()) < 3:
        return FailureInfo("empty", f"Only {len(text.split())} words generated", "warning")
    return None

def detect_repetition(text: str, ngram_size: int = 4, repeat_threshold: int = 3) -> Optional[FailureInfo]:
    """Detect repeating n-gram loops in generated text."""
    tokens = text.split()
    if len(tokens) < ngram_size * repeat_threshold:
        return None
    for i in range(len(tokens) - ngram_size):
        ngram = tuple(tokens[i:i + ngram_size])
        # Count consecutive repeats
        count = 1
        j = i + ngram_size
        while j + ngram_size <= len(tokens):
            if tuple(tokens[j:j + ngram_size]) == ngram:
                count += 1
                j += ngram_size
            else:
                break
        if count >= repeat_threshold:
            snippet = " ".join(ngram)
            return FailureInfo(
                "repetition",
                f'N-gram "{snippet}" repeated {count} times consecutively',
                "critical",
            )
    # Also check character-level repetition (e.g., "}}}}}}")
    for char in ['}', '{', ';', '\n']:
        if text.count(char * 8) > 0:
            return FailureInfo(
                "repetition",
                f"Character '{char}' repeated 8+ times in a row",
                "critical",
            )
    return None

def detect_garbage(text: str, garbage_threshold: float = 0.3) -> Optional[FailureInfo]:
    """Detect high proportion of non-ASCII or suspicious tokens."""
    if not text:
        return None
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / len(text)
    if ratio > garbage_threshold:
        return FailureInfo(
            "garbage",
            f"{ratio:.0%} of output is non-ASCII characters",
            "critical",
        )
    # Check for <unk> tokens
    unk_count = text.count("<unk>") + text.count("[UNK]")
    if unk_count > 3:
        return FailureInfo(
            "garbage",
            f"Found {unk_count} unknown tokens in output",
            "warning",
        )
    return None

def run_all_detectors(text: str) -> list[FailureInfo]:
    """Run all failure detectors, return list of found failures."""
    failures = []
    for detector in [detect_empty, detect_repetition, detect_garbage]:
        result = detector(text)
        if result:
            failures.append(result)
    return failures
```

### Step 3: Syntax validation (lightweight, no tsc required)

```python
# src/evaluation/smoke_test/syntax_check.py
import subprocess
import tempfile
import os

def quick_syntax_check(code: str) -> tuple[bool, str]:
    """
    Check TypeScript syntax using tsc if available,
    fall back to a simple bracket-balance heuristic.
    Returns (is_valid, message).
    """
    # Try tsc first
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--allowJs", "--skipLibCheck", fname],
            capture_output=True, text=True, timeout=8
        )
        os.unlink(fname)
        if result.returncode == 0:
            return True, "tsc: OK"
        # Extract first error message
        first_error = result.stderr.split("\n")[0] if result.stderr else result.stdout.split("\n")[0]
        return False, f"tsc: {first_error[:80]}"
    except FileNotFoundError:
        os.unlink(fname)
        # Fallback: bracket balance heuristic
        return _heuristic_check(code)
    except subprocess.TimeoutExpired:
        os.unlink(fname)
        return None, "tsc: timeout"

def _heuristic_check(code: str) -> tuple[bool, str]:
    """Simple bracket balance check as fallback."""
    opens = code.count('{') + code.count('(') + code.count('[')
    closes = code.count('}') + code.count(')') + code.count(']')
    if opens == 0 and closes == 0:
        return False, "heuristic: no brackets found (possibly empty)"
    imbalance = abs(opens - closes)
    if imbalance > 5:
        return False, f"heuristic: bracket imbalance ({opens} open, {closes} close)"
    return True, "heuristic: bracket balance OK"
```

### Step 4: Main smoke test runner

```python
# src/evaluation/smoke_test/runner.py
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from src.evaluation.smoke_test.prompts import SMOKE_PROMPTS
from src.evaluation.smoke_test.detectors import run_all_detectors
from src.evaluation.smoke_test.syntax_check import quick_syntax_check

console = Console()

def run_smoke_test(
    checkpoint_path: Path,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
    device: str = "cuda",
) -> dict:
    from src.inference import load_model_for_inference, generate_text
    model, tokenizer = load_model_for_inference(checkpoint_path, device)

    console.print(Panel(
        f"[bold]Quick Smoke Test[/bold]\nCheckpoint: {checkpoint_path}",
        style="cyan"
    ))

    results = []
    for i, prompt_def in enumerate(SMOKE_PROMPTS, 1):
        console.print(f"\n[bold cyan][{i}/5] {prompt_def.label}[/bold cyan]")
        console.rule()

        # Generate
        generated = generate_text(
            model, tokenizer,
            prompt=prompt_def.text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        full_code = prompt_def.text + generated

        # Display side-by-side
        console.print(f"[dim]PROMPT:[/dim]    [yellow]{prompt_def.text}[/yellow]")
        syntax = Syntax(full_code, "typescript", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title="Generated", border_style="dim"))

        # Check failures
        failures = run_all_detectors(generated)
        syntax_valid, syntax_msg = quick_syntax_check(full_code)

        # Status line
        if failures:
            for f in failures:
                color = "red" if f.severity == "critical" else "yellow"
                console.print(f"  [{color}]FAIL[/{color}]  {f.failure_type.upper()}: {f.description}")
        elif syntax_valid is False:
            console.print(f"  [yellow]WARN[/yellow]  Syntax: {syntax_msg}")
        else:
            console.print(f"  [green]PASS[/green]  {syntax_msg}")

        results.append({
            "prompt_id": prompt_def.id,
            "prompt_text": prompt_def.text,
            "generated": generated,
            "failures": [vars(f) for f in failures],
            "syntax_valid": syntax_valid,
            "syntax_msg": syntax_msg,
        })

    _render_summary(results)
    return results

def _render_summary(results: list[dict]) -> None:
    total = len(results)
    syntax_ok = sum(1 for r in results if r["syntax_valid"])
    clean = sum(1 for r in results if not r["failures"])
    critical = sum(1 for r in results for f in r["failures"] if f["severity"] == "critical")

    style = "green" if critical == 0 else "red" if critical >= 3 else "yellow"
    console.print(Panel(
        f"{total}/{total} outputs generated  |  "
        f"{syntax_ok}/{total} syntax valid  |  "
        f"{clean}/{total} clean (no failure patterns)  |  "
        f"{critical} critical issues",
        title="Smoke Test Summary",
        style=style,
    ))
```

### Step 5: Post-training hook integration

```python
# src/train.py (modification — add at end of training loop)
from src.evaluation.smoke_test.runner import run_smoke_test
from src.cli import confirm

def post_training_hook(checkpoint_path: Path, config: dict) -> None:
    """Optionally run smoke test after training completes."""
    if not config.get("smoke_test_enabled", False):
        # Ask user if not configured
        if confirm("Run quick smoke test on final checkpoint?", default=True):
            run_smoke_test(checkpoint_path)
    else:
        run_smoke_test(checkpoint_path)
```

In `config.yaml`, add:
```yaml
# Optional post-training features
post_training:
  smoke_test: true          # Run 5 canonical prompts after training
  smoke_test_temp: 0.3      # Temperature for smoke test generation
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/train.py` | Add `post_training_hook()` call at end of training |
| `src/evaluation/smoke_test/__init__.py` | New package |
| `src/evaluation/smoke_test/prompts.py` | The 5 canonical prompts |
| `src/evaluation/smoke_test/detectors.py` | Failure detection logic |
| `src/evaluation/smoke_test/syntax_check.py` | Syntax validation with tsc fallback |
| `src/evaluation/smoke_test/runner.py` | Main runner with Rich display |
| `configs/*.yaml` | Add `post_training.smoke_test` flag |
| `src/menu/menus/generate_menu.py` | Add as standalone menu item |

---

## Testing Strategy

- **Detector unit tests**:
  - `detect_repetition("} } } } } } } } }")` should return a failure
  - `detect_garbage("\x80\x81\x82\x83\x83\x83 hello world")` — test with high non-ASCII ratio
  - `detect_empty("")` should return critical failure
- **Syntax check tests**: Feed known-valid TS, known-invalid TS, verify `quick_syntax_check` returns correct booleans
- **Runner integration test**: Use a tiny saved model checkpoint, run smoke test, verify all 5 results are populated
- **Hook test**: Verify the hook respects `config.post_training.smoke_test = false` and does not run

---

## Performance Considerations

- 5 prompts × 150 tokens = 750 generated tokens total
- At 1000 tok/s (RTX 3080 with a 50M model), this is less than 1 second of generation
- Including model load from disk: ~5-10 seconds total
- If smoke test is run immediately after training (model already in memory), generation is nearly instant
- `tsc` startup adds ~0.5s × 5 = 2.5 seconds; use `--skipLibCheck` and `--transpileOnly` flags to reduce this

---

## Dependencies

| Tool | Use | Install |
|------|-----|---------|
| `rich` | All display | Already installed |
| `tsc` | Syntax checking | `npm install -g typescript` (optional) |
| Python `re` | Repetition detection | stdlib |

No new Python packages required.

---

## Estimated Complexity

**Low** — 1 day.

- Writing 5 prompts: 30 minutes
- Failure detectors: 2 hours
- Syntax check with fallback: 1 hour
- Main runner + Rich display: 2 hours
- Hook integration in train.py: 1 hour
- Testing: 2 hours

Total: ~8.5 hours

---

## 2026 Best Practices

- **Temperature 0.2-0.4 for smoke tests**: Greedy (0.0) can mask repetition issues that appear at normal sampling temperatures. Use slightly above zero for realistic evaluation.
- **Display full code, not just completion**: Show prompt + generated together as a code block. Inspecting just the completion out of context is misleading.
- **Fail fast, fail loud**: Critical failures (empty output, repetition loops) should be shown in red with a prominent warning. The user needs to see this immediately.
- **Separate from benchmarks**: Smoke tests are qualitative + fast. Benchmarks (feature 02) are quantitative + slower. Never conflate them.
- **Persist outputs**: Save smoke test results to `runs/<run_name>/smoke_test_<step>.json` alongside training artifacts for comparison across runs.
