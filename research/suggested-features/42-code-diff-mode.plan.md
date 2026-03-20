# Feature 42: Code Diff Mode

## Overview

Instead of generating a full replacement when editing code, diff mode generates only
the changes — a delta between the original code and the desired modified version.
This produces shorter outputs that are directly applicable and easier to inspect.

Cola-Coder's diff mode takes `(original_code, instruction)` as input and outputs a
diff in unified format (or a custom edit format). The diff is then applied with
`difflib` or `patch`. CLI shows colored diff output.

Status: OPTIONAL — enable via `--feature diff-mode` or CLI menu toggle.

---

## Motivation

- A typical "add error handling" edit changes 3–10 lines out of 50. Full regeneration
  wastes tokens on unchanged code and risks accidental alterations.
- Diffs are explicit about what changed — easier to review than comparing two versions.
- Shorter output = faster generation and easier validation.
- Enables future fine-tuning on git commit data: `(before, after, message)` triples.

---

## Architecture / Design

### Prompt Format for Diff Generation

```
Original code:
```python
def divide(a, b):
    return a / b
```

Instruction: Add error handling for division by zero.

Generate a unified diff to apply this change:
```diff
```

The model is expected to output a unified diff:

```diff
--- a/code.py
+++ b/code.py
@@ -1,2 +1,5 @@
 def divide(a, b):
-    return a / b
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
+    return a / b
```

### DiffGenerator Class

```python
# cola_coder/diff/diff_generator.py

import difflib
from pathlib import Path


DIFF_PROMPT_TEMPLATE = """\
Original code:
```{language}
{original}
```

Instruction: {instruction}

Generate a unified diff to apply this change. Output only the diff, nothing else:
```diff
"""


class DiffGenerator:
    def __init__(self, generator, tokenizer):
        self.generator = generator
        self.tokenizer = tokenizer

    def generate_diff(
        self,
        original: str,
        instruction: str,
        language: str = "python",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Generate a unified diff for the given instruction applied to original."""
        prompt = DIFF_PROMPT_TEMPLATE.format(
            language=language,
            original=original.strip(),
            instruction=instruction,
        )
        raw = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=["```\n"],
        )
        return self._extract_diff(raw)

    def _extract_diff(self, raw: str) -> str:
        """Extract diff content from model output (may be wrapped in code fence)."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Skip the opening fence line
            start = 1 if lines[0].startswith("```") else 0
            # Find closing fence
            end = len(lines)
            for i in range(start, len(lines)):
                if lines[i].strip() == "```":
                    end = i
                    break
            return "\n".join(lines[start:end])
        return raw

    def apply_diff(
        self,
        original: str,
        diff_str: str,
    ) -> tuple[str, bool]:
        """
        Apply a unified diff to the original code.
        Returns (result, success). On failure, returns (original, False).
        """
        try:
            result = apply_unified_diff(original, diff_str)
            return result, True
        except Exception as e:
            return original, False

    def generate_and_apply(
        self,
        original: str,
        instruction: str,
        language: str = "python",
        **gen_kwargs,
    ) -> dict:
        """Generate diff, apply it, return full result with metadata."""
        diff_str = self.generate_diff(original, instruction, language, **gen_kwargs)
        modified, success = self.apply_diff(original, diff_str)
        return {
            "original": original,
            "instruction": instruction,
            "diff": diff_str,
            "modified": modified,
            "applied": success,
        }
```

### Unified Diff Parser and Applier

```python
# cola_coder/diff/patcher.py

import re
from dataclasses import dataclass


@dataclass
class Hunk:
    orig_start: int
    orig_count: int
    new_start: int
    new_count: int
    lines: list[str]   # "+", "-", " " prefixed lines


def parse_unified_diff(diff_str: str) -> list[Hunk]:
    """Parse unified diff format into Hunk objects."""
    hunks = []
    current_hunk: Hunk | None = None
    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for line in diff_str.split("\n"):
        m = hunk_pattern.match(line)
        if m:
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = Hunk(
                orig_start=int(m.group(1)),
                orig_count=int(m.group(2) or 1),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or 1),
                lines=[],
            )
        elif current_hunk is not None:
            if line.startswith(("+", "-", " ")):
                current_hunk.lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def apply_unified_diff(original: str, diff_str: str) -> str:
    """
    Apply a unified diff to original code.
    Raises ValueError if diff cannot be cleanly applied.
    """
    hunks = parse_unified_diff(diff_str)
    if not hunks:
        raise ValueError("No hunks found in diff")

    orig_lines = original.splitlines(keepends=True)
    result_lines = list(orig_lines)
    offset = 0  # track line number shifts from previous hunks

    for hunk in hunks:
        start = hunk.orig_start - 1 + offset  # 0-indexed
        # Validate context lines match
        orig_pos = start
        for line in hunk.lines:
            if line.startswith(" "):
                expected = line[1:]  # strip prefix
                actual = result_lines[orig_pos] if orig_pos < len(result_lines) else ""
                if actual.rstrip("\n") != expected.rstrip("\n"):
                    raise ValueError(
                        f"Context mismatch at line {orig_pos + 1}: "
                        f"expected {repr(expected)}, got {repr(actual)}"
                    )
                orig_pos += 1
            elif line.startswith("-"):
                orig_pos += 1

        # Apply hunk
        new_lines = []
        pos = start
        for line in hunk.lines:
            if line.startswith(" "):
                new_lines.append(result_lines[pos])
                pos += 1
            elif line.startswith("-"):
                pos += 1  # skip this line (remove it)
            elif line.startswith("+"):
                content = line[1:]
                if not content.endswith("\n"):
                    content += "\n"
                new_lines.append(content)

        # Replace original hunk lines with new lines
        replaced_count = hunk.orig_count
        result_lines[start:start + replaced_count] = new_lines
        offset += len(new_lines) - replaced_count

    return "".join(result_lines)


def generate_diff_string(original: str, modified: str, filename: str = "code.py") -> str:
    """Generate a unified diff string between original and modified code."""
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    ))
    return "\n".join(diff)
```

### Custom Edit Format (Alternative to Unified Diff)

Simpler format that is easier for models to generate reliably:

```python
# cola_coder/diff/edit_format.py
"""
Custom edit format (similar to Aider's SEARCH/REPLACE format):

<<<<<<< ORIGINAL
    return a / b
=======
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
>>>>>>> MODIFIED
"""

import re


def apply_edit_format(original: str, edit_str: str) -> tuple[str, bool]:
    """Apply ORIGINAL/MODIFIED edit blocks to original code."""
    result = original
    pattern = re.compile(
        r"<<<<<<< ORIGINAL\n(.*?)\n=======\n(.*?)\n>>>>>>> MODIFIED",
        re.DOTALL,
    )
    for match in pattern.finditer(edit_str):
        search_text = match.group(1)
        replace_text = match.group(2)
        if search_text in result:
            result = result.replace(search_text, replace_text, 1)
        else:
            return original, False
    return result, True
```

### CLI Diff Display with Rich

```python
# cola_coder/cli/diff_display.py

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()


def display_diff(diff_str: str) -> None:
    """Display colored diff output in the terminal."""
    lines = diff_str.split("\n")
    text = Text()
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            text.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"):
            text.append(line + "\n", style="red")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith(("---", "+++")):
            text.append(line + "\n", style="bold")
        else:
            text.append(line + "\n")
    console.print(Panel(text, title="Generated Diff", border_style="blue"))


def display_before_after(original: str, modified: str, language: str = "python") -> None:
    """Side-by-side display of original and modified code."""
    console.print(Panel(
        Syntax(original, language, theme="monokai", line_numbers=True),
        title="[red]Before[/red]", border_style="red"
    ))
    console.print(Panel(
        Syntax(modified, language, theme="monokai", line_numbers=True),
        title="[green]After[/green]", border_style="green"
    ))
```

### Validation: Does Result Compile?

```python
# cola_coder/diff/validate.py

import ast
import subprocess
import tempfile
from pathlib import Path


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def validate_typescript_syntax(code: str) -> tuple[bool, str]:
    """Check TypeScript syntax using tsc --noEmit (requires tsc in PATH)."""
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["tsc", "--noEmit", "--strict", tmp_path],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0, result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True, "tsc not available — skipping TypeScript validation"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

---

## Implementation Steps

1. **Create `cola_coder/diff/` package**: `__init__.py`, `diff_generator.py`,
   `patcher.py`, `edit_format.py`, `validate.py`.

2. **Add diff mode to CLI menu**: "Edit code with diff mode" → prompt for original
   code path or paste, then instruction.

3. **Add HTTP endpoint**:
   ```python
   @app.post("/diff")
   def generate_diff_endpoint(req: DiffRequest):
       result = app.state.diff_generator.generate_and_apply(
           req.original, req.instruction, req.language
       )
       return result
   ```

4. **Training data collection**: optionally mine git history for `(before, after, msg)`
   triples to fine-tune on diff generation:
   ```bash
   git log --no-merges --format="%H %s" | head -1000 | \
   xargs -I{} sh -c 'git show {}...'
   ```

5. **Fallback**: if diff cannot be parsed or applied, fall back to full regeneration.

6. **Add `--diff-format` flag**: `unified` (default) or `search-replace` (easier for model).

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Pass diff-specific stop tokens and prompt format |
| `server.py` | Add `/diff` endpoint |
| `cli/menu.py` | Add "Diff mode" option |
| `cli/diff_display.py` | New file |
| `cola_coder/diff/` | New package |
| `config.py` | Add `DiffConfig` |

---

## Testing Strategy

```python
# tests/test_diff.py

def test_parse_unified_diff():
    diff = """--- a/code.py
+++ b/code.py
@@ -1,3 +1,4 @@
 def f(x):
-    return x
+    if x < 0:
+        raise ValueError
+    return x"""
    hunks = parse_unified_diff(diff)
    assert len(hunks) == 1
    assert hunks[0].orig_start == 1

def test_apply_unified_diff():
    original = "def f(x):\n    return x\n"
    diff = """@@ -1,2 +1,4 @@
 def f(x):
-    return x
+    if x < 0:
+        raise ValueError
+    return x
"""
    result = apply_unified_diff(original, diff)
    assert "raise ValueError" in result
    assert "return x" in result

def test_apply_diff_roundtrip():
    original = "x = 1\ny = 2\n"
    modified = "x = 10\ny = 2\n"
    diff = generate_diff_string(original, modified)
    restored = apply_unified_diff(original, diff)
    assert restored == modified

def test_edit_format_apply():
    original = "def f():\n    pass\n"
    edit = "<<<<<<< ORIGINAL\n    pass\n=======\n    return 42\n>>>>>>> MODIFIED"
    result, ok = apply_edit_format(original, edit)
    assert ok
    assert "return 42" in result

def test_validate_python_syntax():
    ok, err = validate_python_syntax("def f():\n    return 1\n")
    assert ok
    broken, err2 = validate_python_syntax("def f(:\n    return 1")
    assert not broken
```

---

## Performance Considerations

- **Output length**: diffs are typically 20–50% the length of full regeneration.
  This means ~2x faster generation for typical edits.
- **Model reliability**: standard language models trained on code may not reliably
  produce well-formed unified diffs without fine-tuning. The search/replace format
  is more reliably generated.
- **Fallback rate**: if the model frequently generates malformed diffs, the fallback
  to full regeneration negates the benefit. Track fallback rate and tune if needed.
- **Context overhead**: the diff prompt includes the full original code, which costs
  tokens. For very long files, consider only passing the relevant section.

---

## Dependencies

```
difflib          # stdlib — unified diff generation and application
ast              # stdlib — Python syntax validation
rich>=13.0.0     # colored diff display (already required)
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| DiffGenerator + prompt format | 2 hours |
| Unified diff parser + patcher | 4 hours |
| Search/replace edit format | 2 hours |
| CLI diff display | 2 hours |
| Validation (Python + TypeScript) | 2 hours |
| HTTP endpoint | 1 hour |
| Training data collection script | 2 hours |
| Tests | 3 hours |
| **Total** | **~18 hours** |

Complexity rating: **Medium** — diff parsing is tricky to get right; search/replace
format is much easier to parse reliably.

---

## 2026 Best Practices

- **Aider's edit formats**: Aider (the AI coding tool) has extensively experimented with
  edit formats. Their "SEARCH/REPLACE" (similar to this plan's edit format) and "udiff"
  formats are well-tested. Their research shows udiff works best for GPT-4-class models;
  search/replace works better for smaller models.
- **Lazy diff (diff only changed functions)**: instead of diffing the whole file, use
  tree-sitter to identify which functions changed and only output diffs for those
  functions. Reduces output length further.
- **Diff validation before applying**: always validate the diff parses and applies
  cleanly before showing to the user. Show error and retry with fallback if it fails.
- **Streaming diff display**: for long diffs, stream lines as they arrive and color
  them in real-time using the streaming generation feature.
- **Git integration**: after applying a diff, optionally stage with `git add -p` for
  interactive review before committing.
