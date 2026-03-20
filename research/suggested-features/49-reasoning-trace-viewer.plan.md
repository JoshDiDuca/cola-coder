# Feature 49: Reasoning Trace Viewer

**Status:** Proposed
**CLI Flag:** `--trace-viewer`
**Complexity:** Low-Medium

---

## Overview

A CLI tool that renders model reasoning traces with rich formatting: `<think>` sections in gray/italic, `<plan>` sections in blue, code in syntax-highlighted panels. Supports browsing multiple traces from an evaluation run, filtering by pass/fail status, annotating quality issues, and exporting to standalone HTML.

---

## Motivation

Debugging reasoning models is difficult without tooling. When a model generates wrong code, developers need to answer:
- Did the model understand the problem correctly?
- Did it have a sound plan?
- Where did reasoning diverge from the final code?

Currently, raw evaluation output is a wall of text. A structured viewer makes these questions answerable in seconds rather than minutes of manual parsing. It also enables qualitative research: browsing 50 traces from a training checkpoint to spot failure patterns.

---

## Architecture / Design

```
Evaluation run
  └── traces/
       ├── eval_run_001.jsonl    ← one JSON per line: prompt, generation, result
       └── eval_run_002.jsonl

CLI: cola-coder trace-view <path>
  │
  ▼
TraceLoader → [TraceRecord, ...]
  │
  ▼
TraceFilter (pass/fail/all, text search)
  │
  ▼
TraceBrowser (interactive Rich TUI or paged output)
  │
  └── TraceExporter → traces.html
```

### TraceRecord Schema

```python
# src/trace_viewer/models.py
from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class TraceRecord:
    id: str
    prompt: str
    generation: str
    passed: bool
    test_score: float
    type_score: float
    think_quality: Optional[float] = None
    think: str = ""
    plan: str = ""
    code: str = ""
    annotations: list[str] = field(default_factory=list)
    source_file: str = ""
    step: int = 0
```

---

## Implementation Steps

### Step 1: Trace Record I/O

```python
# src/trace_viewer/loader.py
import json
import re
from pathlib import Path
from src.trace_viewer.models import TraceRecord

def _extract_sections(generation: str) -> tuple[str, str, str]:
    think_m = re.search(r"<think>(.*?)</think>", generation, re.DOTALL)
    plan_m  = re.search(r"<plan>(.*?)</plan>",   generation, re.DOTALL)
    code_m  = re.search(r"```(?:\w+)?\n(.*?)```", generation, re.DOTALL)
    return (
        think_m.group(1).strip() if think_m else "",
        plan_m.group(1).strip()  if plan_m  else "",
        code_m.group(1).strip()  if code_m  else generation,
    )

def load_traces(path: str | Path) -> list[TraceRecord]:
    path = Path(path)
    records = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                raw = json.loads(line)
                think, plan, code = _extract_sections(raw.get("generation", ""))
                records.append(TraceRecord(
                    id=raw.get("id", f"{f.stem}:{i}"),
                    prompt=raw.get("prompt", ""),
                    generation=raw.get("generation", ""),
                    passed=raw.get("passed", False),
                    test_score=raw.get("test_score", 0.0),
                    type_score=raw.get("type_score", 0.0),
                    think_quality=raw.get("think_quality"),
                    think=think,
                    plan=plan,
                    code=code,
                    source_file=str(f),
                    step=raw.get("step", 0),
                ))

    return records

def save_trace(record: TraceRecord, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "id": record.id,
            "prompt": record.prompt,
            "generation": record.generation,
            "passed": record.passed,
            "test_score": record.test_score,
            "type_score": record.type_score,
            "think_quality": record.think_quality,
            "step": record.step,
        }) + "\n")
```

### Step 2: Trace Annotator

```python
# src/trace_viewer/annotator.py
from src.trace_viewer.models import TraceRecord

QUALITY_ANNOTATIONS = [
    ("copy_paste",  lambda r: _is_copy_paste(r)),
    ("empty_think", lambda r: len(r.think.strip()) < 20),
    ("no_plan",     lambda r: not r.plan.strip()),
    ("low_quality", lambda r: r.think_quality is not None and r.think_quality < 0.3),
    ("long_think",  lambda r: len(r.think.split()) > 300),
]

def _is_copy_paste(record: TraceRecord) -> bool:
    if not record.think or not record.prompt:
        return False
    prompt_words = set(record.prompt.lower().split())
    think_words  = set(record.think.lower().split())
    overlap = len(prompt_words & think_words) / max(len(prompt_words), 1)
    return overlap > 0.75

def annotate(record: TraceRecord) -> TraceRecord:
    record.annotations = []
    for name, check in QUALITY_ANNOTATIONS:
        if check(record):
            record.annotations.append(name)
    return record

def annotate_all(records: list[TraceRecord]) -> list[TraceRecord]:
    return [annotate(r) for r in records]
```

### Step 3: Rich Terminal Renderer

```python
# src/trace_viewer/renderer.py
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from src.trace_viewer.models import TraceRecord

console = Console()

STATUS_COLORS = {True: "green", False: "red"}
STATUS_LABELS = {True: "PASS", False: "FAIL"}

def render_trace(record: TraceRecord, show_think=True, show_plan=True, show_code=True):
    # Header
    status_color = STATUS_COLORS[record.passed]
    header = Text()
    header.append(f"[{STATUS_LABELS[record.passed]}] ", style=f"bold {status_color}")
    header.append(f"ID: {record.id}", style="bold white")
    if record.annotations:
        header.append(f"  [{', '.join(record.annotations)}]", style="yellow")
    console.print(header)

    # Scores row
    scores = Table.grid(padding=(0, 2))
    scores.add_row(
        f"Test: [cyan]{record.test_score:.2f}[/]",
        f"Types: [cyan]{record.type_score:.2f}[/]",
        f"Think Quality: [cyan]{record.think_quality or 'N/A'}[/]",
    )
    console.print(scores)

    # Prompt
    console.print(Panel(record.prompt, title="Prompt", border_style="white", padding=(0, 1)))

    # Think section
    if show_think and record.think:
        think_text = Text(record.think, style="dim italic")
        console.print(Panel(think_text, title="[dim]Think[/dim]", border_style="dim white"))

    # Plan section
    if show_plan and record.plan:
        console.print(Panel(record.plan, title="[blue]Plan[/blue]", border_style="blue"))

    # Code section
    if show_code and record.code:
        syntax = Syntax(record.code, "typescript", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[green]Code[/green]", border_style="green"))

    console.print()

def render_summary(records: list[TraceRecord]):
    total   = len(records)
    passed  = sum(1 for r in records if r.passed)
    failed  = total - passed
    annotated = {}
    for r in records:
        for a in r.annotations:
            annotated[a] = annotated.get(a, 0) + 1

    table = Table(title="Trace Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total traces", str(total))
    table.add_row("Passed", f"[green]{passed}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]")
    table.add_row("Pass rate", f"{passed/total*100:.1f}%")
    for ann, count in sorted(annotated.items(), key=lambda x: -x[1]):
        table.add_row(f"  [{ann}]", str(count))

    console.print(table)
```

### Step 4: Interactive Browser

```python
# src/trace_viewer/browser.py
from rich.console import Console
from rich.prompt import Prompt, Confirm
from src.trace_viewer.models import TraceRecord
from src.trace_viewer.renderer import render_trace, render_summary

console = Console()

def browse_traces(records: list[TraceRecord], start_index: int = 0):
    idx = start_index
    n = len(records)

    while True:
        console.clear()
        render_trace(records[idx])
        console.print(f"[dim]Trace {idx+1}/{n}[/dim]  [bold]n[/bold]=next  [bold]p[/bold]=prev  [bold]q[/bold]=quit  [bold]f[/bold]=filter  [bold]e[/bold]=export")

        cmd = Prompt.ask("", default="n", console=console)
        if cmd == "n":
            idx = (idx + 1) % n
        elif cmd == "p":
            idx = (idx - 1) % n
        elif cmd == "q":
            break
        elif cmd == "f":
            filter_term = Prompt.ask("Filter (pass/fail/annotation name)", console=console)
            filtered = filter_traces(records, filter_term)
            if filtered:
                browse_traces(filtered, start_index=0)
            else:
                console.print("[yellow]No traces match filter.[/yellow]")
        elif cmd == "e":
            out_path = Prompt.ask("Export path", default="traces.html", console=console)
            export_html(records[:idx+1], out_path)
            console.print(f"[green]Exported to {out_path}[/green]")

def filter_traces(
    records: list[TraceRecord],
    filter_term: str,
) -> list[TraceRecord]:
    term = filter_term.lower().strip()
    if term == "pass":
        return [r for r in records if r.passed]
    if term == "fail":
        return [r for r in records if not r.passed]
    # Filter by annotation name or text search
    return [
        r for r in records
        if term in r.annotations
        or term in r.prompt.lower()
        or term in r.think.lower()
    ]
```

### Step 5: HTML Exporter

```python
# src/trace_viewer/exporter.py
from pathlib import Path
from src.trace_viewer.models import TraceRecord

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cola-Coder Reasoning Traces</title>
<style>
  body {{ font-family: monospace; background: #1e1e2e; color: #cdd6f4; max-width: 900px; margin: 40px auto; }}
  .trace {{ border: 1px solid #45475a; margin: 20px 0; padding: 16px; border-radius: 8px; }}
  .pass {{ border-left: 4px solid #a6e3a1; }}
  .fail {{ border-left: 4px solid #f38ba8; }}
  .header {{ font-size: 1.1em; font-weight: bold; margin-bottom: 8px; }}
  .section {{ margin: 10px 0; padding: 10px; border-radius: 4px; }}
  .think {{ background: #181825; color: #6c7086; font-style: italic; }}
  .plan {{ background: #1e1e2e; border-left: 3px solid #89b4fa; }}
  .code {{ background: #11111b; color: #cba6f7; white-space: pre; overflow-x: auto; }}
  .prompt {{ background: #181825; }}
  .score {{ color: #89dceb; font-size: 0.9em; }}
  .annotation {{ background: #f9e2af; color: #1e1e2e; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; margin: 2px; }}
  .label {{ color: #89b4fa; font-weight: bold; }}
</style>
</head>
<body>
<h1>Reasoning Traces — {n_total} traces, {n_passed} passed</h1>
{body}
</body>
</html>"""

TRACE_TEMPLATE = """<div class="trace {status_class}">
  <div class="header">{status_emoji} {record_id}
    {annotations}
  </div>
  <div class="score">Test: {test_score:.2f} | Types: {type_score:.2f} | Think Quality: {think_quality}</div>
  <div class="section prompt"><span class="label">Prompt</span><br>{prompt}</div>
  {think_section}
  {plan_section}
  {code_section}
</div>"""

def export_html(records: list[TraceRecord], output_path: str):
    n_total  = len(records)
    n_passed = sum(1 for r in records if r.passed)
    parts = []

    for r in records:
        think_section = (
            f'<div class="section think"><span class="label">Think</span><br>{_escape(r.think)}</div>'
            if r.think else ""
        )
        plan_section = (
            f'<div class="section plan"><span class="label">Plan</span><br>{_escape(r.plan)}</div>'
            if r.plan else ""
        )
        code_section = (
            f'<div class="section code"><span class="label">Code</span>\n{_escape(r.code)}</div>'
            if r.code else ""
        )
        annotations = " ".join(
            f'<span class="annotation">{a}</span>' for a in r.annotations
        )
        parts.append(TRACE_TEMPLATE.format(
            status_class="pass" if r.passed else "fail",
            status_emoji="✓" if r.passed else "✗",
            record_id=_escape(r.id),
            annotations=annotations,
            test_score=r.test_score,
            type_score=r.type_score,
            think_quality=f"{r.think_quality:.2f}" if r.think_quality is not None else "N/A",
            prompt=_escape(r.prompt),
            think_section=think_section,
            plan_section=plan_section,
            code_section=code_section,
        ))

    html = HTML_TEMPLATE.format(n_total=n_total, n_passed=n_passed, body="\n".join(parts))
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Exported {n_total} traces to {output_path}")

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
```

### Step 6: CLI Entry Point

```python
# cli/trace_view.py
import argparse
from src.trace_viewer.loader import load_traces
from src.trace_viewer.annotator import annotate_all
from src.trace_viewer.browser import browse_traces, filter_traces
from src.trace_viewer.renderer import render_summary
from src.trace_viewer.exporter import export_html

def main():
    parser = argparse.ArgumentParser(description="Browse reasoning traces from evaluation runs.")
    parser.add_argument("path", help="Path to .jsonl trace file or directory of .jsonl files")
    parser.add_argument("--filter", choices=["pass", "fail", "all"], default="all")
    parser.add_argument("--export", metavar="OUTPUT.html", help="Export traces to HTML")
    parser.add_argument("--summary-only", action="store_true", help="Show summary table, no browser")
    parser.add_argument("--no-think", action="store_true", help="Hide think sections")
    parser.add_argument("--no-plan",  action="store_true", help="Hide plan sections")
    parser.add_argument("--no-code",  action="store_true", help="Hide code sections")
    args = parser.parse_args()

    records = load_traces(args.path)
    records = annotate_all(records)

    if args.filter != "all":
        records = filter_traces(records, args.filter)

    if args.export:
        export_html(records, args.export)
        return

    render_summary(records)

    if not args.summary_only:
        browse_traces(records)

if __name__ == "__main__":
    main()
```

---

## Key Files to Modify / Create

| File | Change |
|---|---|
| `src/trace_viewer/models.py` | TraceRecord dataclass |
| `src/trace_viewer/loader.py` | JSONL load/save |
| `src/trace_viewer/annotator.py` | Quality annotation logic |
| `src/trace_viewer/renderer.py` | Rich terminal renderer |
| `src/trace_viewer/browser.py` | Interactive browsing |
| `src/trace_viewer/exporter.py` | HTML export |
| `cli/trace_view.py` | Entry point |
| `src/eval/evaluator.py` | Save traces as JSONL during eval |

---

## Testing Strategy

```python
# tests/test_trace_viewer.py
import json
import tempfile
from pathlib import Path
from src.trace_viewer.loader import load_traces, save_trace
from src.trace_viewer.annotator import annotate

def test_round_trip_save_load():
    record = TraceRecord(
        id="test:0", prompt="Add two numbers",
        generation="<think>simple add</think>\n```ts\nreturn a+b\n```",
        passed=True, test_score=1.0, type_score=0.9,
    )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        save_trace(record, f.name)
        loaded = load_traces(f.name)
    assert len(loaded) == 1
    assert loaded[0].passed is True

def test_copy_paste_annotation():
    record = TraceRecord(
        id="t1", prompt="Write a sort function",
        generation="<think>Write a sort function</think>\n```ts\n[].sort();\n```",
        passed=True, test_score=1.0, type_score=1.0,
        think="Write a sort function",
    )
    annotated = annotate(record)
    assert "copy_paste" in annotated.annotations

def test_filter_pass_only():
    records = [
        TraceRecord(id="a", prompt="", generation="", passed=True, test_score=1.0, type_score=1.0),
        TraceRecord(id="b", prompt="", generation="", passed=False, test_score=0.0, type_score=0.0),
    ]
    from src.trace_viewer.browser import filter_traces
    filtered = filter_traces(records, "pass")
    assert len(filtered) == 1 and filtered[0].id == "a"

def test_html_export_creates_file(tmp_path):
    records = [TraceRecord(id="x", prompt="p", generation="g", passed=True, test_score=1.0, type_score=1.0)]
    out = str(tmp_path / "out.html")
    export_html(records, out)
    content = Path(out).read_text()
    assert "Cola-Coder" in content
    assert "pass" in content
```

---

## Performance Considerations

- Loading 10k JSONL records takes < 500ms; the viewer is not performance-sensitive.
- HTML export for 1000 traces produces ~5MB of HTML — acceptable, but add a `--max-export` limit for very large runs.
- The Rich TUI does a full redraw on each keypress. For > 1000 traces, add pagination at 50 per page.

---

## Dependencies

```
rich>=13.0.0    # terminal rendering (likely already present)
```

No additional dependencies for HTML export (pure Python string formatting).

---

## Estimated Complexity

**Development time:** 2-3 days
**Risk:** Very low — no model changes, purely tooling.
**Lines of new code:** ~400

---

## 2026 Best Practices

- **JSONL as the interchange format:** JSONL is the de facto standard for ML evaluation logs. Using it makes traces compatible with external tools (wandb artifacts, DVC, LangSmith).
- **Offline-first HTML export:** Rather than a web server, generate self-contained HTML. Works without dependencies, can be shared as a file attachment, and renders in any browser.
- **Annotation-first debugging:** Automatic annotations (copy_paste, empty_think) surface the most common failure modes without manual inspection, making debugging O(minutes) instead of O(hours).
- **Separation of evaluation and viewing:** The evaluator writes JSONL; the viewer reads it. This separation means traces can be generated on a remote training server and viewed locally.
