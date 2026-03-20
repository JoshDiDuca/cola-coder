# Feature 57: Docstring Extraction

**Status:** Proposed
**CLI Flag:** `--extract-docstrings`
**Complexity:** Low-Medium

---

## Overview

Parses TypeScript/JavaScript source files to extract JSDoc-annotated functions, creating (docstring → function body) instruction-tuning pairs. The output is a JSONL dataset used for supervised fine-tuning (SFT) after pretraining. Quality filters ensure only well-documented functions are included.

---

## Motivation

After pretraining on raw code, SFT on instruction-following data dramatically improves the model's ability to generate code from natural language descriptions. JSDoc comments are a high-quality, abundant source of such pairs:

- They describe what a function does in natural language
- They are written by the same developers who wrote the code
- They exist in virtually every high-quality TypeScript codebase
- No human annotation is required

Estimated yield from The Stack v2 (TypeScript): ~5-10M JSDoc-annotated functions, providing a rich SFT dataset at zero annotation cost.

---

## Architecture / Design

```
Source file (.ts)
  │
  ▼
JSDocExtractor
  ├── Find all JSDoc block comments (/** ... */)
  ├── Match each to the immediately following function/method
  ├── Extract: function name, params, return type, JSDoc text
  └── Quality filter
  │
  ▼
InstructionFormatter
  ├── instruction = cleaned docstring
  ├── input = function signature (optional)
  └── output = function body
  │
  ▼
JSONL output: { instruction, input, output }
```

---

## Implementation Steps

### Step 1: JSDoc Extractor

```python
# src/data/docstring_extractor.py
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocstringPair:
    function_name: str
    jsdoc: str
    signature: str
    body: str
    file_path: str
    line_number: int
    quality_score: float = 0.0

JSDOC_PATTERN = re.compile(
    r"/\*\*(.*?)\*/\s*"           # JSDoc block
    r"(?:export\s+)?(?:async\s+)?"
    r"(?:function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{]+))?\s*\{(.*?)(?=\n\}|\n  \}))",
    re.DOTALL,
)

ARROW_JSDOC_PATTERN = re.compile(
    r"/\*\*(.*?)\*/\s*"
    r"(?:export\s+)?const\s+(\w+)\s*(?::\s*[^=]+)?=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*([^=>{]+))?\s*=>\s*\{(.*?)(?=\n\}|\n  \})",
    re.DOTALL,
)

def _clean_jsdoc(raw: str) -> str:
    """Remove * prefixes and @param/@returns tags; keep description."""
    lines = []
    for line in raw.strip().splitlines():
        line = line.strip().lstrip("*").strip()
        if line.startswith("@param") or line.startswith("@returns") or line.startswith("@example"):
            continue
        if line:
            lines.append(line)
    return " ".join(lines).strip()

def _extract_param_descriptions(raw_jsdoc: str) -> dict[str, str]:
    """Extract @param descriptions: {name: description}"""
    result = {}
    for match in re.finditer(r"@param\s+\{[^}]+\}\s+(\w+)\s+-?\s*(.+?)(?=@|\*/|$)", raw_jsdoc, re.DOTALL):
        result[match.group(1)] = match.group(2).strip()
    return result

def extract_docstring_pairs(source: str, file_path: str = "") -> list[DocstringPair]:
    pairs = []

    for pattern in [JSDOC_PATTERN, ARROW_JSDOC_PATTERN]:
        for match in pattern.finditer(source):
            raw_jsdoc = match.group(1)
            fn_name   = match.group(2)
            params    = match.group(3) or ""
            ret_type  = (match.group(4) or "").strip()
            body      = match.group(5)

            cleaned_doc = _clean_jsdoc(raw_jsdoc)
            if not cleaned_doc or not fn_name:
                continue

            signature = f"function {fn_name}({params})"
            if ret_type:
                signature += f": {ret_type}"

            line_num = source[:match.start()].count("\n") + 1

            pairs.append(DocstringPair(
                function_name=fn_name,
                jsdoc=cleaned_doc,
                signature=signature,
                body=body.strip(),
                file_path=file_path,
                line_number=line_num,
            ))

    return pairs
```

### Step 2: Quality Scorer

```python
# src/data/docstring_quality.py
from src.data.docstring_extractor import DocstringPair
import re

MIN_DOC_WORDS    = 5
MIN_BODY_CHARS   = 20
MAX_BODY_CHARS   = 5000

BOILERPLATE_PHRASES = [
    "todo", "fixme", "placeholder", "not implemented",
    "remove this", "stub", "lorem ipsum",
]

def score_docstring_pair(pair: DocstringPair) -> float:
    score = 0.0

    # 1. Doc length
    doc_words = len(pair.jsdoc.split())
    if doc_words < MIN_DOC_WORDS:
        return 0.0
    score += min(doc_words / 20, 1.0) * 0.3

    # 2. Body length
    body_len = len(pair.body)
    if body_len < MIN_BODY_CHARS:
        return 0.0
    if body_len > MAX_BODY_CHARS:
        return 0.0
    score += min(body_len / 200, 1.0) * 0.2

    # 3. Doc describes behavior (not just restating function name)
    fn_name_words = re.sub(r'([A-Z])', r' \1', pair.function_name).lower().split()
    doc_lower = pair.jsdoc.lower()
    doc_minus_name = doc_lower
    for w in fn_name_words:
        doc_minus_name = doc_minus_name.replace(w, "")
    remaining_words = len(doc_minus_name.split())
    if remaining_words >= 3:
        score += 0.3

    # 4. No boilerplate
    if any(bp in pair.jsdoc.lower() for bp in BOILERPLATE_PHRASES):
        return 0.0
    score += 0.1

    # 5. Body has a return statement
    if "return" in pair.body:
        score += 0.1

    return min(score, 1.0)

def filter_pairs(pairs: list[DocstringPair], min_score: float = 0.5) -> list[DocstringPair]:
    scored = []
    for p in pairs:
        p.quality_score = score_docstring_pair(p)
        if p.quality_score >= min_score:
            scored.append(p)
    return scored
```

### Step 3: Instruction Formatter

```python
# src/data/docstring_formatter.py
import json
from src.data.docstring_extractor import DocstringPair

FORMAT_STYLES = ["alpaca", "sharegpt", "raw"]

def format_alpaca(pair: DocstringPair) -> dict:
    return {
        "instruction": pair.jsdoc,
        "input": f"Function signature: {pair.signature}",
        "output": f"```typescript\n{pair.signature} {{\n  {pair.body}\n}}\n```",
    }

def format_sharegpt(pair: DocstringPair) -> dict:
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"{pair.jsdoc}\n\nSignature: `{pair.signature}`",
            },
            {
                "from": "gpt",
                "value": f"```typescript\n{pair.signature} {{\n  {pair.body}\n}}\n```",
            },
        ]
    }

def format_raw(pair: DocstringPair) -> dict:
    return {
        "function_name": pair.function_name,
        "docstring": pair.jsdoc,
        "signature": pair.signature,
        "body": pair.body,
        "quality_score": pair.quality_score,
        "source_file": pair.file_path,
    }

FORMATTERS = {
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
    "raw": format_raw,
}

def write_jsonl(pairs: list[DocstringPair], output_path: str, format_style: str = "alpaca"):
    fmt = FORMATTERS[format_style]
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(fmt(pair), ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} instruction pairs to {output_path}")
```

### Step 4: Batch Processor

```python
# src/data/docstring_pipeline.py
from pathlib import Path
from src.data.docstring_extractor import extract_docstring_pairs
from src.data.docstring_quality import filter_pairs, score_docstring_pair
from src.data.docstring_formatter import write_jsonl

def process_directory(
    input_dir: str,
    output_path: str,
    min_quality: float = 0.5,
    format_style: str = "alpaca",
    extensions: list[str] = None,
    max_files: int = None,
) -> dict:
    if extensions is None:
        extensions = [".ts", ".tsx", ".js", ".jsx"]

    input_path = Path(input_dir)
    all_files  = [f for f in input_path.rglob("*") if f.suffix in extensions]

    if max_files:
        all_files = all_files[:max_files]

    all_pairs = []
    processed  = 0
    errors     = 0

    for file_path in all_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            pairs  = extract_docstring_pairs(source, str(file_path))
            pairs  = filter_pairs(pairs, min_quality)
            all_pairs.extend(pairs)
            processed += 1
        except Exception as e:
            errors += 1
            continue

        if processed % 1000 == 0:
            print(f"Processed {processed}/{len(all_files)} files, {len(all_pairs)} pairs found")

    write_jsonl(all_pairs, output_path, format_style)

    return {
        "files_processed": processed,
        "files_errored": errors,
        "pairs_extracted": len(all_pairs),
        "output_path": output_path,
    }
```

### Step 5: CLI Tool

```python
# cli/extract_docstrings.py
import argparse
from src.data.docstring_pipeline import process_directory

def main():
    parser = argparse.ArgumentParser(description="Extract JSDoc pairs for instruction tuning.")
    parser.add_argument("input_dir", help="Directory of TypeScript/JavaScript files.")
    parser.add_argument("output", help="Output JSONL path.")
    parser.add_argument("--min-quality", type=float, default=0.5,
        help="Minimum quality score for inclusion (default: 0.5).")
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "raw"], default="alpaca",
        help="Output format style (default: alpaca).")
    parser.add_argument("--max-files", type=int, default=None,
        help="Limit number of files processed (useful for testing).")
    parser.add_argument("--extensions", nargs="+", default=[".ts", ".tsx"],
        help="File extensions to process (default: .ts .tsx).")
    args = parser.parse_args()

    stats = process_directory(
        args.input_dir,
        args.output,
        min_quality=args.min_quality,
        format_style=args.format,
        max_files=args.max_files,
        extensions=args.extensions,
    )
    print(f"\nDone: {stats}")

if __name__ == "__main__":
    main()
```

---

## Key Files to Modify

| File | Change |
|---|---|
| `src/data/docstring_extractor.py` | New |
| `src/data/docstring_quality.py` | New |
| `src/data/docstring_formatter.py` | New |
| `src/data/docstring_pipeline.py` | New |
| `cli/extract_docstrings.py` | New CLI entry point |

---

## Testing Strategy

```python
# tests/test_docstring_extractor.py

SAMPLE_TS = '''
/**
 * Calculates the factorial of a non-negative integer.
 * Returns 1 for 0 or 1.
 */
export function factorial(n: number): number {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

/**
 * @param x the value
 */
function noDescription(x: number): number {
  return x;
}

function noDoc(): void {
  console.log("hello");
}
'''

def test_extracts_documented_function():
    pairs = extract_docstring_pairs(SAMPLE_TS)
    names = [p.function_name for p in pairs]
    assert "factorial" in names

def test_skips_undocumented():
    pairs = extract_docstring_pairs(SAMPLE_TS)
    names = [p.function_name for p in pairs]
    assert "noDoc" not in names

def test_quality_filter_rejects_param_only_doc():
    pairs = extract_docstring_pairs(SAMPLE_TS)
    filtered = filter_pairs(pairs, min_score=0.5)
    names = [p.function_name for p in filtered]
    # "@param x the value" should have low score
    assert "noDescription" not in names

def test_alpaca_format():
    pair = DocstringPair(
        function_name="add",
        jsdoc="Adds two numbers and returns the result.",
        signature="function add(a: number, b: number): number",
        body="return a + b;",
        file_path="test.ts",
        line_number=1,
        quality_score=0.8,
    )
    formatted = format_alpaca(pair)
    assert "instruction" in formatted
    assert "typescript" in formatted["output"]

def test_boilerplate_rejected():
    pair = DocstringPair(
        function_name="todo",
        jsdoc="TODO: implement this later",
        signature="function todo(): void",
        body="// not implemented",
        file_path="test.ts",
        line_number=1,
    )
    score = score_docstring_pair(pair)
    assert score == 0.0
```

---

## Performance Considerations

- Regex parsing of 1M TypeScript files takes ~2-4 hours on a single CPU. Parallelize with `multiprocessing.Pool`:

```python
from multiprocessing import Pool
import functools

def _process_one(file_path, min_quality):
    source = Path(file_path).read_text(errors="ignore")
    pairs = extract_docstring_pairs(source, file_path)
    return filter_pairs(pairs, min_quality)

with Pool(processes=16) as pool:
    fn = functools.partial(_process_one, min_quality=0.5)
    results = pool.map(fn, all_files)
```

- Memory: accumulating 5M pairs in memory before writing requires ~2GB RAM. Use streaming writes instead.

---

## Dependencies

No new pip dependencies — uses standard library (`re`, `json`, `pathlib`).

---

## Estimated Complexity

**Development time:** 2-3 days
**Risk:** Low. Pure data processing with no model changes.
**Lines of new code:** ~350

---

## 2026 Best Practices

- **SFT data quality > quantity:** Filter aggressively (min_quality=0.5 keeps ~20-30% of all JSDoc). 500k high-quality pairs outperform 5M low-quality ones for SFT.
- **Alpaca format for compatibility:** The Alpaca instruction format is supported by virtually all SFT training frameworks (Axolotl, LLaMA-Factory, Unsloth). Using it ensures the dataset is reusable.
- **Deduplicate docstrings:** Many open-source projects copy functions from each other. Run MinHash dedup on the (jsdoc, body) pairs before SFT to prevent memorization.
- **Evaluate on HumanEval-Doc:** After SFT, evaluate by using function docstrings as prompts on HumanEval. This directly tests whether docstring extraction improved instruction following for code generation.
