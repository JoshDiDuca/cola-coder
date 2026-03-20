# Feature 45: Hallucination Detection

## Overview

Code models sometimes "hallucinate" APIs that do not exist: invented function names,
non-existent modules, wrong argument counts, or methods that were never part of a
library. Hallucination detection parses the model's generated code, extracts import
statements and API calls, and verifies them against known package APIs.

The result is a per-generation hallucination score: percentage of valid imports,
percentage of valid method calls, and a flagged list of suspected inventions.

Status: OPTIONAL — enable via `--feature hallucination-detection` or CLI menu toggle.

---

## Motivation

- A generated completion that calls `numpy.array_transpose_2d()` (which doesn't exist)
  will fail silently until runtime. Catching this immediately saves debugging time.
- Hallucination rate is a useful quality metric alongside pass@k: a model that writes
  syntactically valid but API-incorrect code scores 0 on tests but may fool human review.
- Building a hallucination detector gives Cola-Coder a feedback signal for RLHF or
  DPO training: penalize completions with invalid API usage.

---

## Architecture / Design

### Pipeline

```
Generated code (string)
       |
       | AST parsing
       v
Extracted APIs:
  - import statements  → ["numpy", "pathlib", "my_fake_lib"]
  - attribute accesses → ["np.array()", "os.path.join()", "np.fakefunc()"]
  - function calls     → ["open()", "len()", "invented_builtin()"]
       |
       | API validation against:
       |   - stdlib module list
       |   - bundled package stubs
       |   - runtime introspection (importlib)
       v
HallucinationReport:
  - valid_imports: ["numpy", "pathlib"]
  - invalid_imports: ["my_fake_lib"]
  - valid_calls: ["np.array()", "os.path.join()"]
  - invalid_calls: ["np.fakefunc()"]
  - hallucination_rate: 0.15
```

### AST Parser for API Extraction

```python
# cola_coder/hallucination/extractor.py

import ast
from dataclasses import dataclass, field


@dataclass
class ExtractedAPIs:
    imports: list[str] = field(default_factory=list)          # module names
    from_imports: list[tuple[str, str]] = field(default_factory=list)  # (module, name)
    attribute_calls: list[str] = field(default_factory=list)  # "obj.method"
    bare_calls: list[str] = field(default_factory=list)       # "function_name"


class APIExtractor(ast.NodeVisitor):
    """Walk AST and collect all import statements and function calls."""

    def __init__(self):
        self.imports: list[str] = []
        self.from_imports: list[tuple[str, str]] = []
        self.attribute_accesses: list[str] = []  # a.b.c chains
        self.bare_calls: list[str] = []
        self._aliases: dict[str, str] = {}  # "np" -> "numpy"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
            if alias.asname:
                self._aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.from_imports.append((module, alias.name))
            if alias.asname:
                self._aliases[alias.asname] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_str = self._get_call_name(node.func)
        if call_str:
            if "." in call_str:
                self.attribute_accesses.append(call_str)
            else:
                self.bare_calls.append(call_str)
        self.generic_visit(node)

    def _get_call_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self._aliases.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            parent = self._get_call_name(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return None

    @classmethod
    def extract(cls, code: str) -> ExtractedAPIs:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ExtractedAPIs()
        extractor = cls()
        extractor.visit(tree)
        return ExtractedAPIs(
            imports=extractor.imports,
            from_imports=extractor.from_imports,
            attribute_calls=extractor.attribute_accesses,
            bare_calls=extractor.bare_calls,
        )
```

### Standard Library Validator

```python
# cola_coder/hallucination/stdlib_validator.py

import sys
import importlib
import pkgutil


# Pre-built list of all Python stdlib module names
_STDLIB_MODULES: frozenset[str] | None = None


def get_stdlib_modules() -> frozenset[str]:
    global _STDLIB_MODULES
    if _STDLIB_MODULES is not None:
        return _STDLIB_MODULES

    if hasattr(sys, "stdlib_module_names"):  # Python 3.10+
        _STDLIB_MODULES = frozenset(sys.stdlib_module_names)
    else:
        # Fallback for older Python
        _STDLIB_MODULES = frozenset(
            m.name for m in pkgutil.iter_modules()
            if m.module_finder.path.startswith(sys.prefix)  # type: ignore
        )
    return _STDLIB_MODULES


def is_stdlib_module(name: str) -> bool:
    top_level = name.split(".")[0]
    return top_level in get_stdlib_modules()


def module_exists(name: str) -> bool:
    """Check if a module can be imported."""
    top_level = name.split(".")[0]
    spec = importlib.util.find_spec(top_level)
    return spec is not None


def attribute_exists_in_module(module_name: str, attr_chain: str) -> bool:
    """
    Check if module.attr.attr... exists.
    Example: module_name="numpy", attr_chain="linalg.inv"
    """
    try:
        mod = importlib.import_module(module_name)
        parts = attr_chain.split(".")
        obj = mod
        for part in parts:
            obj = getattr(obj, part)
        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False
```

### Package API Database

For common packages, maintain a local database of known attributes:

```python
# cola_coder/hallucination/api_database.py

import json
import importlib
from pathlib import Path


KNOWN_PACKAGES_TO_CHECK = [
    "numpy", "pandas", "torch", "sklearn", "fastapi",
    "pathlib", "os", "sys", "json", "re", "datetime",
    "collections", "itertools", "functools", "typing",
]


def build_api_database(packages: list[str] | None = None) -> dict[str, set[str]]:
    """
    Build a database of {package: set of valid attribute names}.
    Only checks top-level attributes (not nested).
    """
    packages = packages or KNOWN_PACKAGES_TO_CHECK
    db: dict[str, set[str]] = {}

    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            attrs = set(dir(mod))
            db[pkg] = attrs
            print(f"Indexed {pkg}: {len(attrs)} attributes")
        except ImportError:
            db[pkg] = set()

    return db


def save_api_database(db: dict[str, set[str]], path: Path) -> None:
    serializable = {k: sorted(v) for k, v in db.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_api_database(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: set(v) for k, v in data.items()}
```

### Hallucination Validator

```python
# cola_coder/hallucination/validator.py

from dataclasses import dataclass, field
from .extractor import APIExtractor, ExtractedAPIs
from .stdlib_validator import module_exists, attribute_exists_in_module
from .api_database import load_api_database
from pathlib import Path
import importlib


@dataclass
class HallucinationReport:
    code: str
    valid_imports: list[str] = field(default_factory=list)
    invalid_imports: list[str] = field(default_factory=list)
    valid_calls: list[str] = field(default_factory=list)
    invalid_calls: list[str] = field(default_factory=list)
    unknown_calls: list[str] = field(default_factory=list)  # can't verify

    @property
    def import_validity_rate(self) -> float:
        total = len(self.valid_imports) + len(self.invalid_imports)
        return len(self.valid_imports) / max(total, 1)

    @property
    def call_validity_rate(self) -> float:
        total = len(self.valid_calls) + len(self.invalid_calls)
        return len(self.valid_calls) / max(total, 1)

    @property
    def hallucination_rate(self) -> float:
        total_checked = (len(self.valid_imports) + len(self.invalid_imports) +
                         len(self.valid_calls) + len(self.invalid_calls))
        total_invalid = len(self.invalid_imports) + len(self.invalid_calls)
        return total_invalid / max(total_checked, 1)

    @property
    def is_clean(self) -> bool:
        return len(self.invalid_imports) == 0 and len(self.invalid_calls) == 0


class HallucinationDetector:
    def __init__(
        self,
        api_db_path: Path | None = None,
        check_attribute_exists: bool = True,
    ):
        self.db: dict[str, set[str]] = {}
        self.check_attrs = check_attribute_exists

        if api_db_path and api_db_path.exists():
            self.db = load_api_database(api_db_path)

    def detect(self, code: str) -> HallucinationReport:
        apis = APIExtractor.extract(code)
        report = HallucinationReport(code=code)

        # Check imports
        for module_name in apis.imports:
            if module_exists(module_name):
                report.valid_imports.append(module_name)
            else:
                report.invalid_imports.append(module_name)

        for module_name, attr_name in apis.from_imports:
            full = f"{module_name}.{attr_name}"
            if module_exists(module_name):
                if self.check_attrs and module_name in self.db:
                    if attr_name in self.db[module_name]:
                        report.valid_imports.append(full)
                    else:
                        report.invalid_imports.append(full)
                else:
                    report.valid_imports.append(full)  # module exists, can't verify attr
            else:
                report.invalid_imports.append(full)

        # Check attribute calls
        for call in apis.attribute_calls:
            parts = call.split(".")
            root = parts[0]
            attr_chain = ".".join(parts[1:])

            if root in self.db:
                # Verify first attribute level
                first_attr = parts[1] if len(parts) > 1 else ""
                if first_attr in self.db[root]:
                    report.valid_calls.append(call)
                else:
                    report.invalid_calls.append(call)
            elif module_exists(root):
                if self.check_attrs:
                    exists = attribute_exists_in_module(root, attr_chain)
                    if exists:
                        report.valid_calls.append(call)
                    else:
                        report.invalid_calls.append(call)
                else:
                    report.unknown_calls.append(call)
            else:
                # Unknown root — could be a local variable, not an imported module
                report.unknown_calls.append(call)

        return report
```

### CLI Display

```python
# cola_coder/hallucination/display.py

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def display_hallucination_report(report: "HallucinationReport") -> None:
    if report.is_clean:
        console.print("[bold green]No hallucinations detected.[/bold green]")
        return

    console.print(Panel(
        f"Hallucination rate: [bold red]{report.hallucination_rate*100:.1f}%[/bold red]\n"
        f"Import validity: {report.import_validity_rate*100:.0f}% | "
        f"Call validity: {report.call_validity_rate*100:.0f}%",
        title="Hallucination Report",
        border_style="red" if report.hallucination_rate > 0.1 else "yellow",
    ))

    if report.invalid_imports:
        console.print("[red]Invalid imports:[/red]")
        for imp in report.invalid_imports:
            console.print(f"  [red]• {imp}[/red] — module/symbol not found")

    if report.invalid_calls:
        console.print("[red]Invalid API calls:[/red]")
        for call in report.invalid_calls:
            console.print(f"  [red]• {call}()[/red] — attribute not found")
```

### Integration into Generation Pipeline

```python
# cola_coder/generator.py  (hallucination check integration)

class CodeGenerator:
    def generate_with_hallucination_check(
        self,
        prompt: str,
        detector: "HallucinationDetector",
        **gen_kwargs,
    ) -> dict:
        completion = self.generate(prompt, **gen_kwargs)
        report = detector.detect(completion)
        return {
            "completion": completion,
            "hallucination_report": {
                "rate": report.hallucination_rate,
                "invalid_imports": report.invalid_imports,
                "invalid_calls": report.invalid_calls,
                "is_clean": report.is_clean,
            }
        }
```

---

## Implementation Steps

1. **Create `cola_coder/hallucination/` package**: `__init__.py`, `extractor.py`,
   `stdlib_validator.py`, `api_database.py`, `validator.py`, `display.py`.

2. **Build and save API database** as a one-time setup step:
   ```bash
   python -m cola_coder.hallucination.api_database --output data/api_db.json
   ```
   This runs once, producing a ~1 MB JSON file.

3. **Integrate into CLI**: add "Check for hallucinations" toggle in the generation
   options. When enabled, display report after each generation.

4. **Integrate into HTTP API**: add optional `check_hallucinations: bool` to generation
   request. Return `hallucination_report` in response when enabled.

5. **Benchmark mode**: run hallucination detection on all HumanEval completions and
   compute aggregate hallucination rate as a model quality metric.

6. **TypeScript support**: add a TypeScript import extractor (use `@typescript-eslint/
   typescript-estree` via subprocess, or a regex-based fallback):
   ```python
   # Simple regex fallback for TS imports:
   import re
   TS_IMPORT_PATTERN = re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
   ```

7. **False positive handling**: attribute access on local variables is flagged as
   unknown (not invalid). Log but do not penalize unknown calls.

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Add `generate_with_hallucination_check()` |
| `server.py` | Add `check_hallucinations` to generation request |
| `cli/menu.py` | Add "Enable hallucination check" toggle |
| `benchmarks/humaneval.py` | Add hallucination rate to benchmark output |
| `config.py` | Add `HallucinationConfig` with db path |
| `cola_coder/hallucination/` | New package |

---

## Testing Strategy

```python
# tests/test_hallucination.py

def test_extract_imports():
    code = "import numpy as np\nfrom pathlib import Path"
    apis = APIExtractor.extract(code)
    assert "numpy" in apis.imports
    assert ("pathlib", "Path") in apis.from_imports

def test_extract_attribute_calls():
    code = "import numpy as np\nx = np.array([1,2,3])\ny = np.fake_func()"
    apis = APIExtractor.extract(code)
    call_roots = [c.split(".")[0] for c in apis.attribute_calls]
    assert "numpy" in call_roots  # alias resolved

def test_valid_stdlib_module():
    assert module_exists("pathlib")
    assert module_exists("os")
    assert module_exists("json")

def test_invalid_module():
    assert not module_exists("completely_fake_package_xyz123")

def test_hallucination_detector_clean_code():
    detector = HallucinationDetector()
    code = "import os\nresult = os.path.join('a', 'b')"
    report = detector.detect(code)
    assert "os" in report.valid_imports
    assert report.hallucination_rate == 0.0

def test_hallucination_detector_flags_fake_module():
    detector = HallucinationDetector()
    code = "import completely_fake_lib\nx = completely_fake_lib.do_thing()"
    report = detector.detect(code)
    assert "completely_fake_lib" in report.invalid_imports
    assert report.hallucination_rate > 0.0

def test_hallucination_detector_flags_fake_attribute():
    detector = HallucinationDetector(check_attribute_exists=True)
    code = "import numpy as np\nx = np.this_function_does_not_exist([1,2,3])"
    report = detector.detect(code)
    # Should flag the fake attribute
    assert len(report.invalid_calls) > 0 or len(report.unknown_calls) > 0

def test_hallucination_rate_property():
    report = HallucinationReport(code="")
    report.valid_imports = ["os", "sys"]
    report.invalid_imports = ["fake_mod"]
    # rate = 1 invalid / (2 valid + 1 invalid) = 0.333...
    assert abs(report.hallucination_rate - 1/3) < 0.01
```

---

## Performance Considerations

- **AST parsing**: fast (< 1 ms for typical completions). Not a bottleneck.
- **Module import checking**: `importlib.util.find_spec()` is fast for installed
  packages. Avoid actually importing packages (that would be slow and have side effects).
- **API database lookup**: set membership check is O(1). No overhead.
- **Attribute existence check via `importlib.import_module`**: slow if the package is
  large and not yet imported. Cache results in `functools.lru_cache`.
- **TypeScript**: requires subprocess call to `tsc` or `ts-node`. Much slower than
  Python AST. Only enable for TypeScript on explicit request.
- **False positive rate**: local variables that shadow module names (e.g., `np = make_array()`)
  will cause false positives. Restrict validation to top-level assignments of
  known import aliases.

---

## Dependencies

```
ast         # stdlib — Python AST parsing (already available)
importlib   # stdlib — module existence checking
pathlib     # stdlib — file paths
```

No new pip dependencies for Python hallucination detection.
Optional: `@typescript-eslint/typescript-estree` (npm) for TypeScript support.

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| APIExtractor (AST walker) | 3 hours |
| stdlib + module validator | 2 hours |
| API database builder | 2 hours |
| HallucinationDetector | 3 hours |
| CLI display + integration | 2 hours |
| HTTP API integration | 1 hour |
| TypeScript extractor (regex) | 2 hours |
| Tests | 3 hours |
| **Total** | **~18 hours** |

Complexity rating: **Medium** — AST walking and module checking are well-understood;
the tricky parts are alias resolution and avoiding false positives on local variables.

---

## 2026 Best Practices

- **Type stub databases**: instead of runtime introspection, check against `.pyi` stub
  files from `typeshed` or individual package type stubs. More complete and faster than
  `import_module + dir()`. The `typeshed` repository covers all stdlib + common packages.
- **Semantic analysis**: tools like `jedi` or `rope` do full semantic analysis including
  local variable types. Much more accurate than AST-only analysis but significantly slower.
- **LLM-as-judge hallucination scoring**: a secondary LLM (small, fast) scores each
  completion for hallucination likelihood without code execution. Shown to correlate
  well with runtime errors.
- **Verified execution**: the gold standard for hallucination detection is execution.
  Run generated code in a sandbox (Docker, subprocess with timeout) and report whether
  imports succeed and basic calls work. More expensive but definitive.
- **Hallucination taxonomy**: distinguish between (a) non-existent modules, (b) existent
  modules but wrong attribute names, (c) correct attribute but wrong argument types,
  (d) correct call but semantically wrong usage. Each requires a different detection strategy.
