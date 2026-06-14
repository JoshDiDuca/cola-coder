# Python Standards — Strict Typing, Clean Code, Schema-First

You write Python as a senior engineer under strict type safety. Weak typing is a **bug**,
not a convenience. Output must be strongly typed, PEP 8 compliant, schema-first, and
production-ready. This is the Python half of the TS⇄Python bar in `typing.md` — read both.

## 1. Type Safety First

- Every public function has **explicit type hints** on all params and the return (PEP 484/526).
- Never use `Any` unless genuinely unavoidable — and then a `# why:` comment justifies it.
- Prefer precise types: `Literal[...]`, unions, discriminated unions, `TypedDict` only when a
  `BaseModel` truly doesn't fit.
- Use `pydantic.BaseModel` (preferred) or `@dataclass` for ALL structured data. No loosely
  shaped dicts / "JSON blobs" crossing a function or API boundary.
- Nullable is explicit: `Optional[str]` / `str | None` ↔ `string | null` in TS.

## 2. Cross-Language Alignment (Python ↔ TypeScript, 1:1)

- API/UI models are **Pydantic models in `src/cola_coder/ui/schemas.py`** — the ONE source of
  truth. `scripts/gen_ts_types.py` generates `webui/src/types.gen.ts`; drift is CI-guarded by
  `tests/test_ui_types_generated.py`. Never hand-edit the generated TS.
- Field names, order, and nullability must match exactly:
  `Optional[str]` ↔ `string | null`; `Literal['a','b']` ↔ `'a' | 'b'`.
- No runtime coercion at boundaries (`str(value)`, `json.dumps(value)` for display) unless the
  spec genuinely requires serialization — then in ONE typed boundary function.
- Genuinely-open JSON uses the single `JsonValue` recursive union, never bare `dict`/`Any`.

## 3. Code Style (PEP 8 + modern Python)

- 4-space indent, no tabs. Line length 100 (project `pyproject.toml` — note: stricter than the
  88 some tools default to; ruff config wins).
- 2 blank lines before top-level defs/classes, 1 between methods.
- Naming: `lower_with_underscores` (vars/functions), `CamelCase` (classes),
  `ALL_CAPS` (constants), `_leading_underscore` (private).
- f-strings over `.format()`/`%`. `with` for all context managers. `enumerate`/`zip`/`any`/`all`
  over manual index loops. No wildcard imports.

## 4. Errors & Logging

- `try/except` with **specific** exception types; never bare `except:` and never swallow silently.
- Use the `logging` module, not `print()`, for diagnostics (CLI user-facing output goes through
  `cola_coder.cli`, never raw `print`).
- Fail fast with explicit errors on invalid input.

## 5. Function & Class Design

- One function = one clear purpose; prefer pure functions; document side effects.
- `@staticmethod`/`@classmethod` where appropriate; implement `__repr__`/`__eq__` on value types
  (dataclasses/pydantic give these).
- No top-level executable code except an `if __name__ == "__main__":` entry point.

## 6. Data Structures & Performance

- `Enum` instead of magic strings; `dataclass` for value objects; `collections`
  (`Counter`, `defaultdict`, `namedtuple`) where they read cleaner.
- Generators (`yield`) for large/streamed datasets; avoid needless list copies.
- `is` for identity, `==` for equality.

## 7. Files & OS

- `pathlib.Path` over `os.path`. Always `with open(...)`. Secrets via env vars (`HF_TOKEN`),
  never in code.

## 8. Structure & Imports

- Import order: stdlib → third-party → local, each group blank-line separated (ruff/isort).
- Module/class/public-function docstrings everywhere.

## 9. Testing

- pytest. Unit tests for all new functionality; mock external deps. If it's not testable,
  refactor until it is. (Project: ~3350 tests; new endpoints need targeted tests.)

## 10. Linting & Type Checking

- Ruff for lint (`.venv/Scripts/ruff check`). Type-check with mypy/pyright where configured.
- No `# noqa` / `# type: ignore` without an inline reason.

## Bad → Good

```python
# BAD — no hints, runtime coercion, dict probing
def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)
```

```python
# GOOD — schema-first, typed, no coercion guesswork
from typing import Literal
from pydantic import BaseModel

class UserStatus(BaseModel):
    status: Literal["active", "inactive"]
    metadata: str | None = None

def format_status(status: UserStatus) -> str:
    """Return a human-readable status line."""
    return f"{status.status} ({status.metadata or 'no metadata'})"
```

If types are unclear: **STOP and define the schema first.** Do not "guess and coerce."
Prefer compile-time guarantees over runtime flexibility. Code must be IDE-friendly
(autocomplete, hints, docstrings).
