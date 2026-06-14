# Strict Typing & Schema-First Rules (TypeScript ⇄ Python)

Weak typing is a **bug**, not a convenience. Code is schema-first, strongly typed, and
fails at **compile time**, not runtime. These rules are mandatory for all `webui/` (TS)
and `src/cola_coder/ui/` (Python) code, and preferred everywhere else.

## The single source of truth (1:1 TS ⇄ Python)

- Every HTTP response/request shape is a **Pydantic `BaseModel`** in
  `src/cola_coder/ui/schemas.py`. This is the ONE source of truth.
- FastAPI endpoints declare `response_model=<Model>` (or annotate the return type) so
  responses are validated and appear in OpenAPI. Endpoints return the model, not a bare dict.
- `scripts/gen_ts_types.py` generates `webui/src/types.gen.ts` from those models —
  **never hand-edit `types.gen.ts`**. The frontend imports types from it.
- `tests/test_ui_types_generated.py` regenerates and asserts **no diff** (drift guard):
  if a Python model changes, the TS must be regenerated or CI fails.
- Field names, nullability, and optionality must match **exactly**:
  `Optional[str]` (Python) ↔ `string | null` (TS); a field with a default that may be
  omitted ↔ `field?: T`. Run `gen_ts_types.py` after any schema edit.

## Banned (treat as a compile error to be removed)

- `any`, `unknown` — including `Record<string, unknown>`, `unknown[]`, `as unknown as`.
- Generic "type-probing" / catch-all formatters: `function fmt(value: unknown)`,
  `typeof x === ...` dispatch, fallback `String(value)` / `JSON.stringify(value)` for
  display. Formatting is **type-specific** — a formatter takes a concrete type.
- Inline anonymous object types for non-trivial structures (`{ a: string; b: ... }` in a
  signature). Name the interface.
- Implicit/mixed return types. Every function has explicit parameter and return types.
- Runtime coercion (`String(value)`, `JSON.stringify(value)`, `Number(x)`) **unless the
  spec genuinely requires serialization** — and then in ONE typed boundary function.

## Required

- If a value can be multiple types, define a **union** or **discriminated union**, and
  handle it **exhaustively** with a `never` check in the `default`/else branch.
- If a structure exists, define a named `interface`/`type` (TS) or `BaseModel` (Python).
- Genuinely open JSON (e.g. a parsed arbitrary YAML config, a previewed JSONL row) is
  **not** an excuse for `unknown`. Either (a) model the known shape, or (b) have the
  **backend coerce values to `str`** at the boundary so the TS type is a concrete
  `Record<string, string>`, or (c) use the single shared principled JSON type
  `JsonValue` (`string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue }`)
  defined once in `types.gen.ts` — never re-invented, never `any`/`unknown`.
- **Shared formatters live in `webui/src/format.ts`**, are named for intent, and are
  type-specific. NEVER copy a `humanBytes`/`fmtInt`/`fmt` helper into a component file.
  - `formatBytes(bytes: number | null): string`
  - `formatInteger(n: number | null): string`
  - `formatPercent(fraction: number | null): string`, etc.
  Components import these; they do not define their own.
- Destructure with intent — don't explode an object into a wall of `const x = obj?.a;`
  lines. Pass the typed object, or destructure the few fields actually used inline.

## Examples

```ts
// GOOD — concrete type in, type-specific formatting, exhaustive union handling
type JobStatus = 'running' | 'done' | 'failed';

function jobBadgeClass(status: JobStatus): string {
  switch (status) {
    case 'running': return 'tag running';
    case 'done':    return 'tag done';
    case 'failed':  return 'tag failed';
    default: { const _exhaustive: never = status; return _exhaustive; }
  }
}
```

```python
# GOOD — Pydantic source of truth (generates the TS interface 1:1)
from pydantic import BaseModel

class TrainingStatus(BaseModel):
    alive: bool
    step: int | None
    loss: float | None
    tok_per_s: float | None
```

## Workflow when types are unclear

STOP and define the schema first. Do not "guess and coerce." Add the Pydantic model,
regenerate `types.gen.ts`, then write code against the concrete type.
