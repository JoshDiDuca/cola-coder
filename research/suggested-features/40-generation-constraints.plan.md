# Feature 40: Generation Constraints (Syntax-Guided Decoding)

## Overview

Generation constraints force the model to produce syntactically valid output by masking
tokens at each step that would create syntax errors. At minimum, this involves tracking
bracket/brace/paren depth to prevent illegal closing tokens. The full version uses
incremental tree-sitter parsing to constrain generation to grammatically valid token
sequences.

Status: OPTIONAL — enable via `--feature constraints` or CLI menu toggle.

---

## Motivation

- Code models occasionally generate syntactically broken completions (mismatched braces,
  premature EOF, illegal identifiers after keywords).
- Constrained decoding guarantees syntactic validity, reducing post-processing filtering.
- Even a simple bracket-matching constraint eliminates a common class of errors in
  generated code.
- Integration with tree-sitter enables language-aware constraints: after `def `, only
  a valid Python identifier is allowed; after `:`, a newline is required.

---

## Architecture / Design

### Constraint Architecture

At each generation step, a `ConstraintEngine` provides a binary mask over the vocabulary:
- `mask[i] = 1` — token `i` is allowed
- `mask[i] = 0` — token `i` is forbidden (set logit to -inf)

```
current state (bracket depth, parser state)
         |
         v
ConstraintEngine.get_mask(state) -> mask (V,)
         |
         v
logits = logits + (-inf * (1 - mask))
         |
         v
sample next token
         |
         v
ConstraintEngine.update(state, token) -> new state
```

### Level 1: Bracket/Brace/Paren Matching

The simplest and cheapest constraint: track open/close counts and forbid closing tokens
when count is already 0.

```python
# cola_coder/constraints/bracket_constraint.py

from dataclasses import dataclass, field
import torch


@dataclass
class BracketState:
    paren_depth: int = 0    # ()
    bracket_depth: int = 0  # []
    brace_depth: int = 0    # {}
    string_mode: str = ""   # "" | "'" | '"' | '"""' | "'''"


class BracketConstraint:
    """
    Tracks bracket/brace/paren depth.
    Forbids closing tokens when depth is already 0.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Build token classification tables at init (not per-step)
        self._build_token_tables()

    def _build_token_tables(self):
        """Pre-classify all tokens into bracket categories."""
        self.token_opens_paren: set[int] = set()
        self.token_closes_paren: set[int] = set()
        self.token_opens_bracket: set[int] = set()
        self.token_closes_bracket: set[int] = set()
        self.token_opens_brace: set[int] = set()
        self.token_closes_brace: set[int] = set()

        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            # Decode to get actual string representation
            decoded = self.tokenizer.convert_tokens_to_string([token_str])
            if "(" in decoded:
                self.token_opens_paren.add(token_id)
            if ")" in decoded:
                self.token_closes_paren.add(token_id)
            if "[" in decoded:
                self.token_opens_bracket.add(token_id)
            if "]" in decoded:
                self.token_closes_bracket.add(token_id)
            if "{" in decoded:
                self.token_opens_brace.add(token_id)
            if "}" in decoded:
                self.token_closes_brace.add(token_id)

    def get_mask(self, state: BracketState, vocab_size: int) -> torch.Tensor:
        """Return allowed token mask (1 = allowed, 0 = forbidden)."""
        mask = torch.ones(vocab_size, dtype=torch.bool)

        if state.paren_depth == 0:
            for tid in self.token_closes_paren:
                if tid < vocab_size:
                    mask[tid] = False

        if state.bracket_depth == 0:
            for tid in self.token_closes_bracket:
                if tid < vocab_size:
                    mask[tid] = False

        if state.brace_depth == 0:
            for tid in self.token_closes_brace:
                if tid < vocab_size:
                    mask[tid] = False

        return mask

    def update(self, state: BracketState, token_id: int) -> BracketState:
        """Return new state after consuming token_id."""
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
        new = BracketState(
            paren_depth=state.paren_depth,
            bracket_depth=state.bracket_depth,
            brace_depth=state.brace_depth,
            string_mode=state.string_mode,
        )
        # Skip constraint updates when inside a string
        if state.string_mode:
            # Simple string mode exit detection
            if state.string_mode in decoded:
                new.string_mode = ""
            return new

        # Check for string openers
        if '"""' in decoded:
            new.string_mode = '"""'
            return new
        if "'''" in decoded:
            new.string_mode = "'''"
            return new
        if '"' in decoded:
            new.string_mode = '"'
            return new
        if "'" in decoded:
            new.string_mode = "'"
            return new

        # Count brackets (net change)
        new.paren_depth += decoded.count("(") - decoded.count(")")
        new.bracket_depth += decoded.count("[") - decoded.count("]")
        new.brace_depth += decoded.count("{") - decoded.count("}")

        # Clamp to 0 (should not go negative if mask is working)
        new.paren_depth = max(0, new.paren_depth)
        new.bracket_depth = max(0, new.bracket_depth)
        new.brace_depth = max(0, new.brace_depth)

        return new
```

### Level 2: Tree-sitter Incremental Parsing

For richer constraints, use tree-sitter to parse the partial output and determine which
tokens are syntactically valid continuations.

```python
# cola_coder/constraints/treesitter_constraint.py

from pathlib import Path
import torch


class TreeSitterConstraint:
    """
    Uses tree-sitter to constrain generation to valid syntax.
    More powerful but ~10x slower than bracket matching.
    Only apply to the final 20% of generation steps or for critical tokens.
    """

    def __init__(self, tokenizer, language: str = "python"):
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_python as tspython

            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)
            self.tokenizer = tokenizer
            self.language = language
            self.enabled = True
        except ImportError:
            print("Warning: tree-sitter not installed. TreeSitterConstraint disabled.")
            self.enabled = False

    def is_valid_prefix(self, code: str) -> bool:
        """Check if `code` is a valid (possibly incomplete) program prefix."""
        if not self.enabled:
            return True
        tree = self.parser.parse(bytes(code, "utf-8"))
        # A tree with no ERROR nodes (or only at the very end) is a valid prefix
        return not self._has_interior_errors(tree.root_node)

    def _has_interior_errors(self, node, depth: int = 0) -> bool:
        """Check for ERROR nodes that are not at the end of input."""
        if node.type == "ERROR" and not node.is_missing:
            # Allow error at the very end (incomplete code)
            return node.end_point[1] < node.parent.end_point[1] - 2 if node.parent else True
        return any(self._has_interior_errors(child, depth + 1) for child in node.children)

    def get_valid_next_tokens(
        self,
        current_code: str,
        tokenizer,
        sample_size: int = 200,   # check a subset of vocab for speed
    ) -> set[int]:
        """
        Find token IDs that produce a valid code prefix when appended.
        Only checks sample_size tokens for performance.
        """
        if not self.enabled:
            return None  # None = no constraint

        vocab = tokenizer.get_vocab()
        valid = set()

        # Sample high-probability tokens to check (not full 50K vocab)
        for token_str, token_id in list(vocab.items())[:sample_size]:
            candidate = current_code + tokenizer.decode([token_id])
            if self.is_valid_prefix(candidate):
                valid.add(token_id)

        return valid
```

### Level 3: Critical Token Constraints (Compromise)

Full tree-sitter parsing at every step is too slow. Constrain only "critical" tokens
where syntax errors are most common:

```python
# cola_coder/constraints/critical_tokens.py

import torch


class CriticalTokenConstraint:
    """
    Only constrain tokens that commonly cause syntax errors:
    - Closing brackets (handled by BracketConstraint)
    - Return/yield outside function
    - Semicolons in Python (often wrong)
    - Invalid indentation tokens (simplified)
    """

    def __init__(self, tokenizer, language: str = "python"):
        self.tokenizer = tokenizer
        self.language = language
        self.bracket = BracketConstraint(tokenizer)
        self._build_critical_sets()

    def _build_critical_sets(self):
        """Pre-compute sets of critical token IDs."""
        vocab = self.tokenizer.get_vocab()
        self.semicolon_ids: set[int] = set()
        if self.language == "python":
            for tok, tid in vocab.items():
                decoded = self.tokenizer.decode([tid])
                # Python semicolons between statements are valid but unusual
                # Only flag standalone semicolons (not in strings)
                if decoded.strip() == ";":
                    self.semicolon_ids.add(tid)

    def get_mask(
        self,
        state: BracketState,
        context: str,   # generated code so far
        vocab_size: int,
        in_function: bool = True,
    ) -> torch.Tensor:
        mask = self.bracket.get_mask(state, vocab_size)
        # Optionally penalize (not hard-forbid) semicolons in Python
        # Use soft constraint: multiply logit by 0.1 rather than masking
        return mask

    def apply_to_logits(
        self,
        logits: torch.Tensor,   # (V,)
        state: BracketState,
        context: str,
        vocab_size: int,
    ) -> torch.Tensor:
        mask = self.get_mask(state, context, vocab_size)
        logits = logits.clone()
        logits[~mask] = float("-inf")
        return logits
```

### Integration into generate_stream()

```python
# cola_coder/generator.py  (constraint integration)

from .constraints.bracket_constraint import BracketConstraint, BracketState

class CodeGenerator:
    def generate_stream(self, prompt, ..., use_constraints=False):
        constraint = BracketConstraint(self.tokenizer) if use_constraints else None
        state = BracketState()
        generated_text = ""

        for step in range(max_new_tokens):
            # ... forward pass, get logits ...

            if constraint is not None:
                mask = constraint.get_mask(state, logits.shape[-1])
                logits[~mask] = float("-inf")

            next_token_id = sample_next_token(logits, ...)

            if constraint is not None:
                state = constraint.update(state, next_token_id)

            token_str = self.tokenizer.decode([next_token_id])
            generated_text += token_str
            yield {...}
```

---

## Implementation Steps

1. **Create `cola_coder/constraints/` package**: `__init__.py`, `bracket_constraint.py`,
   `critical_tokens.py`, `treesitter_constraint.py`.

2. **Pre-build token tables at startup**: `BracketConstraint._build_token_tables()` runs
   once. Cache result to avoid repeated vocab scanning.

3. **Integrate into `generator.py`**: add `use_constraints: bool = False` parameter to
   `generate()` and `generate_stream()`.

4. **Add `--constraints` CLI flag** and menu option.

5. **Soft vs hard constraints**: implement soft version (logit penalty instead of -inf)
   for tokens that are unusual but not always illegal.

6. **String mode tracking**: ensure constraint does not fire inside string literals
   (brackets inside strings are fine).

7. **Benchmark overhead**: measure tokens/sec with and without constraints to quantify
   cost. Target: < 5% overhead for bracket constraint.

8. **Tree-sitter installation guide**: add optional install note. Heavy dependency,
   keep behind feature flag.

---

## Key Files to Modify

| File | Change |
|---|---|
| `generator.py` | Add constraint parameter to generate/generate_stream |
| `sampling.py` | Accept optional mask argument |
| `cli/menu.py` | Add "Enable syntax constraints" toggle |
| `config.py` | Add `ConstraintConfig` |
| `cola_coder/constraints/` | New package |
| `requirements.txt` | Add `tree-sitter` as optional dep |

---

## Testing Strategy

```python
# tests/test_constraints.py

def test_bracket_constraint_forbids_close_when_depth_zero():
    tokenizer = build_test_tokenizer()
    constraint = BracketConstraint(tokenizer)
    state = BracketState()  # all depths at 0

    mask = constraint.get_mask(state, tokenizer.vocab_size)
    # All close-paren token IDs should be masked
    for tid in constraint.token_closes_paren:
        assert not mask[tid], f"Token {tid} should be masked"

def test_bracket_constraint_allows_close_when_depth_positive():
    tokenizer = build_test_tokenizer()
    constraint = BracketConstraint(tokenizer)
    state = BracketState(paren_depth=1)

    mask = constraint.get_mask(state, tokenizer.vocab_size)
    for tid in constraint.token_closes_paren:
        assert mask[tid], f"Token {tid} should be allowed (depth=1)"

def test_state_update_increments_depth():
    tokenizer = build_test_tokenizer()
    constraint = BracketConstraint(tokenizer)
    state = BracketState()

    open_id = next(iter(constraint.token_opens_paren))
    new_state = constraint.update(state, open_id)
    assert new_state.paren_depth == 1

def test_constrained_generation_has_balanced_brackets():
    gen = build_test_generator()
    for _ in range(10):
        result = gen.generate("def f(", max_new_tokens=50, use_constraints=True)
        code = "def f(" + result
        assert code.count("(") >= code.count(")")
        assert code.count("[") >= code.count("]")
        assert code.count("{") >= code.count("}")

def test_treesitter_valid_prefix():
    if not treesitter_available():
        pytest.skip("tree-sitter not installed")
    c = TreeSitterConstraint(tokenizer)
    assert c.is_valid_prefix("def hello(")   # incomplete but valid prefix
    assert c.is_valid_prefix("x = 5\ny = 6")
    assert not c.is_valid_prefix("def def")  # definitely invalid
```

---

## Performance Considerations

- **Token table pre-computation**: classifying all 32K–50K vocabulary tokens once at
  startup takes < 1 second. Store as frozen sets.
- **Mask application**: converting set membership to a bool tensor over 50K elements
  takes ~0.1 ms. Negligible vs generation overhead.
- **Tree-sitter overhead**: parsing 200-token code snippets takes ~2 ms. At 30 tok/s
  generation speed, this is ~6% overhead per step if applied every step. Only apply
  every N steps or for selected critical tokens.
- **String mode edge cases**: Python has `f"..."`, `r"..."`, multi-line strings, raw
  strings. A full string tracker is non-trivial. Start with simple heuristic; refine
  based on observed errors.
- **Batched generation**: when batch_size > 1, each sequence has independent bracket
  state. Track `List[BracketState]` of length B.

---

## Dependencies

```
tree-sitter>=0.23.0           # optional — full incremental parsing
tree-sitter-python>=0.23.0    # optional — Python grammar
torch>=2.2.0                  # base requirement
```

---

## Estimated Complexity

| Aspect | Estimate |
|---|---|
| Token table pre-computation | 2 hours |
| BracketConstraint + state | 3 hours |
| String mode tracking | 2 hours |
| Integration into generator | 2 hours |
| CriticalTokenConstraint | 2 hours |
| Tree-sitter (optional) | 4 hours |
| Tests | 3 hours |
| CLI integration | 1 hour |
| **Total** | **~19 hours** |

Complexity rating: **Medium-Hard** — bracket matching is straightforward; the hard
parts are string mode handling, batch state tracking, and tree-sitter integration.

---

## 2026 Best Practices

- **Outlines / guidance library**: in 2025–2026, `outlines` (by .txt AI) has become
  the standard Python library for constrained text generation. It supports regex-guided,
  JSON-schema-guided, and grammar-guided decoding. Consider using it instead of a
  custom implementation.
- **XGrammar**: specialized constrained decoding runtime designed for LLMs, supports
  context-free grammars with near-zero overhead. Worth evaluating for production use.
- **Logit bias vs hard masking**: soft logit bias (reducing probability of unlikely
  tokens rather than setting to -inf) avoids degenerate outputs when the constraint
  is overly aggressive.
- **Grammar-guided decoding for JSON**: if the model generates JSON (e.g., function
  signatures), JSON schema constraints via `outlines.generate.json()` guarantee
  parse-valid JSON output with no post-processing.
- **Confidence-based constraint activation**: only activate tree-sitter constraints
  when the model's entropy is high (uncertain) — low-entropy steps rarely produce
  syntax errors and don't need constraining.
