# Research: AST-Aware Training

## Status: Novel — Partially explored in academic papers, never fully productionized

## The Idea

Current code LLMs treat code as flat text — just a sequence of tokens. But code has
RICH STRUCTURE that the model has to learn implicitly: function boundaries, scope nesting,
type relationships, import graphs, call chains.

What if we gave the model structural hints during training?

## Approach 1: AST Position Encoding

RoPE encodes linear position (token 0, 1, 2, ...). But code has TREE position:

```typescript
function greet(name: string) {  // depth 0, node: function_declaration
  const msg = `Hello ${name}`;  // depth 1, node: variable_declaration
  if (name === "world") {       // depth 1, node: if_statement
    console.log(msg);           // depth 2, node: expression_statement
  }
  return msg;                   // depth 1, node: return_statement
}
```

**AST-RoPE**: Add a second positional encoding dimension based on AST depth.
Each token gets TWO positions: (linear_pos, ast_depth). The model can attend
to tokens at similar structural depth, even if they're far apart in linear position.

```python
# Extend RoPE to include AST depth
def precompute_ast_rope(dim, max_seq_len, max_depth=64):
    # Split dimensions: half for position, half for depth
    pos_dim = dim // 2
    depth_dim = dim // 2

    pos_freqs = standard_rope(pos_dim, max_seq_len)
    depth_freqs = standard_rope(depth_dim, max_depth)

    return pos_freqs, depth_freqs

def apply_ast_rope(q, k, pos_freqs, depth_freqs, positions, depths):
    q_pos, q_depth = q.chunk(2, dim=-1)
    k_pos, k_depth = k.chunk(2, dim=-1)

    q_pos = rotate(q_pos, pos_freqs[positions])
    q_depth = rotate(q_depth, depth_freqs[depths])
    # ... same for k

    return cat(q_pos, q_depth), cat(k_pos, k_depth)
```

**How to get AST depths**: Use tree-sitter at data prep time. Store depth array
alongside token array. ~2x data size but adds powerful structural signal.

## Approach 2: Scope-Aware Attention Masking

Standard causal attention: every token can attend to all previous tokens.
Scope-aware: every token can attend to all previous tokens BUT tokens in the
same scope get a boost (or tokens in unrelated scopes get a penalty).

```
function A() {
  let x = 1;      // scope: A
  function B() {
    let y = 2;    // scope: A.B — can attend to x, y
    return x + y;
  }
  let z = B();    // scope: A — can attend to x, z, but y is less relevant
}
```

This is like giving the model a "soft" scope boundary — it can still see everything
(causal mask is preserved) but scoped tokens get higher attention weights.

**Implementation**: Modify the attention bias mask. Instead of binary (0 or -inf),
use a continuous mask: 0 for same scope, -alpha for different scope (where alpha
is a learnable parameter).

## Approach 3: Type-Augmented Tokens

For TypeScript specifically, type information is incredibly rich:

```typescript
const users: User[] = await db.user.findMany();
//    ^^^^^ variable   ^^^^  ^^^^^ type
```

Current tokenization: `const`, `users`, `:`, `User`, `[`, `]`, ...
Type-augmented: Each token gets a type tag from the AST:

| Token | Type Tag |
|-------|----------|
| const | keyword |
| users | identifier:variable |
| : | punctuation |
| User | identifier:type |
| [] | type_annotation |
| = | operator |
| await | keyword |
| db | identifier:variable |
| .user | member_access |
| .findMany | method_call |

**Implementation**: During data prep, run tree-sitter, tag each token with its AST
node type. Add a small auxiliary embedding for node types (like token type embeddings
in BERT, but for AST categories). ~10-50 extra embedding entries.

## Approach 4: Multi-Granularity Training (Most Novel)

Train on THREE levels simultaneously:

1. **Token level** (standard): predict next token
2. **Statement level**: predict next statement type (assignment, if, return, call)
3. **Function level**: predict function signature from docstring (or vice versa)

```
Loss = token_loss + 0.1 * statement_loss + 0.1 * function_loss
```

This forces the model to build hierarchical representations — not just character
patterns but structural understanding.

**Implementation**: Add two small prediction heads alongside the main LM head.
Statement head predicts from hidden states at statement boundaries. Function head
predicts from hidden states at function declaration tokens.

## Feasibility Assessment

| Approach | Novelty | Difficulty | Expected Impact | Data Prep Cost |
|----------|---------|------------|----------------|---------------|
| AST-RoPE | High | Medium | Medium-High | 2x data size |
| Scope masking | Medium | Medium | Medium | tree-sitter at prep |
| Type-augmented | High | Low | Medium (TS-specific) | tree-sitter at prep |
| Multi-granularity | Very High | High | High | Complex labeling |

## Recommended Path

Start with **Type-Augmented Tokens** (Approach 3) — it's TypeScript-specific,
relatively simple to implement, and could give your TS specialist model a unique
edge that no other code model has. Then add AST-RoPE (Approach 1) for a structural
position signal.

## Dependencies

- tree-sitter + tree-sitter-typescript (already planned for syntax filter)
- Modifications to preprocess.py to store AST metadata
- Modifications to dataset.py to load AST metadata
- Modifications to model to consume AST embeddings/positions

## Prior Art

- CodeBERT (2020): Token type embeddings, but for masked LM not causal
- GraphCodeBERT (2021): Data flow graph attention, but sequence-to-sequence not causal
- TreeBERT (2021): AST paths in embeddings, encoder-only
- InCoder (2022): Left-to-right + FIM, no structural encoding
- **Nobody has done AST-aware causal training for code generation in 2025-2026.**
