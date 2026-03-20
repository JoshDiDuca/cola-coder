# Research: Execution Feedback Pre-Training (EFP)

## Status: Truly Novel — Combines ideas from RLHF + code execution, never applied to pre-training

## The Big Idea

What if the model learned not just from text, but from RUNNING the code it's learning from?

Current pre-training: Model reads code → predicts next token → cross-entropy loss.
The model has NO IDEA if the code works. It learns syntax and patterns, but not semantics.

**Execution Feedback Pre-Training**: Alongside standard next-token prediction, we ADD
an auxiliary loss based on whether the code, when executed, produces the expected behavior.

## How It Works

### Phase 1: Execution Annotation (Data Prep)

For each code file in training data that has runnable functions:

1. Extract functions with type annotations
2. Generate simple test inputs based on types
3. Execute the function with those inputs
4. Record: did it run? what did it return? any errors?

```python
# Example: annotate this TypeScript function
# function add(a: number, b: number): number { return a + b; }

annotation = {
    "function": "add",
    "inputs": [{"a": 1, "b": 2}, {"a": 0, "b": 0}, {"a": -1, "b": 1}],
    "outputs": [3, 0, 0],
    "executes": True,
    "pure": True,  # No side effects
}
```

### Phase 2: Dual-Head Training

The model has TWO output heads:

1. **LM Head** (standard): Predict next token → cross-entropy loss
2. **Execution Head** (new): Predict execution outcome → binary classification

```python
class ExecutionAwareTransformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        # New: small MLP head that predicts "will this code execute correctly?"
        self.exec_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 4),
            nn.GELU(),
            nn.Linear(config.dim // 4, 3),  # 3 classes: executes, errors, unknown
        )

    def compute_loss(self, token_ids, exec_labels=None):
        logits = self.forward(token_ids)

        # Standard LM loss
        lm_loss = cross_entropy(logits[:, :-1], token_ids[:, 1:])

        # Execution prediction loss (only when labels available)
        if exec_labels is not None:
            # Use hidden state at function boundary tokens
            hidden = self.get_hidden_states()
            exec_pred = self.exec_head(hidden[:, func_boundary_positions])
            exec_loss = cross_entropy(exec_pred, exec_labels)
            return lm_loss + 0.1 * exec_loss

        return lm_loss
```

### Phase 3: Execution-Guided Attention

The most novel part: modify attention weights based on execution traces.

When we execute a function, we know which variables are read/written, which
branches are taken, which functions are called. This creates an EXECUTION GRAPH:

```
function processOrder(order: Order) {
  const total = order.items.reduce((sum, item) => sum + item.price, 0);
  //    ^^^^^   ^^^^^^^^^^^^^^^^^^^   data flows: order → items → price → total
  const tax = total * 0.1;
  //    ^^^   ^^^^^   data flows: total → tax
  return { total, tax, final: total + tax };
  //       ^^^^^  ^^^  ^^^^^   data flows: total + tax → final
}
```

The execution trace tells us: `total` depends on `order.items[*].price`.
Standard attention might not learn this efficiently. But if we add a soft bias
for tokens that are data-flow connected, the model learns semantic relationships faster.

## Why This Could Be Revolutionary

1. **Semantic grounding**: Model learns that `+` means ADDITION, not just "a symbol
   that appears between two identifiers"
2. **Error understanding**: Model learns what causes runtime errors (null refs, type
   errors, out-of-bounds) by seeing code annotated with execution outcomes
3. **Correctness preference**: Model naturally prefers generating code that would
   execute correctly, because it's been trained to predict execution outcomes
4. **Bridge to RL**: The execution head is a proto-reward model. After pre-training,
   you can use it directly as a reward signal for GRPO fine-tuning

## Practical Implementation

### Lightweight Version (Start Here)

Don't try to execute every file. Start with:

1. **Function-level execution only** — Extract standalone pure functions
2. **Type-based input generation** — `number` → random ints, `string` → random strings
3. **Simple success/fail annotation** — Did it throw an error?
4. **Annotate 10-20% of training data** — The rest trains with standard LM loss only

```python
class ExecutionAnnotator:
    """Annotate code files with execution outcomes.

    Only processes:
    - Pure functions (no I/O, no global state)
    - Functions with type annotations (need types for input generation)
    - TypeScript and Python (easiest to sandbox)

    Runs in Docker sandbox (same as test-driven curation).
    """

    def annotate(self, content: str, language: str) -> list[ExecAnnotation]:
        functions = self._extract_functions(content, language)
        annotations = []
        for func in functions:
            if not func.is_pure or not func.has_type_annotations:
                continue
            inputs = self._generate_inputs(func.param_types)
            result = self._execute(func, inputs, timeout=5)
            annotations.append(ExecAnnotation(
                func_name=func.name,
                start_line=func.start,
                end_line=func.end,
                executes=result.success,
                error_type=result.error_type,
                inputs=inputs,
                outputs=result.outputs,
            ))
        return annotations
```

### Full Version (Long-Term)

1. Execute entire test suites (reuse test-driven curation infrastructure)
2. Trace execution at statement level (coverage data)
3. Use execution traces as attention bias during training
4. Train execution head alongside LM head
5. Use execution head as reward model for GRPO

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Security (running arbitrary code) | Docker sandboxing with no network, memory limits |
| Cost (execution is slow) | Only annotate pure functions, cache results |
| Noise (flaky execution) | Retry 3x, mark as "unknown" if inconsistent |
| Complexity (two-head training) | Start with weighted loss, 0.1x for exec head |
| Scale (can't execute all 33M files) | Annotate 10-20%, rest uses standard LM loss |

## Feasibility

| Aspect | Assessment |
|--------|-----------|
| Novelty | Extremely high — nobody has done execution-aware pre-training |
| Difficulty | Very high (execution infra, dual-head model, data pipeline) |
| Expected impact | Potentially transformative for code correctness |
| Prerequisites | Docker sandbox, type-based input gen, function extraction |
| Timeline | 2-4 months for lightweight version |

## Prior Art

- CodeRL (2022): RL fine-tuning with execution feedback — but ONLY for fine-tuning, not pre-training
- AlphaCode (2022): Execution for filtering generated solutions — not for training data
- Self-debugging (2023): Model debugs its own code — inference time, not training time
- **Nobody has used execution outcomes as a training signal during PRE-TRAINING**

## The Dream

Imagine a model that doesn't just predict code that LOOKS right, but code that
IS right — because it learned during pre-training what correct execution means.
This is the difference between a model that memorizes syntax and one that understands semantics.
