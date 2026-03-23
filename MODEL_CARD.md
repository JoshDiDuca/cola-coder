# Cola-Coder Small

**Version:** small  
**License:** Apache 2.0  
**Generated:** 2026-03-23 19:35 UTC

## Model Information

| Field | Value |
|-------|-------|
| Architecture | Decoder-only transformer (RoPE, GQA, SwiGLU, RMSNorm) — same family as LLaMA 3 / Mistral |
| Parameters | 0 |
| Languages | Python, TypeScript, JavaScript |
| License | Apache 2.0 |

## Training Details

| Field | Value |
|-------|-------|
| Dataset | bigcode/starcoderdata |
| Epochs | 1 |
| Learning Rate | 0.0006 |
| Batch Size | 12 |
| Hardware | NVIDIA GeForce RTX 4080 SUPER (16.0 GB VRAM) |
| Training Time | 100,000 steps |

## Performance

| Metric | Value | Dataset |
|--------|-------|---------|
| Training loss (final) | 0.986 | — |
| Best training loss | 0.986 | — |

## Limitations

- Small model (unknown parameters) with limited training budget — struggles with complex multi-file reasoning.
- Base language model — not instruction-tuned.  Feed code prefixes, not natural-language requests.
- Biased toward Python, TypeScript, and JavaScript.  Other languages will see lower quality output.
- Not evaluated for safety or correctness on real-world tasks.  Do not deploy in production without your own evaluation.

## Usage Examples

### Example 1

**Prompt:**

```python
def fibonacci(n):

```

**Output:**

```python
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```


---

*This model card was generated automatically by Cola-Coder's model card generator.*