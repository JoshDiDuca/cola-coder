# 75 - Execution-Guided Beam Search

## Overview

Generate K diverse candidates at high temperature, run each through `tsc` (and optionally execute it), and return the first passing candidate—or the lowest-perplexity candidate if none pass. Uses sandboxed subprocess execution with timeout. Integrates with the existing FastAPI server as an enhanced `/complete` endpoint mode.

**Feature flag:** `config.inference.execution_guided_beam_search.enabled` / `--execution-guided`

---

## Motivation

At inference time, the model's first generation attempt doesn't always produce type-correct code. But generating 5-10 diverse candidates and running the TypeScript compiler on each costs only N×tsc_time ≈ 5-10 seconds—often acceptable for a coding assistant where the alternative is manually fixing a type error.

This technique is called "best-of-N" sampling, "generate-then-verify," or "execution-guided decoding" depending on the community. It's a practical way to improve generation quality without retraining.

**Expected improvement**: if the model has a 40% chance of generating a type-correct snippet on the first try, the probability that at least one of 5 diverse samples passes is `1 - (0.6)^5 ≈ 92%`.

---

## Architecture / Design

### Pipeline

```
prompt
  │
  ▼
Generate K candidates
(high temperature, different random seeds)
  │
  ├─ candidate 1 ──┐
  ├─ candidate 2   │  Run tsc in parallel
  ├─ candidate 3   │  on all candidates
  ├─ ...           │
  └─ candidate K ──┘
        │
        ▼
  Any pass? ──YES──► Return first passing candidate
        │
        NO
        ▼
  Return lowest-perplexity candidate (fallback)
```

### Diversity Strategy

To get K *diverse* candidates from the same prompt, use:
1. **Temperature sampling** with `temperature=0.8-1.2` (high temp increases diversity)
2. **Different random seeds** per candidate (seed the RNG state per candidate)
3. **Nucleus sampling** with `top_p=0.9` for controlled diversity

---

## Implementation Steps

### Step 1: Candidate Generator (`inference/candidate_generator.py`)

```python
import torch
from dataclasses import dataclass

@dataclass
class Candidate:
    text: str
    tokens: list[int]
    log_prob: float        # sum of log probs (for perplexity ranking)
    seed: int

class CandidateGenerator:
    def __init__(self, model, tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate_candidates(
        self,
        prompt: str,
        k: int = 5,
        max_tokens: int = 300,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> list[Candidate]:
        """Generate K diverse candidates from the same prompt."""
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens]).to(self.model.device)

        candidates = []
        self.model.eval()

        for i in range(k):
            seed = 42 + i * 1000
            torch.manual_seed(seed)

            with torch.inference_mode():
                output_ids, log_prob = self._generate_with_log_prob(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            generated_tokens = output_ids[0][len(prompt_tokens):].tolist()
            text = self.tokenizer.decode(generated_tokens)

            candidates.append(Candidate(
                text=text,
                tokens=generated_tokens,
                log_prob=log_prob,
                seed=seed,
            ))

        return candidates

    def _generate_with_log_prob(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, float]:
        """Generate and return (output_ids, cumulative_log_prob)."""
        generated = input_ids.clone()
        total_log_prob = 0.0

        for _ in range(max_new_tokens):
            logits = self.model(generated)[:, -1, :]  # (1, V)
            logits = logits / temperature

            # Nucleus sampling (top-p)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative prob above top_p
            sorted_logits[cumulative_probs > top_p] = -float("inf")

            # Sample
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = sorted_indices.gather(-1, next_token_idx)

            # Track log prob
            log_prob = torch.log_softmax(logits, dim=-1)
            total_log_prob += log_prob[0, next_token.item()].item()

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return generated, total_log_prob
```

### Step 2: Execution Verifier (`inference/execution_verifier.py`)

```python
import subprocess
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

TSCONFIG = {
    "compilerOptions": {
        "strict": True,
        "noEmit": True,
        "target": "ES2020",
        "skipLibCheck": True,
    }
}

@dataclass
class VerificationResult:
    candidate_idx: int
    passed: bool
    error_count: int
    error_codes: list[str]
    execution_time_ms: float

def verify_candidate(
    idx: int,
    full_code: str,
    timeout: int = 15,
) -> VerificationResult:
    """Run tsc on a single candidate."""
    import time
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"cola_beam_{idx}_") as tmpdir:
        ts_file = Path(tmpdir) / "candidate.ts"
        tsconfig = Path(tmpdir) / "tsconfig.json"
        ts_file.write_text(full_code)
        tsconfig.write_text(json.dumps(TSCONFIG))

        try:
            result = subprocess.run(
                ["tsc", "--project", str(tsconfig)],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            output = result.stdout + result.stderr
            import re
            error_codes = re.findall(r'error (TS\d+)', output)
            return VerificationResult(
                candidate_idx=idx,
                passed=result.returncode == 0,
                error_count=len(error_codes),
                error_codes=error_codes,
                execution_time_ms=round(elapsed, 1),
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                candidate_idx=idx, passed=False,
                error_count=-1, error_codes=["TIMEOUT"],
                execution_time_ms=timeout * 1000,
            )
        except FileNotFoundError:
            return VerificationResult(
                candidate_idx=idx, passed=False,
                error_count=-1, error_codes=["TSC_NOT_FOUND"],
                execution_time_ms=0.0,
            )

def verify_all_parallel(
    candidates: list["Candidate"],
    prompt: str,
    max_workers: int = 8,
    early_termination: bool = True,
) -> list[VerificationResult]:
    """
    Verify all candidates in parallel.
    With early_termination=True, cancel remaining when first passes.
    """
    results = {}
    full_codes = [prompt + c.text for c in candidates]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(verify_candidate, i, code): i
            for i, code in enumerate(full_codes)
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result

            if early_termination and result.passed:
                # Cancel remaining futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    # Fill in any cancelled futures as unverified
    for i in range(len(candidates)):
        if i not in results:
            results[i] = VerificationResult(
                candidate_idx=i, passed=False,
                error_count=-1, error_codes=["CANCELLED"],
                execution_time_ms=0.0,
            )

    return [results[i] for i in range(len(candidates))]
```

### Step 3: Best-of-N Selector (`inference/beam_selector.py`)

```python
from inference.candidate_generator import Candidate
from inference.execution_verifier import VerificationResult

def select_best_candidate(
    candidates: list[Candidate],
    verifications: list[VerificationResult],
    strategy: str = "first_passing",
) -> tuple[Candidate, str]:
    """
    Returns (selected_candidate, selection_reason).
    Strategies:
    - first_passing: return first candidate that passes tsc
    - best_passing: return passing candidate with fewest errors (or highest log_prob)
    - fallback_perplexity: if none pass, return lowest perplexity (highest log_prob)
    """
    passing = [
        (c, v) for c, v in zip(candidates, verifications)
        if v.passed
    ]

    if passing:
        if strategy == "first_passing":
            return passing[0][0], "first_passing"
        elif strategy == "best_passing":
            # Rank by log_prob (higher = more likely = better)
            best = max(passing, key=lambda cv: cv[0].log_prob)
            return best[0], "best_passing"

    # Fallback: no candidate passed
    # Return the one with fewest type errors (min error_count)
    non_timeout = [
        (c, v) for c, v in zip(candidates, verifications)
        if v.error_count >= 0
    ]
    if non_timeout:
        best = min(non_timeout, key=lambda cv: (cv[1].error_count, -cv[0].log_prob))
        return best[0], f"fallback_min_errors({best[1].error_count})"

    # Last resort: highest log_prob
    best = max(zip(candidates, verifications), key=lambda cv: cv[0].log_prob)
    return best[0], "fallback_perplexity"
```

### Step 4: FastAPI Integration (`server/guided_completion.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel

class GuidedCompletionRequest(BaseModel):
    prompt: str
    k: int = 5
    max_tokens: int = 300
    temperature: float = 0.9
    strategy: str = "first_passing"
    verify: bool = True         # set False to skip tsc verification
    timeout_sec: int = 15

class GuidedCompletionResponse(BaseModel):
    completion: str
    selection_reason: str
    candidates_generated: int
    candidates_passed: int
    best_candidate_index: int
    verification_time_ms: float

@app.post("/complete/guided", response_model=GuidedCompletionResponse)
async def guided_complete(req: GuidedCompletionRequest):
    generator = CandidateGenerator(model, tokenizer, config={})
    candidates = generator.generate_candidates(
        prompt=req.prompt,
        k=req.k,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    if req.verify:
        import time
        t0 = time.perf_counter()
        verifications = verify_all_parallel(
            candidates, req.prompt, early_termination=True
        )
        verify_ms = (time.perf_counter() - t0) * 1000
    else:
        # No verification: return highest log_prob candidate
        verifications = [
            VerificationResult(i, False, -1, [], 0.0) for i in range(len(candidates))
        ]
        verify_ms = 0.0

    best, reason = select_best_candidate(candidates, verifications, req.strategy)
    passed_count = sum(1 for v in verifications if v.passed)
    best_idx = candidates.index(best)

    return GuidedCompletionResponse(
        completion=best.text,
        selection_reason=reason,
        candidates_generated=len(candidates),
        candidates_passed=passed_count,
        best_candidate_index=best_idx,
        verification_time_ms=round(verify_ms, 1),
    )
```

### Step 5: Config

```yaml
inference:
  execution_guided_beam_search:
    enabled: false
    k: 5                    # number of candidates
    temperature: 0.9
    top_p: 0.95
    strategy: first_passing  # first_passing | best_passing
    early_termination: true  # stop verifying once first candidate passes
    tsc_timeout: 15
    max_workers: 8
    verify: true
```

---

## Key Files to Modify

- `inference/candidate_generator.py` - New file
- `inference/execution_verifier.py` - New file
- `inference/beam_selector.py` - New file
- `server/guided_completion.py` - New file: FastAPI endpoint
- `server/app.py` - Register `/complete/guided` route
- `cli/generate_cmd.py` - Add `--execution-guided` flag
- `config/inference.yaml` - New file or section in training config

---

## Testing Strategy

1. **Candidate diversity test**: generate 5 candidates with different seeds from the same prompt, assert at least 3 are distinct strings.
2. **Log prob ordering test**: for deterministic generation (temperature=0), assert all 5 candidates are identical and log_prob is consistent.
3. **Verify passing test**: write a valid TypeScript snippet, assert `verify_candidate` returns `passed=True`.
4. **Verify failing test**: write a snippet with a deliberate type error, assert `passed=False` and error_codes is non-empty.
5. **Early termination test**: mock verifier so candidate 2 passes; assert futures for candidates 3-5 are cancelled.
6. **Fallback test**: mock all verifiers to fail; assert `select_best_candidate` returns something and reason starts with "fallback".
7. **API integration test**: POST to `/complete/guided` with a simple prompt, assert response has `candidates_generated >= 1`.

---

## Performance Considerations

- K=5 candidates means 5 model forward passes (sequential) + 5 tsc invocations (parallel). On RTX 3080:
  - Generation: 5 × ~500ms = 2.5s total (can be parallelized if running on CPU, but GPU is sequential)
  - Verification: 5 tsc × 1s / 8 workers ≈ 1s with early termination
  - Total latency: ~3.5-4s for K=5
- For a coding assistant, 4s is acceptable. For real-time completion, reduce to K=3.
- Early termination is critical: once one candidate passes, cancel the rest immediately.
- Memory: K=5 candidates stored as token lists, negligible.
- On GPU, generation is sequential (can't run 5 model forwards in parallel without multi-GPU or batching). Batch all K prompts together if the model supports batch generation.

---

## Dependencies

- `tsc` in PATH (TypeScript compiler)
- No new Python dependencies

---

## Estimated Complexity

**Medium.** The individual components (generator, verifier, selector) are clean and well-scoped. The main complexity is the batched parallel verification and the early termination pattern. The FastAPI integration is straightforward. Estimated implementation time: 2-3 days.

---

## 2026 Best Practices

- **Best-of-N over beam search for LLMs**: classical beam search on language models often produces degenerate repetitive outputs. Best-of-N sampling with diverse temperatures produces more natural, varied candidates. The name "execution-guided beam search" is used loosely; the actual method is best-of-N with execution feedback.
- **Early termination is essential**: without early termination, K=5 always waits for all 5 tsc invocations. With early termination, the expected wait is much shorter (often just 1-2 tsc calls).
- **Fallback is a user trust feature**: always return *something*, even if no candidate passes. A fallback based on fewest errors or highest log_prob is better than returning empty. Users trust a system more when it always provides a response.
- **Expose selection metadata**: return `selection_reason` and `candidates_passed` in the API response. This allows the client (VS Code extension) to indicate confidence (e.g., "✓ type-checked" vs "⚠ best guess").
- **Progressive K**: start with K=3 in production, allow users to increase to K=10 in settings. Higher K gives better results but slower response. Let users choose their speed/quality tradeoff.
