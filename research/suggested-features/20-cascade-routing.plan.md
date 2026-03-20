# Feature 20: Cascade Routing

**Status:** Optional | **CLI Flag:** `--cascade-routing` | **Complexity:** Medium-High

---

## Overview

Cascade routing tries the specialist model first, generates output, then evaluates output quality via perplexity scoring. If the specialist's output perplexity exceeds a configurable threshold (the model is "confused"), the output is discarded and generation falls back to the general model. An optional early-exit heuristic aborts the specialist after the first N tokens if perplexity is already too high. Configurable cascade depth enables specialist → general → ensemble chains.

---

## Motivation

Confidence-based routing (Feature 19) decides routing purely based on the input prompt. Cascade routing is more powerful: it uses the actual generated output quality as the routing signal. This catches cases where:

- The router was confident but wrong (e.g., a Prisma query that looks like general TS)
- The specialist begins generating incoherent output for an out-of-domain prompt
- There are ambiguous prompts that need trial-and-error to resolve

The tradeoff is cost: cascade routing runs the specialist forward pass at least partially before potentially discarding the output. The early-exit optimization recovers most of this cost.

---

## Architecture / Design

### Cascade Levels

```
Level 0 (specialist):
  Generate tokens from specialist model
  After every K tokens, check perplexity
  If perplexity > specialist_threshold → abort, discard output
                                       → go to Level 1
  If perplexity OK at sequence end    → return output

Level 1 (general model):
  Generate from general_ts model
  If perplexity > general_threshold   → go to Level 2 (optional)
  Otherwise                           → return output

Level 2 (ensemble, optional):
  Generate from N models, pick lowest perplexity output
  → return best output
```

### Perplexity Calculation

```
perplexity = exp(mean cross-entropy loss over generated tokens)
           = exp(-(1/T) * sum_t log P(x_t | x_{<t}))
```

For early exit, perplexity is estimated on a sliding window of the last K tokens rather than waiting for the full sequence.

### Early Exit Decision Timeline

```
Token:  1   2   3  ... K  K+1 ... 2K ...
         |               |        |
         check not yet   check    check
                         if high → abort
```

---

## Implementation Steps

### Step 1: Perplexity Monitor

```python
# cola_coder/routing/perplexity_monitor.py
import torch
import torch.nn.functional as F
import math
from collections import deque
from typing import Optional

class PerplexityMonitor:
    """
    Tracks token-level log probabilities during generation.
    Supports windowed perplexity for early-exit decisions.
    """
    def __init__(self, window_size: int = 32, threshold: float = 50.0):
        self.window_size = window_size
        self.threshold = threshold
        self._log_probs: deque[float] = deque(maxlen=window_size)
        self._all_log_probs: list[float] = []

    def record_token(self, log_prob: float):
        """Record log probability of a generated token."""
        self._log_probs.append(log_prob)
        self._all_log_probs.append(log_prob)

    def windowed_perplexity(self) -> Optional[float]:
        """Current perplexity over the recent window."""
        if len(self._log_probs) < 4:
            return None
        avg_nll = -sum(self._log_probs) / len(self._log_probs)
        return math.exp(avg_nll)

    def full_perplexity(self) -> Optional[float]:
        """Perplexity over all recorded tokens."""
        if not self._all_log_probs:
            return None
        avg_nll = -sum(self._all_log_probs) / len(self._all_log_probs)
        return math.exp(avg_nll)

    def should_abort(self) -> bool:
        """True if windowed perplexity exceeds threshold."""
        ppl = self.windowed_perplexity()
        return ppl is not None and ppl > self.threshold

    def reset(self):
        self._log_probs.clear()
        self._all_log_probs.clear()
```

### Step 2: CascadeGenerator

```python
# cola_coder/routing/cascade_generator.py
import torch
from dataclasses import dataclass
from typing import Optional
from .perplexity_monitor import PerplexityMonitor

@dataclass
class CascadeResult:
    tokens: list[int]
    text: str
    level_used: int          # 0=specialist, 1=general, 2=ensemble
    domain: str
    final_perplexity: float
    early_exit: bool         # Was specialist aborted early?
    specialist_tokens_generated: int
    total_forward_passes: int


class CascadeGenerator:
    def __init__(
        self,
        registry,
        tokenizer,
        specialist_ppl_threshold: float = 40.0,
        general_ppl_threshold: float = 80.0,
        early_exit_check_interval: int = 16,  # Check every N tokens
        early_exit_min_tokens: int = 8,        # Don't exit before this many tokens
        max_new_tokens: int = 512,
        cascade_depth: int = 2,                # 1=specialist+general, 2=+ensemble
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.registry = registry
        self.tokenizer = tokenizer
        self.specialist_threshold = specialist_ppl_threshold
        self.general_threshold = general_ppl_threshold
        self.check_interval = early_exit_check_interval
        self.min_tokens_before_exit = early_exit_min_tokens
        self.max_new_tokens = max_new_tokens
        self.cascade_depth = cascade_depth
        self.temperature = temperature
        self.top_p = top_p

    def generate(
        self,
        input_ids: torch.Tensor,
        routing_decision,  # RoutingDecision from Feature 19
        device: str = "cuda",
    ) -> CascadeResult:
        domain = routing_decision.domain

        # Level 0: Try specialist
        if domain != "general_ts" and self.cascade_depth >= 1:
            result = self._try_specialist(input_ids, domain, device)
            if result is not None:
                return result

        # Level 1: General model
        result = self._generate_with_model(
            input_ids,
            domain="general_ts",
            threshold=self.general_threshold,
            level=1,
            device=device,
        )
        if result is not None:
            return result

        # Level 2: Ensemble (if depth allows)
        if self.cascade_depth >= 2:
            return self._ensemble_generate(input_ids, device)

        return result  # Best effort from general model

    def _try_specialist(
        self,
        input_ids: torch.Tensor,
        domain: str,
        device: str,
    ) -> Optional[CascadeResult]:
        """Generate with specialist, return None if perplexity too high."""
        try:
            model = self.registry.load_specialist(domain, device)
        except (ValueError, FileNotFoundError) as e:
            print(f"[cascade] Cannot load specialist '{domain}': {e}")
            return None

        monitor = PerplexityMonitor(
            window_size=self.check_interval,
            threshold=self.specialist_threshold,
        )

        generated_ids = input_ids.clone().to(device)
        aborted = False
        specialist_tokens = 0
        forward_passes = 0

        model.eval()
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                logits = model(generated_ids)
                forward_passes += 1
                next_logits = logits[:, -1, :] / self.temperature
                probs = torch.softmax(next_logits, dim=-1)

                # Top-p sampling
                next_token = self._sample_top_p(probs, self.top_p)
                log_prob = torch.log(probs[0, next_token.item()]).item()
                monitor.record_token(log_prob)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                specialist_tokens += 1

                # Early exit check
                if (step >= self.min_tokens_before_exit
                        and step % self.check_interval == 0
                        and monitor.should_abort()):
                    aborted = True
                    print(f"[cascade] Early exit from '{domain}' at token {step}: "
                          f"ppl={monitor.windowed_perplexity():.1f}")
                    break

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        if aborted:
            return None

        final_ppl = monitor.full_perplexity() or 0.0
        if final_ppl > self.specialist_threshold:
            print(f"[cascade] Specialist '{domain}' output rejected: ppl={final_ppl:.1f}")
            return None

        new_tokens = generated_ids[0, input_ids.shape[1]:].tolist()
        return CascadeResult(
            tokens=new_tokens,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            level_used=0,
            domain=domain,
            final_perplexity=final_ppl,
            early_exit=aborted,
            specialist_tokens_generated=specialist_tokens,
            total_forward_passes=forward_passes,
        )

    def _generate_with_model(
        self,
        input_ids: torch.Tensor,
        domain: str,
        threshold: float,
        level: int,
        device: str,
    ) -> Optional[CascadeResult]:
        """Generate from the general model, checking perplexity."""
        model = self.registry.load_specialist(domain, device)
        monitor = PerplexityMonitor(threshold=threshold)
        generated_ids = input_ids.clone().to(device)
        forward_passes = 0

        model.eval()
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                logits = model(generated_ids)
                forward_passes += 1
                next_logits = logits[:, -1, :] / self.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = self._sample_top_p(probs, self.top_p)
                log_prob = torch.log(probs[0, next_token.item()]).item()
                monitor.record_token(log_prob)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        new_tokens = generated_ids[0, input_ids.shape[1]:].tolist()
        final_ppl = monitor.full_perplexity() or 0.0
        return CascadeResult(
            tokens=new_tokens,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            level_used=level,
            domain=domain,
            final_perplexity=final_ppl,
            early_exit=False,
            specialist_tokens_generated=0,
            total_forward_passes=forward_passes,
        )

    def _ensemble_generate(self, input_ids: torch.Tensor, device: str) -> CascadeResult:
        """Generate from multiple models, return lowest-perplexity output."""
        candidates = []
        for domain in ["general_ts"] + [
            e.name for e in self.registry.list_specialists()
            if e.name != "general_ts" and e.is_available()
        ][:2]:  # At most 3 total for VRAM
            result = self._generate_with_model(
                input_ids, domain, threshold=999.0, level=2, device=device
            )
            if result:
                candidates.append(result)

        best = min(candidates, key=lambda r: r.final_perplexity)
        best.level_used = 2
        return best

    @staticmethod
    def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative - sorted_probs > top_p
        sorted_probs[0, remove_mask[0]] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token_idx = torch.multinomial(sorted_probs, 1)
        return sorted_indices[0, next_token_idx[0]].unsqueeze(0)
```

### Step 3: CLI Integration

```python
@app.command()
def generate_cascade(
    prompt: str = typer.Argument(...),
    cascade_depth: int = typer.Option(2, "--depth"),
    specialist_threshold: float = typer.Option(40.0, "--specialist-ppl"),
    general_threshold: float = typer.Option(80.0, "--general-ppl"),
    early_exit_interval: int = typer.Option(16, "--check-every"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate with cascade routing: specialist → general → ensemble."""
    routing_decision = router.route(tokenize(prompt))
    result = cascade_gen.generate(tokenize(prompt), routing_decision)

    level_names = {0: "specialist", 1: "general", 2: "ensemble"}
    if verbose:
        console.print(f"[dim]Level used: {level_names[result.level_used]} ({result.domain})[/dim]")
        console.print(f"[dim]Perplexity: {result.final_perplexity:.2f}[/dim]")
        console.print(f"[dim]Forward passes: {result.total_forward_passes}[/dim]")
    console.print(result.text)
```

---

## Key Files to Modify

- `cola_coder/routing/perplexity_monitor.py` — new file
- `cola_coder/routing/cascade_generator.py` — new file
- `cola_coder/generate.py` — integrate CascadeGenerator as optional generation path
- `cola_coder/cli.py` — `generate-cascade` command
- `configs/cascade.yaml` — thresholds, cascade depth config

---

## Testing Strategy

```python
def test_perplexity_monitor_abort():
    monitor = PerplexityMonitor(window_size=4, threshold=10.0)
    for _ in range(4):
        monitor.record_token(-10.0)  # Very bad log prob → high perplexity
    assert monitor.should_abort()

def test_perplexity_monitor_no_abort():
    monitor = PerplexityMonitor(window_size=4, threshold=1000.0)
    for _ in range(4):
        monitor.record_token(-0.5)  # Good log prob
    assert not monitor.should_abort()

def test_cascade_result_fields():
    # Integration test: run cascade with mock models
    # Verify CascadeResult has all fields populated correctly
    pass
```

---

## Performance Considerations

- **Early exit saves most cost:** If the specialist is wrong, early exit after 16 tokens costs ~3% of full generation. Without early exit, a full failed specialist generation doubles total cost.
- **VRAM:** Keep only one model loaded at a time unless VRAM allows otherwise (see Feature 22). Load/unload between cascade levels.
- **Perplexity threshold tuning:** Calibrate thresholds empirically. A threshold of ~40-50 PPL typically separates "model is confused" from "model is generating normally" for code.
- **Skip cascade for simple prompts:** If the router is very confident (>0.95), skip cascade entirely and trust the specialist. Add `min_router_confidence_for_cascade` config option.
- **Batching:** Cascade routing is inherently sequential per sample. For throughput scenarios, run cascade in parallel threads for different prompts.

---

## Dependencies

- Feature 18 (SpecialistRegistry) — for model loading
- Feature 19 (ConfidenceRouter) — for initial routing decision
- Feature 22 (hot-swap) — for VRAM management during model switches
- Feature 23 (ensemble) — for Level 2 cascade step

---

## Estimated Complexity

| Task                           | Effort   |
|--------------------------------|----------|
| PerplexityMonitor              | 1h       |
| CascadeGenerator core          | 4h       |
| Early exit logic               | 2h       |
| Ensemble level                 | 2h       |
| CLI integration                | 1h       |
| Tests + threshold calibration  | 2h       |
| **Total**                      | **~12h** |

Overall complexity: **Medium-High** (involves multiple model forward passes, state management, careful threshold design)

---

## 2026 Best Practices

- **Perplexity is a noisy signal at low token counts:** Don't make early exit decisions before at least 8-16 tokens; random fluctuations dominate at very short sequences.
- **Log all cascade decisions:** Track which level was ultimately used for each generation. If Level 0 (specialist) succeeds >95% of the time for a domain, cascade overhead may be unjustified — raise the specialist threshold.
- **Configurable per domain:** React specialist may need a higher threshold than Prisma specialist (React code is more variable). Store per-domain thresholds in the registry (Feature 18).
- **Async prefill:** Start prefilling the general model in the background while the specialist generates its first N tokens. Overlap computation to reduce latency at cascade boundary.
- **PPL vs reward comparison:** Feature 23 uses reward scores (type-check) as quality signal. Consider a hybrid: cascade on perplexity first, then optionally re-rank by reward.
