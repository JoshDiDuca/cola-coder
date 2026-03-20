# Feature 23: Ensemble Generation

**Status:** Optional | **CLI Flag:** `--ensemble` | **Complexity:** High

---

## Overview

Generate candidate outputs from multiple specialist models and select the best one using a quality signal. Three selection strategies: (a) lowest perplexity, (b) highest reward score (syntax/type-check passes), (c) majority vote on output structure. Supports parallel generation when VRAM allows, otherwise sequential. Uses router confidence scores as weights when combining signals. Reports which specialist "won" each generation for observability.

---

## Motivation

A single specialist model can fail on prompts that fall on the boundary between domains. Ensemble generation hedges this risk:

- React specialist generates JSX; GraphQL specialist generates type-safe query — pick the one that compiles
- If perplexity scores are close, a fast syntax check breaks the tie
- Majority vote on token structure catches structural failures (malformed function, missing closing bracket)

This is especially valuable for high-stakes completions (full function implementations, complex type definitions) where correctness matters more than latency.

---

## Architecture / Design

### Ensemble Pipeline

```
Router → [domain_1: 0.72, domain_2: 0.21, domain_3: 0.07]
          ↓
Select top-k domains (e.g., top-2 by confidence)
          ↓
Generate from specialist_1   Generate from specialist_2
(parallel if VRAM allows)    (sequential otherwise)
          ↓
Evaluate candidates:
  - Perplexity (log-prob under generating model)
  - Reward score (syntax check, TypeScript tsc pass)
  - Structural majority vote
          ↓
Weighted combination of signals
          ↓
Return best candidate + metadata
```

### Selection Strategies

| Strategy | Signal | Cost | When to Use |
|---|---|---|---|
| Perplexity | Model's own log-probs | Free | Always available baseline |
| Reward (syntax) | Tree-sitter parse | Fast (~5ms) | Code generation tasks |
| Reward (type-check) | tsc --noEmit | Slow (~500ms) | High-quality mode only |
| Majority vote | Token overlap | Medium | When N>2 candidates |

---

## Implementation Steps

### Step 1: Candidate Dataclass

```python
# cola_coder/ensemble/candidate.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Candidate:
    text: str
    tokens: list[int]
    domain: str
    router_confidence: float
    perplexity: float
    reward_score: Optional[float] = None  # Higher = better
    syntax_valid: Optional[bool] = None
    typecheck_passed: Optional[bool] = None
    generation_ms: float = 0.0

    def composite_score(self, weights: dict) -> float:
        """Lower is better (perplexity-based) or higher is better (reward)."""
        score = 0.0
        if "perplexity" in weights:
            # Normalize: lower perplexity → higher score
            score -= weights["perplexity"] * self.perplexity
        if "reward" in weights and self.reward_score is not None:
            score += weights["reward"] * self.reward_score
        if "confidence" in weights:
            score += weights["confidence"] * self.router_confidence
        if "syntax" in weights and self.syntax_valid is not None:
            score += weights["syntax"] * (1.0 if self.syntax_valid else 0.0)
        return score
```

### Step 2: Reward Functions

```python
# cola_coder/ensemble/rewards.py
import subprocess
import tempfile
import pathlib
from typing import Optional

def syntax_reward(code: str, language: str = "typescript") -> tuple[bool, float]:
    """
    Check syntax validity using tree-sitter.
    Returns (is_valid, score) where score = 1.0 if valid, 0.0 if not.
    """
    try:
        import tree_sitter_typescript as tstypescript
        from tree_sitter import Language, Parser
        TS_LANGUAGE = Language(tstypescript.language_typescript())
        parser = Parser(TS_LANGUAGE)
        tree = parser.parse(code.encode())
        # Check for ERROR nodes in the syntax tree
        has_errors = _has_error_nodes(tree.root_node)
        return (not has_errors, 0.0 if has_errors else 1.0)
    except ImportError:
        # Fallback: basic bracket counting
        opens = code.count("{") + code.count("(") + code.count("[")
        closes = code.count("}") + code.count(")") + code.count("]")
        balanced = abs(opens - closes) <= 1
        return (balanced, 1.0 if balanced else 0.0)


def _has_error_nodes(node) -> bool:
    if node.type == "ERROR":
        return True
    return any(_has_error_nodes(child) for child in node.children)


def typecheck_reward(
    code: str,
    timeout_s: float = 5.0,
    tsconfig: Optional[str] = None,
) -> tuple[bool, float]:
    """
    Run TypeScript compiler on generated code. Expensive (~500ms).
    Returns (passed, score).
    Only use in high-quality mode.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_file = pathlib.Path(tmpdir) / "generated.ts"
        ts_file.write_text(code)
        cmd = ["npx", "tsc", "--noEmit", "--strict", "--target", "ES2022",
               "--moduleResolution", "node", str(ts_file)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
            passed = result.returncode == 0
            # Score: penalize by number of type errors
            error_count = result.stdout.count("error TS")
            score = max(0.0, 1.0 - error_count * 0.1)
            return (passed, score)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return (False, 0.0)


def perplexity_reward(perplexity: float, scale: float = 50.0) -> float:
    """Convert perplexity to a [0, 1] reward score (higher = better)."""
    import math
    # Sigmoid-like normalization centered at scale
    return 1.0 / (1.0 + perplexity / scale)
```

### Step 3: EnsembleGenerator

```python
# cola_coder/ensemble/generator.py
import time
import torch
import concurrent.futures
from .candidate import Candidate
from .rewards import syntax_reward, typecheck_reward, perplexity_reward
from ..memory.hot_swap import HotSwapManager
from ..router.routing import RoutingDecision

@dataclass
class EnsembleConfig:
    top_k_domains: int = 2           # Number of specialists to query
    strategy: str = "perplexity"     # "perplexity" | "reward" | "vote" | "weighted"
    use_syntax_reward: bool = True
    use_typecheck_reward: bool = False  # Expensive; disabled by default
    parallel: bool = False           # Requires multi-GPU or large VRAM
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    score_weights: dict = None


class EnsembleGenerator:
    def __init__(
        self,
        hot_swap_manager: HotSwapManager,
        tokenizer,
        config: EnsembleConfig,
        device: str = "cuda",
    ):
        self.manager = hot_swap_manager
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def generate(
        self,
        input_ids: torch.Tensor,
        routing_decision: RoutingDecision,
    ) -> tuple[Candidate, list[Candidate]]:
        """
        Generate from top-k specialists, score all candidates, return best + all.
        """
        # Select top-k domains by confidence
        sorted_domains = sorted(
            routing_decision.all_probs.items(),
            key=lambda x: -x[1]
        )[:self.config.top_k_domains]

        candidates: list[Candidate] = []

        if self.config.parallel and len(sorted_domains) > 1:
            candidates = self._parallel_generate(input_ids, sorted_domains)
        else:
            for domain, confidence in sorted_domains:
                candidate = self._generate_one(input_ids, domain, confidence)
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            raise RuntimeError("All specialists failed to generate")

        # Score and rank candidates
        candidates = self._score_candidates(candidates)
        best = self._select_best(candidates)

        return best, candidates

    def _generate_one(
        self,
        input_ids: torch.Tensor,
        domain: str,
        confidence: float,
    ) -> Candidate | None:
        t0 = time.perf_counter()
        try:
            model = self.manager.get_or_load(domain, self.device)
        except (ValueError, FileNotFoundError):
            return None

        from ..routing.perplexity_monitor import PerplexityMonitor
        monitor = PerplexityMonitor()
        generated_ids = input_ids.clone().to(self.device)

        model.eval()
        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                logits = model(generated_ids)
                next_logits = logits[:, -1, :] / self.config.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = self._sample_top_p(probs, self.config.top_p)
                log_prob = torch.log(probs[0, next_token.item()]).item()
                monitor.record_token(log_prob)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        new_tokens = generated_ids[0, input_ids.shape[1]:].tolist()
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_ms = (time.perf_counter() - t0) * 1000

        return Candidate(
            text=text,
            tokens=new_tokens,
            domain=domain,
            router_confidence=confidence,
            perplexity=monitor.full_perplexity() or 99.0,
            generation_ms=gen_ms,
        )

    def _parallel_generate(self, input_ids, domains_confidences):
        """Generate from multiple specialists in parallel threads (I/O-bound portion)."""
        candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._generate_one, input_ids, domain, conf): domain
                for domain, conf in domains_confidences
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)
        return candidates

    def _score_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Add syntax and type-check scores to candidates."""
        for candidate in candidates:
            if self.config.use_syntax_reward:
                is_valid, score = syntax_reward(candidate.text)
                candidate.syntax_valid = is_valid
                candidate.reward_score = score

            if self.config.use_typecheck_reward:
                passed, score = typecheck_reward(candidate.text)
                candidate.typecheck_passed = passed
                # Blend with syntax score
                candidate.reward_score = (
                    ((candidate.reward_score or 0) + score) / 2
                )
        return candidates

    def _select_best(self, candidates: list[Candidate]) -> Candidate:
        strategy = self.config.strategy
        if strategy == "perplexity":
            return min(candidates, key=lambda c: c.perplexity)
        elif strategy == "reward":
            return max(candidates, key=lambda c: c.reward_score or 0.0)
        elif strategy == "vote":
            return self._majority_vote(candidates)
        elif strategy == "weighted":
            weights = self.config.score_weights or {
                "perplexity": 0.5, "reward": 0.3, "confidence": 0.2
            }
            return max(candidates, key=lambda c: c.composite_score(weights))
        return candidates[0]

    def _majority_vote(self, candidates: list[Candidate]) -> Candidate:
        """
        Select candidate whose first N tokens overlap most with others.
        Useful for detecting structural outliers.
        """
        if len(candidates) <= 1:
            return candidates[0]
        # Compare token-level Jaccard similarity
        best_score = -1
        best = candidates[0]
        for i, c1 in enumerate(candidates):
            total_overlap = 0
            set1 = set(c1.tokens[:64])
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                set2 = set(c2.tokens[:64])
                overlap = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
                total_overlap += overlap
            if total_overlap > best_score:
                best_score = total_overlap
                best = c1
        return best

    @staticmethod
    def _sample_top_p(probs, top_p):
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[0, cumulative[0] - sorted_probs[0] > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx[0, idx[0]].unsqueeze(0)
```

### Step 4: CLI Integration

```python
@app.command()
def generate_ensemble(
    prompt: str = typer.Argument(...),
    top_k: int = typer.Option(2, "--top-k"),
    strategy: str = typer.Option("weighted", "--strategy"),
    typecheck: bool = typer.Option(False, "--typecheck"),
    show_all: bool = typer.Option(False, "--show-all"),
):
    """Generate from multiple specialists and pick best output."""
    routing_decision = router.route(tokenize(prompt))
    best, all_candidates = ensemble_gen.generate(tokenize(prompt), routing_decision)

    console.print(f"[bold green]Winner:[/bold green] {best.domain} "
                  f"(ppl={best.perplexity:.1f}, conf={best.router_confidence:.2f})")
    if show_all:
        for c in all_candidates:
            console.print(f"\n--- {c.domain} (ppl={c.perplexity:.1f}) ---")
            console.print(c.text)
    else:
        console.print(best.text)
```

---

## Key Files to Modify

- `cola_coder/ensemble/__init__.py` — new package
- `cola_coder/ensemble/candidate.py` — Candidate dataclass
- `cola_coder/ensemble/rewards.py` — reward functions
- `cola_coder/ensemble/generator.py` — EnsembleGenerator
- `cola_coder/cli.py` — `generate-ensemble` command
- `configs/ensemble.yaml` — top_k, strategy, reward weights

---

## Testing Strategy

```python
def test_syntax_reward_valid():
    code = "const x: number = 42;\nfunction add(a: number, b: number): number { return a + b; }"
    valid, score = syntax_reward(code)
    assert valid
    assert score == 1.0

def test_syntax_reward_invalid():
    code = "const x = {"  # Unclosed brace
    valid, score = syntax_reward(code)
    assert not valid

def test_candidate_composite_score():
    c = Candidate("code", [], "react", 0.8, 20.0, reward_score=0.9)
    weights = {"perplexity": 0.1, "reward": 0.5, "confidence": 0.4}
    score = c.composite_score(weights)
    assert isinstance(score, float)

def test_ensemble_selects_lower_perplexity():
    c1 = Candidate("a", [], "react", 0.7, 15.0)
    c2 = Candidate("b", [], "prisma", 0.2, 45.0)
    gen = EnsembleGenerator.__new__(EnsembleGenerator)
    gen.config = EnsembleConfig(strategy="perplexity")
    best = gen._select_best([c1, c2])
    assert best.domain == "react"
```

---

## Performance Considerations

- **Sequential by default:** Parallel generation requires both models in VRAM simultaneously. On a 3080 (10GB), two 125M models = ~500MB — feasible. On larger models, sequential is safer.
- **Budget-aware top-k:** Don't blindly use top-2 if the second domain has <5% confidence — the specialist will likely produce garbage. Add a min_confidence_for_ensemble threshold.
- **typecheck_reward is expensive:** Only enable with `--typecheck` flag. In practice, syntax_reward covers 90% of structural failures at 1/100th the cost.
- **Tree-sitter is optional:** Fall back to bracket counting if tree-sitter is not installed. Degrade gracefully.
- **Generation caching:** If the same prompt is generated twice with ensemble, cache the result keyed on prompt hash + strategy.

---

## Dependencies

- Feature 18 (SpecialistRegistry)
- Feature 22 (HotSwapManager) — for model loading
- Feature 19 (ConfidenceRouter) — provides routing decision with per-domain probabilities
- `tree-sitter` + `tree-sitter-typescript` (optional, for syntax reward)
- Node.js / `npx tsc` (optional, for typecheck reward)

---

## Estimated Complexity

| Task                          | Effort   |
|-------------------------------|----------|
| Candidate dataclass           | 0.5h     |
| Reward functions (3)          | 3h       |
| EnsembleGenerator core        | 4h       |
| Parallel generation           | 2h       |
| CLI integration + display     | 1.5h     |
| Tests                         | 2h       |
| **Total**                     | **~13h** |

Overall complexity: **High** (multiple generation paths, reward computation, parallel execution)

---

## 2026 Best Practices

- **Reward model over heuristics:** For highest quality, train a small reward model on (code, quality_label) pairs rather than using rule-based rewards. A 10M param reward model can score TypeScript quality cheaply.
- **Self-play ensemble:** Let the same model generate N diverse outputs at temperature=0.9 and rank by reward. This is cheaper than multi-specialist ensemble and often competitive.
- **RLHF integration:** Ensemble reward scores (syntax pass, type-check pass) are natural signals for RLHF/GRPO training data generation. Log winning candidates for future training.
- **Beam search alternative:** For deterministic scenarios, beam search with N beams is a structured alternative to sampling N times. More expensive but less random.
- **Diversity penalty:** When using perplexity as selection criterion, add a diversity bonus to avoid always selecting the most "average" output. Min-edit distance from other candidates is a simple proxy.
