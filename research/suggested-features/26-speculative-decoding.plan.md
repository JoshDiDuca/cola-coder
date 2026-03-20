# Feature 26: Speculative Decoding

**Status:** Optional | **CLI Flag:** `--speculative` | **Complexity:** High

---

## Overview

Train a tiny draft model (~5M params, 2 transformer layers) that proposes K tokens ahead in a single fast pass. The target (main) model then verifies all K draft tokens in a single forward pass by checking whether its own distribution agrees. Tokens that match are accepted; the first disagreement triggers a rejection + resample. Expected speedup: 2–3x for well-matched drafts. The draft model is distilled from the target model.

Reference: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023).

---

## Motivation

Autoregressive generation is inherently sequential — each token requires one full forward pass through the target model. Speculative decoding breaks this bottleneck:

- The tiny draft model runs many times faster than the target
- The target model can process K tokens in a single parallel forward pass (same cost as 1 token for the KV cache portion)
- If the draft is often correct (high acceptance rate), wall-clock throughput increases 2–3x
- Mathematical guarantee: the output distribution is identical to pure target sampling (no quality loss)

For a 125M parameter target and a 5M parameter draft, the draft runs ~25x faster per token. Even at 70% acceptance rate, the net speedup is substantial.

---

## Architecture / Design

### Draft Model Architecture

```
Draft model: ~5M params
- Embedding: vocab_size × 128
- 2 transformer layers (same GQA+RoPE+SwiGLU+RMSNorm structure)
- n_heads: 4, n_kv_heads: 2, d_model: 128, d_ffn: 384
- Shared vocabulary with target model
- Language model head (unembedding)
```

The draft model uses the same tokenizer as the target. Shared embeddings can optionally be tied to reduce parameter count.

### Speculative Decoding Algorithm

```python
# Pseudocode (see implementation below for full version)
def speculative_generate(prompt, K=5):
    generated = []
    while not done:
        # Step 1: Draft K tokens
        draft_tokens = draft_model.generate_K(context, K)
        draft_probs = draft_model.get_probs(context, draft_tokens)

        # Step 2: Target verifies all K tokens in ONE forward pass
        target_probs = target_model.forward(context + draft_tokens)

        # Step 3: Accept/reject each draft token
        accepted = []
        for i in range(K):
            acceptance_prob = min(1, target_probs[i][draft_tokens[i]] /
                                       draft_probs[i][draft_tokens[i]])
            if random() < acceptance_prob:
                accepted.append(draft_tokens[i])
            else:
                # Rejection: sample from corrected distribution
                corrected = max(0, target_probs[i] - draft_probs[i])
                corrected /= corrected.sum()
                accepted.append(sample(corrected))
                break  # Stop at first rejection

        generated.extend(accepted)
        context = context + accepted
    return generated
```

---

## Implementation Steps

### Step 1: Draft Model Config and Architecture

```python
# cola_coder/speculative/draft_model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
from cola_coder.model.attention import GroupedQueryAttention
from cola_coder.model.feedforward import SwiGLUFFN
from cola_coder.model.rope import precompute_freqs_cis

@dataclass
class DraftConfig:
    vocab_size: int = 32000
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 2
    d_ffn: int = 384
    max_seq_len: int = 2048
    dropout: float = 0.0


class DraftModel(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            DraftBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie embeddings for parameter efficiency
        self.lm_head.weight = self.embedding.weight
        freqs = precompute_freqs_cis(cfg.d_model // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("freqs_cis", freqs)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        freqs = self.freqs_cis[:T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        for block in self.blocks:
            x = block(x, freqs, mask)
        x = self.norm(x)
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DraftBlock(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x, freqs_cis, mask):
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

### Step 2: Speculative Decoder

```python
# cola_coder/speculative/decoder.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpeculativeStats:
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_target_passes: int = 0
    draft_generations: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup vs pure autoregressive."""
        alpha = self.acceptance_rate
        K = self.total_draft_tokens / max(1, self.draft_generations)
        # Expected tokens per target pass = (1 - alpha^(K+1)) / (1 - alpha)
        if alpha >= 1.0:
            return float(K + 1)
        expected_accepted = (1 - alpha ** (K + 1)) / (1 - alpha)
        return expected_accepted


class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: "DraftModel",
        target_model,
        tokenizer,
        K: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.95,
        device: str = "cuda",
    ):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
        self.K = K
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.stats = SpeculativeStats()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> list[int]:
        """
        Generate max_new_tokens using speculative decoding.
        Returns list of generated token IDs.
        """
        input_ids = input_ids.to(self.device)
        context = input_ids.clone()
        generated = []
        eos_id = self.tokenizer.eos_token_id

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(self.K, remaining)

            # === Draft Phase ===
            draft_tokens, draft_probs = self._draft_K_tokens(context, k)

            # === Verify Phase ===
            # Target processes context + all K draft tokens in ONE forward pass
            full_seq = torch.cat([context, draft_tokens.unsqueeze(0)], dim=1)
            target_logits = self.target(full_seq)
            # Extract logits at positions len(context)-1 to len(context)+k-1
            # (these correspond to predicting the K draft tokens)
            target_logits_k = target_logits[
                0,
                context.shape[1] - 1 : context.shape[1] + k - 1,
                :
            ]  # [k, vocab_size]
            target_probs = self._get_probs(target_logits_k)  # [k, vocab_size]

            self.stats.total_target_passes += 1
            self.stats.total_draft_tokens += k
            self.stats.draft_generations += 1

            # === Accept/Reject Phase ===
            accepted_tokens = []
            rejection_idx = k  # Assume all accepted unless rejected earlier

            for i in range(k):
                token_id = draft_tokens[i].item()
                p_target = target_probs[i, token_id].item()
                p_draft = draft_probs[i, token_id].item()

                acceptance_prob = min(1.0, p_target / (p_draft + 1e-10))

                if torch.rand(1).item() < acceptance_prob:
                    accepted_tokens.append(token_id)
                    self.stats.total_accepted_tokens += 1
                else:
                    # Rejection: sample from corrected residual distribution
                    residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                    total = residual.sum()
                    if total > 1e-8:
                        residual /= total
                        corrected_token = torch.multinomial(residual, 1).item()
                    else:
                        corrected_token = torch.multinomial(target_probs[i], 1).item()
                    accepted_tokens.append(corrected_token)
                    self.stats.total_rejected_tokens += 1
                    rejection_idx = i
                    break

            # Add one bonus token from target at the accept/reject boundary
            # (standard speculative decoding bonus)
            if rejection_idx == k:
                # All K accepted: sample one more from target
                bonus_logits = target_logits[0, context.shape[1] + k - 1, :]
                bonus_probs = self._get_probs(bonus_logits.unsqueeze(0)).squeeze(0)
                bonus_token = torch.multinomial(bonus_probs, 1).item()
                accepted_tokens.append(bonus_token)

            generated.extend(accepted_tokens)
            new_tokens = torch.tensor(
                accepted_tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            context = torch.cat([context, new_tokens], dim=1)

            if eos_id in accepted_tokens:
                break

        return generated

    def _draft_K_tokens(
        self,
        context: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Auto-regressively generate K tokens with the draft model."""
        draft_context = context.clone()
        draft_tokens = []
        all_probs = []

        for _ in range(k):
            logits = self.draft(draft_context)
            next_logits = logits[0, -1, :] / self.temperature
            probs = self._get_probs(next_logits.unsqueeze(0)).squeeze(0)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token.squeeze(0))
            all_probs.append(probs)
            draft_context = torch.cat(
                [draft_context, next_token.unsqueeze(0)], dim=1
            )

        return torch.stack(draft_tokens), torch.stack(all_probs)

    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and top-p to get valid probability distribution."""
        scaled = logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumulative - sorted_probs > self.top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Scatter back
            probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
        return probs
```

### Step 3: Draft Model Distillation Training

```python
# cola_coder/speculative/distillation.py
import torch
import torch.nn.functional as F

def distillation_loss(
    student_logits: torch.Tensor,  # [B, T, V]
    teacher_logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,          # [B, T]
    temperature: float = 2.0,
    alpha: float = 0.5,            # Mix of hard labels + soft distillation
) -> torch.Tensor:
    """
    KL divergence distillation + cross-entropy on hard labels.
    temperature: softens teacher/student distributions for distillation
    alpha: weight on hard label loss (1-alpha goes to distillation)
    """
    # Hard label loss
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # Soft distillation loss (KL divergence from teacher)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(
        student_soft.reshape(-1, student_soft.size(-1)),
        teacher_soft.reshape(-1, teacher_soft.size(-1)),
        reduction="batchmean",
    ) * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss
```

### Step 4: CLI Integration

```python
@app.command()
def generate_speculative(
    prompt: str = typer.Argument(...),
    draft_checkpoint: str = typer.Option(None, "--draft-ckpt"),
    target_checkpoint: str = typer.Option(None, "--target-ckpt"),
    k: int = typer.Option(5, "--k", help="Draft tokens per step"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    show_stats: bool = typer.Option(False, "--stats"),
):
    """Generate with speculative decoding (2-3x faster)."""
    decoder = build_speculative_decoder(draft_checkpoint, target_checkpoint, K=k)
    tokens = decoder.generate(tokenize(prompt), max_new_tokens=max_tokens)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    console.print(text)
    if show_stats:
        s = decoder.stats
        console.print(f"\n[dim]Acceptance rate: {s.acceptance_rate:.1%}[/dim]")
        console.print(f"[dim]Estimated speedup: {s.speedup_estimate:.2f}x[/dim]")
        console.print(f"[dim]Target forward passes: {s.total_target_passes}[/dim]")


@app.command()
def train_draft_model(
    target_checkpoint: str = typer.Argument(...),
    data_path: str = typer.Option("data/train.jsonl"),
    output: str = typer.Option("checkpoints/draft-v1.safetensors"),
    epochs: int = typer.Option(3),
    temperature: float = typer.Option(2.0, "--distill-temp"),
):
    """Train a draft model via distillation from the target model."""
    # ... training loop using distillation_loss
```

---

## Key Files to Modify

- `cola_coder/speculative/__init__.py` — new package
- `cola_coder/speculative/draft_model.py` — DraftModel architecture
- `cola_coder/speculative/decoder.py` — SpeculativeDecoder
- `cola_coder/speculative/distillation.py` — distillation training loss
- `cola_coder/cli.py` — `generate-speculative`, `train-draft-model` commands
- `cola_coder/generate.py` — integrate as optional generation path
- `configs/speculative.yaml` — K, temperature, draft checkpoint path

---

## Testing Strategy

```python
def test_speculative_output_distribution_matches_target():
    """Key correctness test: distribution must match pure target sampling."""
    # Use toy models with known distributions
    # Run 1000 samples, compare empirical distribution to target
    # This is a statistical test (KL divergence should be ~0)
    pass  # Requires probabilistic testing framework

def test_draft_generates_K_tokens():
    cfg = DraftConfig(vocab_size=100, d_model=32, n_layers=2)
    model = DraftModel(cfg)
    input_ids = torch.randint(0, 100, (1, 10))
    tokens, probs = decoder._draft_K_tokens(input_ids, K=5)
    assert tokens.shape == (5,)
    assert probs.shape == (5, 100)

def test_speculative_stats_tracking():
    # Run decoder for N steps, verify stats are consistent
    assert decoder.stats.total_accepted_tokens <= decoder.stats.total_draft_tokens

def test_draft_model_param_count():
    cfg = DraftConfig()
    model = DraftModel(cfg)
    assert model.param_count() < 10_000_000  # < 10M
```

---

## Performance Considerations

- **K value tuning:** K=5 is a good default. Increase K if acceptance rate is high (>80%), decrease if low (<60%). The optimal K balances draft cost vs verification batch size.
- **Draft model warmup:** The draft model needs to see the same prompt context as the target. Use KV caching in the draft model to avoid re-processing the prompt on each of the K draft steps.
- **KV cache for both models:** Both draft and target need KV caches for efficient incremental generation. This roughly doubles KV cache VRAM usage.
- **Acceptance rate monitoring:** Log acceptance rate per token position. If it drops sharply at certain positions (e.g., after function signatures), these are high-variance positions where the draft model struggles.
- **Batch size 1:** Speculative decoding is primarily beneficial for batch_size=1 (interactive generation). For large batch inference, standard autoregressive is already efficiently parallelized.

---

## Dependencies

- Existing Cola-Coder model components (GQA, RoPE, SwiGLU, RMSNorm)
- Existing tokenizer
- `safetensors` for draft model checkpoint

---

## Estimated Complexity

| Task                              | Effort   |
|-----------------------------------|----------|
| DraftModel architecture           | 2h       |
| SpeculativeDecoder core algorithm | 5h       |
| Accept/reject + correction math   | 2h       |
| Stats tracking                    | 1h       |
| Distillation training             | 3h       |
| CLI integration                   | 1.5h     |
| Tests + verification              | 3h       |
| **Total**                         | **~17.5h** |

Overall complexity: **High** (mathematical correctness is critical, debugging requires careful probability tracking)

---

## 2026 Best Practices

- **Correctness first:** The accept/reject criterion MUST produce samples from the target distribution. Bugs here produce subtly biased output that is hard to detect. Add distribution matching tests.
- **Medusa heads alternative:** Instead of a separate draft model, add multiple prediction heads to the target model itself (Feature 27). Simpler to train, slightly lower speedup, but no separate model to manage.
- **Draft model sharing:** If using multiple specialist models, train one draft model per specialist checkpoint (or one universal draft for all). The draft must match the target it's verifying.
- **Fallback to autoregressive:** If speculative decoder encounters a bug or the draft produces invalid tokens, fall back gracefully to standard autoregressive generation. Never crash.
- **Benchmark before deploying:** Measure actual wall-clock speedup on your hardware. Theoretical 2-3x assumes GPU-bound computation; on CPU or with very short prompts, overhead dominates.
- **SpecTr / REST extensions:** Newer variants (REST: Retrieval-based Speculative Decoding) use retrieved text as draft tokens instead of a draft model. Consider for future iteration.
