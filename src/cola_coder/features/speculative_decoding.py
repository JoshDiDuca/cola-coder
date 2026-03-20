"""Speculative Decoding: use a small draft model to speed up generation.

The draft model proposes N tokens at once, then the target model verifies
them in a single forward pass. Accepted tokens skip individual forward passes,
giving 2-3x speedup for well-matched draft/target pairs.

Think of it like autocomplete suggestions — the small model guesses what the
big model would say, and the big model just checks "yes/no" on each guess.
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_tokens: int = 5  # How many tokens the draft model proposes
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 256
    # Acceptance threshold — reject draft token if prob ratio is too low
    acceptance_threshold: float = 0.0  # 0 = standard speculative sampling


@dataclass
class SpeculativeStats:
    """Track acceptance rates and speedup metrics."""
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    total_target_forwards: int = 0
    total_draft_forwards: int = 0
    wall_time: float = 0.0
    tokens_generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time == 0:
            return 0.0
        return self.tokens_generated / self.wall_time

    @property
    def estimated_speedup(self) -> float:
        """Estimate speedup vs autoregressive decoding."""
        if self.total_target_forwards == 0:
            return 1.0
        # Without speculation: 1 forward per token
        naive_forwards = self.tokens_generated
        # With speculation: draft_forwards + target_forwards
        actual_forwards = self.total_draft_forwards + self.total_target_forwards
        if actual_forwards == 0:
            return 1.0
        return naive_forwards / actual_forwards

    def summary(self) -> dict:
        return {
            "tokens_generated": self.tokens_generated,
            "acceptance_rate": f"{self.acceptance_rate:.1%}",
            "estimated_speedup": f"{self.estimated_speedup:.2f}x",
            "tokens_per_second": f"{self.tokens_per_second:.1f}",
            "draft_tokens_proposed": self.total_draft_tokens,
            "draft_tokens_accepted": self.accepted_tokens,
            "target_forward_passes": self.total_target_forwards,
        }


def verify_draft_tokens(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> int:
    """Verify draft tokens using speculative sampling.

    For each draft token, compare target vs draft probability.
    Accept with probability min(1, p_target / p_draft).
    On first rejection, also sample a correction token from the
    adjusted distribution.

    Args:
        target_logits: Shape (1, K, vocab) — target model logits for K draft positions.
        draft_logits: Shape (1, K, vocab) — draft model logits for K draft positions.
        draft_token_ids: Shape (K,) — the proposed token IDs.
        temperature: Sampling temperature.

    Returns:
        Number of accepted tokens (0 to K).
    """
    K = draft_token_ids.shape[0]
    if K == 0:
        return 0

    for i in range(K):
        # Get probabilities
        if temperature > 0:
            t_probs = F.softmax(target_logits[0, i] / temperature, dim=-1)
            d_probs = F.softmax(draft_logits[0, i] / temperature, dim=-1)
        else:
            # Greedy: accept only if argmax matches
            t_token = target_logits[0, i].argmax()
            if t_token == draft_token_ids[i]:
                continue
            else:
                return i

        token_id = draft_token_ids[i].item()
        p_target = t_probs[token_id].item()
        p_draft = d_probs[token_id].item()

        # Accept with probability min(1, p_target / p_draft)
        if p_draft == 0:
            return i
        acceptance_prob = min(1.0, p_target / p_draft)
        if torch.rand(1).item() < acceptance_prob:
            continue  # Accepted
        else:
            return i  # Rejected

    return K  # All accepted


def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> int:
    """Sample a single token from logits."""
    if temperature == 0:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[-1]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


class SpeculativeDecoder:
    """Speculative decoding with a draft and target model.

    Usage:
        decoder = SpeculativeDecoder(draft_model, target_model, config)
        tokens, stats = decoder.generate(prompt_ids)
    """

    def __init__(self, draft_model, target_model, config: SpeculativeConfig | None = None):
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or SpeculativeConfig()
        self.device = next(target_model.parameters()).device

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int | None = None,
    ) -> tuple[list[int], SpeculativeStats]:
        """Generate tokens using speculative decoding.

        Args:
            prompt_ids: Shape (1, seq_len) — tokenized prompt.
            max_tokens: Override config max_tokens.

        Returns:
            Tuple of (generated_token_ids, stats).
        """
        max_tokens = max_tokens or self.config.max_tokens
        cfg = self.config
        stats = SpeculativeStats()
        start_time = time.time()

        generated: list[int] = []
        current_ids = prompt_ids.to(self.device)

        self.draft_model.eval()
        self.target_model.eval()

        with torch.no_grad():
            while len(generated) < max_tokens:
                # Phase 1: Draft model proposes K tokens
                draft_ids = current_ids.clone()
                draft_tokens = []
                draft_logits_list = []

                for _ in range(cfg.draft_tokens):
                    d_logits = self.draft_model(draft_ids)
                    last_logits = d_logits[:, -1, :]
                    draft_logits_list.append(last_logits.unsqueeze(1))

                    token = sample_token(last_logits[0], cfg.temperature, cfg.top_k)
                    draft_tokens.append(token)
                    draft_ids = torch.cat([
                        draft_ids,
                        torch.tensor([[token]], device=self.device)
                    ], dim=1)

                stats.total_draft_forwards += cfg.draft_tokens

                if not draft_tokens:
                    break

                draft_token_ids = torch.tensor(draft_tokens, device=self.device)
                draft_logits = torch.cat(draft_logits_list, dim=1)

                # Phase 2: Target model verifies all K tokens in one pass
                verify_ids = torch.cat([
                    current_ids,
                    draft_token_ids.unsqueeze(0)
                ], dim=1)
                target_logits = self.target_model(verify_ids)
                stats.total_target_forwards += 1

                # Get target logits at the draft positions
                prompt_len = current_ids.shape[1]
                target_at_draft = target_logits[:, prompt_len - 1:prompt_len - 1 + cfg.draft_tokens, :]

                # Phase 3: Verify and accept/reject
                n_accepted = verify_draft_tokens(
                    target_at_draft, draft_logits, draft_token_ids, cfg.temperature
                )
                stats.total_draft_tokens += cfg.draft_tokens
                stats.accepted_tokens += n_accepted

                # Add accepted tokens
                for i in range(n_accepted):
                    generated.append(draft_tokens[i])
                    if len(generated) >= max_tokens:
                        break

                # Sample correction token from target if not all accepted
                if n_accepted < cfg.draft_tokens and len(generated) < max_tokens:
                    correction_logits = target_logits[0, prompt_len - 1 + n_accepted]
                    correction_token = sample_token(correction_logits, cfg.temperature, cfg.top_k)
                    generated.append(correction_token)

                # Update current_ids for next iteration
                new_tokens = generated[-(n_accepted + (1 if n_accepted < cfg.draft_tokens else 0)):]
                if new_tokens:
                    new_ids = torch.tensor([new_tokens], device=self.device)
                    current_ids = torch.cat([current_ids, new_ids], dim=1)

        stats.tokens_generated = len(generated)
        stats.wall_time = time.time() - start_time
        return generated, stats


def estimate_speedup(
    draft_time_per_token: float,
    target_time_per_token: float,
    acceptance_rate: float,
    draft_tokens: int = 5,
) -> float:
    """Estimate theoretical speedup from speculative decoding.

    Args:
        draft_time_per_token: Time for one draft model forward pass.
        target_time_per_token: Time for one target model forward pass.
        acceptance_rate: Expected fraction of draft tokens accepted (0-1).
        draft_tokens: Number of draft tokens per speculation round.

    Returns:
        Estimated speedup factor.
    """
    # Expected tokens per round: sum of geometric probabilities
    expected_accepted = sum(acceptance_rate ** i for i in range(1, draft_tokens + 1))

    # Time per round: K draft forwards + 1 target forward
    time_per_round = draft_tokens * draft_time_per_token + target_time_per_token

    # Tokens per round: accepted + 1 correction
    tokens_per_round = expected_accepted + 1

    # Baseline: 1 target forward per token
    baseline_time_per_token = target_time_per_token

    # Speculative time per token
    spec_time_per_token = time_per_round / tokens_per_round

    if spec_time_per_token == 0:
        return 1.0
    return baseline_time_per_token / spec_time_per_token
