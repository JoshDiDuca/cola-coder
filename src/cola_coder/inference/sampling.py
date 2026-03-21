"""Sampling strategies for text generation.

When the model outputs logits (raw scores for each token), we need to
decide which token to actually pick. This is called "sampling."

Different strategies give different results:
- Greedy (temperature=0): always pick the highest-probability token.
  Deterministic but boring/repetitive.
- Temperature: scale the logits before softmax. Higher temperature = more
  random, lower = more focused. Think of it like a confidence dial.
- Top-k: only consider the k most likely tokens. Prevents the model from
  picking extremely unlikely tokens.
- Top-p (nucleus): only consider tokens whose cumulative probability reaches p.
  Adaptive — includes more tokens when the model is uncertain, fewer when confident.
- Repetition penalty: reduce probability of tokens already generated.
  Prevents the "I am a model and I am a model and I am a model..." failure mode.
"""

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_ids: list[int] | None = None,
) -> int:
    """Sample the next token from model output logits.

    Args:
        logits: Raw model output scores, shape (vocab_size,).
                Higher values = model thinks this token is more likely.
        temperature: Controls randomness. 0 = greedy, 1 = default, >1 = more random.
        top_k: Only consider the top k tokens.
        top_p: Only consider tokens until cumulative probability reaches p.
        repetition_penalty: Penalize tokens that appeared before. >1 = penalize.
        generated_ids: List of previously generated token IDs (for repetition penalty).

    Returns:
        The selected token ID (integer).
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Temperature scaling
    if temperature == 0:
        # Greedy: just pick the max
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        logits = _top_k_filter(logits, top_k)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_next_tokens_batch(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample next tokens for a batch of sequences independently.

    Unlike sample_next_token (single sequence), this operates on a full batch
    in one vectorised call — no Python loop over sequences needed.

    Repetition penalty is intentionally omitted here: tracking per-sequence
    history for a batch would require passing ragged lists and would dominate
    the runtime.  Use generate_group (which handles this at the caller level)
    if you need rep-penalty with batched generation.

    Args:
        logits: Raw model output, shape (batch_size, vocab_size).
        temperature: Sampling temperature.  0 → greedy argmax per row.
        top_k: Keep only the top-k logits per row (0 = disabled).
        top_p: Nucleus filtering — keep tokens up to cumulative prob p.

    Returns:
        Tensor of shape (batch_size,) with one sampled token ID per row.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        logits = _top_k_filter_batch(logits, top_k)

    if top_p < 1.0:
        logits = _top_p_filter_batch(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    # torch.multinomial on a 2-D tensor samples independently per row
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _top_k_filter_batch(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k filter for a batch of logit rows.

    Args:
        logits: Shape (batch, vocab).
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with the same shape; values outside top-k → -inf.
    """
    vocab = logits.shape[-1]
    if k >= vocab:
        return logits

    # kth_value: per-row minimum among the top-k values
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(1)  # (batch, 1)
    logits = logits.masked_fill(logits < min_top_k, float("-inf"))
    return logits


def _top_p_filter_batch(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filter for a batch of logit rows.

    Args:
        logits: Shape (batch, vocab).
        p: Cumulative probability threshold.

    Returns:
        Filtered logits with the same shape; tokens outside the nucleus → -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Tokens whose cumulative prob is already above p can be removed,
    # but we always keep the first token (shift by 1).
    sorted_mask = cumulative_probs > p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    # Scatter the mask back to the original (unsorted) token order
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_mask
    )
    logits = logits.masked_fill(mask, float("-inf"))
    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
):
    """Reduce probability of previously generated tokens.

    For each token that has already appeared, divide its logit by the penalty
    if it's positive, or multiply if it's negative. This shifts its probability
    downward relative to other tokens.
    """
    # Get unique token IDs that have been generated
    unique_ids = list(set(generated_ids))
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top k highest logits, set the rest to -infinity.

    This prevents the model from ever picking a token outside the top k,
    regardless of temperature.
    """
    if k >= logits.shape[-1]:
        return logits

    # Find the k-th largest value
    top_k_values, _ = torch.topk(logits, k)
    min_top_k = top_k_values[-1]

    # Zero out everything below the threshold
    logits[logits < min_top_k] = float("-inf")
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep tokens until cumulative probability reaches p (nucleus sampling).

    This is adaptive: when the model is confident (one token has high probability),
    top-p naturally selects fewer tokens. When uncertain (many tokens with similar
    probabilities), it includes more options.

    Example with p=0.9:
    - Confident: token A has 0.85 prob → only A is kept (0.85 < 0.9, next token pushes over)
    - Uncertain: tokens A=0.2, B=0.2, C=0.15, D=0.15, E=0.1 → keep all of these (sum=0.8 < 0.9)
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative prob exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    # Shift: keep at least one token (the most probable one)
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    # Map back to original indices and set filtered logits to -inf
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float("-inf")

    return logits
