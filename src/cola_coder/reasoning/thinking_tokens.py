"""Thinking tokens support for reasoning experiments.

This module adds <think> and </think> special tokens to the model vocabulary.
When the model generates text between these tokens, it's "thinking out loud"
before producing the final answer.

The idea (from DeepSeek-R1 and similar research):
1. Train the model on data where solutions are preceded by step-by-step reasoning
2. The model learns to generate <think>reasoning</think> before the actual code
3. This thinking process helps the model solve harder problems

For a TS dev: think of <think>...</think> like a scratchpad or comments that
the model writes before writing the actual code. During inference, you can
optionally strip these tokens from the output.
"""

import torch

from ..model.transformer import Transformer
from ..tokenizer.tokenizer_utils import CodeTokenizer


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def add_thinking_tokens(
    tokenizer: CodeTokenizer,
    model: Transformer,
) -> tuple[int, int]:
    """Add <think> and </think> tokens to the tokenizer and resize model embeddings.

    Args:
        tokenizer: The BPE tokenizer to extend.
        model: The transformer model whose embedding layer needs resizing.

    Returns:
        (think_open_id, think_close_id) — the token IDs for the new tokens.
    """
    # Add special tokens to tokenizer
    tokenizer.add_special_tokens([THINK_OPEN, THINK_CLOSE])
    new_vocab_size = tokenizer.vocab_size

    # Resize model embeddings to accommodate new tokens
    _resize_embeddings(model, new_vocab_size)

    think_open_id = tokenizer.tokenizer.token_to_id(THINK_OPEN)
    think_close_id = tokenizer.tokenizer.token_to_id(THINK_CLOSE)

    print(f"Added thinking tokens: {THINK_OPEN} (id={think_open_id}), "
          f"{THINK_CLOSE} (id={think_close_id})")
    print(f"New vocab size: {new_vocab_size}")

    return think_open_id, think_close_id


def _resize_embeddings(model: Transformer, new_vocab_size: int):
    """Resize the model's embedding and output layers for new vocab size.

    When we add new tokens, we need to add new rows to the embedding matrix.
    The new rows are initialized with small random values.
    """
    old_vocab_size = model.config.vocab_size
    if new_vocab_size <= old_vocab_size:
        return

    # Resize token embedding
    old_emb = model.tok_emb
    new_emb = torch.nn.Embedding(new_vocab_size, model.config.dim)

    # Copy old weights
    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = old_emb.weight
        # Initialize new token embeddings with small random values
        torch.nn.init.normal_(new_emb.weight[old_vocab_size:], mean=0.0, std=0.02)

    model.tok_emb = new_emb

    # Resize output projection (shares weights with embedding via weight tying)
    new_output = torch.nn.Linear(model.config.dim, new_vocab_size, bias=False)
    with torch.no_grad():
        new_output.weight[:old_vocab_size] = old_emb.weight  # Use old embedding weights
        new_output.weight[old_vocab_size:] = new_emb.weight[old_vocab_size:]

    model.output = new_output
    model.output.weight = model.tok_emb.weight  # Re-tie weights

    # Update config
    model.config.vocab_size = new_vocab_size


def format_thinking_example(thinking: str, code: str) -> str:
    """Format a training example with thinking tokens.

    Args:
        thinking: The step-by-step reasoning.
        code: The actual code solution.

    Returns:
        Formatted string: <think>reasoning</think>\ncode
    """
    return f"{THINK_OPEN}{thinking}{THINK_CLOSE}\n{code}"


def strip_thinking(text: str) -> str:
    """Remove thinking tokens and their content from generated text.

    Used during inference to get just the code output.

    Args:
        text: Generated text that may contain <think>...</think>.

    Returns:
        Text with thinking sections removed.
    """
    result = text
    while THINK_OPEN in result and THINK_CLOSE in result:
        start = result.index(THINK_OPEN)
        end = result.index(THINK_CLOSE) + len(THINK_CLOSE)
        result = result[:start] + result[end:]
    return result.strip()


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract the thinking and code portions from generated text.

    Args:
        text: Generated text containing <think>...</think>.

    Returns:
        (thinking, code) tuple. If no thinking tokens found,
        thinking is empty and code is the full text.
    """
    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return "", text

    start = text.index(THINK_OPEN) + len(THINK_OPEN)
    end = text.index(THINK_CLOSE)
    thinking = text[start:end].strip()
    code = text[end + len(THINK_CLOSE):].strip()

    return thinking, code
