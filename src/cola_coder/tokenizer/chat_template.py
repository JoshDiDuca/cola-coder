"""ChatML format support for instruction tuning.

Implements the ChatML template used by models like GPT-4, Qwen, and
Mistral-Instruct. Each message is wrapped in <|im_start|>role / <|im_end|>
delimiters, making role boundaries unambiguous to the model.

Format example:
    <|im_start|>system
    You are a helpful coding assistant.<|im_end|>
    <|im_start|>user
    Write a fibonacci function in Python.<|im_end|>
    <|im_start|>assistant
    def fibonacci(n):
        ...<|im_end|>

For a TS dev: ChatML is like a tagged template literal that wraps each
message in role markers — similar to how JSX wraps components in open/close
tags so the parser knows where one component ends and another begins.

Usage:
    from cola_coder.tokenizer.chat_template import (
        add_chat_tokens, format_chat, format_chat_training, parse_chat,
    )
"""

from __future__ import annotations

import re

from ..model.transformer import Transformer
from .tokenizer_utils import CodeTokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

CHAT_TOKENS: list[str] = [IM_START, IM_END]


def add_chat_tokens(
    tokenizer: CodeTokenizer,
    model: Transformer,
) -> tuple[int, int]:
    """Add ChatML tokens to the tokenizer and resize model embeddings.

    Follows the same pattern as ``add_thinking_tokens`` in
    ``reasoning/thinking_tokens.py`` — extends the vocabulary and then
    resizes the embedding / output projection layers so the new token IDs
    map to learnable vectors.

    Args:
        tokenizer: The BPE tokenizer to extend.
        model: The transformer model whose embedding layer needs resizing.

    Returns:
        (im_start_id, im_end_id) — the token IDs for the new tokens.
    """
    from cola_coder.cli import cli

    tokenizer.add_special_tokens(CHAT_TOKENS)
    new_vocab_size = tokenizer.vocab_size

    # Resize model embeddings to accommodate new tokens
    _resize_embeddings(model, new_vocab_size)

    im_start_id = tokenizer.tokenizer.token_to_id(IM_START)
    im_end_id = tokenizer.tokenizer.token_to_id(IM_END)

    cli.info(
        "Chat tokens",
        f"{IM_START} (id={im_start_id}), {IM_END} (id={im_end_id})",
    )
    cli.info("New vocab size", str(new_vocab_size))

    return im_start_id, im_end_id


def _resize_embeddings(model: Transformer, new_vocab_size: int) -> None:
    """Resize the model's embedding and output layers for new vocab size.

    New rows are initialised with small random values (std=0.02).
    Weight tying between ``tok_emb`` and ``output`` is preserved.
    """
    import torch

    old_vocab_size = model.config.vocab_size
    if new_vocab_size <= old_vocab_size:
        return

    old_emb = model.tok_emb
    new_emb = torch.nn.Embedding(new_vocab_size, model.config.dim)

    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = old_emb.weight
        torch.nn.init.normal_(
            new_emb.weight[old_vocab_size:], mean=0.0, std=0.02
        )

    model.tok_emb = new_emb

    new_output = torch.nn.Linear(
        model.config.dim, new_vocab_size, bias=False
    )
    with torch.no_grad():
        new_output.weight[:old_vocab_size] = old_emb.weight
        new_output.weight[old_vocab_size:] = new_emb.weight[old_vocab_size:]

    model.output = new_output
    model.output.weight = model.tok_emb.weight  # Re-tie weights

    model.config.vocab_size = new_vocab_size


def format_chat(messages: list[dict[str, str]]) -> str:
    """Format a list of messages into a ChatML string.

    Args:
        messages: List of ``{"role": "...", "content": "..."}`` dicts.
            Typical roles: ``"system"``, ``"user"``, ``"assistant"``.

    Returns:
        A single ChatML-formatted string with all messages.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}")
    return "\n".join(parts)


def format_chat_training(
    messages: list[dict[str, str]],
) -> tuple[str, list[tuple[int, int]]]:
    """Format messages into ChatML and return assistant-content spans.

    This is the training variant of :func:`format_chat`.  In addition to
    the formatted string it returns the character-level (start, end) spans
    of every assistant response.  These spans are used downstream to build
    the loss mask: only tokens inside assistant responses contribute to the
    cross-entropy loss, while system/user tokens are masked with -100.

    Args:
        messages: List of ``{"role": "...", "content": "..."}`` dicts.

    Returns:
        A tuple of ``(formatted_text, assistant_spans)`` where each span
        is a ``(start, end)`` pair of character offsets into the formatted
        text covering the assistant's content (excluding the role line and
        ``<|im_end|>``).
    """
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    offset = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        segment = f"{IM_START}{role}\n{content}{IM_END}"

        if role == "assistant":
            # The content starts after "<|im_start|>assistant\n"
            content_start = offset + len(IM_START) + len(role) + 1  # +1 for \n
            content_end = content_start + len(content)
            assistant_spans.append((content_start, content_end))

        parts.append(segment)
        offset += len(segment) + 1  # +1 for the "\n" join separator

    text = "\n".join(parts)
    return text, assistant_spans


def parse_chat(text: str) -> list[dict[str, str]]:
    """Parse a ChatML-formatted string back into a list of messages.

    Args:
        text: ChatML-formatted string.

    Returns:
        List of ``{"role": "...", "content": "..."}`` dicts.
    """
    pattern = re.compile(
        re.escape(IM_START) + r"(\S+)\n(.*?)" + re.escape(IM_END),
        re.DOTALL,
    )
    messages: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        messages.append({
            "role": match.group(1),
            "content": match.group(2),
        })
    return messages


def has_chat_tokens(tokenizer: CodeTokenizer) -> bool:
    """Check whether the tokenizer already has ChatML tokens.

    Args:
        tokenizer: A CodeTokenizer instance.

    Returns:
        True if both ``<|im_start|>`` and ``<|im_end|>`` are in the vocab.
    """
    im_start_id = tokenizer.tokenizer.token_to_id(IM_START)
    im_end_id = tokenizer.tokenizer.token_to_id(IM_END)
    return im_start_id is not None and im_end_id is not None
