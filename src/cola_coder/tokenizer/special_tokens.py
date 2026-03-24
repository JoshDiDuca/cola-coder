"""Context special tokens for documentation and repository context.

Tokens:
  <|doc|> / <|/doc|>   — documentation context block
  <|repo|> / <|/repo|> — repository context block
  <|file|> / <|/file|> — file path + content marker

These tokens allow the model to distinguish inline documentation context
from regular code and reasoning text — analogous to how <think>...</think>
wraps reasoning steps (see reasoning/thinking_tokens.py).

Usage:
    from cola_coder.tokenizer.special_tokens import add_context_tokens, CONTEXT_TOKENS

    new_vocab_size = add_context_tokens(tokenizer)
"""

from .tokenizer_utils import CodeTokenizer


CONTEXT_TOKENS: list[str] = [
    "<|doc|>",
    "<|/doc|>",
    "<|repo|>",
    "<|/repo|>",
    "<|file|>",
    "<|/file|>",
]

# Feature flag — mirrors the FEATURE_ENABLED pattern used across features/
FEATURE_ENABLED: bool = True


def is_enabled() -> bool:
    """Return True when context special tokens are active.

    Reads the global FEATURE_ENABLED flag.  Override at module level or via
    the feature config to disable without changing call sites.
    """
    return FEATURE_ENABLED


def add_context_tokens(tokenizer: CodeTokenizer) -> int:
    """Add context special tokens to the tokenizer vocabulary.

    Follows the same pattern as add_thinking_tokens() in
    reasoning/thinking_tokens.py — delegates to tokenizer.add_special_tokens()
    so the underlying HuggingFace BPE tokenizer registers each token as a
    non-splittable unit.

    Note: Unlike add_thinking_tokens(), this function does *not* resize model
    embeddings.  That step belongs to the caller so that it can batch-resize
    after adding both thinking tokens and context tokens in one pass.

    Args:
        tokenizer: The BPE tokenizer to extend (CodeTokenizer wrapper).

    Returns:
        New vocabulary size after adding the tokens.
    """
    tokenizer.add_special_tokens(CONTEXT_TOKENS)
    new_vocab_size = tokenizer.vocab_size
    return new_vocab_size


def get_token_ids(tokenizer: CodeTokenizer) -> dict[str, int]:
    """Return a mapping of context token string -> token ID.

    Useful for injecting context tokens at inference time or for building
    special-token masks during training.

    Args:
        tokenizer: A CodeTokenizer that already has context tokens added.

    Returns:
        Dict mapping each token string to its integer ID.
        Tokens not found in the vocabulary map to -1.
    """
    result: dict[str, int] = {}
    for token in CONTEXT_TOKENS:
        tid = tokenizer.tokenizer.token_to_id(token)
        result[token] = tid if tid is not None else -1
    return result


def wrap_doc(content: str) -> str:
    """Wrap content in documentation context delimiters.

    Args:
        content: Raw documentation text.

    Returns:
        String delimited by <|doc|> ... <|/doc|>.
    """
    return f"<|doc|>{content}<|/doc|>"


def wrap_repo(content: str) -> str:
    """Wrap content in repository context delimiters.

    Args:
        content: Repository-level context (e.g. README, package.json summary).

    Returns:
        String delimited by <|repo|> ... <|/repo|>.
    """
    return f"<|repo|>{content}<|/repo|>"


def wrap_file(path: str, content: str) -> str:
    """Wrap a file path and its content in file context delimiters.

    Args:
        path:    Relative file path (e.g. "src/utils/format.ts").
        content: Raw file content.

    Returns:
        String delimited by <|file|> ... <|/file|> with the path on the
        first line so the model can associate content with its location.
    """
    return f"<|file|>{path}\n{content}<|/file|>"


# ---------------------------------------------------------------------------
# ChatML tokens (industry standard from Qwen/Mistral)
# ---------------------------------------------------------------------------

CHATML_TOKENS: list[str] = [
    "<|im_start|>",  # Message start marker
    "<|im_end|>",    # Message end marker
]

# Tool calling tokens
TOOL_TOKENS: list[str] = [
    "<tool_call>",   # Tool call start
    "</tool_call>",  # Tool call end
]


def add_chatml_tokens(tokenizer: CodeTokenizer) -> int:
    """Add ChatML special tokens to the tokenizer.

    Registers <|im_start|>, <|im_end|>, <tool_call>, and </tool_call> as
    non-splittable special tokens in the HuggingFace BPE tokenizer.

    Note: Callers must resize model embeddings separately after calling this
    (same pattern as add_context_tokens — batch-resize after all additions).

    Args:
        tokenizer: The BPE tokenizer to extend (CodeTokenizer wrapper).

    Returns:
        New vocabulary size after adding the tokens.
    """
    tokenizer.add_special_tokens(CHATML_TOKENS + TOOL_TOKENS)
    return tokenizer.vocab_size


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Format messages into a ChatML string.

    Each message dict must have "role" and "content" keys.  Roles are
    typically "system", "user", "assistant", or "tool".

    Output format (one block per message)::

        <|im_start|>system
        You are a code assistant.
        <|im_end|>
        <|im_start|>user
        Write a function...
        <|im_end|>
        <|im_start|>assistant

    The trailing ``<|im_start|>assistant\\n`` is appended when the last
    message is not from the assistant, acting as a generation prompt.

    Args:
        messages: List of {"role": str, "content": str} dicts.

    Returns:
        Formatted ChatML string ready for tokenization.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    # Add final assistant prompt so the model generates from here
    if messages and messages[-1]["role"] != "assistant":
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def get_chatml_token_ids(tokenizer: CodeTokenizer) -> dict[str, int]:
    """Get ChatML token IDs from the tokenizer.

    Useful for building attention masks or special-token filters at
    inference / training time.

    Args:
        tokenizer: A CodeTokenizer that already has ChatML tokens added.

    Returns:
        Dict mapping each ChatML/tool token string to its integer ID.
        Tokens absent from the vocabulary map to -1.
    """
    ids: dict[str, int] = {}
    for token in CHATML_TOKENS + TOOL_TOKENS:
        tid = tokenizer.tokenizer.token_to_id(token)
        ids[token] = tid if tid is not None else -1
    return ids
