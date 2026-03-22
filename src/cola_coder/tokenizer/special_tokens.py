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
