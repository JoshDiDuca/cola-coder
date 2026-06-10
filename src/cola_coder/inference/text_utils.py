"""Shared text helpers for inference/serving."""

from __future__ import annotations


def strip_prompt_prefix(text: str, prompt: str) -> str:
    """Return only the generated continuation, stripping the prompt prefix.

    ``CodeGenerator.generate`` returns ``decode(prompt_tokens + new_tokens)``,
    i.e. the full text including a rendering of the prompt. We must return only
    the completion to clients.

    A naive ``text[len(prompt):] if text.startswith(prompt)`` is fragile: BPE
    ``decode(encode(prompt))`` is not guaranteed byte-identical to ``prompt``
    (BOS rendering, whitespace normalization, cross-boundary token merges), and
    when ``startswith`` is False the naive code returns the WHOLE prompt echo to
    the user — a real leak. This strips the longest common prefix instead, so a
    mismatch costs at most a few boundary characters, never the entire prompt.

    Args:
        text: Full decoded text (prompt rendering + completion).
        prompt: The original prompt string.

    Returns:
        The generated continuation.
    """
    if text.startswith(prompt):
        return text[len(prompt):]
    common = 0
    for a, b in zip(text, prompt):
        if a != b:
            break
        common += 1
    return text[common:]
