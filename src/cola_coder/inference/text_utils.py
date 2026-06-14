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


def trim_suffix_overlap(infill: str, suffix: str, min_overlap: int = 3) -> str:
    """Trim a FIM infill where its tail re-generates the start of the suffix.

    FIM models over-generate, frequently reproducing the document SUFFIX they were
    already given as context (arXiv:2509.24637, 2605.22981) — e.g. regenerating a
    closing ``\\n  return x;\\n}`` that already follows the cursor. In an editor that
    renders as duplicated code after the accepted completion. Since the suffix is
    fixed context the model must NOT reproduce, any verbatim overlap between the END
    of ``infill`` and the START of ``suffix`` is spurious and is cut here.

    Removes the LONGEST such overlap (so a short coincidental match doesn't hide a
    larger real one), but only when it is at least ``min_overlap`` characters — a
    1-2 char coincidence (a lone ``;`` or ``}`` the completion legitimately ends on,
    which also opens the suffix) is left alone.

    Args:
        infill: The generated middle (prompt already stripped).
        suffix: The document text after the cursor (FimRequest.suffix).
        min_overlap: Smallest overlap (chars) worth trimming.

    Returns:
        ``infill`` with any trailing suffix-duplication removed.
    """
    if not infill or not suffix:
        return infill
    max_k = min(len(infill), len(suffix))
    for k in range(max_k, min_overlap - 1, -1):
        if infill[-k:] == suffix[:k]:
            return infill[:-k]
    return infill
