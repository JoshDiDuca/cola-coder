"""Rejection-sampling Fine-Tuning (RFT) / self-verified distillation harness (MODEL-045).

2026 self-improvement recipe (RFT; arXiv:2605.10674, Self-Verified Distillation
2605.26132): the model generates its OWN candidate solutions, an EXTERNAL verifier
keeps only the ones that objectively pass, and you SFT on those — no teacher, no
human labels. cola-coder owns every piece (best-of-N generation + sandbox tsc/tests
verifier + the SFT trainer); this connects them into the loop.

Unlike `generate_distillation_dataset` (which distills an external TEACHER's output),
this distills the STUDENT's own VERIFIED output — self-distillation. It reuses the
battle-tested `generate_best_of_n` so every kept sample is verifier-passed AND
security-screened (it never trains the model on insecure code), and the best-of-N
self-consistency ranking already picks the strongest candidate per prompt.

SECURITY: like generate.py this NEVER executes model output itself — all untrusted
execution happens inside `generate_best_of_n`'s sandboxed verifier (TscRunner /
SandboxedRunner). The injected ``tsc_runner`` / ``execute_fn`` are the same sandbox
hooks the rest of the pipeline uses.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..inference.best_of_n import generate_best_of_n
from .generate import _to_messages


def generate_rft_dataset(
    generator,
    prompts: Sequence[str],
    *,
    num_candidates: int = 4,
    language: str = "auto",
    tests: Sequence[str | None] | None = None,
    keep_only_verified: bool = True,
    require_secure: bool = True,
    system: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    tsc_runner=None,
    execute_fn=None,
) -> tuple[list[dict], dict]:
    """Generate self-verified SFT records via best-of-N rejection sampling.

    For each prompt: generate ``num_candidates`` completions, verify+rank them with
    `generate_best_of_n`, and keep the best one ONLY if it passed the hard verifier
    (when ``keep_only_verified``) and is secure (when ``require_secure``). Emits
    ChatML ``{"messages": [...]}`` records ready for ``scripts/train_sft.py``.

    Args:
        generator: a CodeGenerator (or wrapper) with .generate / .generate_group.
        prompts: the problems to self-solve.
        tests: optional per-prompt test code (same length as prompts) for the
            strongest (execution-based) verification; None entries fall back to
            tsc / syntax checks.
        keep_only_verified: drop a prompt whose best candidate didn't pass (default).
        require_secure: drop a prompt whose best candidate trips the danger scanner.

    Returns:
        ``(records, stats)`` — records are ChatML dicts; stats summarises the run.
    """
    if tests is not None and len(tests) != len(prompts):
        raise ValueError("tests must be None or the same length as prompts")

    records: list[dict] = []
    stats = {
        "prompts": len(prompts),
        "verified": 0,
        "rejected_unverified": 0,
        "rejected_insecure": 0,
        "kept": 0,
    }

    for i, prompt in enumerate(prompts):
        test_code = tests[i] if tests is not None else None
        result = generate_best_of_n(
            generator,
            prompt,
            num_candidates=num_candidates,
            language=language,
            tests=test_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tsc_runner=tsc_runner,
            execute_fn=execute_fn,
        )
        best = result.best
        if best.verified:
            stats["verified"] += 1
        elif keep_only_verified:
            stats["rejected_unverified"] += 1
            continue
        if require_secure and not best.details.get("secure", True):
            stats["rejected_insecure"] += 1
            continue

        messages = _to_messages(prompt, system)
        records.append(
            {"messages": [*messages, {"role": "assistant", "content": best.completion}]}
        )
        stats["kept"] += 1

    return records, stats
