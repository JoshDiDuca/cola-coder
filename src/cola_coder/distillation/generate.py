"""Generate distillation data from a teacher (the Generate→Filter half of MODEL-024).

Runs a prompt set through a :class:`~cola_coder.distillation.teacher.Teacher`
(local Qwen/DeepSeek via Ollama, or cloud), optionally REJECTION-SAMPLES the
completions through a verifier, and emits ChatML ``{"messages": [...]}`` records
that ``scripts/train_sft.py`` consumes directly.

This mirrors the 2026 synthetic-data pipeline (Generate → Critique/Filter → keep):
the teacher generates, the verifier filters, only verified samples are kept.

SECURITY (SEC-014): teacher output is UNTRUSTED code. This module never executes
it — it calls an injected ``verify`` callable, which the CLI wires to the
SANDBOXED verifier (TscScorer / SandboxedRunner). The core stays execution-free and
trivially testable; all untrusted execution happens behind the sandbox.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..security.code_patterns import scan_dangerous
from .teacher import Teacher, TeacherError

# A prompt is either a raw user string or a ready ChatML message list.
Prompt = str | list[dict[str, str]]


def _to_messages(prompt: Prompt, system: str | None) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs
    # Already a message list — prepend system only if not already present.
    if system and not any(m.get("role") == "system" for m in prompt):
        return [{"role": "system", "content": system}, *prompt]
    return list(prompt)


def generate_distillation_dataset(
    teacher: Teacher,
    prompts: Sequence[Prompt],
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system: str | None = None,
    verify: Callable[[str], bool] | None = None,
    keep_only_verified: bool = True,
    screen_security: bool = True,
) -> tuple[list[dict], dict]:
    """Generate (and optionally verify) distillation records.

    Args:
        teacher: any Teacher (text completion in, out).
        prompts: user strings or ChatML message lists.
        verify: optional callable taking the teacher's completion and returning
            True if it passes (e.g. tsc/tests). MUST run untrusted code only in a
            sandbox — this function never executes anything itself. None = no
            functional verification (all functionally kept).
        keep_only_verified: when True and ``verify`` is given, drop completions
            that fail verification (rejection sampling).
        screen_security: when True (default), statically screen every teacher
            completion with the canonical ``scan_dangerous`` scanner and DROP any
            that trip a dangerous pattern — BEFORE functional verification and
            regardless of ``keep_only_verified``. A teacher routinely emits
            working-but-insecure code (the secure-pass@k gap, SEC-018/EVAL-024);
            distilling it teaches the student to write vulnerabilities. This is a
            static (no-execution) defence-in-depth gate that does not rely on the
            caller wiring security into ``verify``. Set False to keep raw output.

    Returns:
        ``(records, stats)`` where records are ChatML ``{"messages": [...]}`` dicts
        ready for train_sft.py, and stats summarises the run.
    """
    records: list[dict] = []
    stats = {
        "prompts": len(prompts),
        "teacher_ok": 0,
        "teacher_errors": 0,
        "rejected_insecure": 0,
        "verified": 0,
        "rejected": 0,
        "kept": 0,
    }
    for prompt in prompts:
        messages = _to_messages(prompt, system)
        try:
            completion = teacher.complete(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        except TeacherError:
            stats["teacher_errors"] += 1
            continue
        if not completion or not completion.strip():
            stats["teacher_errors"] += 1
            continue
        stats["teacher_ok"] += 1

        # Security gate (static, no execution): never distill dangerous code,
        # even if it would pass functional verification. Independent of
        # keep_only_verified — insecure output is never a desirable target.
        if screen_security and scan_dangerous(completion):
            stats["rejected_insecure"] += 1
            continue

        if verify is not None:
            passed = bool(verify(completion))
            if passed:
                stats["verified"] += 1
            else:
                stats["rejected"] += 1
                if keep_only_verified:
                    continue

        records.append(
            {"messages": [*messages, {"role": "assistant", "content": completion}]}
        )
        stats["kept"] += 1

    return records, stats
