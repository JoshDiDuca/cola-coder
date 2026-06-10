"""Best-of-N generation with sandboxed verification.

Inference-time compute scaling: generate N candidate completions for the SAME
prompt, verify each one with real tools, and return the best candidate.

For a TS dev: it's like running `tsc --noEmit` on N codegen outputs in CI and
shipping the one that compiles — except the "CI" is a sandbox that runs in a
few seconds, per request.

The generation step reuses CodeGenerator.generate_group (prefill once, expand
the KV-cache, decode all candidates in parallel), so N candidates cost far
less than N sequential generations. Wrapped generators without generate_group
(e.g. ContextAwareGenerator) transparently fall back to serial generation.

Verification is sandboxed end-to-end and language-aware:

- TypeScript → TscRunner (SandboxedRunner + hardened tsconfig, batched)
- Python + tests → evaluation.runner.execute_code (native/docker sandbox
  per configs/scoring.yaml security settings)
- Python without tests → compile() syntax check (static — never executes)
- No hard verifier available (e.g. tsc not installed) → SelfVerifier
  heuristics only

The final ranking key is (verified, score): a candidate that passed the hard
verifier always beats one that didn't; within a tier, score combines the hard
verifier signal with SelfVerifier's heuristic confidence as a tie-breaker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..data.scorers.language_detect import is_js_ts, is_typescript
from ..data.scorers.utils import ScoreMapper

logger = logging.getLogger(__name__)

# Ranking weights: the hard (tool-based) verdict dominates; heuristics only
# break ties between candidates the tools can't tell apart.
_HARD_WEIGHT = 0.8
_HEURISTIC_WEIGHT = 0.2

# tsc error count → 0.0-1.0 score (fewer errors = closer to compiling)
_TSC_ERROR_SCORE = ScoreMapper([(0, 1.0), (1, 0.7), (3, 0.5), (6, 0.3)], floor=0.1)

# A verifier verdict for one candidate: (passed, score_0_to_1, details)
_Verdict = tuple[bool, float, dict]

# Signature for an injectable Python execution function:
# (code, timeout) -> (success, output). Defaults to the sandboxed
# evaluation.runner.execute_code.
ExecuteFn = Callable[[str, float], tuple[bool, str]]


@dataclass
class CandidateResult:
    """One verified candidate."""

    text: str           # full decoded text (prompt + completion)
    completion: str     # generated part only (prompt stripped)
    verified: bool      # passed the hard verifier (tsc clean / tests pass / parses)
    score: float        # 0.0-1.0 ranking score (hard verdict + heuristic tie-break)
    details: dict = field(default_factory=dict)


@dataclass
class BestOfNResult:
    """Outcome of a best-of-N verified generation."""

    best: CandidateResult
    candidates: list[CandidateResult]   # all candidates, sorted best-first
    language: str                       # resolved language ("python"/"typescript")
    verifier: str                       # "tsc" | "python_exec" | "python_syntax" | "heuristic"


def detect_language(code: str) -> str:
    """Best-effort language detection for verifier routing.

    Routes JS/TS-looking code to tsc (which type-checks plain JS in a .ts
    file just fine); everything else defaults to Python. Checks both
    detectors: is_typescript catches type-annotation-heavy snippets that
    is_js_ts's generic keyword heuristic misses.
    """
    return "typescript" if (is_typescript(code) or is_js_ts(code)) else "python"


def generate_best_of_n(
    generator,
    prompt: str,
    *,
    num_candidates: int = 4,
    language: str = "auto",
    tests: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    timeout: float = 10.0,
    tsc_runner=None,
    execute_fn: ExecuteFn | None = None,
) -> BestOfNResult:
    """Generate N candidates, verify each in a sandbox, return the best.

    Args:
        generator: A CodeGenerator (or wrapper exposing .generate / .generate_group).
        prompt: Shared prompt for all candidates.
        num_candidates: How many candidates to generate (N).
        language: "auto" (detect from prompt), "python", or "typescript".
        tests: Optional Python test code appended to each candidate and
               executed in the sandbox — strongest verification signal.
        max_new_tokens / temperature / top_k / top_p / min_p: sampling params.
        timeout: Per-candidate verification timeout in seconds.
        tsc_runner: Injectable TscRunner (tests); default builds one if tsc exists.
        execute_fn: Injectable Python executor (tests); default is the
                    sandboxed evaluation.runner.execute_code.

    Returns:
        BestOfNResult with candidates sorted best-first.
    """
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")

    texts = _generate_candidates(
        generator,
        prompt,
        num_candidates=num_candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )

    lang = detect_language(prompt) if language == "auto" else language
    verifier_name, verdicts = _run_hard_verifier(
        lang, texts, tests=tests, timeout=timeout,
        tsc_runner=tsc_runner, execute_fn=execute_fn,
    )

    candidates: list[CandidateResult] = []
    for text, (verified, hard_score, details) in zip(texts, verdicts):
        heuristic = _heuristic_confidence(text)
        details["heuristic_confidence"] = round(heuristic, 3)
        candidates.append(
            CandidateResult(
                text=text,
                completion=_strip_prompt(text, prompt),
                verified=verified,
                score=_HARD_WEIGHT * hard_score + _HEURISTIC_WEIGHT * heuristic,
                details=details,
            )
        )

    ranked = sorted(candidates, key=lambda c: (c.verified, c.score), reverse=True)
    logger.info(
        "best-of-%d (%s via %s): %d/%d verified, best score %.3f",
        num_candidates, lang, verifier_name,
        sum(c.verified for c in ranked), len(ranked), ranked[0].score,
    )
    return BestOfNResult(
        best=ranked[0], candidates=ranked, language=lang, verifier=verifier_name
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _generate_candidates(
    generator,
    prompt: str,
    *,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
) -> list[str]:
    """Batched generation when available, serial fallback otherwise."""
    if hasattr(generator, "generate_group"):
        return generator.generate_group(
            prompt=prompt,
            num_completions=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
    logger.debug("generator has no generate_group — serial best-of-N fallback")
    return [
        generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        for _ in range(num_candidates)
    ]


def _strip_prompt(text: str, prompt: str) -> str:
    """Return only the generated part of a decoded candidate.

    BPE decode(encode(prompt)) is not always byte-identical to the prompt, so
    fall back to stripping the longest common prefix when the exact prefix
    doesn't match.
    """
    if text.startswith(prompt):
        return text[len(prompt):]
    common = 0
    for a, b in zip(text, prompt):
        if a != b:
            break
        common += 1
    return text[common:]


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


def _run_hard_verifier(
    language: str,
    texts: list[str],
    *,
    tests: str | None,
    timeout: float,
    tsc_runner,
    execute_fn: ExecuteFn | None,
) -> tuple[str, list[_Verdict]]:
    """Dispatch to the strongest available verifier for `language`."""
    if language == "typescript":
        verdicts = _verify_typescript(texts, tsc_runner=tsc_runner, timeout=timeout)
        if verdicts is not None:
            return "tsc", verdicts
        logger.warning("tsc unavailable — best-of-N falling back to heuristics")
        return "heuristic", _verify_heuristic(texts)

    if language == "python":
        if tests:
            return "python_exec", _verify_python_exec(
                texts, tests, execute_fn=execute_fn, timeout=timeout
            )
        return "python_syntax", _verify_python_syntax(texts)

    logger.warning("unknown language %r — best-of-N using heuristics", language)
    return "heuristic", _verify_heuristic(texts)


def _verify_typescript(
    texts: list[str], *, tsc_runner, timeout: float
) -> list[_Verdict] | None:
    """Type-check all candidates with tsc (sandboxed, batched).

    Returns None when tsc is not available so the caller can fall back.
    """
    if tsc_runner is None:
        from ..reasoning.rewards.tsc_runner import TscRunner

        if not TscRunner.is_available():
            return None
        tsc_runner = TscRunner(timeout=max(1, int(timeout)))

    errors_by_index = tsc_runner.check_batch(list(texts))
    verdicts: list[_Verdict] = []
    for i in range(len(texts)):
        errors = errors_by_index.get(i, [])
        verdicts.append(
            (
                len(errors) == 0,
                _TSC_ERROR_SCORE(len(errors)),
                {
                    "tsc_errors": len(errors),
                    "messages": [getattr(e, "message", str(e)) for e in errors[:3]],
                },
            )
        )
    return verdicts


def _verify_python_exec(
    texts: list[str],
    tests: str,
    *,
    execute_fn: ExecuteFn | None,
    timeout: float,
) -> list[_Verdict]:
    """Run candidate + tests in the sandbox; passing tests is the verdict."""
    execute = execute_fn or _default_execute
    verdicts: list[_Verdict] = []
    for text in texts:
        ok, output = execute(text + "\n\n" + tests, timeout)
        verdicts.append(
            (ok, 1.0 if ok else 0.0, {"tests_passed": ok, "output": output[-400:]})
        )
    return verdicts


def _default_execute(code: str, timeout: float) -> tuple[bool, str]:
    from ..evaluation.runner import execute_code

    return execute_code(code, timeout=timeout)


def _verify_python_syntax(texts: list[str]) -> list[_Verdict]:
    """Static syntax check — compiles to bytecode, never executes."""
    verdicts: list[_Verdict] = []
    for text in texts:
        try:
            compile(text, "<candidate>", "exec")
            verdicts.append((True, 1.0, {"syntax_ok": True}))
        except (SyntaxError, ValueError) as e:
            verdicts.append((False, 0.0, {"syntax_ok": False, "error": str(e)}))
    return verdicts


def _verify_heuristic(texts: list[str]) -> list[_Verdict]:
    """SelfVerifier-only verdicts when no tool-based verifier exists."""
    from ..features.self_verification import SelfVerifier

    verifier = SelfVerifier()
    verdicts: list[_Verdict] = []
    for text in texts:
        result = verifier.verify_code(text)
        verdicts.append(
            (result.passed, result.confidence, {"heuristic_only": True,
                                                "issues": result.issues[:3]})
        )
    return verdicts


def _heuristic_confidence(code: str) -> float:
    from ..features.self_verification import SelfVerifier

    return SelfVerifier().verify_code(code).confidence
