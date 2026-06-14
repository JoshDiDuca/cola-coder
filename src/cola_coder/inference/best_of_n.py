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

Verification — the dominant cost — is cluster-gated by default (IDEA-014,
``cluster_verify=True``): candidates with an identical normalized completion are
byte-identical to the verifier (the prompt is shared), so only ONE representative
per cluster is actually run and its verdict propagates to the rest. N candidates
cost ~(#distinct programs) verifier runs with identical results.

Verification is sandboxed end-to-end and language-aware:

- TypeScript → TscRunner (SandboxedRunner + hardened tsconfig, batched)
- Python + tests → evaluation.runner.execute_code (native/docker sandbox
  per configs/scoring.yaml security settings)
- Python without tests → compile() syntax check (static — never executes)
- No hard verifier available (e.g. tsc not installed) → SelfVerifier
  heuristics only

The final ranking key is (verified, secure, consistency, score): a candidate that
passed the hard verifier always beats one that didn't; within a verified tier, a
SECURE candidate (no dangerous patterns — IDEA-008/SEC-017) beats an insecure one;
then SELF-CONSISTENCY breaks the tie — a completion the model produced repeatedly
(largest normalized-equivalence cluster) is preferred (AlphaCode-style clustering /
self-consistency voting, INFER-026); score (hard verifier + SelfVerifier heuristic)
is the final tiebreak. Consistency is backward-compatible: when every completion is
unique each cluster size is 1, so the prior (verified, secure, score) order stands.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from ..data.scorers.language_detect import is_js_ts, is_typescript
from ..data.scorers.utils import ScoreMapper
from ..security.code_patterns import scan_dangerous
from ..security.import_scanner import classify_unknown_imports, scan_unknown_imports
# Shared with the FastAPI server's non-streaming strip — see text_utils.
from .text_utils import strip_prompt_prefix as _strip_prompt

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
    # Verifier-effort signals (EVAL-026): how much compute this prompt needed before
    # the verifier was satisfied — a free, model-relative difficulty label.
    candidates_used: int = 0            # total candidates actually generated
    rounds: int = 1                     # adaptive escalation rounds (1 = fixed-N)
    final_temperature: float = 0.0      # temperature of the last round
    solved: bool = False                # any candidate passed the hard verifier


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
    no_repeat_ngram_size: int = 0,
    timeout: float = 10.0,
    tsc_runner=None,
    execute_fn: ExecuteFn | None = None,
    cluster_verify: bool = True,
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
        no_repeat_ngram_size: If > 0, hard-block repeated n-grams. The batched
            sampler can't track per-sequence n-gram history, so a positive value
            forces SERIAL candidate generation (slower) to actually honor it
            rather than silently dropping it.
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
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    lang = detect_language(prompt) if language == "auto" else language
    verifier_name, verdicts = _verify_deduplicated(
        lang, texts, prompt, tests=tests, timeout=timeout,
        tsc_runner=tsc_runner, execute_fn=execute_fn, cluster_verify=cluster_verify,
    )

    candidates = _build_candidates(texts, verdicts, prompt, lang)
    ranked = _rank(candidates)
    logger.info(
        "best-of-%d (%s via %s): %d/%d verified, best score %.3f",
        num_candidates, lang, verifier_name,
        sum(c.verified for c in ranked), len(ranked), ranked[0].score,
    )
    return BestOfNResult(
        best=ranked[0], candidates=ranked, language=lang, verifier=verifier_name,
        candidates_used=len(ranked), rounds=1, final_temperature=temperature,
        solved=any(c.verified for c in ranked),
    )


def _build_candidates(texts, verdicts, prompt: str, lang: str) -> "list[CandidateResult]":
    """Build CandidateResults from generated texts + verifier verdicts.

    Shared by the fixed-N and adaptive best-of-N paths. Each candidate gets a
    heuristic-confidence score, a security flag (IDEA-008/SEC-017: scan the
    COMPLETION, not the user-written prompt), and the combined hard/heuristic score.
    """
    candidates: list[CandidateResult] = []
    for text, (verified, hard_score, details) in zip(texts, verdicts):
        heuristic = _heuristic_confidence(text, lang)
        details["heuristic_confidence"] = round(heuristic, 3)
        completion = _strip_prompt(text, prompt)
        dangers = scan_dangerous(completion)
        details["secure"] = not dangers
        if dangers:
            details["dangerous_patterns"] = dangers
        # Slopsquatting review signal (SEC-020): flag imports of packages outside
        # the known-safe allowlist — possible hallucinated/typosquatted deps. Pure
        # signal in details; does not change ranking (a niche-but-real import is
        # common, so it must not down-rank a verified candidate).
        unknown_imports = scan_unknown_imports(completion, lang)
        if unknown_imports:
            details["unknown_imports"] = unknown_imports
            # SEC-023: triage the unknowns — a name confusably close to a popular
            # package is a likely slopsquat/typosquat (higher risk than a niche
            # import). Surfaced as a review signal only (no ranking change).
            triage = classify_unknown_imports(completion, lang)
            if triage.has_typosquat:
                details["typosquat_imports"] = [
                    s.name for s in triage.typosquats
                ]
        candidates.append(
            CandidateResult(
                text=text,
                completion=completion,
                verified=verified,
                score=_HARD_WEIGHT * hard_score + _HEURISTIC_WEIGHT * heuristic,
                details=details,
            )
        )
    return candidates


def _normalize_completion(completion: str) -> str:
    """Normalize a completion for self-consistency clustering (INFER-026).

    AlphaCode-style: collapse intra-line whitespace runs and drop blank lines so
    that solutions differing only in trivial formatting land in the SAME cluster.
    Comments are kept (cheap; per-language comment stripping is out of scope).
    """
    lines = (" ".join(line.split()) for line in completion.splitlines())
    return "\n".join(ln for ln in lines if ln)


def _rank(candidates: "list[CandidateResult]") -> "list[CandidateResult]":
    """Rank best-first by (verified, secure, consistency, score) — see module docstring.

    Self-consistency: cluster candidates by normalized completion and prefer the
    one from the largest cluster (the model "voted" for that solution most often).
    Annotates each candidate's ``details["consistency"]`` with its cluster size.
    """
    counts = Counter(_normalize_completion(c.completion) for c in candidates)
    for c in candidates:
        c.details["consistency"] = counts[_normalize_completion(c.completion)]
    return sorted(
        candidates,
        key=lambda c: (
            c.verified,
            c.details.get("secure", True),
            c.details.get("consistency", 1),
            c.score,
        ),
        reverse=True,
    )


def generate_best_of_n_adaptive(
    generator,
    prompt: str,
    *,
    max_candidates: int = 8,
    initial_candidates: int = 2,
    growth: int = 2,
    language: str = "auto",
    tests: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.0,
    no_repeat_ngram_size: int = 0,
    timeout: float = 10.0,
    tsc_runner=None,
    execute_fn: "ExecuteFn | None" = None,
    cluster_verify: bool = True,
    temperature_growth: float = 1.0,
    max_temperature: float = 1.2,
) -> BestOfNResult:
    """Adaptive-budget best-of-N (IDEA-009): grow the candidate count only as needed.

    Generates an initial small batch and verifies it; if NO candidate both verifies
    AND is secure, it generates more (×growth) up to ``max_candidates``, stopping
    early the moment a verified+secure candidate appears. Cheap/easy prompts cost
    ``initial_candidates``; hard ones get the full budget — trading compute for
    accuracy only when the verifier says it's needed (2026 test-time-compute scaling).

    INFER-029 (verifier-calibrated sampling): with ``temperature_growth > 1.0``, each
    time the verifier rejects a whole round the NEXT round's sampling temperature is
    multiplied by ``temperature_growth`` (capped at ``max_temperature``). Re-sampling
    the SAME distribution rarely escapes a systematic failure — widening it does
    (2026 TTS: modify the sampling distribution, don't just resample it). Default 1.0
    keeps the temperature fixed (backward compatible). Pure inference.
    """
    if max_candidates < 1:
        raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
    if growth < 2:
        raise ValueError(f"growth must be >= 2, got {growth}")
    if temperature_growth < 1.0:
        raise ValueError(f"temperature_growth must be >= 1.0, got {temperature_growth}")
    if max_temperature < temperature:
        raise ValueError(
            f"max_temperature ({max_temperature}) < temperature ({temperature})"
        )
    lang = detect_language(prompt) if language == "auto" else language

    candidates: list[CandidateResult] = []
    verifier_name = "none"
    generated = 0
    rounds = 0
    current_temp = temperature
    target = min(max(1, initial_candidates), max_candidates)
    while generated < max_candidates:
        rounds += 1
        batch = target - generated
        texts = _generate_candidates(
            generator, prompt, num_candidates=batch,
            max_new_tokens=max_new_tokens, temperature=current_temp,
            top_k=top_k, top_p=top_p, min_p=min_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        verifier_name, verdicts = _verify_deduplicated(
            lang, texts, prompt, tests=tests, timeout=timeout,
            tsc_runner=tsc_runner, execute_fn=execute_fn, cluster_verify=cluster_verify,
        )
        candidates.extend(_build_candidates(texts, verdicts, prompt, lang))
        generated = target
        # Early stop: a verified AND secure candidate is good enough — don't spend
        # more compute.
        if any(c.verified and c.details.get("secure", True) for c in candidates):
            break
        next_target = min(generated * growth, max_candidates)
        if next_target <= generated:
            break  # already at the cap
        target = next_target
        # Verifier still unsatisfied → widen the sampling distribution for the next
        # round so retries actually explore new solutions, not the same failure.
        if temperature_growth > 1.0:
            current_temp = min(current_temp * temperature_growth, max_temperature)

    ranked = _rank(candidates)
    logger.info(
        "adaptive best-of-N (%s via %s): used %d/%d candidates @ temp<=%.2f, %d verified, best %.3f",
        lang, verifier_name, generated, max_candidates, current_temp,
        sum(c.verified for c in ranked), ranked[0].score,
    )
    return BestOfNResult(
        best=ranked[0], candidates=ranked, language=lang, verifier=verifier_name,
        candidates_used=generated, rounds=rounds, final_temperature=current_temp,
        solved=any(c.verified for c in ranked),
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
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    """Batched generation when available, serial fallback otherwise.

    n-gram blocking forces the serial path: the batched sampler
    (sample_next_tokens_batch) deliberately skips per-sequence n-gram history
    for throughput, so it cannot honor no_repeat_ngram_size. Rather than drop
    the user's setting silently, generate candidates one at a time where
    generate() applies the constraint.
    """
    if no_repeat_ngram_size <= 0 and hasattr(generator, "generate_group"):
        return generator.generate_group(
            prompt=prompt,
            num_completions=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
    if no_repeat_ngram_size > 0 and hasattr(generator, "generate_group"):
        logger.debug(
            "best-of-N: no_repeat_ngram_size=%d forces serial generation "
            "(batched sampler can't track per-sequence n-grams)",
            no_repeat_ngram_size,
        )
    else:
        logger.debug("generator has no generate_group — serial best-of-N fallback")
    return [
        generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        for _ in range(num_candidates)
    ]


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


def _verify_deduplicated(
    language: str,
    texts: list[str],
    prompt: str,
    *,
    tests: str | None,
    timeout: float,
    tsc_runner,
    execute_fn: ExecuteFn | None,
    cluster_verify: bool,
) -> tuple[str, list[_Verdict]]:
    """Verify candidates, optionally verifying ONE representative per cluster (IDEA-014).

    The sandbox verifier (a tsc/test run per candidate) is best-of-N's dominant cost.
    Candidates with the same normalized completion (the self-consistency clusters from
    `_normalize_completion`) are byte-identical to the verifier — the prompt is shared —
    so they MUST get the same verdict. When ``cluster_verify`` is set, group by
    normalized completion, verify only the first member of each cluster, and propagate
    that verdict (a fresh ``details`` copy per candidate to avoid aliasing) to the rest.
    N candidates then cost ~(#distinct clusters) verifier runs instead of N, with
    identical results. Falls back to verifying everything when disabled or N < 2.
    """
    if not cluster_verify or len(texts) < 2:
        return _run_hard_verifier(
            language, texts, tests=tests, timeout=timeout,
            tsc_runner=tsc_runner, execute_fn=execute_fn,
        )

    keys = [_normalize_completion(_strip_prompt(t, prompt)) for t in texts]
    rep_index: dict[str, int] = {}
    rep_order: list[str] = []
    for i, k in enumerate(keys):
        if k not in rep_index:
            rep_index[k] = i
            rep_order.append(k)

    if len(rep_order) == len(texts):  # all unique — nothing to save
        return _run_hard_verifier(
            language, texts, tests=tests, timeout=timeout,
            tsc_runner=tsc_runner, execute_fn=execute_fn,
        )

    rep_texts = [texts[rep_index[k]] for k in rep_order]
    verifier_name, rep_verdicts = _run_hard_verifier(
        language, rep_texts, tests=tests, timeout=timeout,
        tsc_runner=tsc_runner, execute_fn=execute_fn,
    )
    verdict_by_key = {k: rep_verdicts[i] for i, k in enumerate(rep_order)}
    logger.debug(
        "best-of-N cluster-gated verify: %d candidates -> %d verifier runs",
        len(texts), len(rep_texts),
    )
    verdicts: list[_Verdict] = []
    for k in keys:
        verified, score, details = verdict_by_key[k]
        verdicts.append((verified, score, dict(details)))  # copy: no aliasing
    return verifier_name, verdicts


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
        return "heuristic", _verify_heuristic(texts, language)

    if language == "python":
        if tests:
            return "python_exec", _verify_python_exec(
                texts, tests, execute_fn=execute_fn, timeout=timeout
            )
        return "python_syntax", _verify_python_syntax(texts)

    logger.warning("unknown language %r — best-of-N using heuristics", language)
    return "heuristic", _verify_heuristic(texts, language)


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


def _verify_heuristic(texts: list[str], language: str | None = None) -> list[_Verdict]:
    """SelfVerifier-only verdicts when no tool-based verifier exists."""
    from ..features.self_verification import SelfVerifier

    verifier = SelfVerifier()
    verdicts: list[_Verdict] = []
    for text in texts:
        result = verifier.verify_code(text, language=language)
        verdicts.append(
            (result.passed, result.confidence, {"heuristic_only": True,
                                                "issues": result.issues[:3]})
        )
    return verdicts


def _heuristic_confidence(code: str, language: str | None = None) -> float:
    from ..features.self_verification import SelfVerifier

    return SelfVerifier().verify_code(code, language=language).confidence
