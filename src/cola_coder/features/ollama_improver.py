"""Ollama Improver: use a local LLM to improve code quality before training.

Instead of training on raw, messy GitHub code (which often has poor naming,
missing comments, and no type hints), this feature runs each sample through a
local Ollama model (e.g. CodeLlama) to produce a cleaner version that the
transformer will learn from.

The idea is analogous to "data augmentation" in image ML, except here we're
augmenting *quality* rather than visual variety.  Better training data →
better generated code.

For a TS dev: think of Ollama as a locally-running AI code reviewer that
rewrites code before it enters your dataset.  The REST API is essentially a
`fetch()` call — we just use stdlib `urllib` here to avoid adding any pip
dependencies.

Two classes are exposed:
  OllamaImprover  — full rewriter: comments, naming, type hints, JSDoc, etc.
  OllamaScorer    — lighter weight scorer that rates code quality on 0-1.

Both classes are safe to call even when Ollama is not running — they fall
back gracefully and never raise into the training loop.

Integration example::

    from cola_coder.features.ollama_improver import OllamaImprover, is_enabled

    if is_enabled():
        improver = OllamaImprover(model="codellama")
        if improver.is_available():
            result = improver.improve_code(raw_code, language="python")
            training_sample = result.improved  # drop-in replacement
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature toggle (project-standard pattern)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether the Ollama improver feature is active.

    Flip FEATURE_ENABLED to False at the top of this file to disable the
    entire feature without touching any call sites.
    """
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# The primary improvement prompt.  We ask the model to act as a code quality
# expert and return ONLY the improved code so we can drop the response
# straight back into the dataset without any post-processing.
_IMPROVE_PROMPT = """\
You are a code quality expert. Improve the following {language} code:
1. Add clear, meaningful comments explaining what the code does
2. Improve variable/function names to be more descriptive
3. Add type hints (Python) or JSDoc (JS/TS) where missing
4. Fix any deprecated patterns or anti-patterns
5. Keep ALL functionality exactly the same — only improve readability

Return ONLY the improved code, no explanations.

```{language}
{code}
```"""

# Prompt for the comment-only pass.  Cheaper than full improvement when the
# caller only needs better documentation.
_COMMENT_PROMPT = """\
You are a code documentation expert. Add clear, meaningful comments to the
following {language} code. Do NOT change any logic, variable names, or
structure — only add comments that explain what the code does and why.

Return ONLY the commented code, no explanations.

```{language}
{code}
```"""

# Prompt for the scorer.  We ask for a 1-5 rating because smaller LLMs handle
# bounded integer outputs much more reliably than open-ended 0.0-1.0 floats.
# We normalise to [0, 1] after parsing.
_SCORE_PROMPT = """\
Rate the quality of this {language} code on a scale of 1 to 5:

1 = Very poor: no comments, bad naming, no types, broken logic
2 = Poor: minimal comments, some naming issues, few or no types
3 = Average: some comments, reasonable naming, partial types
4 = Good: clear comments, descriptive naming, most types present
5 = Excellent: thorough comments, excellent naming, full types, clean style

Respond with ONLY:
Score: <number>
Reason: <one sentence>

```{language}
{code}
```"""

# Prompt used by OllamaScorer (even lighter — just a number + brief note).
_SCORER_PROMPT = """\
Rate this {language} code quality from 1 (worst) to 5 (best).
Reply in exactly this format:
Score: <1-5>
Reason: <one short sentence>

Code:
{code}"""


# ---------------------------------------------------------------------------
# ImprovedCode dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImprovedCode:
    """Result of a single code improvement pass.

    Think of this like a structured diff object — it carries both the before
    and after state plus metadata about what changed.

    Attributes:
        original:       The raw input code, unchanged.
        improved:       The LLM-improved version.  If improvement failed, this
                        is identical to ``original``.
        changes:        Human-readable descriptions of what was changed, e.g.
                        ["Added type hints to 3 functions", "Renamed `x` to
                        `token_count`"].  Empty if no changes were made.
        quality_before: Estimated quality score in [0, 1] before improvement.
                        Populated by ``OllamaScorer`` if called; otherwise 0.0.
        quality_after:  Estimated quality score in [0, 1] after improvement.
        was_improved:   True when ``improved != original``.  Useful for
                        logging statistics on how many samples were changed.
    """

    original: str
    improved: str
    changes: list[str] = field(default_factory=list)
    quality_before: float = 0.0
    quality_after: float = 0.0
    was_improved: bool = False

    def __post_init__(self) -> None:
        # Automatically set was_improved based on content comparison.
        # A caller can override it afterwards if needed.
        if not self.was_improved:
            self.was_improved = self.improved.strip() != self.original.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
) -> Optional[str]:
    """Send a prompt to the Ollama /api/generate endpoint and return the text.

    Uses only stdlib urllib — no requests, no httpx, no aiohttp.  This keeps
    the feature zero-dependency.

    For a TS dev: this is equivalent to:
        const res = await fetch(`${baseUrl}/api/generate`, {
            method: 'POST',
            body: JSON.stringify({ model, prompt, stream: false }),
        });
        const { response } = await res.json();

    Args:
        base_url: Ollama server root, e.g. "http://localhost:11434".
        model:    Model tag, e.g. "codellama" or "codellama:7b".
        prompt:   The full prompt string to send.
        timeout:  Request timeout in seconds.

    Returns:
        The ``response`` field from the JSON reply, or None on any error.
    """
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as exc:
        logger.debug("Ollama connection error: %s", exc)
        return None
    except TimeoutError:
        logger.debug("Ollama request timed out after %ds", timeout)
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        logger.debug("Ollama response parse error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001  (broad-except intentional)
        logger.debug("Unexpected Ollama error: %s", exc)
        return None


def _extract_code_from_response(response: str) -> str:
    """Strip markdown fences from an LLM response that returned code.

    Many models wrap their answer in triple-backtick fences even when told not
    to.  This function extracts just the code content.

    Examples handled:
        ```python\\ncode here\\n```   →  "code here"
        ```\\ncode here\\n```         →  "code here"
        code here                    →  "code here"  (passthrough)
    """
    # Match optional language tag after the opening fence
    fence_pattern = re.compile(r"^```[a-zA-Z]*\n(.*?)```", re.DOTALL)
    match = fence_pattern.search(response.strip())
    if match:
        return match.group(1).rstrip()

    # No fence found — return as-is (may already be clean code)
    return response.strip()


def _parse_score_from_response(response: str, fallback: int = 3) -> int:
    """Extract the integer score (1-5) from an LLM scoring response.

    The expected format is:
        Score: 4
        Reason: ...

    We try multiple patterns in order of specificity, then fall back.

    Args:
        response: Raw text from the LLM.
        fallback: Value to return when parsing fails (default 3 = average).

    Returns:
        Integer in [1, 5].
    """
    # Pattern 1: "Score: 4" (our requested format)
    match = re.search(r"[Ss]core\s*:\s*([1-5])", response)
    if match:
        return int(match.group(1))

    # Pattern 2: bare digit on its own line (models sometimes just say "4")
    match = re.search(r"^\s*([1-5])\s*$", response, re.MULTILINE)
    if match:
        return int(match.group(1))

    # Pattern 3: "rating: 4/5" style
    match = re.search(r"([1-5])\s*/\s*5", response)
    if match:
        return int(match.group(1))

    logger.debug("Could not parse score from response, using fallback=%d", fallback)
    return fallback


def _detect_changes(original: str, improved: str) -> list[str]:
    """Produce a simple list of high-level change descriptions.

    This is a lightweight heuristic — it does not do a line-by-line diff.
    Instead it looks for structural signals (comment lines added, type hints
    added, etc.) and reports them at a summary level.

    For a TS dev: think of this as a coarse `git diff --stat` rather than a
    full `git diff`.
    """
    changes: list[str] = []

    orig_lines = original.splitlines()
    impr_lines = improved.splitlines()

    # ---- Comments added? ------------------------------------------------
    def count_comment_lines(lines: list[str], is_python: bool) -> int:
        count = 0
        for line in lines:
            s = line.strip()
            if is_python and s.startswith("#"):
                count += 1
            elif not is_python and (
                s.startswith("//") or s.startswith("/*") or s.startswith("*")
            ):
                count += 1
        return count

    # Detect language from content heuristics (very rough)
    is_python = "def " in original or "import " in original
    orig_comments = count_comment_lines(orig_lines, is_python)
    impr_comments = count_comment_lines(impr_lines, is_python)
    if impr_comments > orig_comments:
        added = impr_comments - orig_comments
        changes.append(f"Added {added} comment line(s)")

    # ---- Type hints added (Python)? -------------------------------------
    orig_type_hints = len(re.findall(r":\s*\w+\s*(?:=|->|\))", original))
    impr_type_hints = len(re.findall(r":\s*\w+\s*(?:=|->|\))", improved))
    if impr_type_hints > orig_type_hints:
        changes.append(f"Added ~{impr_type_hints - orig_type_hints} type hint(s)")

    # ---- JSDoc added (JS/TS)? -------------------------------------------
    orig_jsdoc = original.count("/**")
    impr_jsdoc = improved.count("/**")
    if impr_jsdoc > orig_jsdoc:
        changes.append(f"Added {impr_jsdoc - orig_jsdoc} JSDoc block(s)")

    # ---- Length delta ---------------------------------------------------
    line_delta = len(impr_lines) - len(orig_lines)
    if line_delta > 5:
        changes.append(f"Expanded by ~{line_delta} lines")
    elif line_delta < -5:
        changes.append(f"Reduced by ~{abs(line_delta)} lines")

    # ---- Generic fallback when content changed but nothing specific -----
    if not changes and improved.strip() != original.strip():
        changes.append("Code reformatted/improved")

    return changes


# ---------------------------------------------------------------------------
# OllamaImprover
# ---------------------------------------------------------------------------


class OllamaImprover:
    """Use a locally-running Ollama model to improve code quality.

    This class wraps Ollama's REST API (no SDK needed) and provides methods to
    rewrite code with better comments, naming, and type hints.  It is designed
    to be inserted into the data-preparation pipeline, running each code sample
    through the LLM before it's tokenized and written to the training memmap.

    For a TS dev: this is like running `eslint --fix` + `jsdoc-generator` +
    a variable-renaming tool, all powered by a local LLM instead of static rules.

    Failure modes are handled gracefully:
    - Ollama not running  → returns original code unchanged, logs a debug message
    - Model timeout       → returns original code unchanged
    - Parse error         → returns original code unchanged

    None of these conditions raise exceptions into the training loop.

    Usage::

        improver = OllamaImprover(model="codellama", timeout=60)
        if improver.is_available():
            result = improver.improve_code(raw_python_code, language="python")
            print(result.changes)      # ["Added 5 comment lines", "Added 3 type hints"]
            print(result.quality_after)  # 0.8
    """

    def __init__(
        self,
        model: str = "codellama",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        """Create an OllamaImprover.

        Args:
            model:    Ollama model tag to use.  Must already be pulled locally
                      (run `ollama pull codellama` in a terminal first).
                      Supported options: "codellama", "codellama:7b",
                      "codellama:13b", "deepseek-coder", "starcoder2".
            base_url: Base URL of the Ollama server.  Default is the standard
                      local address.  Change this to point at a remote server.
            timeout:  HTTP request timeout in seconds.  Increase for large
                      models or slow hardware.  The default 30 s works for
                      CodeLlama 7B on an RTX 4080.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._scorer = OllamaScorer(model=model, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if Ollama is running and the configured model is loaded.

        Makes a lightweight request to /api/tags (the Ollama model-list
        endpoint) and checks whether ``self.model`` appears in the response.
        This is cheap — it does not actually run inference.

        Returns:
            True if the model is available, False otherwise.
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                # Each entry has a "name" key like "codellama:latest"
                available_names = [m.get("name", "") for m in models]
                # Check for exact match or prefix match (e.g. "codellama" matches
                # "codellama:latest" and "codellama:7b")
                for name in available_names:
                    base_name = name.split(":")[0]
                    if self.model == name or self.model == base_name:
                        return True
                # If model list is empty but server responded, assume available
                # (some Ollama versions don't pre-list pulled models accurately)
                if not models:
                    return True
                logger.warning(
                    "Ollama is running but model '%s' not found. "
                    "Available: %s",
                    self.model,
                    ", ".join(available_names[:5]),
                )
                return False
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Primary improvement method
    # ------------------------------------------------------------------

    def improve_code(self, code: str, language: str = "") -> ImprovedCode:
        """Run a full quality-improvement pass on a code snippet.

        Sends the code to Ollama with the improvement prompt, parses the
        response, and returns an ImprovedCode object with the before/after
        state and a list of changes.

        If the LLM is unavailable or times out, returns an ImprovedCode where
        ``improved == original`` and ``was_improved == False``.

        Args:
            code:     Source code to improve.  Can be any length, but very
                      long files may hit the model's context window limit.
            language: Language hint, e.g. "python", "typescript", "javascript".
                      Used in the prompt and for change detection heuristics.
                      If empty, the model will infer the language from content.

        Returns:
            ImprovedCode with ``improved`` set to the best version available.
        """
        if not code.strip():
            return ImprovedCode(original=code, improved=code)

        lang_label = language or "code"
        prompt = _IMPROVE_PROMPT.format(language=lang_label, code=code)

        response = _call_ollama(self.base_url, self.model, prompt, self.timeout)

        if response is None:
            logger.debug("improve_code: Ollama unavailable, returning original")
            return ImprovedCode(original=code, improved=code)

        improved = _extract_code_from_response(response)

        # Sanity check: if the model returned something very short or empty,
        # fall back to the original rather than replacing code with garbage.
        if len(improved) < max(10, len(code) // 4):
            logger.debug(
                "improve_code: response suspiciously short (%d chars vs %d), "
                "keeping original",
                len(improved),
                len(code),
            )
            return ImprovedCode(original=code, improved=code)

        changes = _detect_changes(code, improved)

        # Score before and after (best-effort — failures return 0.0)
        quality_before = self._scorer.score_code_normalized(code, language)
        quality_after = self._scorer.score_code_normalized(improved, language)

        return ImprovedCode(
            original=code,
            improved=improved,
            changes=changes,
            quality_before=quality_before,
            quality_after=quality_after,
        )

    # ------------------------------------------------------------------
    # Score method
    # ------------------------------------------------------------------

    def score_code(self, code: str, language: str = "") -> tuple[float, str]:
        """Rate the quality of a code snippet.

        Args:
            code:     Source code to score.
            language: Language hint (optional).

        Returns:
            A tuple of (score, explanation) where score is in [0.0, 1.0] and
            explanation is a one-sentence justification from the model.
            Falls back to (0.5, "unavailable") if Ollama is not running.
        """
        return self._scorer.score_code(code, language)

    # ------------------------------------------------------------------
    # Comments-only pass
    # ------------------------------------------------------------------

    def add_comments(self, code: str, language: str = "") -> str:
        """Add comments to code without changing any other structure.

        This is a cheaper operation than ``improve_code`` because the prompt
        is narrower (comment-only).  Use this when you only want documentation
        improvements without risking variable-rename drift.

        Args:
            code:     Source code to document.
            language: Language hint (optional).

        Returns:
            Code with added comments, or the original if Ollama is unavailable.
        """
        if not code.strip():
            return code

        lang_label = language or "code"
        prompt = _COMMENT_PROMPT.format(language=lang_label, code=code)

        response = _call_ollama(self.base_url, self.model, prompt, self.timeout)

        if response is None:
            logger.debug("add_comments: Ollama unavailable, returning original")
            return code

        commented = _extract_code_from_response(response)

        # Sanity check: commented code should be longer, not shorter
        if len(commented) < len(code) * 0.8:
            logger.debug(
                "add_comments: response shorter than expected, keeping original"
            )
            return code

        return commented

    # ------------------------------------------------------------------
    # Batch improvement
    # ------------------------------------------------------------------

    def improve_batch(
        self, codes: list[str], language: str = ""
    ) -> list[ImprovedCode]:
        """Improve a list of code snippets sequentially.

        Ollama's local API is synchronous, so we call it once per sample.
        Parallelism is intentionally avoided here — running many concurrent
        Ollama requests on a single GPU would thrash VRAM rather than speed
        things up.

        For a TS dev: this is a `Promise.all` that we've deliberately made
        sequential — like a `for...of` with `await` instead of `Promise.all`.

        Args:
            codes:    List of source code strings to improve.
            language: Language hint applied to all samples in the batch.

        Returns:
            List of ImprovedCode objects, one per input, in the same order.
            Samples that failed improvement have ``was_improved == False``.
        """
        results: list[ImprovedCode] = []
        total = len(codes)

        for idx, code in enumerate(codes):
            logger.debug("improve_batch: processing %d/%d", idx + 1, total)
            result = self.improve_code(code, language)
            results.append(result)

        improved_count = sum(1 for r in results if r.was_improved)
        logger.info(
            "improve_batch: %d/%d samples improved", improved_count, total
        )

        return results


# ---------------------------------------------------------------------------
# OllamaScorer
# ---------------------------------------------------------------------------


class OllamaScorer:
    """Lightweight code quality scorer backed by a local Ollama model.

    Compared to OllamaImprover, this class only asks for a rating — it does
    not rewrite the code.  This makes it much faster and cheaper to run, making
    it suitable for scoring large dataset batches where full improvement would
    be too slow.

    For a TS dev: think of this as calling an AI linter that gives you a grade
    (A-F / 1-5) without suggesting specific fixes.

    Typical use-case — filter training data by quality::

        scorer = OllamaScorer()
        high_quality = [
            code for code in raw_samples
            if scorer.score_code_normalized(code, "python") >= 0.6
        ]

    Falls back to a score of 3/5 (normalised: 0.5) when the model is not
    available or when the response cannot be parsed.
    """

    # Default fallback score when parsing fails (3 out of 5 = "average")
    _FALLBACK_SCORE_INT: int = 3

    def __init__(
        self,
        model: str = "codellama",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        """Create an OllamaScorer.

        Args:
            model:    Ollama model tag (same as OllamaImprover).
            base_url: Ollama server URL.
            timeout:  Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def score_code(self, code: str, language: str = "") -> tuple[float, str]:
        """Score code quality and return (normalised_score, explanation).

        Sends a scoring prompt to Ollama, parses the 1-5 integer response, and
        normalises it to [0.0, 1.0] so callers don't need to know the scale.

        The normalisation maps:
            1 → 0.0   (very poor)
            2 → 0.25
            3 → 0.5   (average — also the fallback)
            4 → 0.75
            5 → 1.0   (excellent)

        Args:
            code:     Source code to rate.
            language: Language hint (e.g. "python", "typescript").

        Returns:
            (score, reason) tuple.  score is in [0.0, 1.0].  reason is a
            one-sentence explanation from the model, or a fallback message.
        """
        if not code.strip():
            return 0.0, "Empty code"

        lang_label = language or "code"
        prompt = _SCORER_PROMPT.format(language=lang_label, code=code)

        response = _call_ollama(self.base_url, self.model, prompt, self.timeout)

        if response is None:
            fallback_norm = self._int_to_normalised(self._FALLBACK_SCORE_INT)
            return fallback_norm, "Ollama unavailable — using fallback score"

        score_int = _parse_score_from_response(response, self._FALLBACK_SCORE_INT)
        score_norm = self._int_to_normalised(score_int)

        # Extract the reason line if present
        reason = "No reason provided"
        reason_match = re.search(r"[Rr]eason\s*:\s*(.+)", response)
        if reason_match:
            reason = reason_match.group(1).strip()

        return score_norm, reason

    def score_code_normalized(self, code: str, language: str = "") -> float:
        """Convenience wrapper that returns only the float score.

        Args:
            code:     Source code to rate.
            language: Language hint.

        Returns:
            Score in [0.0, 1.0].
        """
        score, _ = self.score_code(code, language)
        return score

    def score_batch(
        self, codes: list[str], language: str = ""
    ) -> list[tuple[float, str]]:
        """Score a list of code snippets.

        Processes sequentially (same reasoning as OllamaImprover.improve_batch).

        Args:
            codes:    List of source code strings.
            language: Language hint applied to all samples.

        Returns:
            List of (score, reason) tuples in the same order as ``codes``.
        """
        return [self.score_code(code, language) for code in codes]

    @staticmethod
    def _int_to_normalised(score_int: int) -> float:
        """Convert a 1-5 integer score to a [0.0, 1.0] float.

        Uses a linear mapping: score_normalised = (score_int - 1) / 4.

        Args:
            score_int: Integer score in [1, 5].

        Returns:
            Float in [0.0, 1.0].
        """
        clamped = max(1, min(5, score_int))
        return (clamped - 1) / 4.0
