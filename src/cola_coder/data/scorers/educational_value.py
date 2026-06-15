"""Educational-value scorer (FineWeb-Edu / Stack-Edu paradigm, cheap static tier).

The FineWeb-Edu and Stack-Edu lines of work showed that filtering a pre-training
corpus by *educational value* — how well a sample teaches a concept, not merely
whether it is syntactically valid — sharply improves downstream model quality.
Their production scorer is an LLM-distilled classifier. That is the eventual goal
here; this module is the tractable FIRST STEP: a CHEAP, deterministic, CPU-only,
no-LLM, no-subprocess, no-network static proxy for educational value, exposed as a
standard :class:`ScorerProtocol` implementor so it composes with the existing
quality-weight pipeline (``scripts/score_data.py``, ``CompositeScorer``).

It is a *soft* signal: it produces a quality score in ``[0.0, 1.0]`` that the
composite turns into a training weight (reweight-over-filter, the project default).
Well-documented, tested, well-named, structurally-complete code scores high;
obfuscated, comment-free, single-char-identifier, or repetitive boilerplate scores
low.

Signals (each in ``[0.0, 1.0]``) and their weights (sum to 1.0):

  * ``comment_docstring`` (0.25) — proportion of comment lines plus a docstring
    presence bonus. Teaching code explains itself; the FineWeb-Edu rubric rewards
    explanatory prose. Saturates at a healthy density (~20% comment lines) so a
    wall of commented-out code is not rewarded.
  * ``example_or_test`` (0.20) — heuristic presence of a runnable example or test
    (``def test``/``assert``/``if __name__``/``example``/``describe(``/``it(``/
    ``expect(``). Worked examples are the strongest educational signal in the
    Stack-Edu rubric.
  * ``naming_quality`` (0.20) — fraction of *non-trivial* identifiers (length >= 2,
    not pure single-char/obfuscation spam). Penalises minified / obfuscated blobs
    where every name is ``a``, ``b``, ``_0x1f``.
  * ``structural_completeness`` (0.20) — the sample looks like a complete unit, not
    a dangling fragment: has at least one definition, balanced brackets, and a
    plausible body. Uses :class:`ScoreMapper` over a small structural-defect count.
  * ``non_degenerate`` (0.15) — ``1 - duplicate_line_fraction`` from the shared
    Gopher repetition metric (reused from ``data/filters/repetition.py``, DATA-071):
    long runs of identical lines (looping generations, copy-paste boilerplate) read
    as low educational value.

Language-aware where it matters: TypeScript/JavaScript uses ``//`` / ``/* */``
comments and ``it(``/``describe(`` test markers; Python uses ``#`` comments,
triple-quoted docstrings, and ``def test``/``assert``. Detection reuses the shared
``language_detect`` helpers. With no language hint the scorer degrades gracefully to
a language-agnostic union of both comment styles and marker sets.

Pure Python, CPU-only, deterministic, no model load, no subprocess, no network —
safe to run alongside live training.
"""

from __future__ import annotations

import re

from cola_coder.data.filters.repetition import compute_repetition_metrics
from cola_coder.data.scorers.language_detect import is_js_ts
from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.utils import ScoreMapper

# --- Signal weights (must sum to 1.0) ---
_W_COMMENT = 0.25
_W_EXAMPLE = 0.20
_W_NAMING = 0.20
_W_STRUCTURE = 0.20
_W_NON_DEGENERATE = 0.15

# Comment density saturates here: ~1 commenting line per 5 code lines is already a
# well-documented file; beyond that we do not keep rewarding (avoids favouring
# commented-out code dumps).
_COMMENT_DENSITY_SATURATION = 0.20

# Identifier tokens (word starting with a letter/underscore). Numbers excluded so
# numeric literals don't count as "names".
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Common keywords are not user-chosen identifiers; excluding them keeps the naming
# signal focused on names the author actually picked. Union of TS/JS + Python.
_KEYWORDS: frozenset[str] = frozenset({
    # Python
    "def", "class", "return", "import", "from", "as", "if", "elif", "else",
    "for", "while", "in", "is", "not", "and", "or", "with", "try", "except",
    "finally", "raise", "pass", "lambda", "yield", "assert", "global", "nonlocal",
    "True", "False", "None", "self", "cls", "async", "await", "del",
    # TS / JS
    "const", "let", "var", "function", "export", "interface", "type", "enum",
    "extends", "implements", "new", "this", "void", "string", "number", "boolean",
    "any", "unknown", "readonly", "public", "private", "protected", "static",
    "switch", "case", "break", "continue", "default", "throw", "catch", "typeof",
})

# Structural-defect count -> structure score. 0 defects is a clean, complete unit.
_STRUCTURE_SCORE = ScoreMapper([(0, 1.0), (1, 0.6), (2, 0.3)], floor=0.1)

# Example / test markers. Split by language; the language-agnostic path unions both.
_PYTHON_EXAMPLE_MARKERS: tuple[str, ...] = (
    "def test", "assert ", "assert(", "if __name__", "example", ">>>", "pytest",
    "unittest",
)
_JS_TS_EXAMPLE_MARKERS: tuple[str, ...] = (
    "describe(", "it(", "test(", "expect(", "example", "assert(", "console.assert",
)


def _comment_lines(lines: list[str], js_ts: bool, agnostic: bool) -> int:
    """Count lines that are (predominantly) a comment for the detected language."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_hash = stripped.startswith("#")
        is_slash = stripped.startswith("//") or stripped.startswith("/*") \
            or stripped.startswith("*")
        if agnostic and (is_hash or is_slash):
            count += 1
        elif js_ts and is_slash:
            count += 1
        elif not js_ts and not agnostic and is_hash:
            count += 1
    return count


def _has_docstring(code: str, js_ts: bool, agnostic: bool) -> bool:
    """Detect a docstring / doc-comment block."""
    if (not js_ts or agnostic) and ('"""' in code or "'''" in code):
        return True
    if (js_ts or agnostic) and "/**" in code:  # JSDoc / TSDoc block
        return True
    return False


def _comment_docstring_signal(code: str, js_ts: bool, agnostic: bool) -> float:
    """Proportion of comment lines (saturating) plus a docstring presence bonus."""
    lines = code.splitlines()
    code_lines = [ln for ln in lines if ln.strip()]
    if not code_lines:
        return 0.0
    comment_density = _comment_lines(lines, js_ts, agnostic) / len(code_lines)
    density_score = min(1.0, comment_density / _COMMENT_DENSITY_SATURATION)
    has_doc = _has_docstring(code, js_ts, agnostic)
    # 80% density, 20% docstring-presence bonus.
    return 0.8 * density_score + (0.2 if has_doc else 0.0)


def _example_or_test_signal(code: str, js_ts: bool, agnostic: bool) -> float:
    """1.0 if the sample contains a runnable example/test marker, else 0.0."""
    lowered = code.lower()
    markers: tuple[str, ...]
    if agnostic:
        markers = _PYTHON_EXAMPLE_MARKERS + _JS_TS_EXAMPLE_MARKERS
    elif js_ts:
        markers = _JS_TS_EXAMPLE_MARKERS
    else:
        markers = _PYTHON_EXAMPLE_MARKERS
    return 1.0 if any(m in lowered for m in markers) else 0.0


def _naming_quality_signal(code: str) -> float:
    """Fraction of non-trivial author-chosen identifiers.

    Penalises minified/obfuscated code where identifiers are single characters or
    hex-like tokens (``_0x1f``). Keywords are excluded so the signal measures names
    the author actually picked, not language syntax.
    """
    idents = [t for t in _IDENT_RE.findall(code) if t not in _KEYWORDS]
    if not idents:
        return 0.0
    good = 0
    for ident in idents:
        # Single-char names and hex-obfuscation tokens read as low quality.
        if len(ident) <= 1:
            continue
        if ident.startswith("_0x") or ident.startswith("0x"):
            continue
        good += 1
    return good / len(idents)


def _structural_completeness_signal(code: str, js_ts: bool, agnostic: bool) -> float:
    """Score how much the sample resembles a complete, balanced code unit."""
    defects = 0

    # 1. Unbalanced brackets => truncated / fragment.
    for open_b, close_b in (("(", ")"), ("[", "]"), ("{", "}")):
        if code.count(open_b) != code.count(close_b):
            defects += 1
            break  # one bracket-balance defect is enough; don't triple-count

    # 2. No definition at all => probably a fragment, not a teachable unit.
    has_def = False
    if agnostic or not js_ts:
        has_def = has_def or bool(re.search(r"\bdef\s+\w+|\bclass\s+\w+", code))
    if agnostic or js_ts:
        has_def = has_def or bool(
            re.search(r"\bfunction\s+\w+|\bclass\s+\w+|=>\s*[{(]|\bconst\s+\w+", code)
        )
    if not has_def:
        defects += 1

    # 3. Too few non-blank lines to teach anything (a one-liner snippet).
    code_lines = [ln for ln in code.splitlines() if ln.strip()]
    if len(code_lines) < 2:
        defects += 1

    return _STRUCTURE_SCORE(defects)


def _non_degenerate_signal(code: str) -> float:
    """1 - duplicate-line fraction (reuses the shared Gopher repetition metric)."""
    metrics = compute_repetition_metrics(code)
    return max(0.0, 1.0 - metrics.dup_line_frac)


class EducationalValueScorer:
    """Cheap static proxy for educational value (FineWeb-Edu first step).

    Combines five pure-Python signals into a single ``[0.0, 1.0]`` score. See the
    module docstring for the signals, their rationale, and weights.
    """

    name: str = "educational_value"

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        if not code or not code.strip():
            return ScorerResult(
                score=0.0,
                scorer_name=self.name,
                details={"reason": "empty"},
            )

        js_ts = is_js_ts(code, metadata)
        # When metadata gives no language hint AND heuristics don't flag JS/TS, run
        # the language-agnostic union so a Python-or-other sample isn't penalised.
        agnostic = not js_ts and not self._has_language_hint(metadata)

        comment = _comment_docstring_signal(code, js_ts, agnostic)
        example = _example_or_test_signal(code, js_ts, agnostic)
        naming = _naming_quality_signal(code)
        structure = _structural_completeness_signal(code, js_ts, agnostic)
        non_degenerate = _non_degenerate_signal(code)

        score = (
            _W_COMMENT * comment
            + _W_EXAMPLE * example
            + _W_NAMING * naming
            + _W_STRUCTURE * structure
            + _W_NON_DEGENERATE * non_degenerate
        )
        score = max(0.0, min(1.0, score))

        return ScorerResult(
            score=score,
            scorer_name=self.name,
            details={
                "comment_docstring": round(comment, 4),
                "example_or_test": round(example, 4),
                "naming_quality": round(naming, 4),
                "structural_completeness": round(structure, 4),
                "non_degenerate": round(non_degenerate, 4),
                "language": "js_ts" if js_ts else ("agnostic" if agnostic else "other"),
            },
        )

    def score_batch(
        self, items: list[tuple[str, dict[str, object] | None]]
    ) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def _has_language_hint(metadata: dict[str, object] | None) -> bool:
        """True when metadata carries an explicit language/file_path hint."""
        if not metadata:
            return False
        return bool(metadata.get("language")) or bool(metadata.get("file_path"))

    @staticmethod
    def is_available() -> bool:
        return True  # Pure Python, no external dependencies
