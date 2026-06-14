"""Semantically-preserving docstring perturbations for robustness evaluation (EVAL-030).

A robust code model should produce the SAME function whether a problem's docstring
says "Return the sum" or "Output the sum" — the spec is unchanged, only the prose
is reworded. This module applies semantically-preserving transformations to a
``CodingProblem``'s DOCSTRING ONLY and never touches the signature line, the
``entry_point`` name, or the ``test_code``. The downstream verifier
(``runner.evaluate_solution``) then measures functional drift.

HARD INVARIANT — only docstring/comment prose is mutated:
    - The ``def ...(...):`` signature line is byte-identical across every variant.
    - ``entry_point`` and ``test_code`` are byte-identical (we only rewrite ``prompt``).
    - Every emitted variant is AST-parsed and asserted to still define ``entry_point``.
      A perturbation that would change parse structure is silently skipped — we never
      emit a task-changing variant.
    - A problem with no docstring (or an empty one) yields the clean problem only.

For a TS dev: think of this like property-based testing — we generate equivalent
re-phrasings of the same spec and assert the model's behaviour is invariant.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass, replace

from .humaneval import CodingProblem

# All perturbation kinds, in a stable order (used as the default kind set).
ALL_KINDS: tuple[str, ...] = (
    "typo",
    "whitespace",
    "casing",
    "reorder_examples",
    "paraphrase",
)

# Small curated, case-insensitive prose synonym map. Keys are whole words; the
# replacement preserves the original word's leading-capital casing. Deliberately
# conservative — only swaps that keep the spec meaning identical.
_PARAPHRASE_MAP: dict[str, str] = {
    "return": "output",
    "returns": "outputs",
    "given": "provided",
    "list": "sequence",
    "function": "routine",
    "string": "text",
    "check": "verify",
    "find": "locate",
    "compute": "calculate",
    "number": "value",
}

# A "doctest" line is any docstring line whose first non-space char starts a
# ``>>>`` example (the call) — these are reorderable as a set when independent.
_DOCTEST_RE = re.compile(r"^\s*>>>")

# A prose word is a run of letters only — no digits, dots, parens, quotes or
# underscores — so we never touch identifiers, numbers, or code tokens.
_PROSE_WORD_RE = re.compile(r"[A-Za-z]{4,}")


@dataclass
class PerturbedProblem:
    """A single perturbed variant of a coding problem.

    ``problem`` is a full ``CodingProblem`` whose ``prompt`` has had its docstring
    reworded; every other field (``test_code``, ``entry_point``, signature) is
    identical to the base. ``perturbation`` is the kind name ("clean" for the
    unmodified base).
    """

    base_task_id: str
    perturbation: str
    problem: CodingProblem


# ---------------------------------------------------------------------------
# Docstring location — AST-based, never string-guessing
# ---------------------------------------------------------------------------


def _find_function_def(tree: ast.Module, entry_point: str) -> ast.FunctionDef | None:
    """Return the top-level FunctionDef named ``entry_point`` (or None)."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _docstring_span(prompt: str, entry_point: str) -> tuple[int, int, str] | None:
    """Locate the entry-point function's docstring within ``prompt``.

    Returns ``(start_offset, end_offset, docstring_text)`` where the offsets are
    absolute character indices into ``prompt`` spanning the *contents* between the
    triple quotes (NOT the quotes themselves), or ``None`` when the prompt does not
    parse, does not define ``entry_point``, or has no (non-empty) docstring.
    """
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None

    fn = _find_function_def(tree, entry_point)
    if fn is None:
        return None
    if ast.get_docstring(fn, clean=False) is None:
        return None

    # The docstring is the first statement: an Expr wrapping a Constant str.
    first = fn.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
        return None
    const = first.value
    if not isinstance(const.value, str):
        return None

    lines = prompt.splitlines(keepends=True)
    # Convert (lineno, col_offset) -> absolute char offset. lineno is 1-based.
    line_starts: list[int] = []
    running = 0
    for ln in lines:
        line_starts.append(running)
        running += len(ln)

    node_start = line_starts[const.lineno - 1] + const.col_offset
    node_end = line_starts[const.end_lineno - 1] + const.end_col_offset

    raw = prompt[node_start:node_end]
    # Strip the surrounding quotes (''' / """ / ' / ") to expose just the prose.
    for quote in ('"""', "'''", '"', "'"):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= 2 * len(quote):
            inner_start = node_start + len(quote)
            inner_end = node_end - len(quote)
            return inner_start, inner_end, prompt[inner_start:inner_end]
    return None


def _rebuild(prompt: str, start: int, end: int, new_doc: str) -> str:
    """Splice ``new_doc`` back into ``prompt`` between ``start`` and ``end``."""
    return prompt[:start] + new_doc + prompt[end:]


# ---------------------------------------------------------------------------
# Individual perturbation transforms (docstring text -> docstring text)
# ---------------------------------------------------------------------------


def _is_doctest_region(doc: str) -> bool:
    return any(_DOCTEST_RE.match(line) for line in doc.splitlines())


def _typo(doc: str, rng: random.Random) -> str:
    """Swap two adjacent chars inside a single prose word (>=4 letters).

    Doctest (``>>>``) lines are never touched — a typo there could change a call.
    """
    lines = doc.splitlines(keepends=True)
    candidates: list[int] = [
        i for i, ln in enumerate(lines)
        if not _DOCTEST_RE.match(ln) and _PROSE_WORD_RE.search(ln)
    ]
    if not candidates:
        return doc
    idx = rng.choice(candidates)
    line = lines[idx]
    words = list(_PROSE_WORD_RE.finditer(line))
    m = rng.choice(words)
    word = m.group(0)
    # Swap two adjacent interior chars to keep it recognisably the same word.
    if len(word) < 4:
        return doc
    pos = rng.randint(1, len(word) - 3)
    swapped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    lines[idx] = line[: m.start()] + swapped + line[m.end():]
    return "".join(lines)


def _whitespace(doc: str, rng: random.Random) -> str:
    """Jitter indentation / blank lines inside the docstring prose.

    Doctest lines keep their exact leading whitespace (indentation is significant
    for the ``>>>`` continuation convention); only prose lines get jitter, and a
    blank line is inserted between paragraphs.
    """
    lines = doc.splitlines(keepends=True)
    out: list[str] = []
    for ln in lines:
        if _DOCTEST_RE.match(ln) or not ln.strip():
            out.append(ln)
            continue
        # Add 0-2 extra leading spaces to a prose line.
        out.append(" " * rng.randint(0, 2) + ln)
    # Insert one extra blank line near the middle if there is room.
    if len(out) > 1:
        pos = rng.randint(1, len(out) - 1)
        out.insert(pos, "\n")
    return "".join(out)


def _casing(doc: str, rng: random.Random) -> str:
    """Toggle sentence-start casing on a prose line (cosmetic only).

    Picks one non-doctest prose line and flips the case of its first letter.
    """
    lines = doc.splitlines(keepends=True)
    candidates = [
        i for i, ln in enumerate(lines)
        if not _DOCTEST_RE.match(ln) and any(c.isalpha() for c in ln)
    ]
    if not candidates:
        return doc
    idx = rng.choice(candidates)
    line = lines[idx]
    for j, ch in enumerate(line):
        if ch.isalpha():
            flipped = ch.lower() if ch.isupper() else ch.upper()
            lines[idx] = line[:j] + flipped + line[j + 1:]
            break
    return "".join(lines)


def _reorder_examples(doc: str, rng: random.Random) -> str:
    """Shuffle the order of ``>>>`` doctest example BLOCKS (order-invariant).

    Each example is a ``>>>`` line plus its following expected-output line(s) up to
    the next ``>>>`` or end. We permute whole blocks so each call stays attached to
    its expected result — only the order between independent examples changes.
    Needs >=2 example blocks to do anything.
    """
    lines = doc.splitlines(keepends=True)
    # Index of the first doctest line.
    first_dt = next((i for i, ln in enumerate(lines) if _DOCTEST_RE.match(ln)), None)
    if first_dt is None:
        return doc

    head = lines[:first_dt]
    tail = lines[first_dt:]

    blocks: list[list[str]] = []
    for ln in tail:
        if _DOCTEST_RE.match(ln):
            blocks.append([ln])
        elif blocks:
            blocks[-1].append(ln)
        else:
            head.append(ln)
    if len(blocks) < 2:
        return doc

    order = list(range(len(blocks)))
    # Guarantee a real permutation (not the identity) when possible.
    shuffled = order[:]
    rng.shuffle(shuffled)
    if shuffled == order and len(order) > 1:
        shuffled = order[1:] + order[:1]

    reordered: list[str] = []
    for i in shuffled:
        reordered.extend(blocks[i])
    return "".join(head) + "".join(reordered)


def _paraphrase(doc: str, rng: random.Random) -> str:
    """Replace prose words using the curated synonym map (meaning-preserving).

    Only whole prose words on non-doctest lines are swapped; the replacement keeps
    the original leading-capital casing (Return -> Output).
    """
    lines = doc.splitlines(keepends=True)
    out: list[str] = []
    for ln in lines:
        if _DOCTEST_RE.match(ln):
            out.append(ln)
            continue

        def _sub(match: re.Match[str]) -> str:
            word = match.group(0)
            repl = _PARAPHRASE_MAP.get(word.lower())
            if repl is None:
                return word
            if word[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            return repl

        out.append(_PROSE_WORD_RE.sub(_sub, ln))
    return "".join(out)


_TRANSFORMS = {
    "typo": _typo,
    "whitespace": _whitespace,
    "casing": _casing,
    "reorder_examples": _reorder_examples,
    "paraphrase": _paraphrase,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def perturb_docstring(
    problem: CodingProblem,
    kinds: list[str] | None = None,
    seed: int = 42,
) -> list[PerturbedProblem]:
    """Produce semantically-preserving docstring variants of one problem.

    Always returns the unmodified base as the first entry (``perturbation="clean"``).
    For each requested kind, a variant is emitted ONLY when it both (a) actually
    changes the prompt and (b) still AST-parses to a function defining
    ``problem.entry_point`` — variants that would change the task are skipped.

    A problem whose ``prompt`` has no (non-empty) docstring yields the clean entry
    only (nothing to safely perturb).

    Args:
        problem: The base coding problem.
        kinds: Perturbation kinds to attempt (defaults to all). Unknown kinds raise.
        seed: Seed for reproducible perturbation (per-kind derived offset).

    Returns:
        ``[clean, *valid_variants]`` as ``PerturbedProblem`` objects.
    """
    if kinds is None:
        kinds = list(ALL_KINDS)
    unknown = [k for k in kinds if k not in _TRANSFORMS]
    if unknown:
        raise ValueError(f"Unknown perturbation kind(s): {unknown}. Valid: {list(ALL_KINDS)}")

    clean = PerturbedProblem(
        base_task_id=problem.task_id,
        perturbation="clean",
        problem=problem,
    )
    results: list[PerturbedProblem] = [clean]

    span = _docstring_span(problem.prompt, problem.entry_point)
    if span is None:
        return results  # No safely-perturbable docstring — clean only.
    start, end, doc = span

    for offset, kind in enumerate(kinds):
        rng = random.Random(seed + offset)
        new_doc = _TRANSFORMS[kind](doc, rng)
        if new_doc == doc:
            continue  # No-op for this problem — don't emit a duplicate of clean.
        new_prompt = _rebuild(problem.prompt, start, end, new_doc)

        # HARD INVARIANT re-check: the rebuilt prompt must still define entry_point.
        recheck = _docstring_span(new_prompt, problem.entry_point)
        if recheck is None:
            continue  # Perturbation broke the parse / dropped the def — skip it.

        variant = replace(problem, prompt=new_prompt)
        results.append(
            PerturbedProblem(
                base_task_id=problem.task_id,
                perturbation=kind,
                problem=variant,
            )
        )

    return results


def perturb_problem_set(
    problem_set,
    kinds: list[str] | None = None,
    seed: int = 42,
) -> dict[str, list[PerturbedProblem]]:
    """Perturb every problem in an iterable ``problem_set`` (e.g. ``ProblemSet``).

    Args:
        problem_set: Any iterable of ``CodingProblem`` (``ProblemSet`` qualifies).
        kinds: Perturbation kinds to attempt (defaults to all).
        seed: Base seed; each problem reuses the same seed so a given problem is
            perturbed identically regardless of position.

    Returns:
        ``{task_id: [clean, *variants]}``.
    """
    out: dict[str, list[PerturbedProblem]] = {}
    for problem in problem_set:
        out[problem.task_id] = perturb_docstring(problem, kinds=kinds, seed=seed)
    return out
