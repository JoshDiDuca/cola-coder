"""Code-to-instruction pair generator (OSS-Instruct style).

Extracts functions and classes from source code files using regex,
then generates instruction-response pairs in three styles:
  1. Write/Implement/Create instructions from code structure
  2. Explain-the-code pairs
  3. Fix-the-bug pairs (randomly introduce a bug, ask model to fix it)

Output format follows the ChatML messages convention used by SFTDataset:
    {"messages": [
        {"role": "system", "content": "You are a helpful code assistant."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]}

For a TS dev: think of this like a test-data factory that reads real code,
reverse-engineers what the "interview question" would have been, and
produces (question, answer) pairs for training.

Usage:
    from cola_coder.data.sources.instruction_gen import CodeToInstructionGenerator

    gen = CodeToInstructionGenerator(source_dir="./my-code")
    examples = gen.generate(num_samples=500, quality_threshold=0.6)
    CodeToInstructionGenerator.save_jsonl(examples, "sft_data.jsonl")
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = "You are a helpful code assistant."

# ---------------------------------------------------------------------------
# Regex patterns for code extraction (avoids AST import errors on
# incomplete / foreign-language code)
# ---------------------------------------------------------------------------

# Python: def name(...): or async def name(...):
_PY_FUNC_RE = re.compile(
    r"(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:",
    re.MULTILINE,
)

# Python: class Name(...):
_PY_CLASS_RE = re.compile(
    r"class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:",
    re.MULTILINE,
)

# JS/TS: [export] [async] function name(...)
_JS_FUNC_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)

# JS/TS: [export] class Name
_JS_CLASS_RE = re.compile(
    r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

_WRITE_TEMPLATES = [
    "Write a function called `{name}` that {description}.",
    "Implement a function `{name}` that {description}.",
    "Create a function named `{name}` that {description}.",
    "Write a `{name}` function that {description}.",
    "Implement the `{name}` function. It should {description}.",
    "Define a function `{name}` that {description}.",
]

_CLASS_TEMPLATES = [
    "Implement a class called `{name}` that {description}.",
    "Create a class named `{name}` that {description}.",
    "Write a `{name}` class that {description}.",
    "Define a class `{name}` that {description}.",
]

_EXPLAIN_PREFIX = [
    "Explain what the following code does:",
    "Describe the purpose of this code:",
    "What does this code do? Explain step by step:",
    "Read the following code and explain its functionality:",
]

_FIX_PREFIX = [
    "The following code has a bug. Find and fix it:",
    "Fix the bug in this code:",
    "There is an error in the code below. Correct it:",
    "Debug the following code and provide the fixed version:",
]

# ---------------------------------------------------------------------------
# Bug injection strategies for fix-the-bug pairs
# ---------------------------------------------------------------------------

_BUG_INJECTIONS: list[tuple[re.Pattern, str, str]] = [
    # Off-by-one: range(n) -> range(n-1)
    (re.compile(r"range\((\w+)\)"), r"range(\1 - 1)", "off-by-one in range"),
    # Wrong comparison: == -> !=
    (re.compile(r"(\w+)\s*==\s*(\w+)"), r"\1 != \2", "inverted comparison"),
    # Wrong operator: + -> -
    (re.compile(r"(\w+)\s*\+\s*(\w+)"), r"\1 - \2", "wrong arithmetic operator"),
    # Missing return (Python)
    (re.compile(r"(\s+)return "), r"\1# return ", "missing return statement"),
    # Swap True/False
    (re.compile(r"\bTrue\b"), "False", "boolean swap"),
    (re.compile(r"\btrue\b"), "false", "boolean swap"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_indented_block(code: str, start: int) -> str:
    """Extract a Python-style indented block starting from a match position."""
    lines = code[start:].split("\n")
    if not lines:
        return ""

    result_lines = [lines[0]]
    if len(lines) < 2:
        return lines[0]

    body_indent = None
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            result_lines.append(line)
            continue
        indent = len(line) - len(stripped)
        if body_indent is None:
            body_indent = indent
        if indent < body_indent and stripped:
            break
        result_lines.append(line)

    return "\n".join(result_lines)


def _extract_brace_block(code: str, start: int) -> str | None:
    """Extract a brace-delimited block starting at or after *start*."""
    brace_pos = code.find("{", start)
    if brace_pos == -1:
        return None

    depth = 0
    in_string: str | None = None
    i = brace_pos
    while i < len(code):
        ch = code[i]
        if in_string:
            if ch == "\\" and i + 1 < len(code):
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'", "`"):
            in_string = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code[brace_pos: i + 1]
        i += 1

    return None


def _extract_docstring(code: str) -> str:
    """Extract the first docstring or JSDoc comment from a code block."""
    # Python triple-quote docstring
    m = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", code, re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()

    # JSDoc /** ... */
    m = re.search(r"/\*\*\s*(.*?)\s*\*/", code, re.DOTALL)
    if m:
        first = m.group(1).strip().split("\n")[0].strip().lstrip("* ")
        if len(first) > 5:
            return first

    return ""


def _name_to_description(name: str) -> str:
    """Convert camelCase / snake_case name to a rough English phrase."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    words = words.replace("_", " ").lower().strip()
    if not words:
        return "performs the required operations"
    return f"performs {words} operations"


def _detect_language(path: Path) -> str:
    """Return 'python', 'typescript', or 'javascript' from file extension."""
    ext = path.suffix.lower()
    if ext in (".py", ".pyw"):
        return "python"
    if ext in (".ts", ".tsx"):
        return "typescript"
    return "javascript"


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def score_quality(instruction: str, response: str) -> float:
    """Score an instruction-response pair on a 0-1 scale.

    Criteria:
      - Response length (min 20 chars)
      - Instruction non-trivial (min 10 chars)
      - Code syntax validity (ast.parse for Python)
      - No empty content
    """
    if not instruction or not response:
        return 0.0

    score = 0.3  # base

    # Instruction length
    if len(instruction.strip()) >= 10:
        score += 0.15
    if len(instruction.split()) >= 6:
        score += 0.05

    # Response length
    resp_stripped = response.strip()
    if len(resp_stripped) < 20:
        return 0.0  # too short to be useful
    if len(resp_stripped) >= 50:
        score += 0.1
    if len(resp_stripped) >= 200:
        score += 0.1

    # Syntax/"parses" bonus — language-aware so the quality signal is NOT
    # biased toward Python. cola-coder is TypeScript-primary; the old code only
    # awarded this for ast.parse-able Python, so an otherwise-identical TS/JS
    # pair scored 0.2 lower and a short TS response could fall below the keep
    # threshold while the Python equivalent passed. JS/TS earns the same bonus
    # via a balanced-brace check (its analogue of "parses").
    try:
        ast.parse(resp_stripped)
        score += 0.2
    except SyntaxError:
        opens = resp_stripped.count("{")
        closes = resp_stripped.count("}")
        if opens > 0 and opens == closes:
            score += 0.2

    # Has code-like content (indentation, braces, etc.)
    if any(ch in resp_stripped for ch in ("def ", "function ", "class ", "{")):
        score += 0.1

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class CodeToInstructionGenerator:
    """Generate instruction-response pairs from source code files.

    Reads ``.py``, ``.ts``, and ``.js`` files from a directory or a single
    file, extracts functions and classes via regex, and produces three kinds
    of training examples:

    1. **Write** — "Write a function that ..." / original code
    2. **Explain** — "Explain this code: ..." / generated explanation
    3. **Fix** — code with injected bug / original code
    """

    def __init__(
        self,
        source_dir: str | None = None,
        source_file: str | None = None,
    ):
        """
        Args:
            source_dir: Directory of .py/.ts/.js files to read.
            source_file: Path to a single source file.
        """
        if source_dir is None and source_file is None:
            raise ValueError(
                "Provide either source_dir or source_file."
            )
        self.source_dir = Path(source_dir) if source_dir else None
        self.source_file = Path(source_file) if source_file else None

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _iter_files(self) -> list[Path]:
        """Collect all eligible source files."""
        extensions = {".py", ".ts", ".tsx", ".js", ".jsx"}
        if self.source_file is not None:
            p = self.source_file
            if p.is_file() and p.suffix.lower() in extensions:
                return [p]
            return []

        assert self.source_dir is not None
        if not self.source_dir.is_dir():
            logger.warning("Source directory not found: %s", self.source_dir)
            return []

        files: list[Path] = []
        for ext in extensions:
            files.extend(self.source_dir.rglob(f"*{ext}"))
        return sorted(files)

    # ------------------------------------------------------------------
    # Code block extraction
    # ------------------------------------------------------------------

    def _extract_blocks(
        self, code: str, language: str
    ) -> list[dict[str, str]]:
        """Extract function/class blocks from source code.

        Returns list of dicts with keys: name, kind, params, return_type,
        body, docstring.
        """
        blocks: list[dict[str, str]] = []

        if language == "python":
            blocks.extend(self._extract_py_blocks(code))
        else:
            blocks.extend(self._extract_js_blocks(code))

        return blocks

    def _extract_py_blocks(self, code: str) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []

        for m in _PY_FUNC_RE.finditer(code):
            body = _extract_indented_block(code, m.start())
            if len(body.strip().split("\n")) < 3:
                continue
            blocks.append({
                "name": m.group(1),
                "kind": "function",
                "params": m.group(2).strip(),
                "return_type": (m.group(3) or "").strip(),
                "body": body.strip(),
                "docstring": _extract_docstring(body),
            })

        for m in _PY_CLASS_RE.finditer(code):
            body = _extract_indented_block(code, m.start())
            if len(body.strip().split("\n")) < 3:
                continue
            blocks.append({
                "name": m.group(1),
                "kind": "class",
                "params": "",
                "return_type": "",
                "body": body.strip(),
                "docstring": _extract_docstring(body),
            })

        return blocks

    def _extract_js_blocks(self, code: str) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []

        for m in _JS_FUNC_RE.finditer(code):
            brace_block = _extract_brace_block(code, m.start())
            if brace_block is None:
                continue
            brace_start = code.find("{", m.start())
            full = code[m.start(): brace_start + len(brace_block)]
            if len(full.strip().split("\n")) < 3:
                continue
            blocks.append({
                "name": m.group(1),
                "kind": "function",
                "params": m.group(2).strip(),
                "return_type": "",
                "body": full.strip(),
                "docstring": _extract_docstring(full),
            })

        for m in _JS_CLASS_RE.finditer(code):
            brace_block = _extract_brace_block(code, m.start())
            if brace_block is None:
                continue
            brace_start = code.find("{", m.start())
            full = code[m.start(): brace_start + len(brace_block)]
            if len(full.strip().split("\n")) < 3:
                continue
            blocks.append({
                "name": m.group(1),
                "kind": "class",
                "params": "",
                "return_type": "",
                "body": full.strip(),
                "docstring": _extract_docstring(full),
            })

        return blocks

    # ------------------------------------------------------------------
    # Pair generation strategies
    # ------------------------------------------------------------------

    def _make_write_pair(
        self, block: dict[str, str]
    ) -> dict | None:
        """Generate a Write/Implement/Create instruction pair."""
        name = block["name"]
        doc = block["docstring"]
        description = doc if doc else _name_to_description(name)

        if block["kind"] == "class":
            template = random.choice(_CLASS_TEMPLATES)
        else:
            template = random.choice(_WRITE_TEMPLATES)

        instruction = template.format(name=name, description=description)
        response = block["body"]

        q = score_quality(instruction, response)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "_quality": q,
        }

    def _make_explain_pair(
        self, block: dict[str, str]
    ) -> dict | None:
        """Generate an Explain-the-code instruction pair."""
        code = block["body"]
        prefix = random.choice(_EXPLAIN_PREFIX)
        instruction = f"{prefix}\n```\n{code}\n```"

        # Build a simple explanation from what we know
        name = block["name"]
        kind = block["kind"]
        doc = block["docstring"]
        params = block["params"]

        parts = [f"This {kind} `{name}`"]
        if doc:
            parts.append(f" {doc.rstrip('.')}.")
        else:
            parts.append(
                f" {_name_to_description(name).rstrip('.')}."
            )
        if params:
            param_names = [
                p.strip().split(":")[0].split("=")[0].strip()
                for p in params.split(",")
                if p.strip() and p.strip() not in ("self", "cls")
            ]
            if param_names:
                parts.append(
                    f" It takes the following parameters: "
                    f"{', '.join(f'`{p}`' for p in param_names)}."
                )
        lines = len(code.strip().split("\n"))
        parts.append(
            f" The implementation is {lines} lines long."
        )

        response = "".join(parts)
        q = score_quality(instruction, response)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "_quality": q,
        }

    def _make_fix_pair(
        self, block: dict[str, str]
    ) -> dict | None:
        """Generate a Fix-the-bug instruction pair."""
        code = block["body"]

        # Try each bug injection in random order until one matches. Shuffle a
        # COPY — random.shuffle(_BUG_INJECTIONS) would mutate the shared
        # module-level list as a side effect on every call.
        injections = list(_BUG_INJECTIONS)
        random.shuffle(injections)
        for pattern, replacement, _desc in injections:
            if pattern.search(code):
                buggy = pattern.sub(replacement, code, count=1)
                if buggy != code:
                    prefix = random.choice(_FIX_PREFIX)
                    instruction = f"{prefix}\n```\n{buggy}\n```"
                    response = code
                    q = score_quality(instruction, response)
                    return {
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": response},
                        ],
                        "_quality": q,
                    }

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_samples: int = 1000,
        quality_threshold: float = 0.6,
    ) -> list[dict]:
        """Generate instruction-response pairs from source files.

        Args:
            num_samples: Target number of examples to produce.
            quality_threshold: Minimum quality score (0-1) to keep.

        Returns:
            List of dicts, each with a ``"messages"`` key containing
            the ChatML-format conversation (system + user + assistant).
        """
        files = self._iter_files()
        if not files:
            logger.warning("No source files found.")
            return []

        logger.info("Found %d source files.", len(files))

        all_examples: list[dict] = []
        seen_instructions: set[str] = set()

        for path in files:
            if len(all_examples) >= num_samples:
                break

            try:
                code = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)
                continue

            language = _detect_language(path)
            blocks = self._extract_blocks(code, language)

            for block in blocks:
                if len(all_examples) >= num_samples:
                    break

                # Generate all three pair types for each block
                for maker in (
                    self._make_write_pair,
                    self._make_explain_pair,
                    self._make_fix_pair,
                ):
                    if len(all_examples) >= num_samples:
                        break

                    pair = maker(block)
                    if pair is None:
                        continue

                    quality = pair.pop("_quality", 0.0)
                    if quality < quality_threshold:
                        continue

                    # Deduplicate by user instruction
                    user_msg = pair["messages"][1]["content"]
                    key = user_msg[:200].lower().strip()
                    if key in seen_instructions:
                        continue
                    seen_instructions.add(key)

                    all_examples.append(pair)

        logger.info(
            "Generated %d examples (threshold=%.2f).",
            len(all_examples),
            quality_threshold,
        )
        return all_examples

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def save_jsonl(examples: list[dict], output_path: str) -> None:
        """Save examples to a JSONL file.

        Args:
            examples: List of message dicts as returned by ``generate()``.
            output_path: Destination file path.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved %d examples to %s", len(examples), output_path)
