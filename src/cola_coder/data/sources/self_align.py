"""SelfCodeAlign: Generate instruction-tuning data from raw code.

Pipeline:
1. Extract "seed snippets" from raw code (functions, classes, patterns)
2. For each seed, generate a natural language instruction
3. Generate a solution following the instruction
4. Filter: keep only high-quality instruction-solution pairs
5. Output as instruction-tuning training examples

This can use:
- An external LLM (Claude/GPT API) for generation -- best quality
- The model itself (self-instruct) -- for bootstrapping
- Templates + code transformation -- cheapest, no API needed

For a TS dev: think of this like generating coding interview questions
from real code, then generating model answers.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from cola_coder.data.pipeline import DataRecord, DataSource
from cola_coder.data.registry import register_source


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InstructionExample:
    """A single instruction-tuning example.

    Like a coding interview question + model answer, derived from real code.
    """
    instruction: str       # "Write a function that..."
    input_context: str     # Optional context/setup code
    output: str            # The solution code
    seed_code: str         # Original code this was derived from
    quality_score: float   # 0.0-1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "instruction": self.instruction,
            "input": self.input_context,
            "output": self.output,
            "seed_code": self.seed_code,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstructionExample:
        """Deserialize from a dict."""
        return cls(
            instruction=d["instruction"],
            input_context=d.get("input", ""),
            output=d["output"],
            seed_code=d.get("seed_code", ""),
            quality_score=d.get("quality_score", 0.0),
        )

    def to_training_text(self) -> str:
        """Format as a training string for instruction tuning.

        Uses a simple ### Instruction / ### Response format that's common
        in instruction-tuning datasets.
        """
        parts = ["### Instruction", self.instruction]
        if self.input_context:
            parts.extend(["### Input", self.input_context])
        parts.extend(["### Response", self.output])
        return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Seed extraction
# ---------------------------------------------------------------------------

# Regex patterns for extracting code constructs.
# These work for TypeScript/JavaScript and are intentionally simple --
# a full AST parser would be better but adds a heavy dependency.
#
# Strategy: match just the declaration keyword + name, then use
# _extract_brace_block() to find the full body. This avoids the
# complexity of matching multi-line params, nested generics, etc.

# Matches: [export] [async] function name
_FUNC_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)",
    re.MULTILINE,
)

# Matches: [export] [abstract] class Name
_CLASS_RE = re.compile(
    r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
    re.MULTILINE,
)

# Matches: [export] interface Name
_INTERFACE_RE = re.compile(
    r"(?:export\s+)?interface\s+(\w+)",
    re.MULTILINE,
)

# Matches: [export] const/let name = ... =>
_ARROW_RE = re.compile(
    r"(?:export\s+)?(?:const|let)\s+(\w+)\s*(?::[^=]*?)?\s*=",
    re.MULTILINE,
)

# Matches Python: def name(args): or async def name(args):
_PY_FUNC_RE = re.compile(
    r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:",
    re.MULTILINE,
)

# Matches Python: class Name:
_PY_CLASS_RE = re.compile(
    r"class\s+(\w+)\s*(?:\([^)]*\))?\s*:",
    re.MULTILINE,
)


def _extract_brace_block(code: str, start: int) -> str | None:
    """Extract a brace-delimited block starting at the given position.

    Finds the opening '{' at or after `start`, then finds the matching '}'.
    Returns the full block including braces, or None if unbalanced.
    """
    brace_pos = code.find("{", start)
    if brace_pos == -1:
        return None

    depth = 0
    in_string: str | None = None
    i = brace_pos
    while i < len(code):
        ch = code[i]

        # Handle string literals (skip their contents)
        if in_string:
            if ch == "\\" and i + 1 < len(code):
                i += 2  # skip escaped char
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
                return code[brace_pos : i + 1]
        i += 1

    return None  # Unbalanced


def _extract_indented_block(code: str, start: int) -> str:
    """Extract a Python-style indented block starting after a colon.

    Returns lines that are indented more than the definition line.
    """
    lines = code[start:].split("\n")
    if not lines:
        return ""

    # First line is the def/class line
    result_lines = [lines[0]]
    if len(lines) < 2:
        return lines[0]

    # Determine the indentation of the body
    body_indent = None
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:  # blank line
            result_lines.append(line)
            continue
        indent = len(line) - len(stripped)
        if body_indent is None:
            body_indent = indent
        if indent < body_indent and stripped:
            break
        result_lines.append(line)

    return "\n".join(result_lines)


class SeedExtractor:
    """Extract seed snippets from raw code for instruction generation.

    Extracts:
    - Standalone functions with type annotations
    - Classes with methods
    - Interfaces and type definitions
    - Arrow function expressions
    - Algorithm implementations

    Works for TypeScript/JavaScript and Python. Uses regex-based extraction
    (not a full AST) to keep dependencies minimal.
    """

    def __init__(
        self,
        min_lines: int = 3,
        max_lines: int = 100,
        require_types: bool = False,
    ):
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.require_types = require_types

    def extract_seeds(self, code: str, language: str = "typescript") -> list[str]:
        """Extract interesting code snippets as seeds.

        Args:
            code: Full source code content.
            language: Programming language ("typescript", "javascript", "python").

        Returns:
            List of self-contained code snippets suitable as seeds.
        """
        if language in ("typescript", "javascript"):
            return self._extract_ts_seeds(code)
        elif language == "python":
            return self._extract_py_seeds(code)
        else:
            # Fallback: try TypeScript patterns, then Python
            seeds = self._extract_ts_seeds(code)
            if not seeds:
                seeds = self._extract_py_seeds(code)
            return seeds

    def _extract_ts_seeds(self, code: str) -> list[str]:
        """Extract seeds from TypeScript/JavaScript code."""
        seeds: list[str] = []
        seen_ranges: list[tuple[int, int]] = []  # Avoid overlapping extractions

        def _overlaps(start: int, end: int) -> bool:
            return any(s <= start < e or s < end <= e for s, e in seen_ranges)

        # Extract functions, classes, interfaces
        for pattern in [_FUNC_RE, _CLASS_RE, _INTERFACE_RE]:
            for match in pattern.finditer(code):
                block = _extract_brace_block(code, match.start())
                if block is None:
                    continue
                # Find where the opening brace is relative to match start
                brace_start = code.find("{", match.start())
                if brace_start == -1:
                    continue
                snippet_end = brace_start + len(block)
                if _overlaps(match.start(), snippet_end):
                    continue
                snippet = code[match.start() : snippet_end]
                if self._is_valid_seed(snippet):
                    seeds.append(snippet.strip())
                    seen_ranges.append((match.start(), snippet_end))

        # Extract arrow functions (only if they have a brace body)
        for match in _ARROW_RE.finditer(code):
            # Look for => { after the match
            rest = code[match.end():]
            arrow_pos = rest.find("=>")
            if arrow_pos == -1 or arrow_pos > 200:
                continue
            brace_search_start = match.end() + arrow_pos + 2
            brace_pos = code.find("{", brace_search_start)
            if brace_pos == -1 or brace_pos - brace_search_start > 20:
                continue
            block = _extract_brace_block(code, brace_pos)
            if block is None:
                continue
            snippet_end = brace_pos + len(block)
            if _overlaps(match.start(), snippet_end):
                continue
            snippet = code[match.start() : snippet_end]
            if self._is_valid_seed(snippet):
                seeds.append(snippet.strip())
                seen_ranges.append((match.start(), snippet_end))

        return seeds

    def _extract_py_seeds(self, code: str) -> list[str]:
        """Extract seeds from Python code."""
        seeds: list[str] = []

        for pattern in [_PY_FUNC_RE, _PY_CLASS_RE]:
            for match in pattern.finditer(code):
                snippet = _extract_indented_block(code, match.start())
                if self._is_valid_seed(snippet):
                    seeds.append(snippet.strip())

        return seeds

    def _is_valid_seed(self, snippet: str) -> bool:
        """Check if a snippet is worth using as a seed."""
        lines = snippet.strip().split("\n")
        num_lines = len(lines)

        if num_lines < self.min_lines or num_lines > self.max_lines:
            return False

        # Skip trivially short code (getters, empty constructors)
        non_empty = [ln for ln in lines if ln.strip() and not ln.strip().startswith("//")]
        if len(non_empty) < self.min_lines:
            return False

        # Optionally require type annotations (TypeScript)
        if self.require_types:
            has_types = any(
                ":" in ln and not ln.strip().startswith("//")
                for ln in lines[1:]  # skip first line (function signature usually has types)
            )
            if not has_types and ":" not in lines[0]:
                return False

        return True


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

class InstructionGenerator:
    """Generate instructions from code seeds.

    Three modes:
    - template: Use templates to create instructions (free, fast)
    - llm: Use Claude/GPT API to create instructions (best quality)
    - self: Use the model itself (requires a trained base model)

    Template mode is the default and works with zero external dependencies.
    """

    # Templates for different code patterns.
    # {name} = extracted function/class name
    # {seed} = the full seed code
    # {params} = extracted parameter descriptions
    # {return_type} = extracted return type
    FUNCTION_TEMPLATES = [
        "Write a TypeScript function called `{name}` that {description}.",
        "Implement a function `{name}` that takes {params} and returns {return_type}.",
        "Create a TypeScript function that {description}. Name it `{name}`.",
        "Write a function that {description} with proper type annotations.",
    ]

    CLASS_TEMPLATES = [
        "Implement a TypeScript class called `{name}` that {description}.",
        "Create a class `{name}` with the following methods: {methods}.",
        "Write a TypeScript class that {description}.",
    ]

    INTERFACE_TEMPLATES = [
        "Define a TypeScript interface called `{name}` that {description}.",
        "Create a TypeScript interface `{name}` for {description}.",
    ]

    REFACTOR_TEMPLATES = [
        "Refactor the following code to be more readable and type-safe:\n```\n{seed}\n```",
        "Add proper TypeScript type annotations to this code:\n```\n{seed}\n```",
        "Write tests for the following function:\n```\n{seed}\n```",
        "Add error handling to the following code:\n```\n{seed}\n```",
        "Optimize the following code for performance:\n```\n{seed}\n```",
    ]

    def __init__(self, mode: str = "template"):
        """Initialize the instruction generator.

        Args:
            mode: Generation mode — "template", "llm", or "self".
        """
        if mode not in ("template", "llm", "self"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'template', 'llm', or 'self'.")
        self.mode = mode

    def generate(
        self, seed: str, language: str = "typescript"
    ) -> InstructionExample | None:
        """Generate an instruction-solution pair from a seed snippet.

        Args:
            seed: A code snippet extracted by SeedExtractor.
            language: The programming language.

        Returns:
            An InstructionExample, or None if generation failed.
        """
        if self.mode == "template":
            return self._generate_template(seed, language)
        elif self.mode == "llm":
            return self._generate_llm(seed, language)
        elif self.mode == "self":
            return self._generate_self(seed, language)
        return None

    def _generate_template(
        self, seed: str, language: str
    ) -> InstructionExample | None:
        """Generate instruction using pattern matching + templates.

        Analyzes the seed code to determine its type (function, class, etc.)
        and fills in a template with extracted information.
        """
        info = self._analyze_seed(seed, language)
        if info is None:
            return None

        kind = info["kind"]
        name = info.get("name", "unknown")
        description = info.get("description", f"performs the operations shown in `{name}`")

        # Pick template based on code kind
        if kind == "function":
            templates = self.FUNCTION_TEMPLATES
            fill = {
                "name": name,
                "description": description,
                "params": info.get("params", "the appropriate parameters"),
                "return_type": info.get("return_type", "the appropriate type"),
            }
        elif kind == "class":
            templates = self.CLASS_TEMPLATES
            fill = {
                "name": name,
                "description": description,
                "methods": info.get("methods", "appropriate methods"),
            }
        elif kind == "interface":
            templates = self.INTERFACE_TEMPLATES
            fill = {
                "name": name,
                "description": description,
            }
        else:
            # Fallback to refactor templates
            templates = self.REFACTOR_TEMPLATES
            fill = {"seed": seed}

        template = random.choice(templates)
        instruction = template.format(**fill)

        quality = self._score_quality(instruction, seed)

        return InstructionExample(
            instruction=instruction,
            input_context="",
            output=seed,
            seed_code=seed,
            quality_score=quality,
        )

    def _generate_llm(
        self, seed: str, language: str
    ) -> InstructionExample | None:
        """Generate instruction using an external LLM API.

        Requires ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
        Falls back to template mode if no API key is available.
        """
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "  Warning: No ANTHROPIC_API_KEY or OPENAI_API_KEY set. "
                "Falling back to template mode."
            )
            return self._generate_template(seed, language)

        # Use Anthropic if available, otherwise OpenAI
        if os.environ.get("ANTHROPIC_API_KEY"):
            return self._generate_with_anthropic(seed, language, api_key)
        else:
            return self._generate_with_openai(seed, language, api_key)

    def _generate_with_anthropic(
        self, seed: str, language: str, api_key: str
    ) -> InstructionExample | None:
        """Generate using Claude API."""
        try:
            import anthropic
        except ImportError:
            print("  Warning: 'anthropic' package not installed. pip install anthropic")
            return self._generate_template(seed, language)

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"Given this {language} code snippet, write a clear, concise instruction "
            f"that a developer would follow to produce similar code. "
            f"The instruction should be specific enough that a competent developer "
            f"could write the code from the instruction alone.\n\n"
            f"Code:\n```{language}\n{seed}\n```\n\n"
            f"Respond with ONLY the instruction text, nothing else."
        )

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            instruction = response.content[0].text.strip()

            return InstructionExample(
                instruction=instruction,
                input_context="",
                output=seed,
                seed_code=seed,
                quality_score=0.85,  # LLM-generated instructions are generally good
            )
        except Exception as e:
            print(f"  Warning: Anthropic API error: {e}")
            return self._generate_template(seed, language)

    def _generate_with_openai(
        self, seed: str, language: str, api_key: str
    ) -> InstructionExample | None:
        """Generate using OpenAI API."""
        try:
            import openai
        except ImportError:
            print("  Warning: 'openai' package not installed. pip install openai")
            return self._generate_template(seed, language)

        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"Given this {language} code snippet, write a clear, concise instruction "
            f"that a developer would follow to produce similar code.\n\n"
            f"Code:\n```{language}\n{seed}\n```\n\n"
            f"Respond with ONLY the instruction text."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            instruction = response.choices[0].message.content.strip()

            return InstructionExample(
                instruction=instruction,
                input_context="",
                output=seed,
                seed_code=seed,
                quality_score=0.80,
            )
        except Exception as e:
            print(f"  Warning: OpenAI API error: {e}")
            return self._generate_template(seed, language)

    def _generate_self(
        self, seed: str, language: str
    ) -> InstructionExample | None:
        """Generate instruction using cola-coder's own trained model.

        Requires a trained checkpoint. Falls back to template mode if
        no checkpoint is available.
        """
        print(
            "  Warning: Self-instruct mode requires a trained base model. "
            "Train a model first with scripts/train.py, then point to the "
            "checkpoint. Falling back to template mode."
        )
        return self._generate_template(seed, language)

    def _analyze_seed(self, seed: str, language: str) -> dict[str, Any] | None:
        """Analyze a seed snippet to extract structured information.

        Returns a dict with:
        - kind: "function", "class", "interface", or "other"
        - name: extracted name
        - description: best-effort description from docstring or logic
        - params: parameter descriptions (for functions)
        - return_type: return type (for functions)
        - methods: method names (for classes)
        """
        # Try TypeScript/JS patterns (use search, not match, since seeds
        # may start with comments/JSDoc before the declaration)
        func_match = _FUNC_RE.search(seed)
        if func_match:
            name = func_match.group(1)
            # Find the line containing the function signature for param extraction
            sig_line = seed[func_match.start():].split("\n")[0]
            return {
                "kind": "function",
                "name": name,
                "description": self._extract_description(seed, name),
                "params": self._extract_params(sig_line),
                "return_type": self._extract_return_type(seed),
            }

        class_match = _CLASS_RE.search(seed)
        if class_match:
            name = class_match.group(1)
            methods = self._extract_method_names(seed)
            return {
                "kind": "class",
                "name": name,
                "description": self._extract_description(seed, name),
                "methods": ", ".join(methods) if methods else "appropriate methods",
            }

        iface_match = _INTERFACE_RE.search(seed)
        if iface_match:
            name = iface_match.group(1)
            return {
                "kind": "interface",
                "name": name,
                "description": self._extract_description(seed, name),
            }

        arrow_match = _ARROW_RE.search(seed)
        if arrow_match:
            name = arrow_match.group(1)
            sig_line = seed[arrow_match.start():].split("\n")[0]
            return {
                "kind": "function",
                "name": name,
                "description": self._extract_description(seed, name),
                "params": self._extract_params(sig_line),
                "return_type": self._extract_return_type(seed),
            }

        # Try Python patterns
        py_func_match = _PY_FUNC_RE.search(seed)
        if py_func_match:
            name = py_func_match.group(1)
            sig_line = seed[py_func_match.start():].split("\n")[0]
            return {
                "kind": "function",
                "name": name,
                "description": self._extract_description(seed, name),
                "params": self._extract_params(sig_line),
                "return_type": self._extract_return_type(sig_line),
            }

        py_class_match = _PY_CLASS_RE.search(seed)
        if py_class_match:
            name = py_class_match.group(1)
            return {
                "kind": "class",
                "name": name,
                "description": self._extract_description(seed, name),
                "methods": "appropriate methods",
            }

        # Could not identify pattern
        return None

    def _extract_description(self, code: str, name: str) -> str:
        """Extract a description from docstring or JSDoc, or generate from name."""
        # Look for JSDoc: /** ... */
        jsdoc_match = re.search(r"/\*\*\s*(.*?)\s*\*/", code, re.DOTALL)
        if jsdoc_match:
            doc = jsdoc_match.group(1)
            # Take first sentence/line
            first_line = doc.strip().split("\n")[0].strip().lstrip("* ").strip()
            if len(first_line) > 10:
                return first_line

        # Look for Python docstring: """...""" or '''...'''
        pydoc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if not pydoc_match:
            pydoc_match = re.search(r"'''(.*?)'''", code, re.DOTALL)
        if pydoc_match:
            first_line = pydoc_match.group(1).strip().split("\n")[0].strip()
            if len(first_line) > 10:
                return first_line

        # Look for // comment on the line before or after the signature
        lines = code.strip().split("\n")
        for i, line in enumerate(lines[:3]):
            comment_match = re.match(r"\s*//\s*(.+)", line)
            if comment_match:
                return comment_match.group(1).strip()

        # Generate from camelCase/snake_case name
        return self._name_to_description(name)

    def _name_to_description(self, name: str) -> str:
        """Convert a function/class name to a rough description.

        camelCase → "camel case" → "performs camel case operations"
        snake_case → "snake case" → "performs snake case operations"
        """
        # Split camelCase
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        # Split snake_case
        words = words.replace("_", " ").lower()
        return f"performs {words} operations"

    def _extract_params(self, signature: str) -> str:
        """Extract parameter names from a function signature."""
        paren_match = re.search(r"\(([^)]*)\)", signature)
        if not paren_match:
            return "the appropriate parameters"

        params_str = paren_match.group(1).strip()
        if not params_str:
            return "no parameters"

        # Extract just the parameter names
        params = []
        for param in params_str.split(","):
            param = param.strip()
            # Remove type annotations (TypeScript: name: type, Python: name: type)
            name = param.split(":")[0].split("=")[0].strip()
            # Remove decorators, default values
            name = name.lstrip("*").strip()
            if name and name not in ("self", "cls"):
                params.append(f"`{name}`")

        return ", ".join(params) if params else "the appropriate parameters"

    def _extract_return_type(self, signature: str) -> str:
        """Extract return type from a function signature."""
        # TypeScript: ): ReturnType {
        ret_match = re.search(r"\)\s*:\s*([^{]+?)\s*\{", signature)
        if ret_match:
            return f"`{ret_match.group(1).strip()}`"

        # Python: -> ReturnType:
        ret_match = re.search(r"->\s*([^:]+)\s*:", signature)
        if ret_match:
            return f"`{ret_match.group(1).strip()}`"

        return "the appropriate type"

    def _extract_method_names(self, code: str) -> list[str]:
        """Extract method names from a class body."""
        # Match both TS methods (name() {) and Python methods (def name(self):)
        methods = re.findall(
            r"(?:(?:public|private|protected|static|async)\s+)*(\w+)\s*\(",
            code,
        )
        # Filter out constructor and common non-method matches
        skip = {"constructor", "if", "for", "while", "switch", "catch", "class", "function"}
        return [m for m in methods if m not in skip and not m.startswith("_")][:10]

    def _score_quality(self, instruction: str, seed: str) -> float:
        """Score the quality of an instruction-seed pair.

        Heuristic scoring:
        - Longer instructions score higher (more specific)
        - Instructions with backticks (code references) score higher
        - Longer seeds score higher (more substance)
        - Penalize very short or very long examples
        """
        score = 0.5  # Base score

        # Instruction quality
        instr_words = len(instruction.split())
        if instr_words >= 8:
            score += 0.1
        if instr_words >= 15:
            score += 0.05
        if "`" in instruction:
            score += 0.05  # Has code references

        # Seed quality
        seed_lines = len(seed.strip().split("\n"))
        if seed_lines >= 5:
            score += 0.1
        if seed_lines >= 10:
            score += 0.05
        if seed_lines > 80:
            score -= 0.1  # Too long

        # Has type annotations (TypeScript quality indicator)
        if ": " in seed and ("string" in seed or "number" in seed or "boolean" in seed):
            score += 0.1

        # Has docstring/JSDoc
        if "/**" in seed or '"""' in seed:
            score += 0.05

        return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class SelfAlignPipeline:
    """Full self-alignment pipeline.

    Orchestrates: load raw code -> extract seeds -> generate instructions -> filter.

    Like a build pipeline in a CI system — each stage transforms the data
    and the final output is a set of high-quality instruction examples.
    """

    def __init__(
        self,
        source: DataSource | None = None,
        mode: str = "template",
        min_quality: float = 0.5,
        language: str = "typescript",
        seed_extractor: SeedExtractor | None = None,
    ):
        """Initialize the pipeline.

        Args:
            source: A DataSource to read raw code from. If None, you must
                    pass code directly to generate_from_code().
            mode: Instruction generation mode ("template", "llm", "self").
            min_quality: Minimum quality score to keep an example (0.0-1.0).
            language: Programming language of the source code.
            seed_extractor: Custom SeedExtractor, or None for defaults.
        """
        self.source = source
        self.mode = mode
        self.min_quality = min_quality
        self.language = language
        self.extractor = seed_extractor or SeedExtractor()
        self.generator = InstructionGenerator(mode=mode)

    def generate(self, max_examples: int = 1000) -> list[InstructionExample]:
        """Generate instruction-tuning examples from the data source.

        Args:
            max_examples: Maximum number of examples to generate.

        Returns:
            List of filtered InstructionExamples.
        """
        if self.source is None:
            raise ValueError(
                "No data source configured. Pass a DataSource to __init__ "
                "or use generate_from_code() for ad-hoc generation."
            )

        examples: list[InstructionExample] = []
        seen_instructions: set[str] = set()  # Dedup by instruction text

        for record in self.source.stream():
            if len(examples) >= max_examples:
                break

            new_examples = self.generate_from_code(
                record.content,
                max_per_file=5,  # Don't over-extract from one file
            )

            for ex in new_examples:
                if len(examples) >= max_examples:
                    break
                # Dedup
                instr_key = ex.instruction.lower().strip()
                if instr_key in seen_instructions:
                    continue
                seen_instructions.add(instr_key)
                examples.append(ex)

        return examples

    def generate_from_code(
        self, code: str, max_per_file: int = 10
    ) -> list[InstructionExample]:
        """Generate instruction examples from a single code string.

        Useful for ad-hoc generation without setting up a full data source.

        Args:
            code: Raw source code.
            max_per_file: Maximum examples to extract from this code.

        Returns:
            List of filtered InstructionExamples.
        """
        seeds = self.extractor.extract_seeds(code, self.language)
        examples: list[InstructionExample] = []

        for seed in seeds[:max_per_file]:
            example = self.generator.generate(seed, self.language)
            if example is None:
                continue
            if example.quality_score >= self.min_quality:
                examples.append(example)

        return examples

    def save_jsonl(self, examples: list[InstructionExample], path: str | Path) -> None:
        """Save examples to a JSONL file.

        Args:
            path: Output file path.
            examples: List of InstructionExamples to save.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path: str | Path) -> list[InstructionExample]:
        """Load examples from a JSONL file."""
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(InstructionExample.from_dict(json.loads(line)))
        return examples


# ---------------------------------------------------------------------------
# DataSource adapter
# ---------------------------------------------------------------------------

@register_source("self_align")
class SelfAlignSource(DataSource):
    """DataSource that generates instruction-tuning examples on the fly.

    Wraps SelfAlignPipeline as a DataSource so it can be used in the
    standard data pipeline (YAML config, filters, transforms).

    Each yielded DataRecord contains a formatted instruction-response pair
    as its content, suitable for instruction-tuning training.

    Config example (configs/pipeline.yaml):
        sources:
          - type: self_align
            mode: template
            max_examples: 5000
            language: typescript
            source_type: local
            source_paths: ["./data/raw"]
    """

    def __init__(
        self,
        mode: str = "template",
        max_examples: int = 1000,
        language: str = "typescript",
        min_quality: float = 0.5,
        source_type: str | None = None,
        source_paths: list[str] | None = None,
        source_dataset: str | None = None,
    ):
        self._mode = mode
        self._max_examples = max_examples
        self._language = language
        self._min_quality = min_quality
        self._source_type = source_type
        self._source_paths = source_paths
        self._source_dataset = source_dataset

    def name(self) -> str:
        return f"self_align(mode={self._mode}, max={self._max_examples})"

    def _build_inner_source(self) -> DataSource | None:
        """Build the raw code source for seed extraction."""
        if self._source_type == "local" and self._source_paths:
            from cola_coder.data.sources.local import LocalFileSource
            ext_map = {
                "typescript": [".ts", ".tsx"],
                "javascript": [".js", ".jsx"],
                "python": [".py"],
            }
            extensions = ext_map.get(self._language, [".ts", ".tsx", ".js"])
            return LocalFileSource(paths=self._source_paths, extensions=extensions)
        elif self._source_type == "huggingface" and self._source_dataset:
            from cola_coder.data.sources.huggingface import HuggingFaceSource
            # MUST pass languages: HuggingFaceSource defaults to ["python"], so
            # without this a `language: typescript` self-align config would
            # download PYTHON code while the pipeline extracts TYPESCRIPT seeds
            # from it — yielding few/no seeds and empty/garbage SFT data.
            return HuggingFaceSource(
                dataset=self._source_dataset,
                languages=[self._language],
            )
        return None

    def stream(self) -> Iterator[DataRecord]:
        """Stream generated instruction examples as DataRecords."""
        inner_source = self._build_inner_source()
        pipeline = SelfAlignPipeline(
            source=inner_source,
            mode=self._mode,
            min_quality=self._min_quality,
            language=self._language,
        )

        if inner_source is not None:
            examples = pipeline.generate(max_examples=self._max_examples)
        else:
            # No source configured -- yield nothing
            examples = []

        for ex in examples:
            yield DataRecord(
                content=ex.to_training_text(),
                metadata={
                    "source": "self_align",
                    "mode": self._mode,
                    "instruction": ex.instruction,
                    "quality_score": ex.quality_score,
                    "seed_code": ex.seed_code[:200],  # Truncate for metadata
                },
            )

    def estimate_size(self) -> int | None:
        return self._max_examples
