"""Code Pattern Miner: extract frequent AST subtree patterns from code.

Mines common code idioms and language constructs from Python source files
using the built-in `ast` module.  Produces frequency tables of:
- Top-level construct patterns (function signatures, class hierarchies)
- Common expression idioms (list comprehensions, ternary, walrus, etc.)
- AST subtree fingerprints (node-type paths of depth ≤ N)

No external dependencies required.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the code pattern miner feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Pattern representation
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """A single extracted code pattern with its frequency."""

    signature: str  # Human-readable pattern description
    count: int
    examples: list[str] = field(default_factory=list)  # Up to 3 source snippets

    def as_dict(self) -> dict:
        return {
            "signature": self.signature,
            "count": self.count,
            "examples": self.examples,
        }


@dataclass
class MiningResult:
    """Results of mining patterns from a corpus of code snippets."""

    total_files: int
    total_nodes: int
    top_constructs: list[Pattern]  # Most frequent top-level constructs
    top_idioms: list[Pattern]  # Common expression idioms
    top_subtrees: list[Pattern]  # Most frequent AST subtree fingerprints
    anti_patterns: list[Pattern]  # Potential anti-patterns

    def summary(self) -> str:
        lines = [
            f"MiningResult(files={self.total_files}, nodes={self.total_nodes})",
            f"  Top constructs: {[p.signature for p in self.top_constructs[:5]]}",
            f"  Top idioms:     {[p.signature for p in self.top_idioms[:5]]}",
            f"  Top subtrees:   {[p.signature for p in self.top_subtrees[:5]]}",
            f"  Anti-patterns:  {[p.signature for p in self.anti_patterns[:3]]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_type_path(node: ast.AST, depth: int = 2) -> str:
    """Build a string fingerprint for a subtree of *node* up to *depth* levels."""
    parts = [type(node).__name__]
    if depth > 0:
        for child in ast.iter_child_nodes(node):
            parts.append(type(child).__name__)
    return ">".join(parts)


def _get_function_signature(node: ast.FunctionDef) -> str:
    """Create a simplified signature string from a FunctionDef node."""
    args = node.args
    n_args = (
        len(args.args)
        + len(args.posonlyargs)
        + len(args.kwonlyargs)
        + (1 if args.vararg else 0)
        + (1 if args.kwarg else 0)
    )
    has_return = node.returns is not None
    decorators = len(node.decorator_list)
    is_async = isinstance(node, ast.AsyncFunctionDef)
    parts = []
    if is_async:
        parts.append("async")
    parts.append(f"def({n_args}_args)")
    if has_return:
        parts.append("->annot")
    if decorators:
        parts.append(f"@{decorators}decs")
    return ":".join(parts)


_IDIOM_CHECKS = {
    "list_comprehension": ast.ListComp,
    "dict_comprehension": ast.DictComp,
    "set_comprehension": ast.SetComp,
    "generator_expression": ast.GeneratorExp,
    "conditional_expression": ast.IfExp,
    "walrus_operator": ast.NamedExpr,
    "lambda": ast.Lambda,
    "yield": ast.Yield,
    "yield_from": ast.YieldFrom,
    "await": ast.Await,
    "starred_unpack": ast.Starred,
    "fstring": ast.JoinedStr,
}

_ANTI_PATTERN_CHECKS = {
    "bare_except": ("ExceptHandler", lambda n: isinstance(n, ast.ExceptHandler) and n.type is None),
    "mutable_default_arg": ("FunctionDef", _check_mutable_default := None),  # filled below
    "assert_tuple": ("Assert", lambda n: isinstance(n, ast.Assert) and isinstance(n.test, ast.Tuple)),
    "unused_variable_x": ("Assign", lambda n: isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_" for t in n.targets
    )),
}


def _is_mutable_default(node: ast.FunctionDef) -> bool:
    """Detect mutable default arguments (list/dict/set literals)."""
    for default in node.args.defaults + node.args.kw_defaults:
        if default is None:
            continue
        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
            return True
    return False


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

class CodePatternMiner:
    """Mine AST patterns from a list of Python source code strings."""

    def __init__(self, subtree_depth: int = 2, max_examples: int = 3) -> None:
        self.subtree_depth = subtree_depth
        self.max_examples = max_examples

    def mine(self, code_samples: list[str]) -> MiningResult:
        """Mine patterns from multiple code samples.

        Parameters
        ----------
        code_samples:
            List of Python source code strings.

        Returns
        -------
        MiningResult with pattern frequencies.
        """
        construct_counter: Counter[str] = Counter()
        idiom_counter: Counter[str] = Counter()
        subtree_counter: Counter[str] = Counter()
        anti_counter: Counter[str] = Counter()

        # Store examples
        construct_examples: dict[str, list[str]] = defaultdict(list)
        idiom_examples: dict[str, list[str]] = defaultdict(list)
        anti_examples: dict[str, list[str]] = defaultdict(list)

        total_nodes = 0
        parsed_count = 0

        for code in code_samples:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            parsed_count += 1
            source_lines = code.splitlines()

            for node in ast.walk(tree):
                total_nodes += 1

                # Subtree fingerprints
                fp = _node_type_path(node, self.subtree_depth)
                subtree_counter[fp] += 1

                # Top-level constructs
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = _get_function_signature(node)
                    construct_counter[sig] += 1
                    if len(construct_examples[sig]) < self.max_examples and hasattr(node, "lineno"):
                        snippet = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                        construct_examples[sig].append(snippet.strip())

                elif isinstance(node, ast.ClassDef):
                    n_bases = len(node.bases)
                    key = f"class({n_bases}_bases)"
                    construct_counter[key] += 1

                # Idioms
                for idiom_name, node_type in _IDIOM_CHECKS.items():
                    if isinstance(node, node_type):
                        idiom_counter[idiom_name] += 1
                        if len(idiom_examples[idiom_name]) < self.max_examples and hasattr(node, "lineno"):
                            lineno = node.lineno
                            if lineno <= len(source_lines):
                                idiom_examples[idiom_name].append(source_lines[lineno - 1].strip())
                        break

                # Anti-patterns
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    anti_counter["bare_except"] += 1
                    if len(anti_examples["bare_except"]) < self.max_examples and hasattr(node, "lineno"):
                        lineno = node.lineno
                        if lineno <= len(source_lines):
                            anti_examples["bare_except"].append(source_lines[lineno - 1].strip())

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _is_mutable_default(node):
                        anti_counter["mutable_default_arg"] += 1

                if isinstance(node, ast.Assert) and isinstance(node.test, ast.Tuple):
                    anti_counter["assert_tuple"] += 1

        def _to_patterns(counter: Counter, examples_dict: Optional[dict] = None, top_n: int = 20) -> list[Pattern]:
            patterns = []
            for sig, count in counter.most_common(top_n):
                exs = (examples_dict or {}).get(sig, [])
                patterns.append(Pattern(signature=sig, count=count, examples=exs))
            return patterns

        return MiningResult(
            total_files=parsed_count,
            total_nodes=total_nodes,
            top_constructs=_to_patterns(construct_counter, construct_examples),
            top_idioms=_to_patterns(idiom_counter, idiom_examples),
            top_subtrees=_to_patterns(subtree_counter),
            anti_patterns=_to_patterns(anti_counter, anti_examples),
        )

    def find_pattern(self, code: str, pattern_name: str) -> int:
        """Count occurrences of a named idiom in one code snippet.

        Pattern names match the keys in _IDIOM_CHECKS.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0

        node_type = _IDIOM_CHECKS.get(pattern_name)
        if node_type is None:
            return 0

        return sum(1 for node in ast.walk(tree) if isinstance(node, node_type))

    def compare_corpora(self, corpus_a: list[str], corpus_b: list[str]) -> dict:
        """Mine both corpora and return a diff of top idiom frequencies."""
        r_a = self.mine(corpus_a)
        r_b = self.mine(corpus_b)

        a_idioms = {p.signature: p.count for p in r_a.top_idioms}
        b_idioms = {p.signature: p.count for p in r_b.top_idioms}

        all_keys = set(a_idioms) | set(b_idioms)
        diff = {
            k: {
                "a": a_idioms.get(k, 0),
                "b": b_idioms.get(k, 0),
                "delta": b_idioms.get(k, 0) - a_idioms.get(k, 0),
            }
            for k in all_keys
        }
        return diff
