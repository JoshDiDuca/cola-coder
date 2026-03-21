"""Function Extraction Analyzer (improvement #64).

Extract and analyze functions from Python source code:
  - signatures (name, args, return type, decorators)
  - cyclomatic complexity estimate
  - dependencies (names referenced inside the function body)
  - call graph (which functions call which)
  - training sample generation helpers

TypeScript analogy: like a lightweight version of ts-morph's
SourceFile.getFunctions() combined with a dependency graph builder.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if function extraction is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FunctionSignature:
    """Parsed function signature."""

    name: str
    params: List[str]
    param_annotations: Dict[str, Optional[str]]
    return_annotation: Optional[str]
    decorators: List[str]
    is_async: bool
    is_method: bool


@dataclass
class FunctionRecord:
    """Full record for an extracted function."""

    signature: FunctionSignature
    source: str                       # raw source text of the function
    start_line: int
    end_line: int
    complexity: int                   # cyclomatic complexity estimate
    calls: List[str] = field(default_factory=list)        # functions this one calls
    name_refs: Set[str] = field(default_factory=set)      # external names referenced
    docstring: Optional[str] = None

    @property
    def name(self) -> str:
        return self.signature.name

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class CallGraphReport:
    """Call graph for a module."""

    functions: List[FunctionRecord]
    # caller -> set of callees
    edges: Dict[str, Set[str]] = field(default_factory=dict)

    def callers_of(self, name: str) -> List[str]:
        return [fn for fn, callees in self.edges.items() if name in callees]

    def callees_of(self, name: str) -> List[str]:
        return list(self.edges.get(name, set()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _annotation_to_str(node: Optional[ast.expr]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _estimate_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Estimate cyclomatic complexity (1 + branching nodes)."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.While,
                ast.For,
                ast.AsyncFor,
                ast.ExceptHandler,
                ast.With,
                ast.AsyncWith,
                ast.Assert,
                ast.comprehension,
            ),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp) and isinstance(child.op, (ast.And, ast.Or)):
            complexity += len(child.values) - 1
    return complexity


def _collect_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
    """Collect names of functions called inside this function."""
    calls: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def _collect_name_refs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Set[str]:
    """Collect external name references (excluding local vars and params)."""
    local_names: Set[str] = set()
    all_names: Set[str] = set()

    # Params are local
    args = node.args
    for arg in (
        list(args.args)
        + list(args.posonlyargs)
        + list(args.kwonlyargs)
    ):
        local_names.add(arg.arg)
    if args.vararg:
        local_names.add(args.vararg.arg)
    if args.kwarg:
        local_names.add(args.kwarg.arg)

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in ("True", "False", "None"):
            all_names.add(child.id)
        if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        local_names.add(t.id)
            elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                local_names.add(child.target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                local_names.add(child.target.id)

    return all_names - local_names


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FunctionExtractor:
    """Extract and analyse functions from Python source code."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, source: str) -> List[FunctionRecord]:
        """Extract all top-level and class functions from source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines()
        records: List[FunctionRecord] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record = self._build_record(node, lines)
                records.append(record)

        return records

    def build_call_graph(self, source: str) -> CallGraphReport:
        """Build a call graph for the module."""
        records = self.extract(source)
        function_names = {r.name for r in records}
        edges: Dict[str, Set[str]] = {}
        for rec in records:
            # Only include calls to other functions in this module
            internal_calls = {c for c in rec.calls if c in function_names}
            edges[rec.name] = internal_calls
        return CallGraphReport(functions=records, edges=edges)

    def to_training_samples(
        self, source: str, include_signature: bool = True
    ) -> List[Dict]:
        """Convert extracted functions to training sample dicts.

        Each sample has:
          - "prompt": function signature (optionally with docstring)
          - "completion": function body
          - "metadata": complexity, line_count, etc.
        """
        records = self.extract(source)
        samples = []
        for rec in records:
            body_lines = rec.source.splitlines()
            if not body_lines:
                continue
            # Split signature from body
            sig_end = 1
            for i, ln in enumerate(body_lines):
                if ln.rstrip().endswith(":"):
                    sig_end = i + 1
                    break
            sig = "\n".join(body_lines[:sig_end])
            body = textwrap.dedent("\n".join(body_lines[sig_end:]))
            if include_signature and rec.signature.name:
                prompt = sig
            else:
                prompt = f"# Implement function '{rec.signature.name}'"
            samples.append(
                {
                    "prompt": prompt,
                    "completion": body,
                    "metadata": {
                        "name": rec.name,
                        "complexity": rec.complexity,
                        "line_count": rec.line_count,
                        "is_async": rec.signature.is_async,
                        "has_docstring": rec.docstring is not None,
                    },
                }
            )
        return samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_record(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
    ) -> FunctionRecord:
        # Signature
        args = node.args
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        params = [a.arg for a in all_args]
        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")
        param_annotations = {
            a.arg: _annotation_to_str(a.annotation) for a in all_args
        }
        decorators = []
        for d in node.decorator_list:
            try:
                decorators.append(ast.unparse(d))
            except Exception:
                decorators.append("@<unknown>")
        is_method = any(p in ("self", "cls") for p in params)
        sig = FunctionSignature(
            name=node.name,
            params=params,
            param_annotations=param_annotations,
            return_annotation=_annotation_to_str(node.returns),
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
        )

        # Source extraction
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        func_source = "\n".join(lines[start : end + 1])

        # Docstring
        docstring: Optional[str] = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        return FunctionRecord(
            signature=sig,
            source=func_source,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            complexity=_estimate_complexity(node),
            calls=_collect_calls(node),
            name_refs=_collect_name_refs(node),
            docstring=docstring,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def extract_functions(source: str) -> List[FunctionRecord]:
    """Extract function records with default settings."""
    return FunctionExtractor().extract(source)
