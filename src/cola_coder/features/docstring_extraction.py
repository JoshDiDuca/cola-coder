"""Docstring Extraction: parse Python docstrings and TypeScript/JavaScript JSDoc comments.

Extracts documented functions from source code to create (docstring, function) pairs
for training data enrichment and supervised fine-tuning (SFT).

For a TS dev: like a static analysis tool that scrapes all JSDoc-annotated functions
and packages them as instruction-tuning pairs — similar to what you'd need to build
an AI that can "write a function given its docstring."
"""

import ast
import re
from dataclasses import dataclass, field

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocstringInfo:
    """Extracted docstring information for a single function."""
    function_name: str
    docstring: str
    language: str
    params: list[str] = field(default_factory=list)
    return_type: str | None = None
    line_number: int = 0


# ---------------------------------------------------------------------------
# Helpers — Python
# ---------------------------------------------------------------------------

def _parse_python_docstring_params(docstring: str) -> list[str]:
    """Extract parameter names from a Python docstring (Google/NumPy/Sphinx style)."""
    params: list[str] = []

    # Google/NumPy style: "    n: description" under "Args:" / "Parameters:" section
    args_section = re.search(
        r'(?:Args|Arguments|Parameters)\s*:\s*\n(.*?)(?:\n\s*(?:Returns|Raises|Notes|Examples|$))',
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    if args_section:
        for match in re.finditer(r'^\s+(\w+)\s*(?:\(.*?\))?\s*:', args_section.group(1), re.MULTILINE):
            params.append(match.group(1))

    # Sphinx style: ":param name:"
    if not params:
        for match in re.finditer(r':param\s+(?:\w+\s+)?(\w+)\s*:', docstring):
            params.append(match.group(1))

    return params


def _parse_python_return_type(docstring: str) -> str | None:
    """Extract return type hint from a Python docstring."""
    # Sphinx: :rtype: <type>
    m = re.search(r':rtype:\s*(.+)', docstring)
    if m:
        return m.group(1).strip()

    # Google style: "Returns:\n    int: ..." — grab first word after label
    m = re.search(r'Returns\s*:\s*\n\s+(\w[\w\[\], ]*?)(?:\s*:|\n)', docstring, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Helpers — JSDoc
# ---------------------------------------------------------------------------

_JSDOC_BLOCK_RE = re.compile(r'/\*\*(.*?)\*/', re.DOTALL)

# Matches: (export)? (async)? function NAME( ... )
_JSDOC_FUNC_RE = re.compile(
    r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
)
# Matches: (export)? const NAME = (async)? (...) =>
_JSDOC_ARROW_RE = re.compile(
    r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?\(([^)]*)\)\s*(?::[^=>{]+)?\s*=>',
)
# Matches: NAME(...) { — method shorthand
_JSDOC_METHOD_RE = re.compile(
    r'(?:(?:public|private|protected|static|async|override)\s+)*(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{',
)


def _clean_jsdoc_text(raw: str) -> str:
    """Strip leading * characters and whitespace from a raw JSDoc block."""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip().lstrip('*').strip()
        lines.append(stripped)
    return '\n'.join(lines).strip()


def _extract_jsdoc_description(cleaned: str) -> str:
    """Return only the description lines (before the first @tag)."""
    desc_lines = []
    for line in cleaned.splitlines():
        if line.startswith('@'):
            break
        desc_lines.append(line)
    return '\n'.join(desc_lines).strip()


def _extract_jsdoc_params(cleaned: str) -> list[str]:
    """Extract @param names from a cleaned JSDoc block."""
    params: list[str] = []
    # @param {type} name  or  @param name
    for m in re.finditer(r'@param\s+(?:\{[^}]*\}\s+)?(?:\[)?(\w+)', cleaned):
        params.append(m.group(1))
    return params


def _extract_jsdoc_return(cleaned: str) -> str | None:
    """Extract @returns / @return type from a cleaned JSDoc block."""
    m = re.search(r'@returns?\s+\{([^}]+)\}', cleaned)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class DocstringExtractor:
    """Extract docstrings/JSDoc comments from Python and TypeScript/JavaScript source code."""

    def extract(self, code: str, language: str = 'python') -> list[DocstringInfo]:
        """Extract DocstringInfo entries from source code.

        Args:
            code: Source code string.
            language: One of 'python', 'typescript', 'javascript', 'ts', 'js'.

        Returns:
            List of DocstringInfo instances, one per documented function.
        """
        lang = language.lower()
        if lang == 'python':
            return self.extract_python(code)
        if lang in ('typescript', 'javascript', 'ts', 'js', 'tsx', 'jsx'):
            return self.extract_jsdoc(code)
        # Fallback: try Python, then JS/TS
        py_results = self.extract_python(code)
        return py_results if py_results else self.extract_jsdoc(code)

    # ------------------------------------------------------------------
    # Python extraction
    # ------------------------------------------------------------------

    def extract_python(self, code: str) -> list[DocstringInfo]:
        """Extract Python docstrings using the ast module.

        Args:
            code: Python source code.

        Returns:
            List of DocstringInfo for every function/method with a docstring.
        """
        results: list[DocstringInfo] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to regex-based extraction on invalid Python
            return self._extract_python_regex(code)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            raw_docstring = ast.get_docstring(node)
            if not raw_docstring:
                continue

            # Parameter names (skip 'self' and 'cls')
            params = [
                arg.arg for arg in node.args.args
                if arg.arg not in ('self', 'cls')
            ]
            # Augment from docstring body if needed
            if not params:
                params = _parse_python_docstring_params(raw_docstring)

            # Return type: prefer annotation, fall back to docstring
            return_type: str | None = None
            if node.returns is not None:
                try:
                    return_type = ast.unparse(node.returns)
                except Exception:
                    pass
            if return_type is None:
                return_type = _parse_python_return_type(raw_docstring)

            results.append(DocstringInfo(
                function_name=node.name,
                docstring=raw_docstring,
                language='python',
                params=params,
                return_type=return_type,
                line_number=node.lineno,
            ))

        return results

    def _extract_python_regex(self, code: str) -> list[DocstringInfo]:
        """Regex fallback for Python docstring extraction (invalid syntax files)."""
        results: list[DocstringInfo] = []
        pattern = re.compile(
            r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:\s*\n\s*"""(.*?)"""',
            re.DOTALL,
        )
        for m in pattern.finditer(code):
            name = m.group(1)
            raw_params = m.group(2) or ''
            raw_return = (m.group(3) or '').strip() or None
            docstring = m.group(4).strip()
            params = [p.strip().split(':')[0].strip() for p in raw_params.split(',') if p.strip()]
            params = [p for p in params if p and p not in ('self', 'cls')]
            line_number = code[:m.start()].count('\n') + 1
            results.append(DocstringInfo(
                function_name=name,
                docstring=docstring,
                language='python',
                params=params,
                return_type=raw_return,
                line_number=line_number,
            ))
        return results

    # ------------------------------------------------------------------
    # JSDoc extraction
    # ------------------------------------------------------------------

    def extract_jsdoc(self, code: str) -> list[DocstringInfo]:
        """Extract JSDoc comments from TypeScript/JavaScript source.

        Finds every /** ... */ block and attempts to match it to the
        immediately following function declaration, arrow function, or method.

        Args:
            code: TypeScript or JavaScript source code.

        Returns:
            List of DocstringInfo for every JSDoc-annotated callable.
        """
        results: list[DocstringInfo] = []

        for block_match in _JSDOC_BLOCK_RE.finditer(code):
            raw_jsdoc = block_match.group(1)
            cleaned = _clean_jsdoc_text(raw_jsdoc)
            description = _extract_jsdoc_description(cleaned)

            # Skip blocks with no description at all
            if not description:
                continue

            # Look at the code immediately after the closing */
            after = code[block_match.end():block_match.end() + 300].lstrip('\n\r ')
            line_number = code[:block_match.start()].count('\n') + 1

            func_name: str | None = None
            params: list[str] = []

            # Try regular function first
            m = _JSDOC_FUNC_RE.match(after)
            if m:
                func_name = m.group(1)
                raw_params = m.group(2) or ''
            else:
                # Try arrow function / const
                m = _JSDOC_ARROW_RE.match(after)
                if m:
                    func_name = m.group(1)
                    raw_params = m.group(2) or ''
                else:
                    # Try method shorthand
                    m = _JSDOC_METHOD_RE.match(after)
                    if m:
                        func_name = m.group(1)
                        raw_params = m.group(2) or ''

            if not func_name:
                continue

            # Extract params: prefer from signature, fill from @param tags
            sig_params = [
                p.strip().split(':')[0].strip().lstrip('...')
                for p in raw_params.split(',')
                if p.strip()
            ]
            sig_params = [p for p in sig_params if p]
            jsdoc_params = _extract_jsdoc_params(cleaned)
            params = sig_params if sig_params else jsdoc_params

            return_type = _extract_jsdoc_return(cleaned)

            results.append(DocstringInfo(
                function_name=func_name,
                docstring=description,
                language='typescript',
                params=params,
                return_type=return_type,
                line_number=line_number,
            ))

        return results

    # ------------------------------------------------------------------
    # Training pair creation
    # ------------------------------------------------------------------

    def create_pairs(self, code: str, language: str = 'python') -> list[tuple[str, str]]:
        """Create (docstring, function_snippet) pairs for training data.

        Each pair contains the docstring as a natural-language description and
        a minimal function signature as context.  Only pairs with a non-empty
        docstring are included.

        Args:
            code: Source code string.
            language: Source language — 'python', 'typescript', etc.

        Returns:
            List of (docstring, function_signature) tuples.
        """
        infos = self.extract(code, language)
        pairs: list[tuple[str, str]] = []
        for info in infos:
            if not info.docstring:
                continue
            param_str = ', '.join(info.params)
            if language.lower() == 'python':
                ret = f' -> {info.return_type}' if info.return_type else ''
                signature = f'def {info.function_name}({param_str}){ret}:'
            else:
                ret = f': {info.return_type}' if info.return_type else ''
                signature = f'function {info.function_name}({param_str}){ret}'
            pairs.append((info.docstring, signature))
        return pairs

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, results: list[DocstringInfo]) -> dict:
        """Produce a summary dict over a list of DocstringInfo results.

        Args:
            results: Output from extract() or similar.

        Returns:
            Dict with counts and aggregate statistics.
        """
        if not results:
            return {
                'total': 0,
                'languages': {},
                'with_params': 0,
                'with_return_type': 0,
                'avg_docstring_words': 0.0,
                'function_names': [],
            }

        lang_counts: dict[str, int] = {}
        for r in results:
            lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

        with_params = sum(1 for r in results if r.params)
        with_return = sum(1 for r in results if r.return_type)
        avg_words = sum(len(r.docstring.split()) for r in results) / len(results)

        return {
            'total': len(results),
            'languages': lang_counts,
            'with_params': with_params,
            'with_return_type': with_return,
            'avg_docstring_words': round(avg_words, 2),
            'function_names': [r.function_name for r in results],
        }
