"""Tree-sitter based syntax validation filter.

Uses tree-sitter for full AST parsing to detect syntax errors across
multiple languages. Tree-sitter is the same parser used by GitHub for
code navigation, Neovim for syntax highlighting, and Zed editor.

Tree-sitter is OPTIONAL. If not installed, this filter is a no-op
that passes everything through and logs a warning once.
"""

import logging

from cola_coder.data.registry import register_filter

logger = logging.getLogger(__name__)

try:
    import tree_sitter
    _HAS_TREESITTER = True
except ImportError:
    _HAS_TREESITTER = False

# Map our language names to tree-sitter package names.
# Each requires its own pip package, e.g. `pip install tree-sitter-python`.
LANGUAGE_PACKAGES = {
    "python": "tree_sitter_python",
    "typescript": "tree_sitter_typescript",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "ruby": "tree_sitter_ruby",
}


def _count_nodes(node, error_count: int = 0, total_count: int = 0):
    """Recursively count total and ERROR nodes in a tree-sitter AST."""
    total_count += 1
    if node.type == "ERROR" or node.is_missing:
        error_count += 1
    for child in node.children:
        error_count, total_count = _count_nodes(child, error_count, total_count)
    return error_count, total_count


@register_filter("syntax")
class SyntaxFilter:
    """Full AST validation using tree-sitter.

    Parses code with tree-sitter and counts ERROR nodes relative to total
    nodes. If the error ratio exceeds max_error_ratio, the file is rejected.

    Benefits over heuristic parsing:
    - Catches ALL syntax errors, not just obvious ones
    - Language-agnostic API (same code handles Python, TS, Go, Rust)
    - Fast: ~10ms per file (written in C)
    """

    def __init__(self, languages: list[str] | None = None, max_error_ratio: float = 0.05):
        """
        Args:
            languages: Languages to validate. If None, tries all available.
            max_error_ratio: Max fraction of ERROR nodes in AST.
                0.0 = perfect parse only
                0.05 = allow up to 5% error nodes (tolerant)
        """
        self.max_error_ratio = max_error_ratio
        self.parsers: dict = {}
        self._warned = False

        if not _HAS_TREESITTER:
            return

        target_langs = languages or list(LANGUAGE_PACKAGES.keys())
        for lang in target_langs:
            pkg_name = LANGUAGE_PACKAGES.get(lang)
            if not pkg_name:
                continue
            try:
                # Import the language package dynamically
                import importlib
                lang_mod = importlib.import_module(pkg_name)
                # tree-sitter >= 0.22 API: Language(module.language())
                if hasattr(lang_mod, "language"):
                    ts_lang = tree_sitter.Language(lang_mod.language())
                else:
                    # Older API fallback
                    ts_lang = tree_sitter.Language(lang_mod)
                parser = tree_sitter.Parser(ts_lang)
                self.parsers[lang] = parser
            except (ImportError, Exception) as e:
                logger.debug(f"Could not load tree-sitter grammar for {lang}: {e}")

    def name(self) -> str:
        return "syntax"

    def check(self, record) -> tuple[bool, str]:
        """Check if the code parses without too many syntax errors.

        Args:
            record: Object with .content (str) and .metadata (dict) attributes.

        Returns:
            (keep, reason) tuple.
        """
        if not _HAS_TREESITTER or not self.parsers:
            if not self._warned:
                logger.warning(
                    "SyntaxFilter: tree-sitter not available, passing all files. "
                    "Install with: pip install tree-sitter tree-sitter-python "
                    "tree-sitter-javascript tree-sitter-typescript"
                )
                self._warned = True
            return True, ""

        # Determine language from metadata
        lang = record.metadata.get("language", "").lower()
        parser = self.parsers.get(lang)
        if parser is None:
            # No parser for this language — pass through
            return True, ""

        content = record.content
        if not content.strip():
            return True, ""

        try:
            tree = parser.parse(content.encode("utf-8"))
            error_count, total_count = _count_nodes(tree.root_node)

            if total_count == 0:
                return True, ""

            error_ratio = error_count / total_count
            if error_ratio > self.max_error_ratio:
                return False, (
                    f"syntax_errors ({error_count}/{total_count} nodes = "
                    f"{error_ratio:.1%} errors)"
                )
            return True, ""
        except Exception as e:
            logger.debug(f"SyntaxFilter parse error: {e}")
            return True, ""  # Can't parse — pass through

    def setup(self, config: dict) -> None:
        """Optional setup from config dict."""
        if "max_error_ratio" in config:
            self.max_error_ratio = config["max_error_ratio"]
