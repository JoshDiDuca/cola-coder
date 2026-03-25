"""Shared language detection for code scorers."""

from __future__ import annotations

# Language constants
TYPESCRIPT_EXTENSIONS = frozenset({"ts", "tsx", "mts", "cts"})
JAVASCRIPT_EXTENSIONS = frozenset({"js", "jsx", "mjs", "cjs"})
JS_TS_EXTENSIONS = TYPESCRIPT_EXTENSIONS | JAVASCRIPT_EXTENSIONS

TYPESCRIPT_LANGUAGES = frozenset({"typescript", "ts", "tsx"})
JAVASCRIPT_LANGUAGES = frozenset({"javascript", "js", "jsx"})
JS_TS_LANGUAGES = TYPESCRIPT_LANGUAGES | JAVASCRIPT_LANGUAGES


def get_language(metadata: dict[str, object] | None) -> str:
    """Extract normalized language string from metadata."""
    if metadata and "language" in metadata:
        return str(metadata["language"]).lower()
    return ""


def get_file_extension(metadata: dict[str, object] | None) -> str:
    """Extract file extension (without dot) from metadata."""
    if not metadata:
        return ""
    file_path = str(metadata.get("file_path", ""))
    if file_path and "." in file_path:
        return file_path.rsplit(".", 1)[-1].lower()
    return ""


def is_typescript(code: str, metadata: dict[str, object] | None = None) -> bool:
    """Detect if code is TypeScript (not JavaScript)."""
    lang = get_language(metadata)
    if lang in TYPESCRIPT_LANGUAGES:
        return True
    ext = get_file_extension(metadata)
    if ext in TYPESCRIPT_EXTENSIONS:
        return True
    # Heuristic: TypeScript-specific patterns
    ts_indicators = [": string", ": number", ": boolean", "interface ", ": void",
                     "as const", "<T>", "readonly ", "enum "]
    return sum(1 for ind in ts_indicators if ind in code) >= 2


def is_js_ts(code: str, metadata: dict[str, object] | None = None) -> bool:
    """Detect if code is JavaScript or TypeScript."""
    lang = get_language(metadata)
    if lang in JS_TS_LANGUAGES:
        return True
    ext = get_file_extension(metadata)
    if ext in JS_TS_EXTENSIONS:
        return True
    # Heuristic: JS/TS common patterns
    indicators = ["const ", "let ", "import ", "export ", "=> {", "function "]
    return sum(1 for ind in indicators if ind in code) >= 2


def detect_extension(metadata: dict[str, object] | None) -> str:
    """Detect appropriate file extension (with dot) from metadata. Default '.ts'."""
    if metadata:
        ext = get_file_extension(metadata)
        if ext and f".{ext}" in {f".{e}" for e in JS_TS_EXTENSIONS}:
            return f".{ext}"
        lang = get_language(metadata)
        if lang in TYPESCRIPT_LANGUAGES:
            return ".ts"
        if lang in JAVASCRIPT_LANGUAGES:
            return ".js"
    return ".ts"  # Default to TypeScript
