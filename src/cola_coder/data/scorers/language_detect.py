"""Shared language detection for all scorers."""
from __future__ import annotations

# Language constant sets (frozen for safety)
TYPESCRIPT_EXTENSIONS: frozenset[str] = frozenset({".ts", ".tsx", ".mts", ".cts"})
JAVASCRIPT_EXTENSIONS: frozenset[str] = frozenset({".js", ".jsx", ".mjs", ".cjs"})
JS_TS_EXTENSIONS: frozenset[str] = TYPESCRIPT_EXTENSIONS | JAVASCRIPT_EXTENSIONS
TYPESCRIPT_LANGUAGES: frozenset[str] = frozenset({"typescript", "ts", "tsx"})
JAVASCRIPT_LANGUAGES: frozenset[str] = frozenset({"javascript", "js", "jsx"})
JS_TS_LANGUAGES: frozenset[str] = TYPESCRIPT_LANGUAGES | JAVASCRIPT_LANGUAGES


def is_typescript(code: str, metadata: dict[str, object] | None = None) -> bool:
    """Detect if code is TypeScript (not JavaScript)."""
    if metadata:
        lang = str(metadata.get("language", "")).lower()
        if lang in TYPESCRIPT_LANGUAGES:
            return True
        file_path = str(metadata.get("file_path", ""))
        if file_path and "." in file_path:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
            if ext in TYPESCRIPT_EXTENSIONS:
                return True
    # Heuristic: TypeScript-specific keywords
    ts_indicators = [": string", ": number", ": boolean", "interface ", "<T>", "as const"]
    return sum(1 for ind in ts_indicators if ind in code) >= 2


def is_js_ts(code: str, metadata: dict[str, object] | None = None) -> bool:
    """Detect if code is JavaScript or TypeScript."""
    if metadata:
        lang = str(metadata.get("language", "")).lower()
        if lang in JS_TS_LANGUAGES:
            return True
        file_path = str(metadata.get("file_path", ""))
        if file_path and "." in file_path:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
            if ext in JS_TS_EXTENSIONS:
                return True
    # Heuristic: JS/TS keywords
    indicators = ["const ", "let ", "import ", "export ", "=> {", "function "]
    return sum(1 for ind in indicators if ind in code) >= 2


def detect_extension(metadata: dict[str, object] | None) -> str:
    """Detect file extension from metadata. Defaults to '.ts'."""
    if metadata:
        file_path = str(metadata.get("file_path", ""))
        if file_path and "." in file_path:
            ext = "." + file_path.rsplit(".", 1)[-1].lower()
            if ext in JS_TS_EXTENSIONS:
                return ext
        lang = str(metadata.get("language", "")).lower()
        if lang in TYPESCRIPT_LANGUAGES:
            return ".ts"
        if lang in JAVASCRIPT_LANGUAGES:
            return ".js"
    return ".ts"
