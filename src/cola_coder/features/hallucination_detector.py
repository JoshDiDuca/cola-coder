"""Hallucination Detection: detect fake APIs and nonexistent patterns in generated code.

Catches common hallucination patterns in AI-generated code:
- Nonexistent standard library functions
- Made-up npm package APIs
- Invalid method chains
- Fake language features
- Plausible but wrong type annotations

For a TS dev: like having ESLint + TypeScript compiler errors but specifically
tuned to catch things AI models commonly get wrong.
"""

from dataclasses import dataclass, field
import re

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class HallucinationAlert:
    """A single detected hallucination."""
    category: str  # "fake_api", "invalid_method", "nonexistent_module", "syntax_error", "type_error"
    message: str
    line_number: int | None = None
    severity: str = "warning"  # "warning", "error", "info"
    suggestion: str = ""

    def __str__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        sug = f" -> {self.suggestion}" if self.suggestion else ""
        return f"[{self.severity}] {self.category}{loc}: {self.message}{sug}"


# ── Known valid APIs (subset) ───────────────────────────────────────────

# Common JS/TS built-in methods that exist
VALID_ARRAY_METHODS = {
    "push", "pop", "shift", "unshift", "slice", "splice", "concat",
    "join", "reverse", "sort", "indexOf", "lastIndexOf", "find",
    "findIndex", "filter", "map", "reduce", "reduceRight", "every",
    "some", "includes", "flat", "flatMap", "fill", "copyWithin",
    "entries", "keys", "values", "forEach", "at", "findLast",
    "findLastIndex", "toReversed", "toSorted", "toSpliced", "with",
}

VALID_STRING_METHODS = {
    "charAt", "charCodeAt", "concat", "includes", "endsWith",
    "indexOf", "lastIndexOf", "match", "matchAll", "padEnd",
    "padStart", "repeat", "replace", "replaceAll", "search",
    "slice", "split", "startsWith", "substring", "toLowerCase",
    "toUpperCase", "trim", "trimStart", "trimEnd", "at",
}

VALID_OBJECT_STATIC = {
    "keys", "values", "entries", "assign", "create", "defineProperty",
    "defineProperties", "freeze", "fromEntries", "getOwnPropertyDescriptor",
    "getOwnPropertyNames", "getPrototypeOf", "hasOwn", "is",
    "isExtensible", "isFrozen", "isSealed", "preventExtensions",
    "seal", "setPrototypeOf", "groupBy",
}

VALID_PROMISE_STATIC = {
    "all", "allSettled", "any", "race", "reject", "resolve", "withResolvers",
}

VALID_MATH_STATIC = {
    "abs", "ceil", "floor", "round", "max", "min", "pow", "sqrt",
    "random", "sign", "trunc", "log", "log2", "log10", "exp",
    "sin", "cos", "tan", "PI", "E",
}

# Common fake APIs that models hallucinate
FAKE_APIS = {
    # Array methods that don't exist
    "Array.flatten": "Use Array.prototype.flat() instead",
    "Array.contains": "Use Array.prototype.includes() instead",
    "Array.remove": "Use Array.prototype.filter() or splice() instead",
    "Array.first": "Use arr[0] or arr.at(0) instead",
    "Array.last": "Use arr[arr.length-1] or arr.at(-1) instead",
    "Array.unique": "Use [...new Set(arr)] instead",
    "Array.compact": "Use arr.filter(Boolean) instead",
    "Array.groupBy": "Use Object.groupBy(arr, fn) instead (ES2024)",
    "Array.zip": "No built-in zip — use a manual implementation",
    # String methods that don't exist
    "String.reverse": "Use str.split('').reverse().join('') instead",
    "String.contains": "Use String.prototype.includes() instead",
    "String.capitalize": "Use str[0].toUpperCase() + str.slice(1) instead",
    "String.isEmpty": "Use str.length === 0 instead",
    # Object methods that don't exist
    "Object.map": "Use Object.fromEntries(Object.entries(obj).map(...))",
    "Object.filter": "Use Object.fromEntries(Object.entries(obj).filter(...))",
    "Object.merge": "Use Object.assign() or spread operator",
    "Object.deepCopy": "Use structuredClone() instead",
    "Object.isEmpty": "Use Object.keys(obj).length === 0",
    # Promise methods that don't exist
    "Promise.delay": "Use new Promise(r => setTimeout(r, ms))",
    "Promise.map": "Use Promise.all(arr.map(fn))",
    "Promise.retry": "No built-in — implement manually",
    # Console methods that don't exist
    "console.success": "Use console.log() instead",
    "console.fail": "Use console.error() instead",
    # JSON methods
    "JSON.clone": "Use structuredClone() or JSON.parse(JSON.stringify(obj))",
    "JSON.isValid": "Use try { JSON.parse(s); return true } catch { return false }",
}

# Fake Node.js modules
FAKE_MODULES = {
    "utils": "Did you mean 'util' (no 's')?",
    "filesystem": "Did you mean 'fs'?",
    "file-system": "Did you mean 'fs'?",
    "paths": "Did you mean 'path' (no 's')?",
    "http2s": "Did you mean 'http2'?",
    "events-emitter": "Did you mean 'events'?",
    "node-fetch": "Built-in fetch is available since Node 18",
    "lodash.deepclone": "Use structuredClone() instead",
}

# Invalid TypeScript syntax patterns
INVALID_TS_PATTERNS = [
    (r"\binterface\s+\w+\s+implements\b", "Interfaces don't use 'implements' — did you mean 'extends'?"),
    (r"\benum\s+\w+\s+extends\b", "Enums cannot extend other types"),
    (r"\btype\s+\w+\s+implements\b", "Type aliases don't use 'implements'"),
    (r"\babstract\s+interface\b", "Interfaces are inherently abstract"),
    (r"\bstatic\s+constructor\b", "Constructors cannot be static"),
    (r"\basync\s+constructor\b", "Constructors cannot be async"),
    (r"\breturn\s+type\b(?!\s*[:{])", "Possibly confused 'return type' with actual return statement"),
]


class HallucinationDetector:
    """Detect hallucinated APIs and patterns in generated code."""

    def __init__(self):
        self._fake_api_patterns: dict[str, str] = dict(FAKE_APIS)
        self._fake_modules: dict[str, str] = dict(FAKE_MODULES)

    def check(self, code: str, language: str = "typescript") -> list[HallucinationAlert]:
        """Run all hallucination checks on generated code.

        Args:
            code: The generated code to check
            language: Programming language ("typescript" or "javascript")

        Returns:
            List of detected hallucination alerts
        """
        alerts: list[HallucinationAlert] = []

        alerts.extend(self._check_fake_apis(code))
        alerts.extend(self._check_fake_modules(code))
        alerts.extend(self._check_invalid_methods(code))
        if language == "typescript":
            alerts.extend(self._check_invalid_ts_syntax(code))
        alerts.extend(self._check_common_mistakes(code))

        return alerts

    def _check_fake_apis(self, code: str) -> list[HallucinationAlert]:
        """Check for known fake API calls."""
        alerts = []
        for fake_api, suggestion in self._fake_api_patterns.items():
            # Search for the fake API as a method call
            parts = fake_api.split(".")
            if len(parts) == 2:
                obj_name, method_name = parts
                # Match patterns like Array.flatten, .flatten(, etc.
                pattern = rf"\.{method_name}\s*\("
                if obj_name in ("Array", "String"):
                    # Also check for .method() calls on instances
                    for i, line in enumerate(code.splitlines(), 1):
                        if re.search(pattern, line):
                            alerts.append(HallucinationAlert(
                                category="fake_api",
                                message=f"{fake_api}() does not exist",
                                line_number=i,
                                severity="warning",
                                suggestion=suggestion,
                            ))
                            break
                else:
                    # Check for ObjectName.method() pattern
                    pattern = rf"\b{re.escape(obj_name)}\.{method_name}\s*\("
                    for i, line in enumerate(code.splitlines(), 1):
                        if re.search(pattern, line):
                            alerts.append(HallucinationAlert(
                                category="fake_api",
                                message=f"{fake_api}() does not exist",
                                line_number=i,
                                severity="warning",
                                suggestion=suggestion,
                            ))
                            break
        return alerts

    def _check_fake_modules(self, code: str) -> list[HallucinationAlert]:
        """Check for imports of nonexistent modules."""
        alerts = []
        # Match import/require patterns
        import_patterns = [
            r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, code):
                module_name = match.group(1)
                if module_name in self._fake_modules:
                    line_num = code[:match.start()].count("\n") + 1
                    alerts.append(HallucinationAlert(
                        category="nonexistent_module",
                        message=f"Module '{module_name}' does not exist",
                        line_number=line_num,
                        severity="error",
                        suggestion=self._fake_modules[module_name],
                    ))
        return alerts

    def _check_invalid_methods(self, code: str) -> list[HallucinationAlert]:
        """Check for method calls on wrong object types."""
        alerts = []

        # Check for Object.method() calls with invalid methods
        obj_method_pattern = r"\bObject\.(\w+)\s*\("
        for match in re.finditer(obj_method_pattern, code):
            method = match.group(1)
            if method not in VALID_OBJECT_STATIC and method not in {"prototype", "constructor"}:
                line_num = code[:match.start()].count("\n") + 1
                alerts.append(HallucinationAlert(
                    category="invalid_method",
                    message=f"Object.{method}() may not exist",
                    line_number=line_num,
                    severity="info",
                    suggestion=f"Valid Object methods: {', '.join(sorted(list(VALID_OBJECT_STATIC)[:5]))}...",
                ))

        # Check for Promise.method() calls
        promise_pattern = r"\bPromise\.(\w+)\s*\("
        for match in re.finditer(promise_pattern, code):
            method = match.group(1)
            if method not in VALID_PROMISE_STATIC and method not in {"prototype", "constructor"}:
                line_num = code[:match.start()].count("\n") + 1
                alerts.append(HallucinationAlert(
                    category="invalid_method",
                    message=f"Promise.{method}() does not exist",
                    line_number=line_num,
                    severity="warning",
                ))

        return alerts

    def _check_invalid_ts_syntax(self, code: str) -> list[HallucinationAlert]:
        """Check for invalid TypeScript syntax patterns."""
        alerts = []
        for pattern, message in INVALID_TS_PATTERNS:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count("\n") + 1
                alerts.append(HallucinationAlert(
                    category="syntax_error",
                    message=message,
                    line_number=line_num,
                    severity="error",
                ))
        return alerts

    def _check_common_mistakes(self, code: str) -> list[HallucinationAlert]:
        """Check for common plausible-but-wrong patterns."""
        alerts = []

        # Check for `typeof x === "array"` (should be Array.isArray)
        if re.search(r'typeof\s+\w+\s*===?\s*["\']array["\']', code, re.IGNORECASE):
            alerts.append(HallucinationAlert(
                category="fake_api",
                message='typeof does not return "array"',
                severity="error",
                suggestion="Use Array.isArray(x) instead",
            ))

        # Check for `typeof x === "null"` (typeof null is "object")
        if re.search(r'typeof\s+\w+\s*===?\s*["\']null["\']', code):
            alerts.append(HallucinationAlert(
                category="fake_api",
                message='typeof null returns "object", not "null"',
                severity="error",
                suggestion="Use x === null instead",
            ))

        # Check for `typeof x === "undefined"` used correctly but also offer alternative
        # (This is actually valid, so just info level)

        return alerts

    def score(self, code: str, language: str = "typescript") -> float:
        """Score code from 0.0 (many hallucinations) to 1.0 (clean).

        Weighted by severity:
        - error: -0.2 per alert
        - warning: -0.1 per alert
        - info: -0.05 per alert
        """
        alerts = self.check(code, language)
        penalty = 0.0
        for alert in alerts:
            if alert.severity == "error":
                penalty += 0.2
            elif alert.severity == "warning":
                penalty += 0.1
            else:
                penalty += 0.05
        return max(0.0, 1.0 - penalty)

    def print_report(self, code: str, language: str = "typescript") -> None:
        """Print a formatted hallucination report."""
        from cola_coder.cli import cli

        alerts = self.check(code, language)
        score = self.score(code, language)

        if not alerts:
            cli.success(f"No hallucinations detected (score: {score:.0%})")
            return

        cli.warn(f"Found {len(alerts)} potential hallucination(s) (score: {score:.0%})")
        for alert in alerts:
            if alert.severity == "error":
                cli.error(str(alert))
            elif alert.severity == "warning":
                cli.warn(str(alert))
            else:
                cli.dim(str(alert))
