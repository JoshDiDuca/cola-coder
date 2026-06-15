"""Static CWE vulnerability scorer (SEC-024 — non-injection CWE families).

Pure regex/string analysis of code **as untrusted TEXT** — this scorer NEVER
executes, evals, or imports the scanned code. It is the data-quality half of the
"close the eval->data loop" idea in docs/research-log.md (2026-06-15): the same
static CWE screen that measures the model's *generated*-code vulnerability rate
also down-weights training examples that themselves contain those CWE patterns,
directly attacking the root cause (models reproduce CWE patterns they were
trained on).

Relation to the existing safety modules (no duplication):
  * ``security/injection_patterns.py`` / ``injection_scorer.py`` cover prompt
    INJECTION (LLM01) — instruction-override / exfiltration directives in
    retrieved text. This scorer covers the orthogonal NON-injection CWE families
    below and composes with it.
  * ``security/code_patterns.py`` (``scan_dangerous``) is a flat list of
    dangerous-pattern *names* with no CWE id, no severity, and no line number. It
    serves safety_eval / distillation as a boolean-ish screen. This scorer instead
    emits STRUCTURED, per-CWE findings (``cwe`` id + ``severity`` + line/snippet)
    and a graded 0-1 quality score for data weighting — a different shape for a
    different consumer (the composite data scorer).

Detected CWE families (language-aware: Python + TS/JS), each documented inline:
  * CWE-78  OS command injection  — os.system(, subprocess(..., shell=True),
                                     os.popen(, JS child_process.exec/execSync(
  * CWE-94/95 code injection / eval-exec — bare eval(/exec(, JS eval(,
                                     new Function(, setTimeout/Interval string arg
  * CWE-502 unsafe deserialization — pickle.load(s), marshal.loads, yaml.load(
                                     without a safe Loader
  * CWE-89  SQL injection          — SQL built via f-string / % / + / template
                                     literal and passed to .execute( (heuristic)
  * CWE-327/328 weak crypto/hash   — hashlib.md5/sha1(, crypto.createHash('md5'|'sha1')
  * CWE-330 insecure randomness    — random.* used in a secret/token/password/key
                                     context (heuristic)
  * CWE-22  path traversal         — open()/fs path built from concatenation that
                                     contains a ".." segment (heuristic, high-precision)

False-positive caveats (documented, deliberately conservative):
  * Single-line ``#`` / ``//`` comments are stripped before scanning so
    commented-out vulnerable code is not flagged. Block comments (``/* */``,
    triple-quoted strings) and string literals are NOT stripped (cheap-precision
    tradeoff) — a vulnerable pattern inside a string may still match.
  * CWE-330 / CWE-22 are line-local heuristics (the risky call and its context
    must co-occur on the same line) to keep precision high at the cost of recall.

Deterministic, fast, no network / subprocess / model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from cola_coder.data.scorers.language_detect import is_js_ts
from cola_coder.data.scorers.protocol import ScorerResult
from cola_coder.data.scorers.utils import ScoreMapper

Severity = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CwePattern:
    """A single static CWE detection rule."""

    cwe: str            # e.g. "CWE-78"
    name: str           # human-readable weakness name
    severity: Severity  # high | medium | low
    regex: re.Pattern[str]
    language: Literal["python", "js_ts", "any"]  # gating: where the rule applies


def _rx(pattern: str) -> re.Pattern[str]:
    """Compile a pattern case-insensitively (single place, consistent flags)."""
    return re.compile(pattern, re.IGNORECASE)


# --- Pattern table. Each entry documents its CWE id + the construct it matches. ---
_PATTERNS: list[CwePattern] = [
    # === CWE-78: OS command injection ===
    CwePattern("CWE-78", "os.system() shell execution", "high",
               _rx(r"\bos\.system\s*\("), "python"),
    CwePattern("CWE-78", "os.popen() shell execution", "high",
               _rx(r"\bos\.popen\s*\("), "python"),
    CwePattern("CWE-78", "subprocess with shell=True", "high",
               _rx(r"\bsubprocess\.(?:call|run|Popen|check_output|check_call)\s*\([^)]*shell\s*=\s*True"),
               "python"),
    CwePattern("CWE-78", "child_process.exec/execSync (shell execution)", "high",
               _rx(r"\bchild_process\.(?:exec|execSync)\s*\(|\b(?:exec|execSync)\s*\(\s*`"), "js_ts"),

    # === CWE-94 / CWE-95: code injection via eval/exec ===
    # Negative lookbehind for '.' so JS method calls like `regex.exec(...)`
    # (benign) aren't flagged — only the bare builtin eval/exec.
    CwePattern("CWE-95", "eval() dynamic code execution", "high",
               _rx(r"(?<!\.)\beval\s*\("), "any"),
    CwePattern("CWE-95", "exec() dynamic code execution", "high",
               _rx(r"(?<!\.)\bexec\s*\("), "python"),
    CwePattern("CWE-94", "new Function() dynamic code", "high",
               _rx(r"\bnew\s+Function\s*\("), "js_ts"),
    CwePattern("CWE-95", "setTimeout/Interval string-eval", "medium",
               _rx(r"\bset(?:Timeout|Interval)\s*\(\s*['\"]"), "js_ts"),

    # === CWE-502: unsafe deserialization ===
    CwePattern("CWE-502", "pickle deserialization of untrusted data", "high",
               _rx(r"\bpickle\.loads?\s*\("), "python"),
    CwePattern("CWE-502", "marshal.loads deserialization", "high",
               _rx(r"\bmarshal\.loads?\s*\("), "python"),
    # yaml.load WITHOUT an explicit Loader= (SafeLoader) is RCE-capable.
    CwePattern("CWE-502", "yaml.load without SafeLoader", "high",
               _rx(r"\byaml\.load\s*\((?![^)]*Loader)"), "python"),

    # === CWE-89: SQL injection (string-built queries) ===
    # SQL keyword in an f-string passed to .execute(  -> f"...SELECT..."
    CwePattern("CWE-89", "SQL built via f-string passed to execute()", "high",
               _rx(r"\.execute\s*\(\s*f['\"][^'\"]*\b(?:SELECT|INSERT|UPDATE|DELETE|DROP)\b"),
               "python"),
    # SQL built with % or + concatenation passed to .execute(
    CwePattern("CWE-89", "SQL built via string concatenation/format passed to execute()", "high",
               _rx(r"\.execute\s*\(\s*['\"][^'\"]*\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^)]*['\"]\s*(?:%|\+)"),
               "python"),
    # JS template-literal SQL: `SELECT ... ${...}` interpolation.
    CwePattern("CWE-89", "SQL injection via template-literal interpolation", "high",
               _rx(r"`[^`]*\b(?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\b[^`]*\$\{"),
               "js_ts"),

    # === CWE-327 / CWE-328: weak cryptographic hash ===
    CwePattern("CWE-327", "weak hash (MD5/SHA-1) via hashlib", "medium",
               _rx(r"\bhashlib\.(?:md5|sha1)\s*\("), "python"),
    CwePattern("CWE-327", "weak hash (MD5/SHA-1) via crypto.createHash", "medium",
               _rx(r"\bcreateHash\s*\(\s*['\"](?:md5|sha1)['\"]"), "js_ts"),

    # === CWE-330: insecure randomness for secrets (heuristic, line-local) ===
    # random.<fn>(...) on a line that also mentions secret/token/password/key.
    CwePattern("CWE-330", "non-cryptographic random used for a secret/token", "medium",
               _rx(r"\brandom\.\w+\s*\([^)]*\)[^\n]*\b(?:secret|token|password|passwd|api[_\s-]?key|nonce|salt|otp)\b"
                   r"|\b(?:secret|token|password|passwd|api[_\s-]?key|nonce|salt|otp)\b[^\n]*\brandom\.\w+\s*\("),
               "python"),
    # JS Math.random() in a secret/token context (heuristic, line-local).
    CwePattern("CWE-330", "Math.random() used for a secret/token", "medium",
               _rx(r"\bMath\.random\s*\(\s*\)[^\n]*\b(?:secret|token|password|api[_\s-]?key|nonce|salt|otp)\b"
                   r"|\b(?:secret|token|password|api[_\s-]?key|nonce|salt|otp)\b[^\n]*\bMath\.random\s*\(\s*\)"),
               "js_ts"),

    # === CWE-22: path traversal (heuristic, high-precision) ===
    # open()/fs call whose argument is a concatenation that contains a ".." segment.
    CwePattern("CWE-22", "path traversal via unsanitized concatenation in open()", "high",
               _rx(r"\bopen\s*\([^)]*\.\.[\\/][^)]*(?:\+|%|\.format|f['\"])"
                   r"|\bopen\s*\([^)]*(?:\+|%|\.format)[^)]*\.\.[\\/]"),
               "python"),
    CwePattern("CWE-22", "path traversal via concatenation in fs path", "high",
               _rx(r"\bfs\.(?:readFile|readFileSync|writeFile|writeFileSync|createReadStream)\s*\("
                   r"[^)]*\.\.[\\/][^)]*(?:\+|\$\{)"
                   r"|\bfs\.(?:readFile|readFileSync|writeFile|writeFileSync|createReadStream)\s*\("
                   r"[^)]*(?:\+|\$\{)[^)]*\.\.[\\/]"),
               "js_ts"),

    # === CWE-295: improper certificate / TLS verification (disabled) ===
    # Secrets/TLS misconfig is a top class in AI-generated code (2026 surveys):
    # disabling cert verification silently exposes every request to MITM.
    CwePattern("CWE-295", "TLS certificate verification disabled (verify=False)", "high",
               _rx(r"\bverify\s*=\s*False\b"), "python"),
    CwePattern("CWE-295", "unverified SSL context / CERT_NONE", "high",
               _rx(r"\bssl\._create_unverified_context\s*\(|\bssl\.CERT_NONE\b|"
                   r"\bcheck_hostname\s*=\s*False\b"), "python"),
    CwePattern("CWE-295", "Node TLS verification disabled (rejectUnauthorized: false)", "high",
               _rx(r"\brejectUnauthorized\s*:\s*false\b|"
                   r"\bNODE_TLS_REJECT_UNAUTHORIZED\b\s*[=:]\s*['\"]?0"), "js_ts"),

    # === CWE-327: weak symmetric cipher / insecure mode (DES, 3DES, RC4, ECB) ===
    CwePattern("CWE-327", "weak/broken symmetric cipher (DES/3DES/RC4/Blowfish) or ECB mode", "medium",
               _rx(r"\b(?:DES|DES3|ARC4|Blowfish)\.new\s*\(|\bMODE_ECB\b|"
                   r"\bmodes\.ECB\s*\(|\balgorithms\.(?:TripleDES|ARC4|Blowfish)\s*\("), "python"),
    CwePattern("CWE-327", "weak cipher (DES/RC4) or ECB mode via createCipheriv", "medium",
               _rx(r"\bcreateCipheriv\s*\(\s*['\"](?:des|des3|rc4|rc2)[^'\"]*['\"]|"
                   r"['\"][a-z0-9]+-ecb['\"]"), "js_ts"),
]

# Severity weights: a higher-severity finding subtracts more "demerit points",
# which the ScoreMapper then turns into a quality score. high=3, medium=2, low=1.
_SEVERITY_WEIGHT: dict[Severity, int] = {"high": 3, "medium": 2, "low": 1}

# Weighted-demerit -> quality score. 0 demerits (clean) stays 1.0; each
# high-severity finding (=3) pushes the score (and thus training weight) down.
# Mirrors the InjectionScorer's graded approach via the shared ScoreMapper.
_CWE_SCORE = ScoreMapper(
    [(0, 1.0), (2, 0.7), (4, 0.45), (6, 0.25), (9, 0.12)],
    floor=0.05,
)

# Strip single-line comments so commented-out vulnerable code isn't flagged.
_PY_COMMENT = re.compile(r"#.*$", re.MULTILINE)
_JS_LINE_COMMENT = re.compile(r"//.*$", re.MULTILINE)


@dataclass(frozen=True)
class CweFinding:
    """One CWE match: the weakness id/name, its severity, and where it occurred."""

    cwe: str
    name: str
    severity: Severity
    line: int
    snippet: str


def _strip_line_comments(code: str, js_ts: bool) -> str:
    """Remove single-line comments for the code's LANGUAGE (precision: skip dead code).

    Language-gated: only ``//`` for JS/TS, only ``#`` for Python. Applying the JS
    ``//`` rule to Python (BUG-134) clobbered any line containing ``//`` — most
    importantly URLs like ``https://...`` — eating the rest of the line and causing
    false negatives for EVERY pattern on that line (e.g. ``requests.get(url, verify=False)``).
    Block comments and string literals are left intact (documented tradeoff).
    """
    return _JS_LINE_COMMENT.sub("", code) if js_ts else _PY_COMMENT.sub("", code)


def _line_of(code: str, index: int) -> int:
    """1-based line number of a character offset."""
    return code.count("\n", 0, index) + 1


def scan_cwe(code: str, metadata: dict[str, object] | None = None) -> list[CweFinding]:
    """Statically scan ``code`` for CWE patterns. Returns [] when clean.

    Language gating reuses the shared ``is_js_ts`` detector: js/ts-only rules are
    skipped for Python and vice-versa, so e.g. a JS ``new Function(`` rule never
    fires on Python and Python's ``pickle`` rule never fires on a .ts file.
    """
    if not code or not code.strip():
        return []

    js_ts = is_js_ts(code, metadata)
    scanned = _strip_line_comments(code, js_ts)

    findings: list[CweFinding] = []
    for pat in _PATTERNS:
        if pat.language == "python" and js_ts:
            continue
        if pat.language == "js_ts" and not js_ts:
            continue
        match = pat.regex.search(scanned)
        if match is None:
            continue
        line = _line_of(scanned, match.start())
        snippet = match.group(0).strip()[:120]
        findings.append(
            CweFinding(
                cwe=pat.cwe,
                name=pat.name,
                severity=pat.severity,
                line=line,
                snippet=snippet,
            )
        )
    return findings


def _weighted_demerits(findings: list[CweFinding]) -> int:
    """Sum severity weights across findings (drives the score downward)."""
    return sum(_SEVERITY_WEIGHT[f.severity] for f in findings)


class CweSecurityScorer:
    """Down-weight training samples containing static CWE vulnerability patterns.

    Conforms to ``ScorerProtocol``: ``name``, ``score``, ``score_batch``,
    ``is_available``. Pure static analysis — never executes the scanned code.
    """

    name: str = "cwe_security"

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        findings = scan_cwe(code, metadata)
        demerits = _weighted_demerits(findings)
        return ScorerResult(
            score=_CWE_SCORE(demerits),
            scorer_name=self.name,
            details={
                "findings": [
                    {
                        "cwe": f.cwe,
                        "name": f.name,
                        "severity": f.severity,
                        "line": f.line,
                        "snippet": f.snippet,
                    }
                    for f in findings
                ],
                "num_findings": len(findings),
                "weighted_demerits": demerits,
            },
        )

    def score_batch(
        self, items: list[tuple[str, dict[str, object] | None]]
    ) -> list[ScorerResult]:
        return [self.score(code, meta) for code, meta in items]

    @staticmethod
    def is_available() -> bool:
        return True  # Pure Python, no external dependencies
