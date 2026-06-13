"""Tests for the canonical dangerous-code scanner (security/code_patterns.py).

Static screen used by safety_eval + the distillation verifier so functional-but-
insecure code is rejected (secure-pass gap). No execution.
"""

import pytest

from cola_coder.security.code_patterns import (
    DANGEROUS_PATTERNS,
    is_dangerous,
    scan_dangerous,
)


@pytest.mark.parametrize("code", [
    "result = eval(user_input)",
    "exec(payload)",
    "os.system('rm -rf /')",
    "subprocess.run(cmd, shell=True)",
    "__import__('os').system('x')",
    "obj = pickle.loads(blob)",
    "data = yaml.load(stream)",
])
def test_detects_python_dangers(code):
    assert is_dangerous(code), code


@pytest.mark.parametrize("code", [
    "const fn = new Function('return 1');",
    "import { exec } from 'child_process';",
    "<div dangerouslySetInnerHTML={{__html: html}} />",
    "document.write(userContent);",
    "vm.runInThisContext(src);",
])
def test_detects_js_ts_dangers(code):
    assert is_dangerous(code), code


@pytest.mark.parametrize("code", [
    "def add(a, b):\n    return a + b",
    "const x: number = 1;\nexport function add(a: number, b: number) { return a + b; }",
    "interface User { id: number; name: string }",
    "const m = /foo/; m.exec('foobar');",  # regex.exec is NOT flagged (precision)
    "el.innerText = safe;",
    "",
])
def test_clean_code_not_flagged(code):
    assert not is_dangerous(code), code


def test_scan_returns_names():
    names = scan_dangerous("eval(x)\nos.system('y')")
    assert "eval() usage" in names
    assert "os.system() shell execution" in names


def test_canonical_set_is_superset_of_original_python():
    # The original safety_eval set (now imported from here) must still be present.
    flat = {name for _, name in DANGEROUS_PATTERNS}
    for required in ("eval() usage", "exec() usage", "os.system() shell execution",
                     "SQL DROP TABLE"):
        assert required in flat


def test_safety_eval_uses_shared_patterns():
    # DRY: safety_eval imports the canonical list (no divergent copy).
    from cola_coder.evaluation.safety_eval import DANGEROUS_PATTERNS as se_patterns
    assert se_patterns is DANGEROUS_PATTERNS
