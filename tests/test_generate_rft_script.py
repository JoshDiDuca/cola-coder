"""MODEL-046: generate_rft_data.py prompt loading (JSONL path, no GPU/checkpoint)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_rft_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_rft_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, jsonl=None, max_prompts=None):
        self.jsonl = jsonl
        self.max_prompts = max_prompts


def test_load_prompts_from_jsonl(tmp_path):
    mod = _load_module()
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"prompt": "def f():", "test_code": "assert f()"}\n'
        "\n"  # blank line skipped
        '{"prompt": "def g():"}\n',
        encoding="utf-8",
    )
    prompts, tests = mod._load_prompts(_Args(jsonl=str(p)))
    assert prompts == ["def f():", "def g():"]
    assert tests == ["assert f()", None]  # missing test_code → None (tsc/syntax fallback)


def test_builtin_problems_respect_max_prompts():
    mod = _load_module()
    prompts, tests = mod._load_prompts(_Args(max_prompts=3))
    assert len(prompts) == 3
    assert len(tests) == 3
    assert all(isinstance(p, str) and p for p in prompts)
