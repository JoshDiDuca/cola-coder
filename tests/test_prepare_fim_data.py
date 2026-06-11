"""DATA-030: prepare_fim_data must FAIL LOUD on a tokenizer lacking FIM tokens.

The old code called setup_fim_tokenizer, which ADDS the <|fim_*|> tokens to the
in-memory tokenizer when absent — but never re-saves tokenizer.json. The FIM
marker ids baked into the output .npy would then be out-of-vocab for the model
at training time (silent poison, the DATA-003/013 class). `_resolve_fim_ids`
reads the ids without mutating, and raises so the caller fails loud.
"""

import importlib.util
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "prepare_fim_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_fim_data_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_tokenizer(tmp_path):
    from cola_coder.tokenizer import train_tokenizer as tt
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 100),
        vocab_size=320, output_path=out,
    )
    return CodeTokenizer(out)


class TestResolveFimIds:
    def test_real_tokenizer_resolves_and_caches(self, tmp_path):
        m = _load()
        tok = _real_tokenizer(tmp_path)
        ids = m._resolve_fim_ids(tok)
        assert set(ids) == {"fim_prefix", "fim_suffix", "fim_middle"}
        assert all(isinstance(v, int) for v in ids.values())
        # Cached on the tokenizer where FIMTransform.apply reads them.
        assert tok.fim_prefix_id == ids["fim_prefix"]
        assert tok.fim_suffix_id == ids["fim_suffix"]
        assert tok.fim_middle_id == ids["fim_middle"]

    def test_all_missing_raises_with_full_list(self):
        m = _load()
        fake = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(token_to_id=lambda t: None)
        )
        with pytest.raises(m._MissingFimTokens) as exc:
            m._resolve_fim_ids(fake)
        assert "<|fim_prefix|>" in exc.value.missing
        assert "<|fim_middle|>" in exc.value.missing

    def test_partial_missing_lists_only_absent(self):
        m = _load()
        present = {"<|fim_prefix|>": 5, "<|fim_middle|>": 7}  # suffix absent
        fake = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(token_to_id=lambda t: present.get(t))
        )
        with pytest.raises(m._MissingFimTokens) as exc:
            m._resolve_fim_ids(fake)
        assert exc.value.missing == ["<|fim_suffix|>"]
        # The present ones were still cached before the failure.
        assert fake.fim_prefix_id == 5
