"""INFER-016: no_repeat_ngram_size (INFER-015) must be reachable from the API.

The core generator gained no_repeat_ngram_size, but a feature is only useful if
users can set it. All four code-generation request bodies must expose it (default
0 = off, backward compatible), and the server must forward it to base_gen.generate
at every direct call site.
"""

import ast
from pathlib import Path

from cola_coder.inference.server import (
    ChatCompletionRequest,
    CompletionRequest,
    FimRequest,
    GenerateRequest,
)

_SERVER = Path(__file__).parent.parent / "src" / "cola_coder" / "inference" / "server.py"


class TestRequestModelsExposeField:
    def test_all_models_have_field_default_off(self):
        for model in (GenerateRequest, ChatCompletionRequest, CompletionRequest, FimRequest):
            assert "no_repeat_ngram_size" in model.model_fields, model.__name__
            assert model.model_fields["no_repeat_ngram_size"].default == 0, model.__name__

    def test_field_is_settable(self):
        assert FimRequest(prefix="a", suffix="b", no_repeat_ngram_size=3).no_repeat_ngram_size == 3
        assert GenerateRequest(prompt="x", no_repeat_ngram_size=4).no_repeat_ngram_size == 4


class TestServerForwardsField:
    def test_every_direct_generate_call_passes_it(self):
        """Each base_gen.generate / generate_stream call must forward the field,
        else the request param is silently ignored."""
        src = _SERVER.read_text(encoding="utf-8")
        # Count direct generator calls (not the best_of_n path, which has its own
        # signature) vs. how many forward the field.
        forwards = src.count("no_repeat_ngram_size=request.no_repeat_ngram_size")
        # 4 request bodies → /generate, chat (non-stream + stream), completions
        # (non-stream + stream), FIM = 6 direct call sites.
        assert forwards >= 6, f"only {forwards} call sites forward no_repeat_ngram_size"

    def test_server_still_parses(self):
        ast.parse(_SERVER.read_text(encoding="utf-8"))
