"""INFER-011: interactive chat must use ChatML against an SFT model, and the
generator must expose completion-only text so reply extraction survives the
special-token stripping that ChatML markers undergo on decode.

Three layers are locked here:
1. CodeGenerator.generate(return_new_only=True) returns ONLY the completion
   (decode of the new tokens), even when a string stop fires.
2. ChatSession renders ChatML (<|im_start|>role…<|im_end|> + an assistant
   generation prompt) when chat_format="chatml", and Alpaca otherwise.
3. InteractiveChat._generate_reply uses return_new_only + the <|im_end|> stop
   in ChatML mode (no fragile prompt string-diff).
"""

import torch

from cola_coder.features.multi_turn_chat import ChatSession, InteractiveChat
from cola_coder.inference.generator import CodeGenerator
from cola_coder.tokenizer.chat_template import IM_END, IM_START


# --- Stub harness (char-level tokenizer; specials stripped on decode) ---------
class _StubTokenizer:
    eos_id = 0
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [ord(c) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        specials = {self.eos_id, self.bos_id}
        return "".join(chr(i) for i in ids if not (skip_special and i in specials))


class _ScriptedModel:
    def __init__(self, scripted_ids, vocab=256):
        self.scripted = scripted_ids
        self.vocab = vocab
        self.calls = 0

    def eval(self):
        return self

    def clear_caches(self):
        pass

    def __call__(self, input_ids, start_pos=0, use_cache=True):
        batch, seq = input_ids.shape[0], input_ids.shape[1]
        idx = self.calls
        self.calls += 1
        tok = self.scripted[idx] if idx < len(self.scripted) else _StubTokenizer.eos_id
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[:, -1, tok] = 10.0
        return logits


def _gen(completion: str):
    model = _ScriptedModel([ord(c) for c in completion])
    return CodeGenerator(model, _StubTokenizer(), device="cpu")


# --- Layer 1: completion-only return -----------------------------------------
class TestReturnNewOnly:
    def test_returns_completion_not_prompt(self):
        gen = _gen("HELLO")
        out = gen.generate(
            "PROMPT> ", max_new_tokens=20, temperature=0,
            repetition_penalty=1.0, return_new_only=True,
        )
        assert out == "HELLO"  # no prompt echo

    def test_default_still_returns_prompt_plus_completion(self):
        gen = _gen("HELLO")
        out = gen.generate(
            "PROMPT> ", max_new_tokens=20, temperature=0, repetition_penalty=1.0,
        )
        assert out == "PROMPT> HELLO"  # legacy behavior unchanged

    def test_completion_only_excludes_string_stop(self):
        # "ab\n\ncd": string stop "\n\n" fires; completion-only keeps "ab".
        gen = _gen("ab\n\ncd")
        out = gen.generate(
            "P", max_new_tokens=20, temperature=0, repetition_penalty=1.0,
            stop_tokens=["\n\n"], return_new_only=True,
        )
        assert out == "ab"

    def test_string_stop_in_prompt_ignored_for_completion_only(self):
        gen = _gen("xyz")
        out = gen.generate(
            "has\n\nstop", max_new_tokens=20, temperature=0, repetition_penalty=1.0,
            stop_tokens=["\n\n"], return_new_only=True,
        )
        assert out == "xyz"  # the prompt's "\n\n" must not truncate


# --- Layer 2: ChatSession ChatML rendering -----------------------------------
class TestChatMLRendering:
    def test_chatml_user_turn_has_markers_and_generation_prompt(self):
        s = ChatSession(chat_format="chatml")
        s.add_user_message("hi")
        p = s.format_prompt()
        assert p == f"{IM_START}user\nhi{IM_END}\n{IM_START}assistant\n"

    def test_chatml_includes_system_as_role(self):
        s = ChatSession(chat_format="chatml", system_prompt="be nice")
        s.add_user_message("hi")
        p = s.format_prompt()
        assert p.startswith(f"{IM_START}system\nbe nice{IM_END}\n")
        assert p.endswith(f"{IM_START}assistant\n")

    def test_chatml_multi_turn_order(self):
        s = ChatSession(chat_format="chatml")
        s.add_user_message("q1")
        s.add_assistant_message("a1")
        s.add_user_message("q2")
        p = s.format_prompt()
        expected = (
            f"{IM_START}user\nq1{IM_END}\n"
            f"{IM_START}assistant\na1{IM_END}\n"
            f"{IM_START}user\nq2{IM_END}\n"
            f"{IM_START}assistant\n"
        )
        assert p == expected

    def test_alpaca_is_default_and_unchanged(self):
        s = ChatSession()  # default alpaca
        s.add_user_message("hi")
        p = s.format_prompt()
        assert p == "### User:\nhi\n\n### Assistant:\n"
        assert IM_START not in p


class TestTruncationFormatConsistent:
    class _CharTok:
        def encode(self, text, add_bos=False, add_eos=False):
            return [0] * len(text)

    def test_chatml_truncation_drops_oldest_keeps_format(self):
        s = ChatSession(chat_format="chatml", max_context_tokens=200)
        for i in range(6):
            s.add_user_message(f"user message number {i}")
            s.add_assistant_message(f"assistant reply number {i}")
        p = s.format_prompt(tokenizer=self._CharTok())
        # Fits the window, still valid ChatML ending in a generation prompt,
        # and the OLDEST turn was dropped while the newest is kept.
        assert len(p) <= 200
        assert p.endswith(f"{IM_START}assistant\n")
        assert "number 0" not in p
        assert "number 5" in p
        # Every kept message stays well-formed (balanced markers).
        assert p.count(IM_START) == p.count(IM_END) + 1  # +1 for the open gen turn

    def test_alpaca_truncation_unchanged(self):
        s = ChatSession(max_context_tokens=40)
        for i in range(6):
            s.add_user_message(f"q{i}")
            s.add_assistant_message(f"a{i}")
        p = s.format_prompt(tokenizer=self._CharTok())
        assert len(p) <= 40
        assert p.endswith("### Assistant:\n")
        assert "q0" not in p


# --- Layer 3: InteractiveChat reply path -------------------------------------
class _RecordingGenerator:
    def __init__(self, reply):
        self.reply = reply
        self.last_kwargs = None

    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        return self.reply  # in chatml mode the generator already returns reply


class TestInteractiveChatReplyPath:
    def test_chatml_uses_return_new_only_and_im_end_stop(self):
        gen = _RecordingGenerator("def f(): pass")
        chat = InteractiveChat(gen, chat_format="chatml")
        reply = chat._generate_reply("<|im_start|>assistant\n")
        assert reply == "def f(): pass"
        assert gen.last_kwargs["return_new_only"] is True
        assert gen.last_kwargs["stop_tokens"] == [IM_END]

    def test_alpaca_uses_prompt_strip_path(self):
        prompt = "### User:\nhi\n\n### Assistant:\n"
        gen = _RecordingGenerator(prompt + "Hello")  # prompt+completion
        chat = InteractiveChat(gen, chat_format="alpaca")
        reply = chat._generate_reply(prompt)
        assert reply == "Hello"
        assert "return_new_only" not in gen.last_kwargs
