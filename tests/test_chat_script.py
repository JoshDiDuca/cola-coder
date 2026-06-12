"""UX-014: live multi-turn chat entry point (scripts/chat.py).

Locks the format-resolution rule (auto → chatml for _sft checkpoints, else
alpaca; explicit override wins) and that InteractiveChat actually forwards its
per-turn sampling params to the generator (so the script's --temperature /
--max-tokens flags are meaningful, not ignored).
"""

import importlib.util
from pathlib import Path

import pytest

from cola_coder.features.multi_turn_chat import InteractiveChat

_SCRIPT = Path(__file__).parent.parent / "scripts" / "chat.py"


def _load():
    spec = importlib.util.spec_from_file_location("chat_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResolveChatFormat:
    def test_sft_parent_dir_autoselects_chatml(self, tmp_path):
        m = _load()
        ckpt = tmp_path / "checkpoints" / "small_sft" / "step_00010000"
        ckpt.mkdir(parents=True)
        assert m.resolve_chat_format(ckpt, "auto") == "chatml"

    def test_base_checkpoint_autoselects_alpaca(self, tmp_path):
        m = _load()
        ckpt = tmp_path / "checkpoints" / "small" / "step_00010000"
        ckpt.mkdir(parents=True)
        assert m.resolve_chat_format(ckpt, "auto") == "alpaca"

    def test_reasoning_sft_dir_is_chatml(self, tmp_path):
        m = _load()
        ckpt = tmp_path / "checkpoints" / "4080_max_sft" / "latest"
        ckpt.mkdir(parents=True)
        assert m.resolve_chat_format(ckpt, "auto") == "chatml"

    def test_explicit_override_wins_over_auto_signal(self, tmp_path):
        m = _load()
        sft = tmp_path / "small_sft" / "step_1"
        sft.mkdir(parents=True)
        # _sft path but user forced alpaca
        assert m.resolve_chat_format(sft, "alpaca") == "alpaca"
        base = tmp_path / "small" / "step_1"
        base.mkdir(parents=True)
        assert m.resolve_chat_format(base, "chatml") == "chatml"

    def test_invalid_override_raises(self, tmp_path):
        m = _load()
        with pytest.raises(ValueError, match="auto|alpaca|chatml"):
            m.resolve_chat_format(tmp_path, "yolo")


class _RecordingGenerator:
    def __init__(self):
        self.last_kwargs = None

    def generate(self, prompt, **kwargs):
        self.last_kwargs = kwargs
        return "reply"


class TestInteractiveChatForwardsSamplingParams:
    def test_chatml_uses_configured_params(self):
        gen = _RecordingGenerator()
        chat = InteractiveChat(
            gen, chat_format="chatml", max_new_tokens=99, temperature=0.42,
        )
        chat._generate_reply("<|im_start|>assistant\n")
        assert gen.last_kwargs["max_new_tokens"] == 99
        assert gen.last_kwargs["temperature"] == 0.42

    def test_alpaca_uses_configured_params(self):
        gen = _RecordingGenerator()
        chat = InteractiveChat(
            gen, chat_format="alpaca", max_new_tokens=33, temperature=0.11,
        )
        chat._generate_reply("### User:\nhi\n\n### Assistant:\n")
        assert gen.last_kwargs["max_new_tokens"] == 33
        assert gen.last_kwargs["temperature"] == 0.11

    def test_defaults_are_sane(self):
        gen = _RecordingGenerator()
        chat = InteractiveChat(gen, chat_format="chatml")
        chat._generate_reply("<|im_start|>assistant\n")
        assert gen.last_kwargs["max_new_tokens"] == 256
        assert gen.last_kwargs["temperature"] == 0.7
