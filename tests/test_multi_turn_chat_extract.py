"""INFER-010: InteractiveChat reply extraction must be robust.

The old extraction did rsplit on "### Assistant:" with a fallback of
`assistant_text = response` (the WHOLE prompt+completion) when the marker
wasn't found, and would drop the reply's start if the model emitted an
assistant marker itself. Now it uses the canonical strip_prompt_prefix.
"""

from cola_coder.features.multi_turn_chat import InteractiveChat
from cola_coder.inference.text_utils import strip_prompt_prefix


def _chat():
    return InteractiveChat(generator=object())


class TestExtractReply:
    def test_clean_completion(self):
        chat = _chat()
        prompt = "### User:\nhi\n\n### Assistant:\n"
        reply = chat._extract_reply(prompt + "Hello there", prompt)
        assert reply == "Hello there"

    def test_drift_does_not_echo_full_prompt(self):
        # Simulate BPE round-trip drift (last prompt char changed): the OLD
        # fallback returned the whole prompt+completion. Now we strip the
        # longest common prefix → no full-prompt echo.
        chat = _chat()
        prompt = "### User:\nhi\n\n### Assistant:\n"
        drifted = prompt[:-1] + " REPLY"
        reply = chat._extract_reply(drifted, prompt)
        assert prompt not in reply
        assert reply == strip_prompt_prefix(drifted, prompt).strip()

    def test_truncates_at_model_emitted_user_marker(self):
        # If the model runs past its turn into a new user marker, keep only the
        # reply up to that marker.
        chat = _chat()
        prompt = "### User:\nq\n\n### Assistant:\n"
        response = prompt + "the answer\n\n### User:\nnext question"
        reply = chat._extract_reply(response, prompt)
        assert reply == "the answer"

    def test_no_full_echo_when_marker_absent(self):
        # Even if the prompt's marker somehow isn't literally present, the reply
        # must be the completion, never the entire conversation.
        chat = _chat()
        prompt = "PROMPTBODY"
        reply = chat._extract_reply(prompt + "COMPLETION", prompt)
        assert reply == "COMPLETION"
