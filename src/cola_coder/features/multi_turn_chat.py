"""Multi-Turn Chat: conversation history management for code generation.

Manages a sequence of user/assistant turns, formats them into a single prompt
for the model, and handles context window limits by truncating old turns.

For a TS dev: like managing a chat state with messages array, except we need
to carefully pack everything into a fixed-size context window.
"""

from dataclasses import dataclass, field
from typing import Literal
import json

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMessage":
        return cls(
            role=d["role"],
            content=d["content"],
            metadata=d.get("metadata"),
        )


@dataclass
class ChatSession:
    """A multi-turn conversation session.

    Manages the conversation history and formats it into prompts
    that fit within the model's context window.
    """
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str = ""
    max_context_tokens: int = 1024
    turn_separator: str = "\n\n"
    user_prefix: str = "### User:\n"
    assistant_prefix: str = "### Assistant:\n"
    system_prefix: str = "### System:\n"

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content, metadata=metadata))

    def add_user_message(self, content: str) -> None:
        """Shortcut to add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Shortcut to add an assistant message."""
        self.add_message("assistant", content)

    def format_prompt(self, tokenizer=None) -> str:
        """Format the conversation history into a single prompt string.

        If a tokenizer is provided, truncates old messages to fit within
        max_context_tokens. Without a tokenizer, includes all messages.

        Args:
            tokenizer: Optional tokenizer for token counting and truncation

        Returns:
            Formatted prompt string ready for the model
        """
        parts = []

        # System prompt first
        if self.system_prompt:
            parts.append(f"{self.system_prefix}{self.system_prompt}")

        # Format each message
        for msg in self.messages:
            if msg.role == "user":
                parts.append(f"{self.user_prefix}{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"{self.assistant_prefix}{msg.content}")
            elif msg.role == "system":
                parts.append(f"{self.system_prefix}{msg.content}")

        # Add the assistant prefix for the model to continue
        parts.append(self.assistant_prefix)

        full_prompt = self.turn_separator.join(parts)

        # Truncate if tokenizer available and prompt too long
        if tokenizer:
            full_prompt = self._truncate_to_fit(full_prompt, tokenizer)

        return full_prompt

    def _truncate_to_fit(self, prompt: str, tokenizer) -> str:
        """Truncate oldest messages to fit within context window.

        Strategy: keep system prompt + last N turns that fit.
        """
        token_count = len(tokenizer.encode(prompt, add_bos=False))
        if token_count <= self.max_context_tokens:
            return prompt

        # Rebuild with fewer messages
        # Keep system prompt, drop oldest messages first
        for start_idx in range(1, len(self.messages)):
            parts = []
            if self.system_prompt:
                parts.append(f"{self.system_prefix}{self.system_prompt}")

            for msg in self.messages[start_idx:]:
                if msg.role == "user":
                    parts.append(f"{self.user_prefix}{msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"{self.assistant_prefix}{msg.content}")
                elif msg.role == "system":
                    parts.append(f"{self.system_prefix}{msg.content}")

            parts.append(self.assistant_prefix)
            truncated = self.turn_separator.join(parts)

            token_count = len(tokenizer.encode(truncated, add_bos=False))
            if token_count <= self.max_context_tokens:
                return truncated

        # If even a single turn doesn't fit, return just the last turn truncated
        last_msg = self.messages[-1]
        return f"{self.user_prefix}{last_msg.content}\n\n{self.assistant_prefix}"

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def pop_last(self) -> ChatMessage | None:
        """Remove and return the last message."""
        if self.messages:
            return self.messages.pop()
        return None

    @property
    def turn_count(self) -> int:
        """Number of complete user-assistant turn pairs."""
        user_count = sum(1 for m in self.messages if m.role == "user")
        assistant_count = sum(1 for m in self.messages if m.role == "assistant")
        return min(user_count, assistant_count)

    @property
    def last_user_message(self) -> str | None:
        """Get the last user message content."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    @property
    def last_assistant_message(self) -> str | None:
        """Get the last assistant message content."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_json(self) -> str:
        """Serialize session to JSON."""
        data = {
            "system_prompt": self.system_prompt,
            "max_context_tokens": self.max_context_tokens,
            "messages": [m.to_dict() for m in self.messages],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ChatSession":
        """Deserialize session from JSON."""
        data = json.loads(json_str)
        session = cls(
            system_prompt=data.get("system_prompt", ""),
            max_context_tokens=data.get("max_context_tokens", 1024),
        )
        for msg_data in data.get("messages", []):
            session.messages.append(ChatMessage.from_dict(msg_data))
        return session


class InteractiveChat:
    """Interactive CLI chat loop using a code generation model.

    Provides a REPL-like interface for multi-turn conversations.
    """

    def __init__(
        self,
        generator,
        system_prompt: str = "You are a helpful coding assistant.",
        max_context_tokens: int = 1024,
    ):
        """
        Args:
            generator: A CodeGenerator or StreamingGenerator instance
            system_prompt: System instructions for the model
            max_context_tokens: Max tokens for context window
        """
        self.generator = generator
        self.session = ChatSession(
            system_prompt=system_prompt,
            max_context_tokens=max_context_tokens,
        )
        self.commands = {
            "/clear": self._cmd_clear,
            "/history": self._cmd_history,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/turns": self._cmd_turns,
            "/help": self._cmd_help,
        }

    def run(self) -> None:
        """Start the interactive chat loop."""
        from ..cli import cli
        cli.header("Cola-Coder", "Interactive Chat")
        cli.dim("Type /help for commands, Ctrl+C to exit")
        print()

        while True:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/"):
                cmd = user_input.split()[0]
                args = user_input[len(cmd):].strip()
                if cmd in self.commands:
                    self.commands[cmd](args)
                    continue
                else:
                    print(f"Unknown command: {cmd}. Type /help for commands.")
                    continue

            # Add user message and generate response
            self.session.add_user_message(user_input)
            prompt = self.session.format_prompt()

            # Generate response
            response = self.generator.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                stop_tokens=[self.session.user_prefix.strip()],
            )

            # Extract just the assistant's response (after the last assistant prefix)
            assistant_text = response
            prefix = self.session.assistant_prefix
            if prefix in response:
                # Get text after the LAST occurrence of assistant prefix
                parts = response.rsplit(prefix, 1)
                if len(parts) > 1:
                    assistant_text = parts[1].strip()

            self.session.add_assistant_message(assistant_text)
            print(f"\nAssistant> {assistant_text}\n")

    def _cmd_clear(self, args: str) -> None:
        self.session.clear()
        print("Chat history cleared.")

    def _cmd_history(self, args: str) -> None:
        if not self.session.messages:
            print("No messages yet.")
            return
        for i, msg in enumerate(self.session.messages):
            role = msg.role.upper()
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"  [{i}] {role}: {content}")

    def _cmd_save(self, args: str) -> None:
        path = args or "chat_session.json"
        from pathlib import Path
        Path(path).write_text(self.session.to_json())
        print(f"Session saved to {path}")

    def _cmd_load(self, args: str) -> None:
        path = args or "chat_session.json"
        from pathlib import Path
        if not Path(path).exists():
            print(f"File not found: {path}")
            return
        self.session = ChatSession.from_json(Path(path).read_text())
        print(f"Session loaded from {path} ({len(self.session.messages)} messages)")

    def _cmd_turns(self, args: str) -> None:
        print(f"Turns: {self.session.turn_count}")
        print(f"Messages: {len(self.session.messages)}")

    def _cmd_help(self, args: str) -> None:
        print("Commands:")
        print("  /clear    - Clear chat history")
        print("  /history  - Show message history")
        print("  /save [f] - Save session to file")
        print("  /load [f] - Load session from file")
        print("  /turns    - Show turn count")
        print("  /help     - Show this help")
