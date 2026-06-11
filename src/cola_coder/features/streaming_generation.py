"""Streaming Generation: yield tokens as they are generated.

Instead of waiting for the full sequence, this streams tokens one at a time.
Useful for interactive CLI, web servers (SSE), and real-time display.

For a TS dev: this is like returning a ReadableStream or async generator
instead of a Promise that resolves with the full string.
"""

import time
from dataclasses import dataclass
from typing import Generator, Callable

import torch
from torch.amp import autocast

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class StreamStats:
    """Statistics from a streaming generation run."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    stopped_by: str = ""  # "eos", "stop_token", "max_tokens"

    @property
    def tokens_per_second(self) -> float:
        if self.decode_time_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.decode_time_ms / 1000.0)

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill_time_ms <= 0:
            return 0.0
        return self.prompt_tokens / (self.prefill_time_ms / 1000.0)

    def summary(self) -> str:
        lines = [
            f"Prompt: {self.prompt_tokens} tokens ({self.prefill_time_ms:.0f}ms, "
            f"{self.prefill_tokens_per_second:.0f} tok/s)",
            f"Generated: {self.generated_tokens} tokens ({self.decode_time_ms:.0f}ms, "
            f"{self.tokens_per_second:.1f} tok/s)",
            f"Total: {self.total_time_ms:.0f}ms | Stopped by: {self.stopped_by}",
        ]
        return "\n".join(lines)


@dataclass
class StreamToken:
    """A single streamed token with metadata."""
    text: str
    token_id: int
    position: int  # position in generated sequence (0-indexed)
    elapsed_ms: float  # time since generation started
    is_final: bool = False


class StreamingGenerator:
    """Generate code with token-by-token streaming output.

    Usage:
        gen = StreamingGenerator(model, tokenizer, device="cuda")

        # As a generator
        for token in gen.stream("def hello"):
            print(token.text, end="", flush=True)

        # With callback
        gen.generate_with_callback(
            "def hello",
            on_token=lambda t: print(t.text, end="", flush=True),
        )
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self._last_stats: StreamStats | None = None

    @property
    def last_stats(self) -> StreamStats | None:
        """Stats from the most recent generation."""
        return self._last_stats

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
    ) -> Generator[StreamToken, None, None]:
        """Stream tokens as they are generated.

        Yields StreamToken objects one at a time. The full text can be
        reconstructed by concatenating all token.text values.

        Args:
            prompt: Input text to continue from
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            stop_tokens: Stop when any of these strings are generated

        Yields:
            StreamToken with text, token_id, position, timing
        """
        from ..inference.sampling import sample_next_token
        from ..inference.generator import partition_stops, _earliest_stop_index

        stats = StreamStats()
        start_time = time.perf_counter()

        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        stats.prompt_tokens = len(token_ids)

        # Stops: single-token (EOS + special tokens) matched exactly; multi-token
        # stops matched at the STRING level on the decoded completion. Reducing a
        # multi-token stop to its first token halts far too early (INFER-006).
        single_stop_ids, string_stops = partition_stops(self.tokenizer, stop_tokens)

        # Clear caches
        self.model.clear_caches()
        generated_ids = list(token_ids)

        # Prefill: process entire prompt at once
        prefill_start = time.perf_counter()
        use_amp = self.device == "cuda"
        with autocast(device_type=self.device if use_amp else "cpu",
                       dtype=torch.bfloat16, enabled=use_amp):
            logits = self.model(input_ids, start_pos=0, use_cache=True)
        next_logits = logits[0, -1, :]
        stats.prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

        # Decode: generate tokens one at a time
        decode_start = time.perf_counter()
        stopped_by = "max_tokens"

        # Text is computed by decoding the FULL sequence and yielding only the
        # incremental new characters (full-decode-diff). Per-token decode
        # ([next_token]) mangles byte-level BPE: a single token can be a partial
        # multi-byte UTF-8 sequence, so it decodes to replacement chars / wrong
        # spacing in isolation. (Matches CodeGenerator.generate_stream.)
        current_decoded = self.tokenizer.decode(generated_ids)
        prev_decoded_len = len(current_decoded)
        prompt_char_len = prev_decoded_len
        # With string stops active, hold back the last (max_stop_len-1) chars
        # before emitting so a stop sequence can't leak its own prefix (e.g.
        # "\n\n" leaking its first "\n"). Matches CodeGenerator.generate_stream.
        max_stop_len = max((len(s) for s in string_stops), default=0)
        last_token = None

        for i in range(max_new_tokens):
            next_token = sample_next_token(
                next_logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            # Token-level stop (EOS / single-token stops) — excluded from output.
            if next_token in single_stop_ids:
                stopped_by = "eos" if next_token == self.tokenizer.eos_id else "stop_token"
                break

            generated_ids.append(next_token)
            last_token = next_token
            current_decoded = self.tokenizer.decode(generated_ids)
            elapsed = (time.perf_counter() - start_time) * 1000

            if string_stops:
                idx = _earliest_stop_index(current_decoded, string_stops, prompt_char_len)
                if idx is not None:
                    if idx > prev_decoded_len:
                        yield StreamToken(
                            text=current_decoded[prev_decoded_len:idx],
                            token_id=next_token, position=i,
                            elapsed_ms=elapsed, is_final=True,
                        )
                    stopped_by = "stop_token"
                    break
                # Only emit text that can't be the prefix of a future stop.
                safe_len = len(current_decoded) - (max_stop_len - 1)
                token_text = (
                    current_decoded[prev_decoded_len:safe_len]
                    if safe_len > prev_decoded_len else ""
                )
                prev_decoded_len = max(prev_decoded_len, safe_len)
            else:
                token_text = current_decoded[prev_decoded_len:]
                prev_decoded_len = len(current_decoded)

            is_final = (i == max_new_tokens - 1)
            yield StreamToken(
                text=token_text,
                token_id=next_token,
                position=i,
                elapsed_ms=elapsed,
                is_final=is_final,
            )

            # Feed the new token to get the next token's logits (KV-cache).
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            start_pos = len(generated_ids) - 1
            with autocast(device_type=self.device if use_amp else "cpu",
                           dtype=torch.bfloat16, enabled=use_amp):
                logits = self.model(next_input, start_pos=start_pos, use_cache=True)
            next_logits = logits[0, -1, :]

        # Flush any held-back tail (generation ended via EOS / max tokens without
        # hitting a string stop).
        if string_stops and len(current_decoded) > prev_decoded_len and stopped_by != "stop_token":
            yield StreamToken(
                text=current_decoded[prev_decoded_len:],
                token_id=last_token if last_token is not None else self.tokenizer.eos_id,
                position=len(generated_ids) - len(token_ids) - 1,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                is_final=True,
            )

        # Finalize stats
        stats.decode_time_ms = (time.perf_counter() - decode_start) * 1000
        stats.total_time_ms = (time.perf_counter() - start_time) * 1000
        stats.generated_tokens = len(generated_ids) - len(token_ids)
        stats.stopped_by = stopped_by
        self._last_stats = stats
        self.model.clear_caches()

    def generate_with_callback(
        self,
        prompt: str,
        on_token: Callable[[StreamToken], None] | None = None,
        on_complete: Callable[[StreamStats], None] | None = None,
        **kwargs,
    ) -> str:
        """Generate with callbacks — returns the full text.

        Args:
            prompt: Input prompt
            on_token: Called for each generated token
            on_complete: Called when generation finishes with stats
            **kwargs: Passed to stream()

        Returns:
            Full generated text (prompt + generated)
        """
        all_text_parts = []
        for token in self.stream(prompt, **kwargs):
            all_text_parts.append(token.text)
            if on_token:
                on_token(token)

        if on_complete and self._last_stats:
            on_complete(self._last_stats)

        return prompt + "".join(all_text_parts)

    def print_stream(
        self,
        prompt: str,
        show_stats: bool = True,
        **kwargs,
    ) -> str:
        """Generate and print tokens in real-time to stdout.

        Args:
            prompt: Input prompt
            show_stats: Print throughput stats after generation
            **kwargs: Passed to stream()

        Returns:
            Full generated text
        """

        print(prompt, end="", flush=True)
        parts = []
        for token in self.stream(prompt, **kwargs):
            print(token.text, end="", flush=True)
            parts.append(token.text)
        print()  # Final newline

        if show_stats and self._last_stats:
            print(f"\n--- {self._last_stats.summary()}")

        return prompt + "".join(parts)
