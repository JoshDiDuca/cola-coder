"""Text generation with KV-cache.

This is where you actually USE the trained model to generate code.

The generation process:
1. Encode the prompt into token IDs
2. Feed the prompt through the model (populate the KV-cache)
3. Get the logits for the last token
4. Sample the next token
5. Feed that single token back in (using the cached K/V from previous tokens)
6. Repeat steps 3-5 until we hit a stop condition

The KV-cache is what makes generation fast. Without it, every new token
would require re-processing the entire sequence from scratch. With the cache,
we only process the single new token and look up the cached attention state
for all previous tokens.

For a TS dev: the KV-cache is like memoization. Once you've computed the
attention state for a token, you cache it and never recompute it.
"""

from typing import Generator

import torch
from torch.amp import autocast

from ..model.transformer import Transformer
from ..tokenizer.tokenizer_utils import CodeTokenizer
import logging

from .sampling import sample_next_token, sample_next_tokens_batch

logger = logging.getLogger(__name__)


def _earliest_stop_index(text: str, stop_strings: list[str], start: int = 0) -> int | None:
    """Return the earliest index at/after ``start`` where any stop string begins.

    Used for STRING-level stop detection (multi-token stop sequences). ``start``
    is the character length of the decoded prompt so a stop string only halts
    generation when it begins in the completion, never inside the prompt.
    """
    earliest: int | None = None
    for s in stop_strings:
        if not s:
            continue
        i = text.find(s, start)
        if i != -1 and (earliest is None or i < earliest):
            earliest = i
    return earliest


def partition_stops(tokenizer, stop_tokens: list[str] | None) -> tuple[set[int], list[str]]:
    """Split requested stops into token-level and string-level matchers.

    EOS and any stop that encodes to exactly ONE token (this covers special
    tokens such as ``<|im_end|>`` / ``<|fim_suffix|>``, which the decoder strips
    and so cannot be matched as text) go in a set of token IDs for exact, fast
    matching. Stops that encode to MULTIPLE tokens are returned as strings for
    substring matching on the decoded output — reducing them to their first
    token stops generation far too early (e.g. ``";\\n"`` halts at the first
    ``;``). Shared by CodeGenerator and StreamingGenerator so both behave
    identically (INFER-006).
    """
    single_stop_ids: set[int] = {tokenizer.eos_id}
    string_stops: list[str] = []
    for st in stop_tokens or []:
        if not st:
            continue
        encoded = tokenizer.encode(st, add_bos=False)
        if len(encoded) == 1:
            single_stop_ids.add(encoded[0])
        elif encoded:
            string_stops.append(st)
    return single_stop_ids, string_stops


def _fit_context_window(
    token_ids: list[int],
    max_new_tokens: int,
    max_seq_len: int,
) -> tuple[list[int], int]:
    """Clamp prompt + generation length to the model's KV-cache capacity.

    The KV-cache and causal mask are allocated for exactly ``config.max_seq_len``
    positions (see ``CausalSelfAttention._init_cache`` / ``Transformer.causal_mask``).
    Two failure modes exist without this guard, both reachable from the FIM
    endpoint (a long file easily exceeds ``seq_len``):

    * **Prompt longer than the window** — prefill does
      ``cache_k[:, 0:seq_len] = k`` with ``seq_len > max_seq_len``, which raises a
      cryptic ``RuntimeError`` ("expanded size ... must match existing size") and
      500s the request.
    * **Generation crosses the window mid-decode** — once
      ``start_pos >= max_seq_len`` the write ``cache_k[:, start_pos:start_pos+1]``
      targets a zero-size slice, so the new token's K/V is **silently dropped**
      and the model reads stale cache — garbage output with no error.

    Fix: left-truncate the prompt to the most recent ``max_seq_len - 1`` tokens
    (keep at least one slot for a generated token, standard sliding-window
    behaviour) and cap ``max_new_tokens`` so ``start_pos`` never reaches the
    cache bound.

    Returns ``(possibly_truncated_token_ids, capped_max_new_tokens)``.
    """
    if max_seq_len <= 0:
        return token_ids, max_new_tokens
    # Keep at least one position free for generation when truncating.
    if len(token_ids) > max_seq_len - 1:
        token_ids = token_ids[-(max_seq_len - 1):]
    # start_pos for the i-th generated token is len(prompt) + i; the cache has
    # room for indices [0, max_seq_len). Cap so the last write stays in range.
    room = max_seq_len - len(token_ids)
    capped = max(0, min(max_new_tokens, room))
    return token_ids, capped


class CodeGenerator:
    """Generate code using a trained transformer model."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: CodeTokenizer,
        device: str = "cuda",
    ):
        """
        Args:
            model: Trained transformer model (already loaded with weights).
            tokenizer: Trained BPE tokenizer.
            device: "cuda" for GPU, "cpu" for CPU inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Disable dropout for deterministic inference

    def _partition_stops(self, stop_tokens: list[str] | None) -> tuple[set[int], list[str]]:
        """Split requested stops into token-level and string-level matchers.

        EOS and any stop that encodes to exactly ONE token (this covers special
        tokens such as ``<|im_end|>`` / ``<|fim_suffix|>``, which the decoder
        strips and so cannot be matched as text) are returned as a set of token
        IDs for exact, fast matching. Stops that encode to MULTIPLE tokens are
        returned as strings for substring matching on the decoded output —
        reducing them to their first token (the old behavior) stopped generation
        far too early (e.g. ``";\\n"`` halted at the first ``;``).
        """
        return partition_stops(self.tokenizer, stop_tokens)

    def _max_seq_len(self) -> int:
        """KV-cache / causal-mask capacity = ``model.config.max_seq_len``.

        Resolved defensively (0 disables the guard) so lightweight test stubs
        without a full ``config`` still work; real ``Transformer`` always has it.
        """
        config = getattr(self.model, "config", None)
        return int(getattr(config, "max_seq_len", 0) or 0)

    @torch.no_grad()  # Disable gradient computation (saves memory, faster)
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
        return_new_only: bool = False,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """Generate code given a prompt.

        Args:
            prompt: The input text/code to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy, higher = more random).
            top_k: Top-k filtering threshold.
            top_p: Top-p (nucleus) filtering threshold.
            min_p: Confidence-scaled floor — drop tokens below
                   min_p * max_token_prob (0 = disabled, try 0.05-0.1).
            repetition_penalty: Penalty for repeating tokens.
            stop_tokens: Stop generation when any of these tokens are generated.
            no_repeat_ngram_size: If > 0, hard-block any token that would repeat
                an n-gram of this size (fixes verbatim repetition loops; 3 typical,
                0 = off).
            return_new_only: When True, return ONLY the completion (decode of the
                newly generated tokens), not ``prompt + completion``. This is the
                robust way to recover the reply when the prompt contains special
                tokens (e.g. ChatML ``<|im_start|>``) that ``decode`` strips —
                string-diffing the decoded prompt then fails (INFER-011). Default
                False preserves the legacy prompt+completion return.

        Returns:
            The generated text — ``prompt + new tokens`` by default, or just the
            completion when ``return_new_only`` is True.
        """
        # Encode the prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        # Clamp prompt + generation to the KV-cache window. Without this, a prompt
        # longer than max_seq_len crashes prefill, and crossing the window mid-decode
        # silently drops K/V writes → garbage output (see _fit_context_window).
        token_ids, max_new_tokens = _fit_context_window(
            token_ids, max_new_tokens, self._max_seq_len()
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Partition stops: single-token stops (EOS + special tokens like
        # <|im_end|>/<|fim_suffix|>) match exactly at the token level; multi-token
        # stops ("\nclass ", ";\n", ...) match at the STRING level on the decoded
        # text. The old code reduced every stop to its FIRST token, so ";\n"
        # halted at the first ";" — truncating code after a single statement.
        single_stop_ids, string_stops = self._partition_stops(stop_tokens)

        # Clear any existing cache
        self.model.clear_caches()

        generated_ids = list(token_ids)
        # Char length of the decoded prompt → string stops only fire in completion
        prompt_char_len = len(self.tokenizer.decode(token_ids)) if string_stops else 0

        # Phase 1: Process the prompt (prefill)
        # Feed the entire prompt at once to populate the KV-cache
        with autocast(device_type="cuda", dtype=torch.bfloat16,
                       enabled=self.device == "cuda"):
            logits = self.model(input_ids, start_pos=0, use_cache=True)

        # Get logits for the last prompt token (the prediction for the first new token)
        next_logits = logits[0, -1, :]

        # Phase 2: Generate tokens one by one
        for i in range(max_new_tokens):
            # Sample next token
            next_token = sample_next_token(
                next_logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            # Token-level stop (EOS / single-token stops) — exclude from output
            if next_token in single_stop_ids:
                break

            generated_ids.append(next_token)

            # String-level stop: if a multi-token stop string appears in the
            # completion, truncate there and finish (the stop text is excluded).
            if string_stops:
                full = self.tokenizer.decode(generated_ids)
                idx = _earliest_stop_index(full, string_stops, prompt_char_len)
                if idx is not None:
                    self.model.clear_caches()
                    if return_new_only:
                        # Re-decode the completion alone and cut the stop there:
                        # char offsets differ between the full and new-only
                        # decodings, so we can't reuse `idx`.
                        comp = self.tokenizer.decode(generated_ids[len(token_ids):])
                        cidx = _earliest_stop_index(comp, string_stops, 0)
                        return comp[:cidx] if cidx is not None else comp
                    return full[:idx]

            # Feed the new token through the model (with KV-cache)
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            start_pos = len(generated_ids) - 1

            with autocast(device_type="cuda", dtype=torch.bfloat16,
                           enabled=self.device == "cuda"):
                logits = self.model(next_input, start_pos=start_pos, use_cache=True)

            next_logits = logits[0, -1, :]

        # Decode all generated tokens
        self.model.clear_caches()
        if return_new_only:
            return self.tokenizer.decode(generated_ids[len(token_ids):])
        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
        no_repeat_ngram_size: int = 0,
    ) -> Generator[str, None, None]:
        """Generate code given a prompt, yielding tokens incrementally as they're produced.

        Uses the same KV-cache logic as generate(), but yields the new text after each
        token rather than returning everything at the end. To handle BPE merge edge cases
        and multi-byte characters cleanly, it decodes the full generated sequence each
        step and yields only the incremental difference (new characters since last yield).

        Args:
            prompt: The input text/code to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy, higher = more random).
            top_k: Top-k filtering threshold.
            top_p: Top-p (nucleus) filtering threshold.
            repetition_penalty: Penalty for repeating tokens.
            stop_tokens: Stop generation when any of these tokens are generated.

        Yields:
            Incremental text chunks as new tokens are generated.
        """
        # Encode the prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        # Clamp prompt + generation to the KV-cache window (see generate() /
        # _fit_context_window): a prompt longer than max_seq_len crashes prefill,
        # and crossing the window mid-decode silently corrupts the cache.
        token_ids, max_new_tokens = _fit_context_window(
            token_ids, max_new_tokens, self._max_seq_len()
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Partition stops (see generate() for why): single-token stops match at
        # the token level, multi-token stops at the string level on decoded text.
        single_stop_ids, string_stops = self._partition_stops(stop_tokens)

        # Clear any existing cache
        self.model.clear_caches()

        generated_ids = list(token_ids)
        # Track what we've already yielded so we can compute the incremental diff.
        # This also marks where the completion begins, so string stops only fire
        # in completion text, never inside the prompt.
        prev_decoded_len = len(self.tokenizer.decode(generated_ids))
        prompt_char_len = prev_decoded_len
        # When string stops are active we must hold back the last (max_stop_len-1)
        # decoded chars before yielding: they might be the START of a stop
        # sequence that completes on a later token. Without this, "\n\n" would
        # leak its first "\n" before the stop fires. (vLLM/TGI do the same.)
        max_stop_len = max((len(s) for s in string_stops), default=0)
        current_decoded = self.tokenizer.decode(generated_ids)

        try:
            # Phase 1: Process the prompt (prefill)
            # Feed the entire prompt at once to populate the KV-cache
            with autocast(device_type="cuda", dtype=torch.bfloat16,
                           enabled=self.device == "cuda"):
                logits = self.model(input_ids, start_pos=0, use_cache=True)

            # Get logits for the last prompt token (the prediction for the first new token)
            next_logits = logits[0, -1, :]

            # Phase 2: Generate tokens one by one, yielding each as it arrives
            for i in range(max_new_tokens):
                # Sample next token
                next_token = sample_next_token(
                    next_logits.clone(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    generated_ids=generated_ids,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )

                # Token-level stop (EOS / single-token stops)
                if next_token in single_stop_ids:
                    break

                generated_ids.append(next_token)

                # Decode full sequence and yield only the new characters.
                # This correctly handles BPE merges where a single token ID can
                # decode differently depending on surrounding context, and
                # multi-byte UTF-8 sequences that may span token boundaries.
                current_decoded = self.tokenizer.decode(generated_ids)

                if string_stops:
                    # String-level stop: emit up to the stop string, then finish
                    # (the stop text itself is not emitted).
                    idx = _earliest_stop_index(current_decoded, string_stops, prompt_char_len)
                    if idx is not None:
                        if idx > prev_decoded_len:
                            yield current_decoded[prev_decoded_len:idx]
                        return
                    # Only emit text that can't be the prefix of a future stop.
                    safe_len = len(current_decoded) - (max_stop_len - 1)
                    if safe_len > prev_decoded_len:
                        yield current_decoded[prev_decoded_len:safe_len]
                        prev_decoded_len = safe_len
                else:
                    new_text = current_decoded[prev_decoded_len:]
                    if new_text:
                        yield new_text
                    prev_decoded_len = len(current_decoded)

                # Feed the new token through the model (with KV-cache)
                next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                start_pos = len(generated_ids) - 1

                with autocast(device_type="cuda", dtype=torch.bfloat16,
                               enabled=self.device == "cuda"):
                    logits = self.model(next_input, start_pos=start_pos, use_cache=True)

                next_logits = logits[0, -1, :]

            # Flush any held-back tail (generation ended via EOS / max tokens
            # without hitting a string stop).
            if string_stops and len(current_decoded) > prev_decoded_len:
                yield current_decoded[prev_decoded_len:]

        finally:
            # Always clear the KV cache, even if generation was interrupted
            self.model.clear_caches()

    @torch.no_grad()
    def generate_group(
        self,
        prompt: str,
        num_completions: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
    ) -> list[str]:
        """Generate multiple completions for the SAME prompt in a single batched pass.

        Optimised for GRPO: because all completions share the same prompt, we:
        1. Prefill the KV-cache once (batch=1, full prompt).
        2. Expand the KV-cache along the batch dimension to num_completions.
        3. Generate all completions in parallel (one forward pass per token step).
        4. Each completion samples independently via sample_next_tokens_batch.
        5. Track per-sequence EOS with a mask so stopped sequences don't grow.

        If the full batch does not fit in VRAM (torch.cuda.OutOfMemoryError), the
        method transparently retries with progressively smaller mini-batches
        (halving each time) and stitches results together.  If even batch=1
        fails it falls back to the serial generate() path.

        Args:
            prompt: The input text/code shared by all completions.
            num_completions: How many independent completions to produce (G).
            max_new_tokens: Maximum number of new tokens per completion.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k filtering (0 = disabled).
            top_p: Nucleus sampling threshold.
            min_p: Confidence-scaled floor (0 = disabled).

        Returns:
            List of num_completions decoded strings (prompt + generated tokens).
        """
        # Try batched generation, falling back to smaller batches on OOM.
        return self._generate_group_with_fallback(
            prompt=prompt,
            num_completions=num_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            batch_size=num_completions,
        )

    def _generate_group_with_fallback(
        self,
        prompt: str,
        num_completions: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        batch_size: int,
        min_p: float = 0.0,
    ) -> list[str]:
        """Internal helper — attempts batched generation, retries with smaller batches on OOM."""
        if batch_size <= 1:
            # Last-resort: fully serial fallback
            logger.warning(
                "generate_group: falling back to serial generation "
                "(VRAM insufficient for batched mode)"
            )
            return [
                self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )
                for _ in range(num_completions)
            ]

        try:
            return self._generate_group_batched(
                prompt=prompt,
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                batch_size=batch_size,
            )
        except torch.cuda.OutOfMemoryError:
            # Clear the (possibly expanded) KV-cache BEFORE empty_cache, or the
            # still-referenced cache tensors can't be freed and the retry runs
            # with the same VRAM pressure that caused the OOM.
            self.model.clear_caches()
            torch.cuda.empty_cache()
            new_batch = max(1, batch_size // 2)
            logger.warning(
                "generate_group: OOM with batch_size=%d, retrying with %d",
                batch_size,
                new_batch,
            )
            return self._generate_group_with_fallback(
                prompt=prompt,
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                batch_size=new_batch,
            )

    @torch.no_grad()
    def _generate_group_batched(
        self,
        prompt: str,
        num_completions: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        batch_size: int,
        min_p: float = 0.0,
    ) -> list[str]:
        """Core batched generation logic.

        Runs multiple mini-batches of size `batch_size` when batch_size < num_completions
        and stitches results together.
        """
        results: list[str] = []
        remaining = num_completions

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            batch_results = self._generate_group_single_batch(
                prompt=prompt,
                batch_size=current_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            results.extend(batch_results)
            remaining -= current_batch

        return results

    @torch.no_grad()
    def _generate_group_single_batch(
        self,
        prompt: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float = 0.0,
    ) -> list[str]:
        """Generate a single mini-batch of completions for the same prompt.

        Steps:
            1. Encode the prompt and prefill the KV-cache (batch=1).
            2. Expand the KV-cache to batch_size.
            3. Loop up to max_new_tokens:
               a. Sample batch_size next tokens in one call.
               b. Mark finished sequences (EOS hit).
               c. Run a forward pass with the new token for all unfinished seqs.
            4. Decode each sequence and return.
        """
        eos_id = self.tokenizer.eos_id

        # --- Phase 1: encode prompt ---
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        # Clamp to the KV-cache window: in Phase 4 start_pos = prompt_len + step
        # must stay < max_seq_len, else the per-step write cache_k[:, start_pos:
        # start_pos+1] = k targets a zero-size slice — silently dropping the new
        # token's K/V and reading stale cache → garbage. Same guard generate()/
        # generate_stream() use (INFER-014). Prompt shared across the group, so
        # clamp once here.
        token_ids, max_new_tokens = _fit_context_window(
            token_ids, max_new_tokens, self._max_seq_len()
        )
        prompt_len = len(token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        self.model.clear_caches()

        # The KV-cache is expanded to batch_size below; a try/finally guarantees
        # it's cleared even if prefill or decode raises (OOM, etc.). Otherwise an
        # exception leaves the cache in expanded (batch>1) state, wasting VRAM
        # until the next request and risking a shape mismatch on a serial call.
        seq_tokens: list[list[int]] = [list(token_ids) for _ in range(batch_size)]
        try:
            # --- Phase 2: prefill (batch=1) ---
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                logits = self.model(input_ids, start_pos=0, use_cache=True)

            # logits for the last prompt token → first generation step
            # Expand from (1, vocab) to (batch_size, vocab)
            next_logits = logits[:, -1, :].expand(batch_size, -1).clone()

            # --- Phase 3: expand KV-cache to batch_size ---
            self.model.expand_caches(batch_size)

            finished = [False] * batch_size

            # --- Phase 4: autoregressive decode ---
            for step in range(max_new_tokens):
                # Sample next tokens for all sequences in one vectorised call
                sampled = sample_next_tokens_batch(
                    next_logits.clone(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )  # (batch_size,)

                # Check EOS and append tokens
                all_done = True
                for i in range(batch_size):
                    if finished[i]:
                        continue
                    tok = sampled[i].item()
                    if tok == eos_id:
                        finished[i] = True
                    else:
                        seq_tokens[i].append(tok)
                        all_done = False

                if all_done:
                    break

                # Build next input: (batch_size, 1) — use EOS as a dummy token
                # for already-finished sequences (their logits are discarded).
                next_token_ids = sampled.unsqueeze(1)  # (batch_size, 1)

                start_pos = prompt_len + step

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    logits = self.model(next_token_ids, start_pos=start_pos, use_cache=True)

                next_logits = logits[:, -1, :]  # (batch_size, vocab)
        finally:
            self.model.clear_caches()

        # Decode each sequence
        return [self.tokenizer.decode(toks) for toks in seq_tokens]

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
    ) -> list[str]:
        """Generate code for multiple prompts.

        Note: batch generation is more complex because different prompts
        have different lengths. For simplicity, we generate one at a time.
        A production implementation would pad prompts and track per-sequence
        stop conditions.

        Args:
            prompts: List of input texts.
            max_new_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
            top_k: Top-k threshold.
            top_p: Top-p threshold.
            min_p: Confidence-scaled floor (0 = disabled).
            repetition_penalty: Penalty for repeating tokens.
            stop_tokens: Stop generation when any of these tokens are produced.

        Returns:
            List of generated texts.
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens,
            )
            results.append(result)
        return results
