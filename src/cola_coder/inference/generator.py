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

    @torch.no_grad()  # Disable gradient computation (saves memory, faster)
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate code given a prompt.

        Args:
            prompt: The input text/code to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy, higher = more random).
            top_k: Top-k filtering threshold.
            top_p: Top-p (nucleus) filtering threshold.
            repetition_penalty: Penalty for repeating tokens.
            stop_tokens: Stop generation when any of these tokens are generated.

        Returns:
            The generated text (prompt + new tokens).
        """
        # Encode the prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Get stop token IDs
        stop_ids = set()
        stop_ids.add(self.tokenizer.eos_id)
        if stop_tokens:
            for st in stop_tokens:
                encoded = self.tokenizer.encode(st, add_bos=False)
                if encoded:
                    stop_ids.add(encoded[0])

        # Clear any existing cache
        self.model.clear_caches()

        generated_ids = list(token_ids)

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
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            # Check stop condition
            if next_token in stop_ids:
                break

            generated_ids.append(next_token)

            # Feed the new token through the model (with KV-cache)
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            start_pos = len(generated_ids) - 1

            with autocast(device_type="cuda", dtype=torch.bfloat16,
                           enabled=self.device == "cuda"):
                logits = self.model(next_input, start_pos=start_pos, use_cache=True)

            next_logits = logits[0, -1, :]

        # Decode all generated tokens
        self.model.clear_caches()
        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: list[str] | None = None,
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
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Get stop token IDs
        stop_ids = set()
        stop_ids.add(self.tokenizer.eos_id)
        if stop_tokens:
            for st in stop_tokens:
                encoded = self.tokenizer.encode(st, add_bos=False)
                if encoded:
                    stop_ids.add(encoded[0])

        # Clear any existing cache
        self.model.clear_caches()

        generated_ids = list(token_ids)
        # Track what we've already yielded so we can compute the incremental diff
        prev_decoded_len = len(self.tokenizer.decode(generated_ids))

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
                    repetition_penalty=repetition_penalty,
                    generated_ids=generated_ids,
                )

                # Check stop condition
                if next_token in stop_ids:
                    break

                generated_ids.append(next_token)

                # Decode full sequence and yield only the new characters.
                # This correctly handles BPE merges where a single token ID can
                # decode differently depending on surrounding context, and
                # multi-byte UTF-8 sequences that may span token boundaries.
                current_decoded = self.tokenizer.decode(generated_ids)
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
                batch_size=batch_size,
            )
        except torch.cuda.OutOfMemoryError:
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
        prompt_len = len(token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        self.model.clear_caches()

        # --- Phase 2: prefill (batch=1) ---
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits = self.model(input_ids, start_pos=0, use_cache=True)

        # logits for the last prompt token → first generation step
        # Expand from (1, vocab) to (batch_size, vocab)
        next_logits = logits[:, -1, :].expand(batch_size, -1).clone()

        # --- Phase 3: expand KV-cache to batch_size ---
        self.model.expand_caches(batch_size)

        # Per-sequence token buffers (start with the prompt tokens)
        # Shape: (batch_size, dynamic_len) — stored as a list of lists for flexibility
        seq_tokens: list[list[int]] = [list(token_ids) for _ in range(batch_size)]
        finished = [False] * batch_size

        # --- Phase 4: autoregressive decode ---
        for step in range(max_new_tokens):
            # Sample next tokens for all sequences in one vectorised call
            sampled = sample_next_tokens_batch(
                next_logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
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
            )
            results.append(result)
        return results
