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

import torch
from torch.cuda.amp import autocast

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.tokenizer_utils import CodeTokenizer
from .sampling import sample_next_token


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
