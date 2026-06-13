"""Tokenizer utilities for encoding and decoding text.

This wraps the raw HuggingFace tokenizer with convenience methods
specific to our model's needs (special token IDs, batch encoding, etc.).
"""


from .train_tokenizer import load_tokenizer


class CodeTokenizer:
    """Wrapper around the BPE tokenizer with convenience methods.

    For a TS dev: this is like a service class that wraps a lower-level
    library with a cleaner API specific to our use case.
    """

    def __init__(self, tokenizer_path: str = "tokenizer.json"):
        self.tokenizer = load_tokenizer(tokenizer_path)

        # Cache special token IDs for fast access
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.unk_id = self.tokenizer.token_to_id("<|unk|>")
        self.fim_prefix_id = self.tokenizer.token_to_id("<|fim_prefix|>")
        self.fim_middle_id = self.tokenizer.token_to_id("<|fim_middle|>")
        self.fim_suffix_id = self.tokenizer.token_to_id("<|fim_suffix|>")

        # Fail loud on a mismatched tokenizer. The core tokens are required —
        # encode() does `[self.bos_id] + ids`, so a None id would silently
        # corrupt every sequence with a None token (crash or garbage logits)
        # instead of erroring here. (FIM tokens are optional — only encode_fim
        # needs them, and it checks separately.)
        missing = [
            name for name, tid in (
                ("<|pad|>", self.pad_id), ("<|bos|>", self.bos_id),
                ("<|eos|>", self.eos_id), ("<|unk|>", self.unk_id),
            ) if tid is None
        ]
        if missing:
            raise ValueError(
                f"Tokenizer at {tokenizer_path} is missing required special "
                f"tokens {missing}. It was not trained with cola-coder's "
                f"SPECIAL_TOKENS (see tokenizer/train_tokenizer.py). Retrain it "
                f"or point at the correct tokenizer.json."
            )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The code/text to tokenize.
            add_bos: Whether to prepend the beginning-of-sequence token.
            add_eos: Whether to append the end-of-sequence token.

        Returns:
            List of integer token IDs.
        """
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special: Whether to remove special tokens from output.

        Returns:
            The decoded text string.
        """
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            ids = [i for i in ids if i not in special_ids]
        return self.tokenizer.decode(ids)

    def encode_batch(
        self, texts: list[str], add_bos: bool = True, add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode multiple texts at once (faster than encoding one by one).

        The HuggingFace tokenizer's encode_batch uses Rust-level parallelism
        internally, so this is significantly faster than calling encode() in a loop.

        Args:
            texts: List of strings to encode.
            add_bos: Whether to prepend BOS to each.
            add_eos: Whether to append EOS to each.

        Returns:
            List of token ID lists.
        """
        encodings = self.tokenizer.encode_batch(texts)
        results = [enc.ids for enc in encodings]
        if add_bos and add_eos:
            results = [[self.bos_id] + ids + [self.eos_id] for ids in results]
        elif add_bos:
            results = [[self.bos_id] + ids for ids in results]
        elif add_eos:
            results = [ids + [self.eos_id] for ids in results]
        return results

    def has_fim_tokens(self) -> bool:
        """Return True only when all three FIM marker tokens are in the vocab.

        FIM tokens are optional (unlike pad/bos/eos/unk, which the constructor
        requires). Callers that need FIM — ``encode_fim``/``fim_prompt`` and the
        dynamic-FIM auto-disable path in the trainer — gate on this so a
        tokenizer trained without ``<|fim_*|>`` degrades gracefully instead of
        emitting ``None`` token IDs.
        """
        return (
            self.fim_prefix_id is not None
            and self.fim_middle_id is not None
            and self.fim_suffix_id is not None
        )

    def _require_fim_tokens(self) -> None:
        """Raise a clear error if any FIM marker token is missing.

        Without this guard a missing FIM token (id ``None``) flows straight into
        the returned id list (``[self.fim_prefix_id] + ...``), silently
        corrupting the sequence with a ``None`` that later crashes the model or
        produces garbage — exactly the failure mode the constructor's required-
        token check prevents for the core tokens.
        """
        if not self.has_fim_tokens():
            missing = [
                name for name, tid in (
                    ("<|fim_prefix|>", self.fim_prefix_id),
                    ("<|fim_middle|>", self.fim_middle_id),
                    ("<|fim_suffix|>", self.fim_suffix_id),
                ) if tid is None
            ]
            raise ValueError(
                f"Tokenizer is missing FIM special tokens {missing}, so "
                f"Fill-in-the-Middle encoding is unavailable. Retrain the "
                f"tokenizer with cola-coder's SPECIAL_TOKENS, or gate FIM on "
                f"has_fim_tokens() before calling this method."
            )

    def encode_fim(self, prefix: str, suffix: str) -> list[int]:
        """Encode for Fill-in-the-Middle (FIM) format.

        FIM lets the model generate code that goes IN BETWEEN existing code,
        instead of just continuing from the end. This is how code completion
        works in IDEs — you have code before and after the cursor.

        Format: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>
        The model then generates what goes in the middle.

        Args:
            prefix: Code before the gap.
            suffix: Code after the gap.

        Returns:
            Token IDs in FIM format.

        Raises:
            ValueError: If the tokenizer lacks the ``<|fim_*|>`` marker tokens.
        """
        self._require_fim_tokens()
        prefix_ids = self.tokenizer.encode(prefix).ids
        suffix_ids = self.tokenizer.encode(suffix).ids
        return (
            [self.fim_prefix_id] + prefix_ids
            + [self.fim_suffix_id] + suffix_ids
            + [self.fim_middle_id]
        )

    def fim_prompt(self, prefix: str, suffix: str) -> str:
        """Build a FIM prompt STRING with the literal marker tokens intact.

        Use this (NOT ``decode(encode_fim(...))``) whenever a STRING is needed
        to feed back into ``generate()`` / ``encode()``. ``decode`` skips
        special tokens, so ``decode(encode_fim(p, s))`` returns just ``p + s``
        with the ``<|fim_prefix|>``/``<|fim_suffix|>``/``<|fim_middle|>`` markers
        STRIPPED — the model then sees no fill-in-the-middle structure at all.
        Keeping the marker strings means ``encode()`` re-recognises them as the
        FIM special tokens, so the model receives the intended FIM layout.

        Raises:
            ValueError: If the tokenizer lacks the ``<|fim_*|>`` marker tokens.
        """
        self._require_fim_tokens()
        pfx = self.tokenizer.id_to_token(self.fim_prefix_id)
        sfx = self.tokenizer.id_to_token(self.fim_suffix_id)
        mid = self.tokenizer.id_to_token(self.fim_middle_id)
        return f"{pfx}{prefix}{sfx}{suffix}{mid}"

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add new special tokens to the vocabulary.

        Used when adding <think> and </think> tokens for reasoning experiments.

        Args:
            tokens: List of special token strings to add.

        Returns:
            New vocabulary size.
        """
        from tokenizers import AddedToken
        for token in tokens:
            self.tokenizer.add_special_tokens([AddedToken(token, special=True)])
        return self.vocab_size

    def encode_chatml(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Encode ChatML messages into token IDs.

        Formats the messages list into a ChatML string (see
        ``special_tokens.format_chatml``) and tokenizes the result.  A BOS
        token is prepended and no EOS token is appended so the model can
        continue generating.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            add_generation_prompt: If True (default), a trailing
                ``<|im_start|>assistant\\n`` is appended so the model
                generates the assistant reply.  Pass False when the last
                message is already an assistant turn (e.g. for training
                labels that include the full assistant response).

        Returns:
            List of token IDs suitable for passing directly to the model.
        """
        from cola_coder.tokenizer.special_tokens import format_chatml

        formatted = format_chatml(messages)
        if not add_generation_prompt and formatted.endswith("<|im_start|>assistant\n"):
            # Remove the trailing generation prompt
            formatted = formatted.rsplit("<|im_start|>assistant\n", 1)[0]
        return self.encode(formatted, add_bos=True, add_eos=False)
