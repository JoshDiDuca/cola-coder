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
        """
        prefix_ids = self.tokenizer.encode(prefix).ids
        suffix_ids = self.tokenizer.encode(suffix).ids
        return (
            [self.fim_prefix_id] + prefix_ids
            + [self.fim_suffix_id] + suffix_ids
            + [self.fim_middle_id]
        )

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
