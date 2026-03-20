"""
Byte-level tokenization fallback for unknown tokens and rare characters.

When a BPE tokenizer cannot handle a token, this module provides a fallback
to byte-level encoding, ensuring every possible input can be represented.
"""

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


class ByteLevelEncoder:
    """
    Encodes text to byte-level token IDs with an optional offset.

    Each byte value (0-255) is mapped to a token ID by adding byte_token_offset,
    so byte tokens occupy the range [offset, offset+255] inclusive.
    """

    def __init__(self, offset: int = 0):
        self._byte_token_offset = offset

    @property
    def byte_token_offset(self) -> int:
        return self._byte_token_offset

    def encode(self, text: str) -> list[int]:
        """Encode text to byte-level token IDs (byte value + offset)."""
        return [b + self._byte_token_offset for b in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        """Decode byte-level token IDs back to text."""
        raw = bytes(tid - self._byte_token_offset for tid in token_ids)
        return raw.decode("utf-8")

    def is_byte_token(self, token_id: int) -> bool:
        """Return True if token_id falls within the byte-token range."""
        return self._byte_token_offset <= token_id <= self._byte_token_offset + 255


class FallbackTokenizer:
    """
    Tokenizer that uses a primary vocabulary for known tokens and falls back
    to byte-level encoding for unknown tokens.

    Primary vocab lookup is word-level (whitespace-split). Unknown words are
    encoded byte-by-byte with the configured offset so they remain
    distinguishable from primary vocab IDs.
    """

    def __init__(
        self,
        primary_vocab: dict[str, int] | None = None,
        byte_offset: int = 50000,
    ):
        self._vocab: dict[str, int] = primary_vocab if primary_vocab is not None else {}
        self._byte_encoder = ByteLevelEncoder(offset=byte_offset)
        # Build reverse map for decoding primary tokens
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Splits on whitespace and looks up each word in the primary vocab.
        Words not found are encoded byte-by-byte via ByteLevelEncoder.
        Spaces between words are encoded as a byte token for round-trip fidelity.
        """
        token_ids: list[int] = []
        words = text.split(" ")
        for i, word in enumerate(words):
            if i > 0:
                # Encode the space separator as a byte token
                token_ids.extend(self._byte_encoder.encode(" "))
            if word in self._vocab:
                token_ids.append(self._vocab[word])
            else:
                token_ids.extend(self._byte_encoder.encode(word))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Primary vocab IDs are converted directly; byte tokens are accumulated
        and decoded together to correctly handle multi-byte UTF-8 sequences.
        """
        result_parts: list[str] = []
        byte_buffer: list[int] = []

        def flush_bytes() -> None:
            if byte_buffer:
                result_parts.append(
                    bytes(
                        tid - self._byte_encoder.byte_token_offset
                        for tid in byte_buffer
                    ).decode("utf-8", errors="replace")
                )
                byte_buffer.clear()

        for tid in token_ids:
            if tid in self._id_to_token:
                flush_bytes()
                result_parts.append(self._id_to_token[tid])
            elif self._byte_encoder.is_byte_token(tid):
                byte_buffer.append(tid)
            else:
                # Unknown ID — emit replacement character
                flush_bytes()
                result_parts.append("\ufffd")

        flush_bytes()
        return "".join(result_parts)

    def vocab_size(self) -> int:
        """Total vocabulary size: primary tokens + 256 byte tokens."""
        return len(self._vocab) + 256

    def stats(self, text: str) -> dict:
        """
        Return a dict describing how primary vs byte tokens are used for text.

        Keys:
            primary_tokens  - count of tokens resolved via primary vocab
            byte_tokens     - count of tokens resolved via byte encoding
            total_tokens    - total token count
            primary_words   - words found in primary vocab
            unknown_words   - words not found in primary vocab
        """
        primary_count = 0
        byte_count = 0
        primary_words: list[str] = []
        unknown_words: list[str] = []

        words = text.split(" ")
        for word in words:
            if not word:
                # Empty string from multiple spaces — counts as byte tokens
                byte_count += len(self._byte_encoder.encode(" "))
                continue
            if word in self._vocab:
                primary_count += 1
                primary_words.append(word)
            else:
                byte_count += len(self._byte_encoder.encode(word))
                unknown_words.append(word)

        # Account for space separators between words
        if len(words) > 1:
            byte_count += len(words) - 1  # one space byte token per gap

        return {
            "primary_tokens": primary_count,
            "byte_tokens": byte_count,
            "total_tokens": primary_count + byte_count,
            "primary_words": primary_words,
            "unknown_words": unknown_words,
        }
