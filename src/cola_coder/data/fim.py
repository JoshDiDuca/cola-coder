"""Fill-in-the-Middle (FIM) training support.

FIM lets the model complete code given both prefix AND suffix context.
This is critical for IDE autocomplete where the cursor is in the middle
of existing code.

Paper: "Efficient Training of Language Models to Fill in the Middle"
       Bavarian et al., 2022 (https://arxiv.org/abs/2207.14255)

Two orderings:
  PSM (Prefix-Suffix-Middle):
      <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle

  SPM (Suffix-Prefix-Middle):
      <fim_suffix> suffix <fim_prefix> prefix <fim_middle> middle

PSM is the primary format (psm_rate controls the mix).
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tokenizer.tokenizer_utils import CodeTokenizer

# Special token strings for FIM
# Note: the tokenizer already has <|fim_prefix|> etc. in its vocabulary.
# These plain-string constants are used only when working with raw text.
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"


class FIMTransform:
    """Transform training token sequences (or raw text) into FIM format.

    During training, each sample is independently decided:
      - With probability (1 - fim_rate): left as-is (standard causal LM)
      - With probability fim_rate: rearranged into PSM or SPM format

    The sequence is split into prefix | middle | suffix, then rearranged so the
    middle (the part the model learns to predict) comes LAST, after both
    surrounding contexts:

      PSM:  [fim_prefix_id] prefix [fim_suffix_id] suffix [fim_middle_id] middle
      SPM:  [fim_suffix_id] suffix [fim_prefix_id] prefix [fim_middle_id] middle

    (StarCoder / OpenAI FIM convention; psm_rate controls the PSM↔SPM mix.)
    """

    # Middle must be between 10% and 90% of the full sequence length,
    # so splits near the very beginning or very end are skipped.
    MIN_MIDDLE_FRAC = 0.10
    MAX_MIDDLE_FRAC = 0.90

    def __init__(
        self,
        fim_rate: float = 0.5,
        psm_rate: float = 0.5,
        truncate_or_pad: bool = True,
        seed: int | None = None,
    ):
        """
        Args:
            fim_rate: Probability of applying FIM to a sample.
                      0.0 = never apply FIM, 1.0 = always apply FIM.
            psm_rate: Probability of PSM vs SPM ordering.
                      1.0 = always PSM, 0.0 = always SPM.
            truncate_or_pad: When True (default), the output is the SAME length
                             as the input: the 3 FIM special tokens take the place
                             of the last 3 content tokens (content is sliced to
                             len-3 before the markers are inserted). When False,
                             the 3 markers are ADDED with no content removed, so
                             the output is LONGER than the input by exactly 3
                             tokens. Use True for fixed-window training tensors.
            seed: Optional RNG seed for reproducible transforms (useful in tests).
        """
        if not 0.0 <= fim_rate <= 1.0:
            raise ValueError(f"fim_rate must be in [0, 1], got {fim_rate}")
        if not 0.0 <= psm_rate <= 1.0:
            raise ValueError(f"psm_rate must be in [0, 1], got {psm_rate}")

        self.fim_rate = fim_rate
        self.psm_rate = psm_rate
        self.truncate_or_pad = truncate_or_pad
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Token-level API (used during training with already-tokenized data)
    # ------------------------------------------------------------------

    def apply(self, token_ids: list[int], tokenizer: "CodeTokenizer") -> list[int]:
        """Apply FIM transformation to an already-tokenized sequence.

        With probability fim_rate, splits the sequence at a random point,
        then rearranges it into PSM or SPM format using the tokenizer's
        fim_prefix_id / fim_suffix_id / fim_middle_id.

        The three special token IDs are inserted so the total length stays
        equal to the original when truncate_or_pad=True: we use the first
        (len - 3) tokens of content and add 3 special-token slots back.

        Args:
            token_ids: List of integer token IDs (already-tokenized training sample).
            tokenizer: CodeTokenizer instance; provides fim_*_id attributes.

        Returns:
            List of token IDs, same length as input when truncate_or_pad=True.
        """
        if self._rng.random() >= self.fim_rate:
            return token_ids  # Not selected for FIM this sample

        n = len(token_ids)
        # Need at least 4 tokens (1 for each piece + 1 to spare)
        if n < 4:
            return token_ids

        # We'll insert 3 special tokens.  Reserve 3 content positions so the
        # output fits in the same window when truncate_or_pad=True.
        content = token_ids[: n - 3] if self.truncate_or_pad else token_ids

        min_idx = max(1, int(len(content) * self.MIN_MIDDLE_FRAC))
        max_idx = min(len(content) - 1, int(len(content) * self.MAX_MIDDLE_FRAC))

        if min_idx >= max_idx:
            return token_ids  # Sequence too short for a meaningful split

        # Pick two split points: prefix | middle | suffix
        split1 = self._rng.randint(min_idx, max_idx - 1)
        split2 = self._rng.randint(split1 + 1, max_idx)

        prefix = content[:split1]
        middle = content[split1:split2]
        suffix = content[split2:]

        fim_prefix_id = tokenizer.fim_prefix_id
        fim_suffix_id = tokenizer.fim_suffix_id
        fim_middle_id = tokenizer.fim_middle_id

        if self._rng.random() < self.psm_rate:
            # PSM: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle
            result = (
                [fim_prefix_id] + prefix
                + [fim_suffix_id] + suffix
                + [fim_middle_id] + middle
            )
        else:
            # SPM: <fim_suffix> suffix <fim_prefix> prefix <fim_middle> middle
            result = (
                [fim_suffix_id] + suffix
                + [fim_prefix_id] + prefix
                + [fim_middle_id] + middle
            )

        # Truncate to the original length (adds exactly 0 extra tokens vs
        # original because we removed 3 from content already)
        if self.truncate_or_pad:
            result = result[:n]

        return result

    # ------------------------------------------------------------------
    # Text-level API (used by prepare_fim_data.py on raw strings)
    # ------------------------------------------------------------------

    def apply_to_text(self, text: str) -> tuple[str, bool]:
        """Apply FIM transformation to raw text.

        Splits at a random *line* boundary when possible (to avoid breaking
        identifiers mid-way).  Falls back to a character boundary if there
        are not enough lines.

        Args:
            text: Raw source code string.

        Returns:
            (transformed_text, was_transformed) tuple.
            was_transformed is False when fim_rate skips the sample or the
            text is too short to split meaningfully.
        """
        if self._rng.random() >= self.fim_rate:
            return text, False

        # Prefer splitting at line boundaries
        lines = text.splitlines(keepends=True)

        if len(lines) >= 3:
            # Need at least 3 lines for a 3-part split
            min_line = max(1, int(len(lines) * self.MIN_MIDDLE_FRAC))
            max_line = min(len(lines) - 1, int(len(lines) * self.MAX_MIDDLE_FRAC))

            if min_line < max_line:
                split1 = self._rng.randint(min_line, max_line - 1)
                split2 = self._rng.randint(split1 + 1, max_line)

                prefix = "".join(lines[:split1])
                middle = "".join(lines[split1:split2])
                suffix = "".join(lines[split2:])
            else:
                return text, False
        else:
            # Fall back to character boundaries for short texts
            n = len(text)
            if n < 4:
                return text, False
            min_c = max(1, int(n * self.MIN_MIDDLE_FRAC))
            max_c = min(n - 1, int(n * self.MAX_MIDDLE_FRAC))
            if min_c >= max_c:
                return text, False

            split1 = self._rng.randint(min_c, max_c - 1)
            split2 = self._rng.randint(split1 + 1, max_c)

            prefix = text[:split1]
            middle = text[split1:split2]
            suffix = text[split2:]

        if self._rng.random() < self.psm_rate:
            result = FIM_PREFIX + prefix + FIM_SUFFIX + suffix + FIM_MIDDLE + middle
        else:
            result = FIM_SUFFIX + suffix + FIM_PREFIX + prefix + FIM_MIDDLE + middle

        return result, True

    # ------------------------------------------------------------------
    # Tokenizer setup helper
    # ------------------------------------------------------------------

    @staticmethod
    def add_special_tokens(tokenizer: "CodeTokenizer") -> dict[str, int]:
        """Add FIM special tokens to a CodeTokenizer if not already present.

        The tokenizer already ships with <|fim_prefix|> etc.  This method is
        a safety net: it checks whether the IDs are valid and, if not, adds
        the tokens so they never collide with existing vocabulary.

        Args:
            tokenizer: A CodeTokenizer instance to (possibly) mutate.

        Returns:
            Mapping of human-readable name to token ID:
            {"fim_prefix": id, "fim_suffix": id, "fim_middle": id}
        """
        return setup_fim_tokenizer(tokenizer)


# ------------------------------------------------------------------
# Module-level helper (can be imported directly)
# ------------------------------------------------------------------

def setup_fim_tokenizer(tokenizer: "CodeTokenizer") -> dict[str, int]:
    """Add FIM special tokens to an existing CodeTokenizer.

    The cola-coder tokenizer already reserves <|fim_prefix|>, <|fim_middle|>,
    and <|fim_suffix|> as special tokens.  This function verifies they are
    present and caches their IDs on the tokenizer object.  If for some reason
    they are missing (e.g., an older tokenizer.json), it adds them.

    Args:
        tokenizer: A CodeTokenizer instance.

    Returns:
        dict with keys "fim_prefix", "fim_suffix", "fim_middle" mapping to int IDs.
    """
    # The canonical token strings used inside the vocabulary
    _FIM_TOKENS = {
        "fim_prefix": "<|fim_prefix|>",
        "fim_suffix": "<|fim_suffix|>",
        "fim_middle": "<|fim_middle|>",
    }

    ids: dict[str, int] = {}

    for name, token_str in _FIM_TOKENS.items():
        token_id = tokenizer.tokenizer.token_to_id(token_str)
        if token_id is None:
            # Token not present — add it
            tokenizer.add_special_tokens([token_str])
            token_id = tokenizer.tokenizer.token_to_id(token_str)

        if token_id is None:
            raise RuntimeError(
                f"Failed to add FIM special token {token_str!r} to tokenizer"
            )

        # Cache on the tokenizer object using the attribute names it already
        # expects (fim_prefix_id, fim_suffix_id, fim_middle_id)
        attr = f"{name}_id"  # e.g. "fim_prefix_id"
        setattr(tokenizer, attr, token_id)
        ids[name] = token_id

    return ids
