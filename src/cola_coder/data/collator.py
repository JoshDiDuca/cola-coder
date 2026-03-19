"""Data collation utilities.

The main collator is in dataset.py (CodeCollator). This module provides
additional collation functions for specialized training modes like FIM.
"""

import random

import torch


class FIMCollator:
    """Collator that randomly converts some examples to Fill-in-the-Middle format.

    FIM (Fill-in-the-Middle) teaches the model to complete code given both
    a prefix AND suffix — not just generate from left to right. This is
    essential for IDE-style code completion where you have code above and
    below the cursor.

    During training, a percentage of examples are randomly converted to
    FIM format. The rest stay as normal left-to-right sequences.
    """

    def __init__(
        self,
        fim_rate: float = 0.5,
        fim_prefix_id: int = 4,
        fim_middle_id: int = 5,
        fim_suffix_id: int = 6,
    ):
        """
        Args:
            fim_rate: Fraction of examples to convert to FIM format (0.0 to 1.0).
            fim_prefix_id: Token ID for <|fim_prefix|>.
            fim_middle_id: Token ID for <|fim_middle|>.
            fim_suffix_id: Token ID for <|fim_suffix|>.
        """
        self.fim_rate = fim_rate
        self.fim_prefix_id = fim_prefix_id
        self.fim_middle_id = fim_middle_id
        self.fim_suffix_id = fim_suffix_id

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate examples, randomly applying FIM to some.

        For each example, with probability fim_rate:
        1. Pick a random split point in the token sequence
        2. Rearrange as: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
        3. The model learns to generate 'middle' given prefix and suffix

        Args:
            examples: List of {"input_ids": tensor} dictionaries.

        Returns:
            Batched dictionary with FIM-transformed sequences.
        """
        batch = []
        for ex in examples:
            tokens = ex["input_ids"]

            if random.random() < self.fim_rate:
                tokens = self._apply_fim(tokens)

            batch.append(tokens)

        return {"input_ids": torch.stack(batch)}

    def _apply_fim(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transform a sequence into FIM format.

        Original:  [a, b, c, d, e, f, g, h]
        Split at position 3:
          prefix = [a, b, c]
          middle = [d, e]
          suffix = [f, g, h]
        FIM: [<prefix>, a, b, c, <suffix>, f, g, h, <middle>, d, e]

        The model then learns to predict 'd, e' given the prefix and suffix context.
        """
        seq_len = len(tokens)

        # Pick a random split region
        # We split into three parts: prefix | middle | suffix
        split_start = random.randint(1, seq_len - 2)
        split_end = random.randint(split_start + 1, min(split_start + seq_len // 4, seq_len - 1))

        prefix = tokens[:split_start]
        middle = tokens[split_start:split_end]
        suffix = tokens[split_end:]

        # Construct FIM sequence
        fim_tokens = torch.cat([
            torch.tensor([self.fim_prefix_id]),
            prefix,
            torch.tensor([self.fim_suffix_id]),
            suffix,
            torch.tensor([self.fim_middle_id]),
            middle,
        ])

        # Truncate or pad to original sequence length
        if len(fim_tokens) > seq_len:
            fim_tokens = fim_tokens[:seq_len]
        elif len(fim_tokens) < seq_len:
            # Pad with the pad token (ID 0)
            pad = torch.zeros(seq_len - len(fim_tokens), dtype=fim_tokens.dtype)
            fim_tokens = torch.cat([fim_tokens, pad])

        return fim_tokens
