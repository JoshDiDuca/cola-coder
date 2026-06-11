"""Data collation utilities.

The main collator is in dataset.py (CodeCollator). This module provides
additional collation functions for specialized training modes like FIM.

NOTE: ``FIMCollator`` is an optional collate_fn for applying Fill-in-the-Middle
*dynamically at training time* (different random splits each epoch). It is NOT
wired into ``create_dataloader`` by default — the default trainer path applies
FIM at data-prep time via ``scripts/prepare_fim_data.py`` (``FIMTransform``).
To use dynamic FIM you must pass a ``FIMCollator`` as the DataLoader's
``collate_fn`` explicitly (see DATA-012 for the optional trainer wiring).
"""

from types import SimpleNamespace

import torch

from .fim import FIMTransform


class FIMCollator:
    """Collator that randomly converts some examples to Fill-in-the-Middle format.

    FIM (Fill-in-the-Middle) teaches the model to complete code given both
    a prefix AND suffix — not just generate from left to right. This is
    essential for IDE-style code completion where you have code above and
    below the cursor.

    During training, a percentage of examples are randomly converted to
    FIM format. The rest stay as normal left-to-right sequences.

    The actual PSM/SPM rearrangement is delegated to the canonical, tested
    ``FIMTransform`` (data/fim.py) so the two implementations can never diverge
    (DRY). Crucially, ``FIMTransform`` reserves 3 content slots up front, so the
    output length exactly equals the input and the prediction target (``middle``,
    the LAST segment) is never truncated. The previous hand-rolled version built
    ``seq_len + 3`` tokens then truncated to ``seq_len``, silently chopping the
    end of ``middle`` (and, for short middles, the ``<fim_middle>`` marker
    itself) — corrupting the very target FIM is meant to teach.
    """

    def __init__(
        self,
        fim_rate: float = 0.5,
        fim_prefix_id: int = 4,
        fim_middle_id: int = 5,
        fim_suffix_id: int = 6,
        psm_rate: float = 0.5,
        seed: int | None = None,
    ):
        """
        Args:
            fim_rate: Fraction of examples to convert to FIM format (0.0 to 1.0).
            fim_prefix_id: Token ID for <|fim_prefix|>.
            fim_middle_id: Token ID for <|fim_middle|>.
            fim_suffix_id: Token ID for <|fim_suffix|>.
            psm_rate: Probability of PSM vs SPM ordering (1.0 = always PSM).
            seed: Optional RNG seed for reproducible transforms (useful in tests).
                  Leave None for fresh per-epoch randomness during training.
        """
        self.fim_rate = fim_rate
        self.fim_prefix_id = fim_prefix_id
        self.fim_middle_id = fim_middle_id
        self.fim_suffix_id = fim_suffix_id
        self.psm_rate = psm_rate

        # The transform gates on fim_rate internally and preserves length.
        self._transform = FIMTransform(fim_rate=fim_rate, psm_rate=psm_rate, seed=seed)
        # FIMTransform reads fim_*_id off a tokenizer-like object; a tiny
        # namespace satisfies that interface without a real tokenizer.
        self._fim_ids = SimpleNamespace(
            fim_prefix_id=fim_prefix_id,
            fim_suffix_id=fim_suffix_id,
            fim_middle_id=fim_middle_id,
        )

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate examples, randomly applying FIM to some.

        For each example, with probability fim_rate:
        1. Pick a random split point in the token sequence
        2. Rearrange as: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
           (or the SPM ordering, per psm_rate)
        3. The model learns to generate 'middle' given prefix and suffix

        Args:
            examples: List of {"input_ids": tensor} dictionaries.

        Returns:
            Batched dictionary with FIM-transformed sequences.
        """
        batch = [self._apply_fim(ex["input_ids"]) for ex in examples]
        return {"input_ids": torch.stack(batch)}

    def _apply_fim(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transform a sequence into FIM format via the canonical FIMTransform.

        The transform internally rolls fim_rate (skipping → returns the sequence
        unchanged) and keeps the output length exactly equal to the input, so
        the batch can always be ``torch.stack``-ed and the FIM target is never
        truncated. dtype/device are preserved.
        """
        ids = self._transform.apply(tokens.tolist(), self._fim_ids)
        return torch.tensor(ids, dtype=tokens.dtype, device=tokens.device)
