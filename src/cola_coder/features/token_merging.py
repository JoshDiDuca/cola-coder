"""
Token merging: merge similar tokens to reduce sequence length and speed up inference.
Based on cosine similarity between token representations (inspired by ToMe - Token Merging).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class MergeConfig:
    merge_ratio: float = 0.25
    similarity_threshold: float = 0.9
    min_tokens: int = 4


def cosine_similarity_matrix(a: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity for all tokens in a sequence.

    Args:
        a: Tensor of shape (seq_len, dim)

    Returns:
        Tensor of shape (seq_len, seq_len) with pairwise cosine similarities.
    """
    # Normalize each token vector to unit length
    normed = F.normalize(a, p=2, dim=-1)  # (seq_len, dim)
    # Dot product of unit vectors = cosine similarity
    return torch.mm(normed, normed.t())  # (seq_len, seq_len)


class TokenMerger:
    """Merges similar tokens in a sequence to reduce length and speed up inference."""

    def find_merge_candidates(
        self, hidden_states: torch.Tensor, threshold: float
    ) -> list[tuple[int, int]]:
        """Find pairs of tokens whose cosine similarity exceeds the threshold.

        Only returns the upper-triangle pairs (i < j) to avoid duplicates.
        The lower-index token in each pair is the one that will be kept (merged into).

        Args:
            hidden_states: Tensor of shape (seq_len, dim)
            threshold: Minimum cosine similarity to be considered a merge candidate.

        Returns:
            List of (i, j) index pairs where token j will be merged into token i.
        """
        seq_len = hidden_states.shape[0]
        sim = cosine_similarity_matrix(hidden_states)  # (seq_len, seq_len)

        # Only look at upper triangle (i < j) to avoid self-pairs and duplicates
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        above_threshold = (sim >= threshold) & mask

        pairs = above_threshold.nonzero(as_tuple=False).tolist()
        return [(int(i), int(j)) for i, j in pairs]

    def merge_tokens(
        self, hidden_states: torch.Tensor, pairs: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Average merged token pairs and remove the redundant tokens.

        For each pair (i, j), token j is averaged into token i, then token j is dropped.
        If a token index appears in multiple pairs, each merge is applied independently
        (the base token accumulates all merges).

        Args:
            hidden_states: Tensor of shape (seq_len, dim)
            pairs: List of (i, j) pairs — token j merges into token i.

        Returns:
            Tensor of shape (new_seq_len, dim) with merged tokens removed.
        """
        if not pairs:
            return hidden_states

        seq_len = hidden_states.shape[0]
        # Work on a mutable copy
        merged = hidden_states.clone()

        # Track which token indices are being absorbed (will be removed)
        absorbed: set[int] = set()

        # Track how many tokens have been merged into each "base" token
        merge_counts = torch.ones(seq_len, device=hidden_states.device)

        for i, j in pairs:
            if j in absorbed or i in absorbed:
                # Skip if either token has already been consumed
                continue
            # Weighted average: merge j into i
            total = merge_counts[i] + merge_counts[j]
            merged[i] = (merged[i] * merge_counts[i] + merged[j] * merge_counts[j]) / total
            merge_counts[i] = total
            absorbed.add(j)

        # Keep only non-absorbed tokens, preserving order
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
        for idx in absorbed:
            keep_mask[idx] = False

        return merged[keep_mask]

    def merge(
        self,
        hidden_states: torch.Tensor,
        config: Optional[MergeConfig] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Full token-merging pipeline.

        Handles batched input (batch_size, seq_len, dim) or unbatched (seq_len, dim).

        Args:
            hidden_states: Tensor of shape (batch, seq_len, dim) or (seq_len, dim).
            config: MergeConfig controlling merge behavior. Defaults to MergeConfig().

        Returns:
            Tuple of:
              - merged hidden states with same batch dims but potentially shorter seq_len
              - info dict with keys: original_len, merged_len, num_pairs, speedup
        """
        if config is None:
            config = MergeConfig()

        batched = hidden_states.dim() == 3
        if batched:
            # Process each batch element independently; pad to common length at the end
            batch_size = hidden_states.shape[0]
            results = []
            total_pairs = 0
            for b in range(batch_size):
                out, info = self.merge(hidden_states[b], config)
                results.append(out)
                total_pairs += info["num_pairs"]

            # Pad to longest sequence in batch
            max_len = max(r.shape[0] for r in results)
            dim = hidden_states.shape[2]
            padded = torch.zeros(batch_size, max_len, dim, dtype=hidden_states.dtype, device=hidden_states.device)
            for b, r in enumerate(results):
                padded[b, : r.shape[0]] = r

            original_len = hidden_states.shape[1]
            merged_len = max_len
            return padded, {
                "original_len": original_len,
                "merged_len": merged_len,
                "num_pairs": total_pairs // batch_size,
                "speedup": self.estimate_speedup(original_len, merged_len),
            }

        # --- Unbatched path (seq_len, dim) ---
        seq_len = hidden_states.shape[0]

        # Enforce minimum token count
        if seq_len <= config.min_tokens:
            return hidden_states, {
                "original_len": seq_len,
                "merged_len": seq_len,
                "num_pairs": 0,
                "speedup": 1.0,
            }

        candidates = self.find_merge_candidates(hidden_states, config.similarity_threshold)

        # Limit how many tokens we may remove based on merge_ratio
        max_removals = max(0, int(seq_len * config.merge_ratio))
        # Ensure we don't go below min_tokens
        max_removals = min(max_removals, seq_len - config.min_tokens)

        # Greedily pick pairs that don't conflict (each token used at most once as "j")
        used_j: set[int] = set()
        selected_pairs: list[tuple[int, int]] = []
        for i, j in candidates:
            if len(selected_pairs) >= max_removals:
                break
            if j not in used_j:
                selected_pairs.append((i, j))
                used_j.add(j)

        merged = self.merge_tokens(hidden_states, selected_pairs)
        merged_len = merged.shape[0]

        return merged, {
            "original_len": seq_len,
            "merged_len": merged_len,
            "num_pairs": len(selected_pairs),
            "speedup": self.estimate_speedup(seq_len, merged_len),
        }

    def estimate_speedup(self, original_len: int, merged_len: int) -> float:
        """Estimate attention speedup from sequence length reduction.

        Attention is O(n^2), so speedup ≈ (original / merged)^2.

        Args:
            original_len: Original sequence length.
            merged_len: Merged sequence length.

        Returns:
            Estimated speedup factor (>= 1.0).
        """
        if merged_len <= 0:
            return 1.0
        return (original_len / merged_len) ** 2
