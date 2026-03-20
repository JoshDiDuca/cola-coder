"""Beam Search for Code Generation with Syntax-Awareness.

Standard greedy decoding always picks the single most likely next token, which
can lead to locally-optimal but globally-suboptimal sequences. Beam search
maintains the top-K candidate sequences simultaneously, exploring a broader
portion of the output space before committing.

Think of it like a BFS/DFS hybrid in TS terms — instead of following one path
greedily (DFS), we keep the K best paths alive at each step (bounded BFS).

Syntax-awareness adds a bonus to beams whose partial token sequences form
valid (or at least parseable) Python/code, steering the search toward
syntactically well-formed outputs.
"""

import ast
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Configuration & data structures
# ---------------------------------------------------------------------------


@dataclass
class BeamConfig:
    """Configuration for beam search decoding."""

    beam_width: int = 5
    """Number of beams (candidate sequences) to maintain in parallel."""

    max_length: int = 256
    """Maximum number of tokens to generate (beyond the prompt)."""

    length_penalty: float = 0.6
    """Exponent applied to sequence length when normalising scores.
    Values < 1 favour shorter sequences; values > 1 favour longer ones.
    Google's original formula: score / (length ** alpha).
    """

    syntax_bonus: float = 0.5
    """Log-space bonus added to a beam's score when its decoded tokens form
    syntactically valid Python. Acts as a soft reward signal."""


@dataclass
class Beam:
    """A single candidate sequence being tracked during beam search."""

    token_ids: list  # list[int]
    """Token IDs generated so far (prompt tokens not included)."""

    score: float
    """Cumulative log-probability score (higher is better)."""

    finished: bool
    """True once an EOS token has been emitted."""


# ---------------------------------------------------------------------------
# Core BeamSearcher
# ---------------------------------------------------------------------------


class BeamSearcher:
    """Runs beam search over an autoregressive model.

    The model is expected to accept a 1-D or 2-D integer tensor of token IDs
    and return logits of shape (..., vocab_size) for the last position.
    """

    def __init__(self, config: BeamConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        model: torch.nn.Module,
        prompt_ids: list,  # list[int]
        vocab_size: int,
        eos_token_id: int = 0,
        decode_fn: Optional[Callable] = None,
    ) -> list:  # list[Beam]
        """Run beam search and return the final list of Beam objects.

        Args:
            model: Autoregressive model. Called as model(input_ids) where
                   input_ids is a LongTensor of shape (batch, seq_len).
                   Must return a tensor whose last dimension is vocab_size.
            prompt_ids: Prompt token IDs (integers).
            vocab_size: Size of the vocabulary.
            eos_token_id: Token ID that signals end-of-sequence.
            decode_fn: Optional callable that maps list[int] -> str for
                       syntax checking. Falls back to chr()-based heuristic
                       when None.

        Returns:
            List of Beam objects sorted by normalised score (best first).
        """
        cfg = self.config
        device = next(model.parameters()).device

        # Initialise with a single beam containing the empty generation.
        beams: list = [Beam(token_ids=[], score=0.0, finished=False)]

        for _step in range(cfg.max_length):
            # Split beams into active / finished.
            active = [b for b in beams if not b.finished]
            finished = [b for b in beams if b.finished]

            if not active:
                break

            # Build a batched input tensor: (num_active, prompt_len + gen_len)
            all_candidates: list = []

            for beam in active:
                full_ids = prompt_ids + beam.token_ids
                input_tensor = torch.tensor(
                    full_ids, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, seq_len)

                with torch.no_grad():
                    output = model(input_tensor)

                # Support models returning tensors directly or tuples/objects.
                if isinstance(output, torch.Tensor):
                    logits = output
                elif hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output[0]

                # Take the logits at the last position: (vocab_size,)
                last_logits = logits[0, -1, :]  # (vocab_size,)

                # Top-K expansion: pick the best beam_width next tokens.
                log_probs = F.log_softmax(last_logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, cfg.beam_width)

                for log_prob, token_id in zip(
                    top_log_probs.tolist(), top_indices.tolist()
                ):
                    new_token_ids = beam.token_ids + [token_id]
                    new_score = self.score_beam(
                        Beam(new_token_ids, beam.score + log_prob, False),
                        last_logits,
                    )
                    syntax_bonus = self.check_syntax(new_token_ids, decode_fn)
                    new_score += syntax_bonus

                    is_finished = token_id == eos_token_id
                    all_candidates.append(
                        Beam(
                            token_ids=new_token_ids,
                            score=new_score,
                            finished=is_finished,
                        )
                    )

            # Prune to beam_width best candidates + carry forward finished beams.
            pruned = self.get_top_k(all_candidates, cfg.beam_width)
            beams = pruned + finished

        # Final normalisation by length to remove length bias.
        beams = self._normalise_scores(beams)
        beams.sort(key=lambda b: b.score, reverse=True)
        return beams

    def score_beam(self, beam: "Beam", logits: torch.Tensor) -> float:
        """Compute a (potentially updated) score for a beam.

        Currently returns the beam's existing cumulative log-prob score,
        which already incorporates the latest token's log-probability from
        the caller. Subclasses can override this to add custom penalties
        (repetition, length, etc.).

        Args:
            beam: Beam whose score we are computing.
            logits: Raw logits (vocab_size,) at the current step (unused in
                    the base implementation but available for extensions).

        Returns:
            Float score (higher = better).
        """
        return beam.score

    def check_syntax(
        self,
        tokens: list,  # list[int]
        decode_fn: Optional[Callable] = None,
    ) -> float:
        """Return a syntax validity bonus (0.0 or config.syntax_bonus).

        Attempts to decode the token IDs to a string and parse it as Python
        using ast.parse. If parsing succeeds, the syntax bonus is returned.

        Args:
            tokens: List of integer token IDs.
            decode_fn: Optional callable mapping list[int] -> str.
                       When None, a simple chr()-based fallback is used that
                       is unlikely to produce valid Python but will not crash.

        Returns:
            self.config.syntax_bonus if the decoded text is valid Python,
            otherwise 0.0.
        """
        if not tokens:
            return 0.0

        try:
            if decode_fn is not None:
                text = decode_fn(tokens)
            else:
                # Fallback: interpret token IDs as Unicode code points where
                # possible. This is a rough heuristic — in practice a real
                # tokenizer's decode_fn should be supplied.
                text = "".join(
                    chr(t) if 0 < t < 0x110000 else "?" for t in tokens
                )

            ast.parse(text)
            return float(self.config.syntax_bonus)
        except Exception:
            return 0.0

    def get_top_k(self, beams: list, k: int) -> list:  # list[Beam]
        """Return the top-k beams sorted by score descending.

        Args:
            beams: Candidate beams to rank.
            k: Number of beams to keep.

        Returns:
            Up to k Beam objects, best score first.
        """
        sorted_beams = sorted(beams, key=lambda b: b.score, reverse=True)
        return sorted_beams[:k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_scores(self, beams: list) -> list:
        """Apply length penalty normalisation to all beams in-place."""
        alpha = self.config.length_penalty
        normalised = []
        for beam in beams:
            length = max(len(beam.token_ids), 1)
            norm_score = beam.score / (length ** alpha)
            normalised.append(
                Beam(
                    token_ids=beam.token_ids,
                    score=norm_score,
                    finished=beam.finished,
                )
            )
        return normalised


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def beam_search(
    model: torch.nn.Module,
    prompt_ids: list,  # list[int]
    config: Optional[BeamConfig] = None,
    vocab_size: int = 32000,
    eos_token_id: int = 0,
    decode_fn: Optional[Callable] = None,
) -> list:  # list[list[int]]
    """Run beam search and return the generated token ID sequences.

    This is a thin wrapper around BeamSearcher.search that returns plain
    lists of token IDs rather than Beam objects.

    Args:
        model: Autoregressive PyTorch model.
        prompt_ids: Prompt token IDs.
        config: BeamConfig (uses defaults if None).
        vocab_size: Vocabulary size passed to the searcher.
        eos_token_id: End-of-sequence token ID.
        decode_fn: Optional decode function for syntax checking.

    Returns:
        List of token ID sequences (best first), one per beam.
    """
    if config is None:
        config = BeamConfig()

    searcher = BeamSearcher(config)
    beams = searcher.search(
        model,
        prompt_ids,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        decode_fn=decode_fn,
    )
    return [beam.token_ids for beam in beams]
