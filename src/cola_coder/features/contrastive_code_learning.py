"""
Contrastive learning for code embeddings.

Learn embeddings where similar code (same semantics, different surface form)
is close together, and different code is far apart.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveConfig:
    temperature: float = 0.07
    embedding_dim: int = 128
    margin: float = 1.0


# ---------------------------------------------------------------------------
# Code augmentation helpers
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r'\b([a-z_][a-z0-9_]*)\b')

_RENAME_MAP: dict[str, str] = {
    'a': 'x', 'b': 'y', 'c': 'z',
    'x': 'u', 'y': 'v', 'z': 'w',
    'i': 'idx', 'j': 'jdx', 'k': 'kdx',
    'n': 'num', 'm': 'cnt', 'result': 'out',
    'val': 'value', 'tmp': 'temp', 'ret': 'retval',
}

_PYTHON_KEYWORDS = frozenset({
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield', 'self', 'cls', 'print', 'len', 'range',
    'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'bool',
    'type', 'object', 'super', 'property', 'staticmethod', 'classmethod',
})


def _rename_vars(code: str) -> str:
    """Rename a subset of local variable names using a fixed mapping."""
    tokens = re.split(r'(\W+)', code)
    out = []
    for tok in tokens:
        if tok in _RENAME_MAP and tok not in _PYTHON_KEYWORDS:
            out.append(_RENAME_MAP[tok])
        else:
            out.append(tok)
    return ''.join(out)


def _reformat(code: str) -> str:
    """Apply minor whitespace / style changes."""
    # Collapse multiple spaces to one (except indentation)
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        # Normalise spaces around operators
        stripped = re.sub(r' {2,}', ' ', stripped)
        stripped = re.sub(r'\s*([+\-*/=,:])\s*', r'\1', stripped)
        stripped = re.sub(r'\s*([+\-*/=,:]) ', r'\1 ', stripped)
        lines.append(indent + stripped)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CodePairGenerator
# ---------------------------------------------------------------------------

class CodePairGenerator:
    """Generate positive and negative code pairs for contrastive learning."""

    def create_positive_pair(self, code: str) -> tuple[str, str]:
        """Return (original, augmented) — semantically equivalent code."""
        augmented = _rename_vars(code)
        if augmented == code:
            augmented = _reformat(code)
        # If still identical, add a harmless trailing newline variant
        if augmented == code:
            augmented = code.rstrip('\n') + '\n'
        return code, augmented

    def create_negative_pair(self, codes: list[str]) -> tuple[str, str]:
        """Return two distinct code snippets from the list."""
        if len(codes) < 2:
            raise ValueError("Need at least 2 code snippets to form a negative pair.")
        idx_a, idx_b = random.sample(range(len(codes)), 2)
        return codes[idx_a], codes[idx_b]

    def generate_pairs(
        self,
        codes: list[str],
        n_pairs: int,
    ) -> list[tuple[str, str, bool]]:
        """
        Generate a list of (code_a, code_b, is_positive) tuples.

        Positive pairs: augmented versions of the same snippet.
        Negative pairs: two different snippets.
        """
        if not codes:
            return []

        pairs: list[tuple[str, str, bool]] = []
        half = max(1, n_pairs // 2)

        # Positive pairs
        for _ in range(half):
            src = random.choice(codes)
            a, b = self.create_positive_pair(src)
            pairs.append((a, b, True))

        # Negative pairs (only possible if we have >= 2 snippets)
        neg_count = n_pairs - half
        if len(codes) >= 2:
            for _ in range(neg_count):
                a, b = self.create_negative_pair(codes)
                pairs.append((a, b, False))
        else:
            # Fall back to more positive pairs when only one snippet exists
            for _ in range(neg_count):
                src = codes[0]
                a, b = self.create_positive_pair(src)
                pairs.append((a, b, True))

        return pairs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def contrastive_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss.

    Args:
        embeddings_a: (N, D) float tensor
        embeddings_b: (N, D) float tensor
        labels:       (N,)   float tensor — 1 for positive pair, 0 for negative
        temperature:  softmax temperature

    Returns:
        Scalar loss tensor.
    """
    # L2-normalise
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)

    # Cosine similarity matrix: (N, N)
    sim = torch.mm(a, b.t()) / temperature  # (N, N)

    n = sim.size(0)
    # For each anchor in `a`, its positive is the matching row in `b`
    # We use cross-entropy with the diagonal as target class for positives,
    # then mask out negative pairs from being used as anchors.
    targets = torch.arange(n, device=sim.device)
    loss_a = F.cross_entropy(sim, targets, reduction='none')       # (N,)
    loss_b = F.cross_entropy(sim.t(), targets, reduction='none')   # (N,)
    loss = (loss_a + loss_b) / 2.0

    # Weight by label: positives contribute fully, negatives contribute 0
    # (standard InfoNCE only trains on positive pairs as anchors)
    labels = labels.to(sim.device)
    weighted = (loss * labels).sum() / (labels.sum() + 1e-8)
    return weighted


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Triplet margin loss.

    Loss = mean(max(0, d(anchor, positive) - d(anchor, negative) + margin))

    Args:
        anchor:   (N, D)
        positive: (N, D)
        negative: (N, D)
        margin:   float

    Returns:
        Scalar loss tensor.
    """
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)
