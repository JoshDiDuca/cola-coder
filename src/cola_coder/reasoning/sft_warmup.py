"""SFT (Supervised Fine-Tuning) warmup before GRPO training.

Implements the cold-start SFT phase from DeepSeek-R1's training pipeline.
Fine-tunes the model on a small set of curated chain-of-thought examples so it
learns the <think>...</think> format and basic reasoning patterns before GRPO
reinforcement learning begins.

Without SFT warmup, the model generates incoherent thinking traces early in
GRPO training, wasting many RL steps on format learning instead of reasoning.
Even 5-50 examples with 3-10 epochs gives GRPO a much better starting point.

For a TS dev: think of this as running a "type check + linting pass" before
deploying — it doesn't change the logic, it just ensures the model speaks the
right format before the real training starts.

Feature toggle pattern (project-wide convention):
    Set FEATURE_ENABLED = False to skip SFT warmup entirely.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from ..cli import cli
from .cot_data import COT_EXAMPLES, CoTExample, get_cot_training_data, generate_cot_from_solutions
from .thinking_tokens import format_thinking_example


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if SFT warmup is active.

    When False, calling SFTWarmup.train() is a no-op and returns 0.0.
    """
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# SFTWarmup
# ---------------------------------------------------------------------------


class SFTWarmup:
    """Supervised fine-tuning warmup phase before GRPO.

    Trains the model on curated <think>...</think> examples so it learns:
    1. The thinking token format (<think> / </think> delimiters)
    2. Basic step-by-step reasoning patterns that precede code
    3. That reasoning should always come BEFORE the actual solution

    This is a lightweight training loop — 3-10 epochs on 5-50 examples.
    The goal is NOT to reach peak performance, but to give GRPO a non-degenerate
    starting point (avoid the "all rewards = 0" cold-start problem).

    Args:
        model: The transformer model to fine-tune (must already have thinking tokens).
        tokenizer: Trained tokenizer (with <think>/<think> tokens added).
        device: "cuda" or "cpu".
        learning_rate: AdamW learning rate. Keep very small (1e-5 to 5e-5).
        max_seq_len: Maximum sequence length to truncate examples to.
        precision: "bf16" (RTX 4080) or "fp16" (RTX 3080) or "fp32" (CPU).
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        tokenizer: "object",
        device: str = "cuda",
        learning_rate: float = 5e-5,
        max_seq_len: int = 1024,
        precision: str = "bf16",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len

        # Mixed precision — mirrors the pattern in trainer.py
        if device == "cpu":
            self.use_bf16 = False
            self.use_fp16 = False
        else:
            self.use_bf16 = precision == "bf16"
            self.use_fp16 = precision == "fp16"

        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        # AdamW optimizer with low LR (fine-tuning, not pre-training)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(
        self,
        examples: list[dict] | None = None,
        num_epochs: int = 5,
        batch_size: int = 1,
    ) -> float:
        """Run SFT on chain-of-thought examples.

        If no examples are provided, uses the 5 seed examples from cot_data.py.
        Training is skipped entirely when is_enabled() returns False.

        Args:
            examples: List of dicts with a 'text' key (full training string).
                      If None, loads the built-in CoT seed examples.
            num_epochs: Number of passes over the data (default 5).
                        Keep low (3-10) to avoid overfitting the tiny dataset.
            batch_size: Examples per gradient step (default 1).
                        Batch > 1 requires padding — kept at 1 for simplicity.

        Returns:
            Final average loss over the last epoch. Returns 0.0 when disabled.
        """
        if not is_enabled():
            cli.dim("SFT warmup is disabled (FEATURE_ENABLED=False). Skipping.")
            return 0.0

        if examples is None:
            examples = get_cot_training_data()

        if not examples:
            cli.warn("No SFT examples provided — skipping warmup.")
            return 0.0

        cli.info("SFT warmup", f"{len(examples)} examples, {num_epochs} epochs")

        # Build a cosine LR scheduler over all steps
        total_steps = max(1, len(examples) * num_epochs)
        scheduler = _cosine_scheduler(self.optimizer, total_steps)

        self.model.train()
        final_loss = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for example in examples:
                loss = self._train_step(example["text"])
                epoch_loss += loss
                scheduler.step()

            avg_loss = epoch_loss / max(1, len(examples))
            final_loss = avg_loss
            cli.dim(f"  SFT epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

        cli.info("SFT warmup complete", f"final loss={final_loss:.4f}")
        return final_loss

    def generate_synthetic_examples(
        self,
        problems: list[dict] | None = None,
        num_per_problem: int = 3,
    ) -> list[dict]:
        """Generate synthetic CoT examples via self-play data augmentation.

        Takes a list of coding problems and produces additional chain-of-thought
        training examples by generating reasoning traces from code solutions.
        This is a lightweight version of DeepSeek-R1's rejection-sampling SFT.

        The method uses generate_cot_from_solutions() from cot_data.py to build
        reasoning traces — it does NOT run the model for generation (that would
        require a full inference loop). Instead, it uses heuristic trace generation
        from the solutions embedded in the problem dict.

        Args:
            problems: List of dicts, each with:
                - 'task_id': problem identifier (str)
                - 'prompt': function signature / docstring (str)
                - 'solution': correct code solution (str)
              If None, derives synthetic examples from the built-in COT_EXAMPLES.
            num_per_problem: How many variations to create per problem (default 3).
                             Currently generates 1 per problem; the parameter is
                             reserved for future multi-temperature sampling.

        Returns:
            List of dicts with 'text' key (same format as get_cot_training_data()).
        """
        if problems is None:
            # Fall back to augmenting the built-in seed examples
            problems = [
                {
                    "task_id": ex.task_id,
                    "prompt": ex.prompt,
                    "solution": ex.solution,
                }
                for ex in COT_EXAMPLES
            ]

        if not problems:
            return []

        # Extract parallel lists for generate_cot_from_solutions
        solutions = [p.get("solution", "") for p in problems]
        valid_pairs = [
            (p, s) for p, s in zip(problems, solutions) if s.strip()
        ]
        if not valid_pairs:
            return []

        filtered_problems, filtered_solutions = zip(*valid_pairs)

        synthetic_examples: list[dict] = []
        for _ in range(num_per_problem):
            cot_examples: list[CoTExample] = generate_cot_from_solutions(
                list(filtered_problems), list(filtered_solutions)
            )
            for ex in cot_examples:
                text = ex.prompt + format_thinking_example(
                    thinking=ex.thinking,
                    code=ex.solution,
                )
                synthetic_examples.append({"text": text})

        return synthetic_examples

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _train_step(self, text: str) -> float:
        """One gradient step on a single text example.

        Args:
            text: Full training string (prompt + <think>reasoning</think> + code).

        Returns:
            Scalar loss for this step.
        """
        # Tokenize and truncate
        token_ids = self.tokenizer.encode(text, add_bos=True)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        # Need at least 2 tokens for a meaningful (input, target) pair
        if len(token_ids) < 2:
            return 0.0

        input_ids = torch.tensor([token_ids[:-1]], device=self.device)
        target_ids = torch.tensor([token_ids[1:]], device=self.device)

        self.optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        amp_enabled = self.use_bf16 or self.use_fp16

        with autocast(device_type="cuda" if self.device != "cpu" else "cpu",
                      dtype=amp_dtype, enabled=amp_enabled):
            logits = self.model(input_ids)
            # logits: [1, seq_len, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

        if self.use_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing LR scheduler with no warmup (tiny SFT datasets).

    Args:
        optimizer: The AdamW optimizer to wrap.
        total_steps: Total number of gradient steps (epochs * examples).
        min_lr_ratio: Minimum LR as a fraction of the base LR (default 0.1).

    Returns:
        A LambdaLR scheduler that decays from 1.0 to min_lr_ratio.
    """
    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        progress = step / total_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
