"""Group Relative Policy Optimization (GRPO) for reasoning improvement.

GRPO is a simplified version of PPO (Proximal Policy Optimization) that
doesn't need a separate critic/value model. Instead, it uses the GROUP
of generated solutions as its own baseline.

How GRPO works (simplified):
1. For each coding problem, generate G solutions (e.g., G=8)
2. Score each solution with the reward function (did the code pass tests?)
3. Compute "advantages" relative to the group mean:
   advantage[i] = reward[i] - mean(rewards)
   This means: "was this solution better or worse than average?"
4. Update the model to make high-advantage solutions more likely
   and low-advantage solutions less likely
5. Use a clipped objective (from PPO) to prevent too-large updates

The key simplification vs full PPO:
- No critic network (saves memory and complexity)
- Advantages come from group comparison, not learned value estimates
- No KL penalty against a reference model (in our simplified version)

Why this works for code:
- Code has a binary, verifiable reward (tests pass or fail)
- We don't need a learned reward model
- The group baseline provides a natural comparison

For a TS dev: think of GRPO like A/B testing with multiple variants.
Generate several solutions, see which ones work, and adjust the model
to produce more solutions like the working ones.
"""

import logging
from typing import TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from ..model.transformer import Transformer
from ..tokenizer.tokenizer_utils import CodeTokenizer
from ..inference.generator import CodeGenerator
from .reward import compute_batch_rewards_parallel
from .reward_registry import RewardFunction, RewardRegistry

if TYPE_CHECKING:
    from ..evaluation.problem_loader import ProblemSet

logger = logging.getLogger(__name__)

# Difficulty ordering for curriculum temperature scaling
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
# Temperature multipliers per difficulty tier for curriculum learning
_CURRICULUM_TEMP = {"easy": 0.7, "medium": 0.8, "hard": 0.9}


class GRPOTrainer:
    """Simplified GRPO trainer for reasoning experiments."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: CodeTokenizer,
        learning_rate: float = 1e-5,
        group_size: int = 8,
        clip_epsilon: float = 0.2,
        max_new_tokens: int = 512,
        max_thinking_tokens: int = 256,
        device: str = "cuda",
        reward_fn: Union[str, RewardFunction, None] = None,
        parallel_generation: bool = False,
        parallel_rewards: bool = False,
        reward_workers: int = 4,
    ):
        """
        Args:
            model: The transformer model to train.
            tokenizer: Trained tokenizer.
            learning_rate: Learning rate for GRPO updates (should be very small).
            group_size: Number of solutions to generate per problem (G).
            clip_epsilon: PPO-style clipping parameter.
            max_new_tokens: Maximum tokens to generate per solution.
            max_thinking_tokens: Maximum thinking trace length for reward.
            device: "cuda" or "cpu".
            reward_fn: Reward function to use.  Can be:
                - None (default): use the built-in Python execution reward.
                - A string name registered in RewardRegistry
                  (e.g. "python_exec", "typescript", "combined").
                - A callable conforming to the RewardFunction protocol.
            parallel_generation: When True, use generate_group() to generate all
                group completions in a single batched forward pass instead of a
                serial loop.  Falls back to serial automatically on OOM.
            parallel_rewards: When True, compute rewards in parallel using
                ProcessPoolExecutor (requires reward function to be picklable).
                Falls back to serial on any error.
            reward_workers: Number of worker processes for parallel reward
                computation.  Ignored when parallel_rewards=False.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.max_new_tokens = max_new_tokens
        self.max_thinking_tokens = max_thinking_tokens

        # Resolve the reward function
        if reward_fn is None:
            # Default: existing Python execution reward (backward compatible)
            self._reward_fn: RewardFunction = RewardRegistry.get("python_exec")
            self._reward_name = "python_exec"
        elif isinstance(reward_fn, str):
            self._reward_fn = RewardRegistry.get(reward_fn)
            self._reward_name = reward_fn
            logger.info("GRPOTrainer: using reward function '%s'", reward_fn)
        else:
            # Assume it is already a callable RewardFunction
            self._reward_fn = reward_fn
            self._reward_name = getattr(reward_fn, "__name__", "custom")
            logger.info("GRPOTrainer: using custom reward function '%s'", self._reward_name)

        self.parallel_generation = parallel_generation
        self.parallel_rewards = parallel_rewards
        self.reward_workers = reward_workers

        # Generator for producing solutions
        self.generator = CodeGenerator(model, tokenizer, device)

        # Optimizer (separate from base training — much smaller LR)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    def train_step(
        self,
        prompt: str,
        test_code: str,
        temperature: float = 0.8,
    ) -> dict:
        """One GRPO training step on a single problem.

        Args:
            prompt: The coding problem (function signature + docstring).
            test_code: Test cases for verifying solutions.
            temperature: Sampling temperature for generation.

        Returns:
            Dictionary with step metrics (loss, rewards, etc.).
        """
        self.model.eval()

        # Step 1: Generate G solutions
        # ------------------------------------------------------------------
        # Batched path: one prefill + G parallel decode passes (faster).
        # Serial path: G independent generate() calls (lower VRAM, fallback).
        # ------------------------------------------------------------------
        if self.parallel_generation:
            try:
                generations = self.generator.generate_group(
                    prompt=prompt,
                    num_completions=self.group_size,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "train_step: generate_group failed, falling back to serial generation"
                )
                self.parallel_generation = False  # Disable for future steps
                generations = [
                    self.generator.generate(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                    )
                    for _ in range(self.group_size)
                ]
        else:
            generations = [
                self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                )
                for _ in range(self.group_size)
            ]

        # Compute log probabilities of the generated tokens (old policy, pi_old)
        log_probs_list = []
        for output in generations:
            token_ids = self.tokenizer.encode(output, add_bos=True)
            input_tensor = torch.tensor([token_ids], device=self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                log_probs = F.log_softmax(logits, dim=-1)
                # Get the log prob of each actual generated token
                token_log_probs = log_probs[0, :-1].gather(
                    1, input_tensor[0, 1:].unsqueeze(1)
                ).squeeze(1)
                total_log_prob = token_log_probs.sum().item()
                log_probs_list.append(total_log_prob)

        # Step 2: Compute rewards
        # ------------------------------------------------------------------
        # Parallel path: rewards computed concurrently via ProcessPoolExecutor.
        # Serial path: rewards computed in-process one by one.
        # The parallel path only applies to the built-in serial reward functions
        # (compute_batch_rewards).  Custom reward_fn callables always run serially
        # here — callers that need parallel custom rewards should handle it themselves.
        # ------------------------------------------------------------------
        if self.parallel_rewards and self._reward_name in ("python_exec",):
            rewards, infos = compute_batch_rewards_parallel(
                generations,
                test_code,
                max_thinking_tokens=self.max_thinking_tokens,
                workers=self.reward_workers,
            )
        else:
            rewards, infos = self._reward_fn(
                generations,
                test_code,
                max_thinking_tokens=self.max_thinking_tokens,
            )

        # Step 3: Compute advantages (relative to group mean)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8  # Prevent division by zero
        advantages = (rewards_tensor - mean_reward) / std_reward

        # Step 4: Policy gradient update
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        for i in range(self.group_size):
            token_ids = self.tokenizer.encode(generations[i], add_bos=True)
            input_tensor = torch.tensor([token_ids], device=self.device)

            # Forward pass to get current policy log probs
            with autocast(device_type="cuda", dtype=torch.bfloat16,
                           enabled=self.device == "cuda"):
                logits = self.model(input_tensor)
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[0, :-1].gather(
                    1, input_tensor[0, 1:].unsqueeze(1)
                ).squeeze(1)
                current_log_prob = token_log_probs.sum()

            # Compute probability ratio (pi_new / pi_old)
            old_log_prob = log_probs_list[i]
            ratio = torch.exp(current_log_prob - old_log_prob)

            # Clipped surrogate objective (PPO-style)
            advantage = advantages[i]
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            loss = -torch.min(unclipped, clipped)

            # Accumulate loss
            total_loss += loss / self.group_size

        # Backward and update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Return metrics
        num_correct = sum(1 for info in infos if info["correct"])
        return {
            "loss": total_loss.item(),
            "mean_reward": mean_reward.item(),
            "num_correct": num_correct,
            "group_size": self.group_size,
            "pass_rate": num_correct / self.group_size,
        }

    def _problems_to_dicts(
        self,
        problems: "list[dict] | ProblemSet",
    ) -> list[dict]:
        """Normalize problems to a list of {'prompt': str, 'test_code': str, ...} dicts.

        Accepts either the legacy list[dict] format or a ProblemSet instance.
        This keeps backward compatibility: existing callers passing list[dict]
        continue to work without any changes.
        """
        # Check for ProblemSet by duck-typing (avoids circular import)
        if hasattr(problems, "to_training_dicts"):
            base = problems.to_training_dicts()  # type: ignore[union-attr]
            # Enrich with difficulty from the CodingProblem objects
            enriched = []
            for p, d in zip(problems, base):  # type: ignore[call-overload]
                row = dict(d)
                if hasattr(p, "difficulty"):
                    row.setdefault("difficulty", p.difficulty)
                enriched.append(row)
            return enriched

        # Already a list of dicts
        return list(problems)  # type: ignore[arg-type]

    def _apply_curriculum(self, problems: list[dict]) -> list[dict]:
        """Sort problem dicts easy → medium → hard.

        Problems that don't have a 'difficulty' key are treated as 'medium'.
        """
        return sorted(
            problems,
            key=lambda p: _DIFFICULTY_ORDER.get(p.get("difficulty", "medium"), 1),
        )

    def train(
        self,
        problems: "list[dict] | ProblemSet",
        num_epochs: int = 3,
        temperature: float = 0.8,
        curriculum: bool = False,
        problem_set: "ProblemSet | None" = None,
    ) -> None:
        """Train on a set of coding problems using GRPO.

        Args:
            problems: Either a list of dicts with 'prompt' and 'test_code' keys
                      (legacy format, fully backward-compatible) OR a ProblemSet
                      instance (new format with difficulty metadata).
            num_epochs: Number of passes over all problems.
            temperature: Base sampling temperature for generation.
            curriculum: If True, sort problems easy → medium → hard before
                        training and apply per-difficulty temperature scaling.
            problem_set: Deprecated alias for ``problems``; takes lower priority.
                         Provided for call-site backward compatibility.
        """
        # Resolve the problem source (problem_set is a legacy alias)
        if problem_set is not None and problems is None:
            problems = problem_set

        training_problems = self._problems_to_dicts(problems)

        if curriculum:
            training_problems = self._apply_curriculum(training_problems)

        print("\nStarting GRPO training:")
        print(f"  Problems: {len(training_problems)}")
        print(f"  Group size: {self.group_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Curriculum: {curriculum}")
        print()

        for epoch in range(num_epochs):
            epoch_metrics = {
                "loss": 0.0,
                "mean_reward": 0.0,
                "total_correct": 0,
                "total_generated": 0,
            }
            # Per-difficulty counters for curriculum reporting
            diff_correct: dict[str, int] = {}
            diff_total: dict[str, int] = {}

            for problem in tqdm(training_problems, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Curriculum learning: vary temperature by difficulty
                difficulty = problem.get("difficulty", "medium")
                if curriculum:
                    step_temp = _CURRICULUM_TEMP.get(difficulty, temperature)
                else:
                    step_temp = temperature

                metrics = self.train_step(
                    prompt=problem["prompt"],
                    test_code=problem["test_code"],
                    temperature=step_temp,
                )

                epoch_metrics["loss"] += metrics["loss"]
                epoch_metrics["mean_reward"] += metrics["mean_reward"]
                epoch_metrics["total_correct"] += metrics["num_correct"]
                epoch_metrics["total_generated"] += metrics["group_size"]

                diff_correct[difficulty] = (
                    diff_correct.get(difficulty, 0) + metrics["num_correct"]
                )
                diff_total[difficulty] = (
                    diff_total.get(difficulty, 0) + metrics["group_size"]
                )

            n = len(training_problems)
            overall_pass = (
                epoch_metrics["total_correct"] / epoch_metrics["total_generated"]
                if epoch_metrics["total_generated"] > 0
                else 0.0
            )
            print(
                f"Epoch {epoch + 1}: "
                f"loss={epoch_metrics['loss']/n:.4f}, "
                f"mean_reward={epoch_metrics['mean_reward']/n:.3f}, "
                f"pass_rate={overall_pass:.1%}"
            )
            if curriculum and len(diff_total) > 1:
                for diff in ("easy", "medium", "hard"):
                    if diff in diff_total and diff_total[diff] > 0:
                        dr = diff_correct[diff] / diff_total[diff]
                        print(f"  {diff}: pass_rate={dr:.1%}")
