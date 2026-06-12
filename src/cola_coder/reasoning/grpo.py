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


def compute_group_advantages(
    rewards: torch.Tensor,
    norm: str = "std",
) -> torch.Tensor:
    """Group-relative advantages for GRPO.

    Args:
        rewards: Shape (group_size,) — one reward per generated solution.
        norm: "std" — original GRPO, (r - mean) / (std + eps).
              "mean" — Dr. GRPO (Liu et al. 2025, "GRPO Done Right"):
              r - mean only. Dividing by the group std inflates the update
              for groups where nearly everything passed or nearly everything
              failed (tiny std), biasing training toward the easiest and
              hardest problems instead of the informative middle.

    Returns:
        Advantage tensor, same shape as rewards.
    """
    centered = rewards - rewards.mean()
    if norm == "mean":
        return centered
    return centered / (rewards.std() + 1e-8)


def _completion_logprobs(
    token_log_probs: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Per-token log-probs of the COMPLETION tokens only (mask the prompt).

    `token_log_probs[j]` scores token j+1 given tokens [:j+1]. The prompt is
    fixed context, not a sampled action, so under the policy only the completion
    tokens (indices >= prompt_len) count — they are scored by token_log_probs
    indices >= prompt_len - 1. Masking the prompt (MODEL-004) matches reference
    GRPO; it was harmless before only because advantages are mean-centered and
    the shared prompt log-probs cancel in (current - old).

    Returns a 1-D tensor of the completion token log-probs (empty when there are
    no completion tokens).
    """
    start = max(prompt_len - 1, 0)
    if start >= token_log_probs.shape[0]:
        return token_log_probs.new_zeros((0,))
    return token_log_probs[start:]


def _completion_logprob_sum(
    token_log_probs: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum of the completion-token log-probs (see _completion_logprobs).

    Returns a 0-D tensor (0.0 when there are no completion tokens).
    """
    return _completion_logprobs(token_log_probs, prompt_len).sum()


def grpo_clipped_surrogate(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantage: torch.Tensor | float,
    clip_low: float,
    clip_high: float,
) -> torch.Tensor:
    """Per-token PPO-clipped surrogate, SUMMED over completion tokens.

    For each completion token: ratio r_t = exp(logπ_new,t - logπ_old,t), and
    surrogate_t = min(r_t·A, clip(r_t, 1-clip_low, 1+clip_high)·A). Returns
    Σ_t surrogate_t (a MAXIMIZATION objective — the caller negates it for the
    loss).

    Why per-token (not sequence-level): the old code used ONE ratio for the whole
    sequence, exp(Σ_t Δlogp) = the PRODUCT of per-token ratios. Over a long
    completion that product explodes/vanishes and saturates the clip on nearly
    every sample, destroying PPO's per-token credit assignment. Reference
    GRPO/Dr.GRPO/DAPO all clip PER TOKEN.

    Why SUM (not mean): at ppo_epochs=1 the new policy == the old policy, so every
    r_t ≡ 1 and the per-token gradient Σ_t A·∇logπ_new,t exactly equals the old
    sequence-level gradient A·∇(Σ_t logπ_new,t) — i.e. the default behavior is
    unchanged. The clip only diverges from a no-op once the weights move
    (ppo_epochs > 1), which is precisely when DAPO clip-higher should engage.

    Empty completion → 0.
    """
    if new_logp.numel() == 0:
        return new_logp.new_zeros(())
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantage
    return torch.min(unclipped, clipped).sum()


class GRPOTrainer:
    """Simplified GRPO trainer for reasoning experiments."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: CodeTokenizer,
        learning_rate: float = 1e-5,
        group_size: int = 8,
        clip_epsilon: float = 0.2,
        clip_epsilon_high: float | None = None,
        advantage_norm: str = "std",
        ppo_epochs: int = 1,
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
            clip_epsilon: PPO-style clipping parameter (lower bound).
            clip_epsilon_high: Optional separate UPPER clip bound (DAPO
                "clip-higher", e.g. 0.28 with clip_epsilon=0.2). A looser
                upper bound lets low-probability tokens grow, which fights
                the entropy collapse symmetric clipping causes. None =
                symmetric (original GRPO behavior).
            advantage_norm: "std" (original GRPO: divide by group std) or
                "mean" (Dr. GRPO, Liu et al. 2025: subtract mean only —
                dividing by std over-weights near-zero-variance groups,
                i.e. problems that are nearly always right or always wrong).
            ppo_epochs: Number of gradient steps per generated group (PPO inner
                epochs, μ). 1 (default) = single on-policy step, where the
                importance ratio is exactly 1 and clipping is a no-op. >1 reuses
                the (expensive) generations for several updates against the fixed
                old policy, which is the ONLY regime where clip_epsilon /
                clip_epsilon_high actually engage.
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
        self.clip_epsilon_high = clip_epsilon_high
        self.advantage_norm = advantage_norm
        # PPO inner epochs: gradient steps taken per generated group, reusing the
        # FIXED old log-probs. With 1 (default) the ratio stays 1 and clipping is
        # inert (pure REINFORCE+baseline). >1 reuses the expensive generations and
        # lets the DAPO clip-higher bound actually take effect.
        self.ppo_epochs = max(1, int(ppo_epochs))
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

        # Number of prompt tokens (shared by all generations) — used to mask the
        # prompt out of the policy log-prob so only completion tokens count.
        prompt_len = len(self.tokenizer.encode(prompt, add_bos=True))

        # Per-token completion log-probs under the OLD policy (pi_old), held fixed
        # across all ppo_epochs. Stored as detached 1-D tensors (one per group
        # member) so the PPO ratio can be computed PER TOKEN in the update.
        old_token_logps: list[torch.Tensor] = []
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
                old_token_logps.append(
                    _completion_logprobs(token_log_probs, prompt_len).detach().float()
                )

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
        std_reward = rewards_tensor.std()

        # Collapse guard: when all rewards are identical (std≈0) the advantage
        # signal is degenerate — all advantages collapse to ~0 and the gradient
        # step is pure noise that can corrupt weights without improving the policy.
        # Skip the update; log the skip so it's visible in training output.
        num_correct = sum(1 for info in infos if info["correct"])
        if std_reward < 1e-4:
            print(
                f"  [GRPO] Skipping update — zero reward variance "
                f"(all {self.group_size} rewards = {mean_reward.item():.3f})"
            )
            return {
                "loss": 0.0,
                "mean_reward": mean_reward.item(),
                "num_correct": num_correct,
                "group_size": self.group_size,
                "pass_rate": num_correct / self.group_size,
                "skipped": True,
            }

        advantages = compute_group_advantages(rewards_tensor, norm=self.advantage_norm)

        # Step 4: PPO policy update. Take ppo_epochs gradient steps reusing the
        # FIXED old log-probs. Upper clip bound may be looser than the lower one
        # (DAPO clip-higher) to counteract entropy collapse. At epoch 0 the ratio
        # is exactly 1 (weights == pi_old) so the clip is inert; later epochs move
        # the weights, making the per-token ratio diverge from 1 and the clip act.
        self.model.train()
        eps_high = (
            self.clip_epsilon_high
            if self.clip_epsilon_high is not None
            else self.clip_epsilon
        )

        last_loss = 0.0
        for _epoch in range(self.ppo_epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for i in range(self.group_size):
                token_ids = self.tokenizer.encode(generations[i], add_bos=True)
                input_tensor = torch.tensor([token_ids], device=self.device)

                # Forward pass to get current policy per-token log probs
                with autocast(device_type="cuda", dtype=torch.bfloat16,
                               enabled=self.device == "cuda"):
                    logits = self.model(input_tensor)
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs[0, :-1].gather(
                        1, input_tensor[0, 1:].unsqueeze(1)
                    ).squeeze(1)
                    new_logp = _completion_logprobs(token_log_probs, prompt_len)

                # Per-token clipped surrogate (fp32 for a stable exp).
                surrogate = grpo_clipped_surrogate(
                    new_logp.float(), old_token_logps[i], advantages[i],
                    self.clip_epsilon, eps_high,
                )
                total_loss = total_loss + (-surrogate) / self.group_size

            # Degenerate group: if no group member produced any completion tokens
            # (e.g. the model emitted EOS immediately), total_loss is a grad-less
            # zero — there is nothing to optimize, and calling backward() would
            # raise. Skip the step instead of crashing the run.
            if not (torch.is_tensor(total_loss) and total_loss.requires_grad):
                last_loss = float(total_loss)
                break

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            last_loss = total_loss.item()

        # Return metrics
        num_correct = sum(1 for info in infos if info["correct"])
        return {
            "loss": last_loss,
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
