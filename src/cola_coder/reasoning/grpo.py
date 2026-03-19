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

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..model.transformer import Transformer
from ..tokenizer.tokenizer_utils import CodeTokenizer
from ..inference.generator import CodeGenerator
from .reward import compute_batch_rewards


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
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.max_new_tokens = max_new_tokens
        self.max_thinking_tokens = max_thinking_tokens

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
        generations = []
        log_probs_list = []

        for _ in range(self.group_size):
            # Generate a solution
            output = self.generator.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
            )
            generations.append(output)

            # Compute log probabilities of the generated tokens
            # This is the "old policy" probability (pi_old)
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
        rewards, infos = compute_batch_rewards(
            generations, test_code,
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

    def train(
        self,
        problems: list[dict],
        num_epochs: int = 3,
        temperature: float = 0.8,
    ):
        """Train on a set of coding problems using GRPO.

        Args:
            problems: List of dicts with 'prompt' and 'test_code' keys.
            num_epochs: Number of passes over all problems.
            temperature: Sampling temperature for generation.
        """
        print(f"\nStarting GRPO training:")
        print(f"  Problems: {len(problems)}")
        print(f"  Group size: {self.group_size}")
        print(f"  Epochs: {num_epochs}")
        print()

        for epoch in range(num_epochs):
            epoch_metrics = {
                "loss": 0.0,
                "mean_reward": 0.0,
                "total_correct": 0,
                "total_generated": 0,
            }

            for problem in tqdm(problems, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                metrics = self.train_step(
                    prompt=problem["prompt"],
                    test_code=problem["test_code"],
                    temperature=temperature,
                )

                epoch_metrics["loss"] += metrics["loss"]
                epoch_metrics["mean_reward"] += metrics["mean_reward"]
                epoch_metrics["total_correct"] += metrics["num_correct"]
                epoch_metrics["total_generated"] += metrics["group_size"]

            n = len(problems)
            print(
                f"Epoch {epoch + 1}: "
                f"loss={epoch_metrics['loss']/n:.4f}, "
                f"mean_reward={epoch_metrics['mean_reward']/n:.3f}, "
                f"pass_rate={epoch_metrics['total_correct']/epoch_metrics['total_generated']:.1%}"
            )
