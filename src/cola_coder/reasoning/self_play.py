"""Self-play / iterative improvement trainer for code generation.

The model generates solutions, tests them, then improves based on
test results. This is a simplified RL loop that doesn't need a
separate critic model.

Research backing:
- AlphaCode (2022): Generate many, filter by tests, cluster, submit best
- CodeRL (Le et al., 2022): RL for code with execution feedback
- StepCoder (ACL 2024): Curriculum of completion subtasks
- SPIN (2024): Self-play without external data, >10% improvement

Flow per problem:
1. Generate N candidate solutions
2. Execute tests on each
3. If any pass: use as positive examples
4. If none pass: use error messages as feedback, regenerate
5. Repeat for max_iterations
6. Update model on (prompt + error context → correct solution) pairs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    num_candidates: int = 8  # Solutions per problem per iteration
    max_iterations: int = 3  # Max improve cycles per problem
    temperature: float = 0.8  # Sampling temperature
    temperature_decay: float = 0.9  # Reduce temp each iteration
    max_new_tokens: int = 512
    learning_rate: float = 1e-5
    include_error_context: bool = True  # Feed errors back as context


@dataclass
class SelfPlayResult:
    """Result from one self-play episode."""

    problem_id: str
    iterations_used: int
    solutions_found: int
    best_reward: float
    total_candidates: int
    improved: bool = False  # Did later iterations find better solutions?


class SelfPlayTrainer:
    """Iterative self-play trainer for code generation.

    Unlike GRPO which updates on a single batch of solutions,
    self-play iterates: generate → test → improve → test again.
    The model gets feedback from failed attempts.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        reward_fn: Callable,
        config: SelfPlayConfig | None = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model: Transformer model
            tokenizer: CodeTokenizer
            reward_fn: Reward function (generations, test_code) -> (rewards, infos)
            config: Self-play configuration
            device: Device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or SelfPlayConfig()
        self.device = device

        # Optimizer for online updates
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def train_episode(
        self,
        prompt: str,
        test_code: str,
        problem_id: str = "",
    ) -> SelfPlayResult:
        """Run one self-play episode on a single problem.

        Args:
            prompt: Problem prompt
            test_code: Test code for validation
            problem_id: Problem identifier

        Returns:
            SelfPlayResult with episode statistics
        """
        best_reward = 0.0
        best_solution = ""
        total_candidates = 0
        solutions_found = 0
        improved = False

        current_prompt = prompt
        temperature = self.config.temperature
        iteration = 0

        for iteration in range(self.config.max_iterations):
            # Generate candidates
            candidates = self._generate_candidates(
                current_prompt,
                num_candidates=self.config.num_candidates,
                temperature=temperature,
            )
            total_candidates += len(candidates)

            # Evaluate
            rewards, infos = self.reward_fn(candidates, test_code)

            # Find best
            for i, reward in enumerate(rewards):
                if reward > best_reward:
                    best_reward = reward
                    best_solution = candidates[i]
                    if iteration > 0:
                        improved = True

                if reward > 0.5:
                    solutions_found += 1

            # If we found a passing solution, update model and stop
            if best_reward > 0.9:
                self._update_on_solution(prompt, best_solution)
                break

            # Build error context for next iteration
            if self.config.include_error_context and infos:
                error_context = self._build_error_context(candidates, rewards, infos)
                current_prompt = (
                    f"{prompt}\n\n# Previous attempts had these issues:\n"
                    f"{error_context}\n\n# Try again, fixing the issues above:"
                )

            # Decay temperature (more focused on later iterations)
            temperature *= self.config.temperature_decay

        # Final update if we found any solution
        if best_solution and best_reward > 0.3:
            self._update_on_solution(prompt, best_solution)

        return SelfPlayResult(
            problem_id=problem_id,
            iterations_used=min(iteration + 1, self.config.max_iterations),
            solutions_found=solutions_found,
            best_reward=best_reward,
            total_candidates=total_candidates,
            improved=improved,
        )

    def train(
        self,
        problems: list[dict],
        num_epochs: int = 1,
    ) -> dict:
        """Train on a list of problems.

        Args:
            problems: List of {"prompt": str, "test_code": str, "task_id": str}
            num_epochs: Number of passes over the problem set

        Returns:
            Training metrics dict
        """
        from cola_coder.cli import cli

        cli.header("Self-Play Training", f"{len(problems)} problems, {num_epochs} epochs")

        total_results: list[SelfPlayResult] = []

        for epoch in range(num_epochs):
            epoch_results: list[SelfPlayResult] = []

            for i, problem in enumerate(problems):
                prompt = problem.get("prompt", "")
                test_code = problem.get("test_code", "")
                task_id = problem.get("task_id", f"problem_{i}")

                result = self.train_episode(prompt, test_code, task_id)
                epoch_results.append(result)

                if (i + 1) % 5 == 0:
                    avg_reward = sum(r.best_reward for r in epoch_results) / len(epoch_results)
                    pass_rate = (
                        sum(1 for r in epoch_results if r.best_reward > 0.9) / len(epoch_results)
                    )
                    cli.step(
                        i + 1,
                        len(problems),
                        f"avg_reward={avg_reward:.3f} pass_rate={pass_rate:.1%}",
                    )

            total_results.extend(epoch_results)

            # Epoch summary
            n = max(len(epoch_results), 1)
            avg_reward = sum(r.best_reward for r in epoch_results) / n
            pass_rate = sum(1 for r in epoch_results if r.best_reward > 0.9) / n
            improved_rate = sum(1 for r in epoch_results if r.improved) / n

            cli.info(
                f"Epoch {epoch + 1}/{num_epochs}",
                f"reward={avg_reward:.3f} pass={pass_rate:.1%} improved={improved_rate:.1%}",
            )

        n_total = max(len(total_results), 1)
        metrics = {
            "total_problems": len(total_results),
            "avg_reward": sum(r.best_reward for r in total_results) / n_total,
            "pass_rate": sum(1 for r in total_results if r.best_reward > 0.9) / n_total,
            "improved_rate": sum(1 for r in total_results if r.improved) / n_total,
            "avg_iterations": sum(r.iterations_used for r in total_results) / n_total,
            "avg_candidates": sum(r.total_candidates for r in total_results) / n_total,
        }

        cli.done(
            "Self-play training complete",
            {
                "Avg reward": f"{metrics['avg_reward']:.3f}",
                "Pass rate": f"{metrics['pass_rate']:.1%}",
                "Improved rate": f"{metrics['improved_rate']:.1%}",
            },
        )

        return metrics

    def _generate_candidates(
        self,
        prompt: str,
        num_candidates: int,
        temperature: float,
    ) -> list[str]:
        """Generate multiple candidate solutions."""
        from cola_coder.inference.generator import CodeGenerator

        self.model.eval()
        generator = CodeGenerator(self.model, self.tokenizer, self.device)

        candidates: list[str] = []
        for _ in range(num_candidates):
            try:
                output = generator.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                )
                candidates.append(output)
            except Exception:
                logger.debug("Candidate generation failed; appending empty string.")
                candidates.append("")

        self.model.train()
        return candidates

    def _build_error_context(
        self,
        candidates: list[str],
        rewards: list[float],
        infos: list[dict],
    ) -> str:
        """Build error context from failed attempts."""
        errors: list[str] = []
        for i, (reward, info) in enumerate(zip(rewards, infos)):
            if reward < 0.5 and info:
                error_msg = info.get("error", info.get("stderr", ""))
                if error_msg:
                    errors.append(f"Attempt {i + 1}: {error_msg[:200]}")

        return "\n".join(errors[:3])  # Top 3 error messages

    def _update_on_solution(self, prompt: str, solution: str) -> None:
        """Update model weights on a successful solution.

        Simple supervised update: minimize loss on (prompt → solution).
        """
        self.model.train()

        # Tokenize prompt + solution
        full_text = f"{prompt}\n{solution}"
        tokens = self.tokenizer.encode(full_text, add_bos=True, add_eos=True)

        if len(tokens) < 2:
            return

        input_ids = torch.tensor([tokens], device=self.device)

        # Compute loss
        logits = self.model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
