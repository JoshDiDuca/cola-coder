"""Reward functions for reinforcement learning on code generation.

In GRPO (and RL in general), the model generates code, and we need a way
to score how good that code is. This is the "reward function."

The key insight from recent research (DeepSeek-R1, etc.): for code generation,
we have a VERIFIABLE reward — we can actually RUN the code and see if it passes
tests. This is much simpler than training a separate reward model (which is
what chat-focused RL does).

Our reward function:
- +1.0 if the code passes all test cases (correct solution)
- +0.0 if the code fails any test case (incorrect)
- +0.1 bonus if the output uses proper <think>...</think> format
- -0.1 penalty if the thinking trace is excessively long (> max tokens)
"""

from .thinking_tokens import THINK_OPEN, THINK_CLOSE, extract_thinking
from ..evaluation.runner import execute_code


def compute_reward(
    generated_text: str,
    test_code: str,
    max_thinking_tokens: int = 512,
    timeout: float = 10.0,
) -> tuple[float, dict]:
    """Compute reward for a generated solution.

    Args:
        generated_text: The model's full output (may include <think> tokens).
        test_code: Python test code to verify the solution.
        max_thinking_tokens: Maximum allowed thinking trace length.
        timeout: Maximum execution time for tests.

    Returns:
        (reward, info) tuple.
        reward: Float reward value.
        info: Dictionary with reward breakdown details.
    """
    reward = 0.0
    info = {
        "correct": False,
        "has_thinking": False,
        "thinking_length": 0,
        "format_bonus": 0.0,
        "length_penalty": 0.0,
        "execution_output": "",
    }

    # Extract thinking and code portions
    thinking, code = extract_thinking(generated_text)
    info["has_thinking"] = bool(thinking)
    info["thinking_length"] = len(thinking.split()) if thinking else 0

    # Correctness reward (the main signal)
    # Combine the generated code with the test cases and run
    full_code = code + "\n\n" + test_code
    success, output = execute_code(full_code, timeout=timeout)
    info["correct"] = success
    info["execution_output"] = output

    if success:
        reward += 1.0

    # Format bonus: reward proper use of thinking tokens
    if THINK_OPEN in generated_text and THINK_CLOSE in generated_text:
        # Check that thinking comes BEFORE the code
        think_end = generated_text.index(THINK_CLOSE)
        if think_end < len(generated_text) - 10:  # Code follows thinking
            info["format_bonus"] = 0.1
            reward += 0.1

    # Length penalty: discourage excessively long thinking
    if info["thinking_length"] > max_thinking_tokens:
        excess = info["thinking_length"] - max_thinking_tokens
        penalty = min(0.1, excess * 0.001)  # Small penalty, capped
        info["length_penalty"] = -penalty
        reward -= penalty

    return reward, info


def compute_batch_rewards(
    generations: list[str],
    test_code: str,
    max_thinking_tokens: int = 512,
) -> tuple[list[float], list[dict]]:
    """Compute rewards for a batch of generated solutions.

    Args:
        generations: List of model outputs.
        test_code: Test code to verify solutions.
        max_thinking_tokens: Maximum thinking trace length.

    Returns:
        (rewards, infos) lists.
    """
    rewards = []
    infos = []

    for gen in generations:
        reward, info = compute_reward(
            gen, test_code,
            max_thinking_tokens=max_thinking_tokens,
        )
        rewards.append(reward)
        infos.append(info)

    return rewards, infos
