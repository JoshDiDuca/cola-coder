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

import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

from .thinking_tokens import THINK_OPEN, THINK_CLOSE, extract_thinking
from ..evaluation.runner import execute_code

logger = logging.getLogger(__name__)


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


def _compute_reward_worker(args: tuple) -> tuple[float, dict]:
    """Top-level worker function for ProcessPoolExecutor.

    Must be defined at module scope so it can be pickled on Windows (spawn
    start method).  Receives a single tuple argument so it works with
    executor.map without extra overhead.

    Args:
        args: (generated_text, test_code, max_thinking_tokens, timeout)

    Returns:
        (reward, info) pair from compute_reward.
    """
    generated_text, test_code, max_thinking_tokens, timeout = args
    return compute_reward(
        generated_text,
        test_code,
        max_thinking_tokens=max_thinking_tokens,
        timeout=timeout,
    )


def compute_batch_rewards_parallel(
    generations: list[str],
    test_code: str,
    max_thinking_tokens: int = 512,
    workers: int = 4,
    per_task_timeout: float = 30.0,
) -> tuple[list[float], list[dict]]:
    """Compute rewards in parallel using ProcessPoolExecutor.

    Each reward computation runs code in a subprocess (via execute_code), so
    using processes rather than threads gives true parallelism and isolates
    crashes from individual test runs.

    Falls back to the serial compute_batch_rewards path when:
    - workers <= 1 (caller explicitly asked for serial)
    - ProcessPoolExecutor raises any exception (e.g. Windows spawn issues)
    - Individual futures time out (those completions get reward=0)

    Args:
        generations: List of model outputs to evaluate.
        test_code: Test code to verify solutions against.
        max_thinking_tokens: Maximum allowed thinking trace length.
        workers: Maximum number of worker processes.
        per_task_timeout: Seconds to wait for each individual future before
            treating it as timed-out (reward=0, correct=False).

    Returns:
        (rewards, infos) lists in the same order as `generations`.
    """
    if workers <= 1:
        return compute_batch_rewards(
            generations, test_code, max_thinking_tokens=max_thinking_tokens
        )

    task_args = [
        (gen, test_code, max_thinking_tokens, per_task_timeout)
        for gen in generations
    ]

    try:
        rewards: list[float] = []
        infos: list[dict] = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_compute_reward_worker, args) for args in task_args]

            for future in futures:
                try:
                    reward, info = future.result(timeout=per_task_timeout)
                    rewards.append(reward)
                    infos.append(info)
                except FuturesTimeoutError:
                    logger.warning("compute_batch_rewards_parallel: future timed out")
                    rewards.append(0.0)
                    infos.append(
                        {
                            "correct": False,
                            "has_thinking": False,
                            "thinking_length": 0,
                            "format_bonus": 0.0,
                            "length_penalty": 0.0,
                            "execution_output": "timeout",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("compute_batch_rewards_parallel: future raised %r", exc)
                    rewards.append(0.0)
                    infos.append(
                        {
                            "correct": False,
                            "has_thinking": False,
                            "thinking_length": 0,
                            "format_bonus": 0.0,
                            "length_penalty": 0.0,
                            "execution_output": f"error: {exc}",
                        }
                    )

        return rewards, infos

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "compute_batch_rewards_parallel: executor failed (%r), falling back to serial", exc
        )
        return compute_batch_rewards(
            generations, test_code, max_thinking_tokens=max_thinking_tokens
        )
