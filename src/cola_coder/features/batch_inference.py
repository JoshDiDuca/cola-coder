"""Batch Inference: process multiple prompts efficiently.

Processes a batch of prompts through the model, managing the queue,
tracking progress, and collecting results. Useful for benchmarking,
evaluation, and bulk code generation.

For a TS dev: like Promise.all() for model inference — process many
prompts concurrently instead of one at a time.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import json

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


@dataclass
class InferenceResult:
    """Result from a single inference."""
    prompt: str
    output: str
    tokens_generated: int = 0
    time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.time_ms / 1000.0)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "output": self.output,
            "tokens_generated": self.tokens_generated,
            "time_ms": round(self.time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "metadata": self.metadata,
        }


@dataclass
class BatchResult:
    """Results from a batch inference run."""
    results: list[InferenceResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    batch_size: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated for r in self.results)

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_time_ms <= 0:
            return 0.0
        return self.total_tokens / (self.total_time_ms / 1000.0)

    @property
    def avg_time_per_prompt_ms(self) -> float:
        if not self.results:
            return 0.0
        return self.total_time_ms / len(self.results)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        non_empty = sum(1 for r in self.results if r.output.strip())
        return non_empty / len(self.results)

    def to_dict(self) -> dict:
        return {
            "total_prompts": len(self.results),
            "total_tokens": self.total_tokens,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "avg_time_per_prompt_ms": round(self.avg_time_per_prompt_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        """Save batch results to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "BatchResult":
        """Load batch results from a JSON file."""
        data = json.loads(Path(path).read_text())
        batch = cls(
            total_time_ms=data.get("total_time_ms", 0.0),
            batch_size=data.get("total_prompts", 0),
        )
        for r_data in data.get("results", []):
            batch.results.append(InferenceResult(
                prompt=r_data["prompt"],
                output=r_data["output"],
                tokens_generated=r_data.get("tokens_generated", 0),
                time_ms=r_data.get("time_ms", 0.0),
                metadata=r_data.get("metadata", {}),
            ))
        return batch

    def summary(self) -> str:
        """Generate a text summary of the batch results."""
        lines = [
            "Batch Inference Results",
            f"  Prompts: {len(self.results)}",
            f"  Total tokens: {self.total_tokens:,}",
            f"  Total time: {self.total_time_ms:.0f}ms ({self.total_time_ms/1000:.1f}s)",
            f"  Throughput: {self.avg_tokens_per_second:.1f} tok/s",
            f"  Avg per prompt: {self.avg_time_per_prompt_ms:.0f}ms",
            f"  Success rate: {self.success_rate:.0%}",
        ]
        return "\n".join(lines)


class BatchInference:
    """Process multiple prompts through a model efficiently."""

    def __init__(
        self,
        generator=None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """
        Args:
            generator: A CodeGenerator instance (or None for testing)
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
        """
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def run(
        self,
        prompts: list[str],
        on_progress: Optional[Callable] = None,
        metadata: Optional[list[dict]] = None,
    ) -> BatchResult:
        """Run inference on a batch of prompts.

        Args:
            prompts: List of prompt strings
            on_progress: Callback(current, total, result) called after each prompt
            metadata: Optional metadata dict per prompt

        Returns:
            BatchResult with all results and timing
        """
        if metadata is None:
            metadata = [{}] * len(prompts)

        batch_result = BatchResult(batch_size=len(prompts))
        batch_start = time.perf_counter()

        for i, (prompt, meta) in enumerate(zip(prompts, metadata)):
            start = time.perf_counter()

            if self.generator:
                output = self.generator.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                # Extract just the generated part (after prompt)
                generated = output[len(prompt):] if output.startswith(prompt) else output
            else:
                # No generator — return empty (for testing)
                generated = ""

            elapsed = (time.perf_counter() - start) * 1000

            # Estimate token count (rough: ~4 chars per token)
            est_tokens = max(1, len(generated) // 4)

            result = InferenceResult(
                prompt=prompt,
                output=generated,
                tokens_generated=est_tokens,
                time_ms=elapsed,
                metadata=meta,
            )
            batch_result.results.append(result)

            if on_progress:
                on_progress(i + 1, len(prompts), result)

        batch_result.total_time_ms = (time.perf_counter() - batch_start) * 1000
        return batch_result

    def run_from_file(self, path: str, **kwargs) -> BatchResult:
        """Run inference on prompts from a text file (one per line).

        Args:
            path: Path to text file with prompts
            **kwargs: Passed to run()

        Returns:
            BatchResult
        """
        prompts = [
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        return self.run(prompts, **kwargs)

    def print_progress(self, current: int, total: int, result: InferenceResult) -> None:
        """Default progress callback — prints to stdout."""
        preview = result.output[:50].replace("\n", " ")
        print(f"  [{current}/{total}] {result.time_ms:.0f}ms | {preview}...")

    def print_report(self, batch_result: BatchResult) -> None:
        """Print a formatted report of batch results."""
        from cola_coder.cli import cli

        cli.header("Batch Inference", f"{len(batch_result.results)} prompts")
        cli.info("Total tokens", f"{batch_result.total_tokens:,}")
        cli.info("Total time", f"{batch_result.total_time_ms:.0f}ms")
        cli.info("Throughput", f"{batch_result.avg_tokens_per_second:.1f} tok/s")
        cli.info("Avg per prompt", f"{batch_result.avg_time_per_prompt_ms:.0f}ms")
        cli.info("Success rate", f"{batch_result.success_rate:.0%}")
