"""Inference Benchmark Suite: measure tokens/sec at different settings.

Benchmarks:
    - Tokens/sec across temperatures (0.0, 0.3, 0.7, 1.0)
    - Tokens/sec across sequence lengths (64, 128, 256, 512)
    - Tokens/sec across batch sizes (1, 2, 4, 8)
    - Precision comparison: bf16 vs fp16 (vs int8 if bitsandbytes available)

Generates a formatted performance table and optional JSON report.

Usage:
    python scripts/inference_benchmark.py --checkpoint checkpoints/tiny/latest
    python scripts/inference_benchmark.py --checkpoint checkpoints/tiny/latest --quick
    python scripts/inference_benchmark.py --checkpoint checkpoints/tiny/latest --json
    python scripts/inference_benchmark.py --checkpoint checkpoints/tiny/latest --output bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRun:
    """A single benchmark measurement."""

    label: str
    category: str  # "temperature", "seq_len", "batch_size", "precision"
    param_name: str
    param_value: str
    tokens_per_sec: float
    latency_ms_first_token: float
    total_tokens: int
    duration_sec: float
    error: str = ""


@dataclass
class BenchmarkReport:
    """Full inference benchmark report."""

    checkpoint: str
    device: str
    model_params: int = 0
    runs: list[BenchmarkRun] = field(default_factory=list)
    total_sec: float = 0.0

    def summary_table(self) -> str:
        """Format results as an ASCII table."""
        if not self.runs:
            return "No results."

        lines = [
            f"Inference Benchmark — {Path(self.checkpoint).name}  [{self.device}]",
            "",
            f"{'Category':<16} {'Parameter':<12} {'Tokens/sec':>12} {'First token':>14} {'Note':<20}",
            "─" * 80,
        ]
        prev_cat = ""
        for r in self.runs:
            if r.error:
                note = f"ERROR: {r.error[:20]}"
                tps = "—"
                ft = "—"
            else:
                note = ""
                tps = f"{r.tokens_per_sec:,.1f}"
                ft = f"{r.latency_ms_first_token:.1f} ms"

            cat_col = r.category if r.category != prev_cat else ""
            prev_cat = r.category
            lines.append(f"{cat_col:<16} {r.param_value:<12} {tps:>12} {ft:>14} {note:<20}")
        lines.append("─" * 80)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class InferenceBenchmarker:
    """Run inference benchmarks on a cola-coder model.

    The benchmarker lazy-loads the model once and reuses it across runs to
    avoid repeated load overhead.
    """

    def __init__(
        self,
        checkpoint: str,
        config_path: str | None = None,
        tokenizer_path: str | None = None,
        n_warmup: int = 2,
        n_measure: int = 3,
    ) -> None:
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path
        self.n_warmup = n_warmup
        self.n_measure = n_measure
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"

    def _load(self) -> None:
        """Load model and tokenizer (done once)."""
        import torch

        from cola_coder.model.config import ModelConfig
        from cola_coder.training.checkpoint import load_model_only

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        # Resolve config from checkpoint metadata
        import json as _json
        from pathlib import Path as _Path

        meta_file = _Path(self.checkpoint) / "metadata.json"
        config_dict: dict = {}
        if meta_file.exists():
            config_dict = _json.loads(meta_file.read_text())

        raw_cfg = config_dict.get("config", config_dict)
        model_cfg_dict = raw_cfg.get("model", raw_cfg)
        model_cfg = ModelConfig(**{
            k: v for k, v in model_cfg_dict.items()
            if k in ModelConfig.__dataclass_fields__
        })

        self._model = load_model_only(self.checkpoint, model_cfg, device=self._device)
        self._model.eval()

        # Load tokenizer
        tok_path = self.tokenizer_path
        if not tok_path:
            storage = get_storage_config()
            tok_path = storage.tokenizer_path
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        self._tokenizer = Tokenizer.from_file(tok_path)

    def _generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[float, float, int]:
        """Generate tokens and return (tokens_per_sec, first_token_ms, n_tokens)."""
        import torch

        ids = self._tokenizer.encode(prompt).ids  # type: ignore[union-attr]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

        t0 = time.perf_counter()
        generated = 0
        first_token_ms = 0.0

        with torch.no_grad():
            for i in range(max_new_tokens):
                logits = self._model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[0, -1, :]

                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = next_logits.argmax(keepdim=True)

                if i == 0:
                    first_token_ms = (time.perf_counter() - t0) * 1000

                input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
                generated += 1

        total_sec = time.perf_counter() - t0
        tps = generated / total_sec if total_sec > 0 else 0.0
        return tps, first_token_ms, generated

    def _measure(
        self,
        label: str,
        category: str,
        param_name: str,
        param_value: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> BenchmarkRun:
        """Run warmup + measurement passes and return a BenchmarkRun."""
        try:
            # Warmup
            for _ in range(self.n_warmup):
                self._generate_tokens(prompt, min(max_new_tokens, 16), temperature)

            # Measure
            tps_list: list[float] = []
            ft_list: list[float] = []
            t0 = time.monotonic()
            total_tokens = 0

            for _ in range(self.n_measure):
                tps, ft, n = self._generate_tokens(prompt, max_new_tokens, temperature)
                tps_list.append(tps)
                ft_list.append(ft)
                total_tokens += n

            total_sec = time.monotonic() - t0
            avg_tps = sum(tps_list) / len(tps_list)
            avg_ft = sum(ft_list) / len(ft_list)

            return BenchmarkRun(
                label=label,
                category=category,
                param_name=param_name,
                param_value=param_value,
                tokens_per_sec=avg_tps,
                latency_ms_first_token=avg_ft,
                total_tokens=total_tokens,
                duration_sec=total_sec,
            )
        except Exception as exc:
            return BenchmarkRun(
                label=label,
                category=category,
                param_name=param_name,
                param_value=param_value,
                tokens_per_sec=0.0,
                latency_ms_first_token=0.0,
                total_tokens=0,
                duration_sec=0.0,
                error=str(exc),
            )

    def run(self, quick: bool = False) -> BenchmarkReport:
        """Run the full benchmark suite."""
        self._load()
        report = BenchmarkReport(
            checkpoint=self.checkpoint,
            device=self._device,
        )

        prompt = "def fibonacci(n: int) -> int:\n    "
        base_tokens = 32 if quick else 64

        orig_n = self.n_measure
        if quick:
            self.n_measure = 1
            self.n_warmup = 1

        t0 = time.monotonic()

        # ── Temperature sweep ──────────────────────────────────────────────
        temps = [0.0, 0.7] if quick else [0.0, 0.3, 0.7, 1.0]
        for temp in temps:
            run = self._measure(
                label=f"temperature={temp}",
                category="temperature",
                param_name="temperature",
                param_value=str(temp),
                prompt=prompt,
                max_new_tokens=base_tokens,
                temperature=temp,
            )
            report.runs.append(run)

        # ── Sequence length sweep ──────────────────────────────────────────
        seq_lens = [64, 128] if quick else [64, 128, 256, 512]
        for seq_len in seq_lens:
            run = self._measure(
                label=f"seq_len={seq_len}",
                category="seq_len",
                param_name="max_new_tokens",
                param_value=str(seq_len),
                prompt=prompt,
                max_new_tokens=seq_len,
                temperature=0.7,
            )
            report.runs.append(run)

        # ── Precision comparison ───────────────────────────────────────────
        import torch

        current_dtype = next(self._model.parameters()).dtype  # type: ignore[union-attr]
        precisions = [("bf16", torch.bfloat16), ("fp16", torch.float16)]
        if not quick:
            precisions.append(("fp32", torch.float32))

        for prec_name, dtype in precisions:
            try:
                self._model.to(dtype)  # type: ignore[union-attr]
                run = self._measure(
                    label=f"precision={prec_name}",
                    category="precision",
                    param_name="dtype",
                    param_value=prec_name,
                    prompt=prompt,
                    max_new_tokens=base_tokens,
                    temperature=0.7,
                )
                report.runs.append(run)
            except Exception as exc:
                report.runs.append(
                    BenchmarkRun(
                        label=f"precision={prec_name}",
                        category="precision",
                        param_name="dtype",
                        param_value=prec_name,
                        tokens_per_sec=0.0,
                        latency_ms_first_token=0.0,
                        total_tokens=0,
                        duration_sec=0.0,
                        error=str(exc),
                    )
                )
            finally:
                # Restore original dtype
                self._model.to(current_dtype)  # type: ignore[union-attr]

        report.total_sec = time.monotonic() - t0
        self.n_measure = orig_n
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cola-coder inference throughput.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--quick", action="store_true", help="Run a faster subset of benchmarks.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file.")
    args = parser.parse_args()

    # Auto-detect checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        try:
            storage = get_storage_config()
            from cola_coder.training.checkpoint import detect_latest_checkpoint

            result = detect_latest_checkpoint(storage.checkpoints_dir)
            if result:
                checkpoint, _ = result
        except Exception:
            pass

    if not checkpoint:
        cli.fatal("No checkpoint found.", hint="Pass --checkpoint.")
        sys.exit(1)

    if not args.json:
        cli.header("Cola-Coder", "Inference Benchmark")
        cli.info("Checkpoint", checkpoint)
        if args.quick:
            cli.info("Mode", "quick")

    try:
        benchmarker = InferenceBenchmarker(
            checkpoint=checkpoint,
            config_path=args.config,
            tokenizer_path=args.tokenizer,
        )
        report = benchmarker.run(quick=args.quick)
    except Exception as exc:
        cli.fatal(f"Benchmark failed: {exc}")
        sys.exit(1)

    if not args.json:
        print()
        print(report.summary_table())
        print()
        cli.info("Total time", f"{report.total_sec:.1f}s")

    if args.json or args.output:
        report_dict = {
            "checkpoint": report.checkpoint,
            "device": report.device,
            "total_sec": round(report.total_sec, 2),
            "runs": [asdict(r) for r in report.runs],
        }
        json_str = json.dumps(report_dict, indent=2)
        if args.json:
            print(json_str)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            if not args.json:
                cli.info("Saved", args.output)


if __name__ == "__main__":
    main()
