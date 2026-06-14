#!/usr/bin/env python
"""Generate self-verified SFT data via RFT (rejection-sampling fine-tuning).

MODEL-046 — the runnable front-end for the MODEL-045 harness. Loads a checkpoint,
generates best-of-N candidates per problem, keeps ONLY the verifier-passed + secure
ones (self-distillation: the student's own verified output), and writes ChatML SFT
records ready for ``scripts/train_sft.py``.

Examples:
    # Self-solve the built-in problems, keep verified+secure:
    python scripts/generate_rft_data.py --checkpoint checkpoints/small/latest --config configs/small.yaml

    # Custom prompts (JSONL with {"prompt": ..., "test_code": ...} per line):
    python scripts/generate_rft_data.py --checkpoint ckpt --config cfg --jsonl prompts.jsonl --output data/sft/rft.jsonl

Security: model output is NEVER executed here — all execution happens inside the
sandboxed verifier used by generate_best_of_n (TscRunner / SandboxedRunner).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli  # noqa: E402


def _load_prompts(args) -> tuple[list[str], list[str | None]]:
    """Return (prompts, tests). Tests may contain None where unavailable."""
    if args.jsonl:
        prompts: list[str] = []
        tests: list[str | None] = []
        for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
            tests.append(obj.get("test_code"))
        return prompts, tests

    # Built-in coding problems (prompt + test_code).
    from cola_coder.evaluation.humaneval import get_all_problems
    problems = get_all_problems()
    if args.max_prompts:
        problems = problems[: args.max_prompts]
    return [p.prompt for p in problems], [p.test_code for p in problems]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RFT / self-verified SFT data generation (MODEL-046)"
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or `latest`")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (auto-resolved if omitted)")
    parser.add_argument("--jsonl", default=None, help="Prompts JSONL (default: built-in problems)")
    parser.add_argument("--output", default="data/sft/rft.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--language", default="auto", choices=["auto", "python", "typescript"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep the best candidate even if it failed verification (default: drop).",
    )
    args = parser.parse_args()

    cli.header("RFT — self-verified SFT data generation")

    try:
        from cola_coder.inference.loading import load_generator
        generator, _config, _tok = load_generator(
            args.checkpoint, args.config, tokenizer_path=args.tokenizer,
        )
    except Exception as e:  # noqa: BLE001 — surface a clean message, don't traceback
        cli.error(f"Could not load checkpoint: {e}")
        sys.exit(1)
    cli.info("Checkpoint", str(args.checkpoint))

    prompts, tests = _load_prompts(args)
    cli.info("Prompts", str(len(prompts)))
    cli.info("Candidates/prompt", str(args.num_candidates))
    cli.info("Verification", "keep-all" if args.keep_all else "verified + secure only")

    from cola_coder.distillation import generate_rft_dataset
    records, stats = generate_rft_dataset(
        generator,
        prompts,
        num_candidates=args.num_candidates,
        language=args.language,
        tests=tests,
        keep_only_verified=not args.keep_all,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    cli.kv_table("Results", {
        "prompts": stats["prompts"],
        "verified": stats["verified"],
        "rejected (unverified)": stats["rejected_unverified"],
        "rejected (insecure)": stats["rejected_insecure"],
        "kept (written)": stats["kept"],
        "output": str(out),
    })
    cli.success(f"Wrote {stats['kept']} RFT SFT records → {out}")


if __name__ == "__main__":
    main()
