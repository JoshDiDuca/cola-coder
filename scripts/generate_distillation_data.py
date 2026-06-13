"""Generate distillation SFT data from a teacher model (MODEL-024 / MODEL-028).

Runs prompts through the teacher in configs/distillation.yaml (local Qwen/DeepSeek
via Ollama, or cloud), optionally rejection-samples completions through the
SANDBOXED TypeScript verifier (tsc --strict — teacher output is UNTRUSTED, SEC-014),
and writes ChatML JSONL that scripts/train_sft.py consumes.

Examples:
    # local Qwen via Ollama, verify TS completions before keeping them
    .venv/Scripts/python scripts/generate_distillation_data.py \
        --config configs/distillation.yaml \
        --prompts data/sft/seed_prompts.jsonl \
        --output data/sft/distilled.jsonl --language ts --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cola_coder.cli import cli
from cola_coder.distillation import build_teacher, generate_distillation_dataset


def _load_prompts(path: Path) -> list:
    """Read a prompts JSONL. Each line: {"messages":[...]} | {"prompt":...} |
    {"instruction":...}."""
    prompts: list = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if isinstance(rec.get("messages"), list):
            prompts.append(rec["messages"])
        elif rec.get("prompt"):
            prompts.append(str(rec["prompt"]))
        elif rec.get("instruction"):
            prompts.append(str(rec["instruction"]))
    return prompts


def _make_ts_verifier():
    """Sandboxed TS verifier: accept a completion only if tsc actually ran and
    found zero errors. SEC-014/SEC-016: untrusted teacher code runs ONLY in the
    sandbox, and a sandbox-unavailable result is treated as NOT verified."""
    from cola_coder.data.scorers.tsc_scorer import TscScorer

    scorer = TscScorer()

    def verify(completion: str) -> bool:
        res = scorer.score(completion, {"file_path": "completion.ts"})
        if res.details.get("not_verified") or res.details.get("skipped"):
            return False
        return res.details.get("num_errors", 1) == 0

    return verify


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate distillation SFT data from a teacher.")
    ap.add_argument("--config", default="configs/distillation.yaml")
    ap.add_argument("--prompts", required=True, help="JSONL of prompts")
    ap.add_argument("--output", required=True, help="output ChatML JSONL")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--language", choices=["ts", "none"], default="none")
    ap.add_argument("--verify", action="store_true", help="rejection-sample via sandboxed verifier")
    ap.add_argument("--limit", type=int, default=None, help="cap number of prompts")
    args = ap.parse_args()

    cli.header("Distillation data generation")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    gen_cfg = cfg.get("generation", {})
    teacher = build_teacher(cfg["teacher"])
    cli.info(f"Teacher: {teacher.name}")

    prompts = _load_prompts(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]
    cli.info(f"Prompts: {len(prompts)}")

    verify = None
    if args.verify or gen_cfg.get("verify"):
        from cola_coder.security.code_patterns import is_dangerous, scan_dangerous

        ts_verify = _make_ts_verifier() if args.language == "ts" else None

        def verify(completion: str) -> bool:
            # SECURITY SCREEN (always when verifying): reject completions with
            # dangerous patterns so we never DISTILL insecure code into the
            # student — functional code is often insecure (secure-pass@1 is low),
            # so tsc/tests alone aren't enough. Static, no execution.
            if is_dangerous(completion):
                return False
            # Functional verification (TS only): sandboxed tsc --strict.
            return ts_verify(completion) if ts_verify is not None else True

        _ = scan_dangerous  # (available for richer logging if needed)
        if ts_verify is not None:
            cli.info("Verification: security screen + sandboxed tsc --strict (rejection sampling)")
        else:
            cli.info("Verification: dangerous-pattern security screen (rejection sampling)")

    records, stats = generate_distillation_dataset(
        teacher,
        prompts,
        max_tokens=args.max_tokens or gen_cfg.get("max_tokens", 512),
        temperature=args.temperature if args.temperature is not None
        else gen_cfg.get("temperature", 0.7),
        verify=verify,
        keep_only_verified=bool(gen_cfg.get("keep_only_verified", True)),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cli.kv_table("Results", {
        "prompts": stats["prompts"],
        "teacher_ok": stats["teacher_ok"],
        "teacher_errors": stats["teacher_errors"],
        "verified": stats["verified"],
        "rejected": stats["rejected"],
        "kept (written)": stats["kept"],
        "output": str(out),
    })
    cli.success(f"Wrote {stats['kept']} distillation records → {out}")


if __name__ == "__main__":
    main()
