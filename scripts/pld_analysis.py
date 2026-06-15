"""Offline Prompt-Lookup-Decoding (PLD) acceptance analysis (INFER-035).

Replays the draft-free PLD drafter over a prepared ``.npy`` token corpus and
reports the acceptance statistics that predict speculative-decoding speedup —
acceptance rate, mean accepted length, draft hit rate, and an idealised
step-count speedup estimate. Pure offline: NO model load, NO GPU, NO network. It
only string-matches token ids, so the numbers tell you whether wiring PLD into
the live decoder is worth it BEFORE you build the hot-path version.

Usage:
    python scripts/pld_analysis.py --data data/processed/train_data.npy
    python scripts/pld_analysis.py --data train.npy --sample 200 --max-ngram 4
    python scripts/pld_analysis.py --data train.npy --num-pred 8 --min-ngram 2 --seed-len 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cola_coder.cli import cli  # noqa: E402
from cola_coder.inference.prompt_lookup import (  # noqa: E402
    PromptLookupConfig,
    PromptLookupDrafter,
    analyze_acceptance,
)


def _load_sequences(data_path: Path, sample: int | None) -> list[list[int]]:
    """Load a ``.npy`` token array as a list of per-sequence token-id lists.

    A 2-D ``[N, seq]`` array yields one list per row; a 1-D ``[tokens]`` array
    yields a single sequence. ``sample`` caps the number of sequences returned.
    """
    import numpy as np

    arr = np.load(str(data_path), mmap_mode="r")
    if arr.ndim == 2:
        rows = arr.shape[0]
        limit = min(rows, sample) if sample else rows
        return [[int(t) for t in arr[i]] for i in range(limit)]
    # 1-D: a single flat sequence.
    return [[int(t) for t in np.asarray(arr)]]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline PLD acceptance analysis over a tokenized .npy corpus (no model)."
    )
    parser.add_argument(
        "--data", required=True, metavar="PATH",
        help="Path to a tokenized .npy array (shape [N, seq] or [tokens])",
    )
    parser.add_argument(
        "--max-ngram", type=int, default=3, metavar="N",
        help="Longest suffix n-gram the drafter tries (default: 3)",
    )
    parser.add_argument(
        "--min-ngram", type=int, default=1, metavar="N",
        help="Shortest suffix n-gram the drafter may match (default: 1)",
    )
    parser.add_argument(
        "--num-pred", type=int, default=10, metavar="N",
        help="Max continuation tokens proposed per draft (default: 10)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Limit analysis to the first N sequences (default: all)",
    )
    parser.add_argument(
        "--seed-len", type=int, default=1, metavar="N",
        help="Primed-context tokens before prediction begins (default: 1)",
    )
    args = parser.parse_args()

    cli.header("Cola-Coder", "Prompt-Lookup Acceptance Analysis")

    data_path = Path(args.data)
    if not data_path.exists():
        cli.error(
            f"Data file not found: {data_path}",
            hint="Run prepare_data.py first, or pass an existing --data <path.npy>",
        )
        return 1

    try:
        config = PromptLookupConfig(
            max_ngram_size=args.max_ngram,
            num_pred_tokens=args.num_pred,
            min_ngram_size=args.min_ngram,
        )
    except ValueError as exc:
        cli.error(f"Invalid drafter config: {exc}")
        return 1

    try:
        sequences = _load_sequences(data_path, args.sample)
    except ImportError:
        cli.error("numpy is required", hint="pip install numpy")
        return 1
    except Exception as exc:  # malformed / unreadable .npy
        cli.error(f"Failed to load data: {exc}")
        return 1

    if not sequences:
        cli.error("No sequences found in the data file.")
        return 1

    drafter = PromptLookupDrafter(config)

    # Aggregate across sequences long enough to predict from.
    total_tokens = 0
    decode_steps = 0
    baseline_steps = 0
    total_drafted = 0
    total_accepted = 0
    steps_with_draft = 0
    analyzed = 0
    skipped = 0

    for seq in sequences:
        if len(seq) <= args.seed_len:
            skipped += 1
            continue
        report = analyze_acceptance(seq, drafter, seed_len=args.seed_len)
        total_tokens += report.total_tokens
        decode_steps += report.decode_steps
        baseline_steps += report.baseline_steps
        total_drafted += report.total_drafted
        total_accepted += report.total_accepted
        steps_with_draft += sum(1 for s in report.steps if s.drafted > 0)
        analyzed += 1

    if analyzed == 0:
        cli.error(
            f"No sequence was longer than seed_len ({args.seed_len}); nothing to analyze.",
            hint="Lower --seed-len or provide longer sequences.",
        )
        return 1

    acceptance_rate = total_accepted / total_drafted if total_drafted else 0.0
    mean_accepted_length = total_accepted / decode_steps if decode_steps else 0.0
    draft_hit_rate = steps_with_draft / decode_steps if decode_steps else 0.0
    speedup_estimate = baseline_steps / decode_steps if decode_steps else 1.0

    cli.info("Data file", str(data_path))
    cli.info("Sequences analyzed", f"{analyzed:,}" + (f" ({skipped:,} too short)" if skipped else ""))

    cli.kv_table({
        "max_ngram_size": str(config.max_ngram_size),
        "min_ngram_size": str(config.min_ngram_size),
        "num_pred_tokens": str(config.num_pred_tokens),
        "seed_len": str(args.seed_len),
    }, title="Drafter Config")

    cli.kv_table({
        "Tokens predicted": f"{total_tokens:,}",
        "Decode steps": f"{decode_steps:,}",
        "Baseline steps": f"{baseline_steps:,}",
        "Acceptance rate": f"{acceptance_rate:.1%}",
        "Mean accepted length": f"{mean_accepted_length:.2f}",
        "Draft hit rate": f"{draft_hit_rate:.1%}",
        "Speedup estimate": f"{speedup_estimate:.2f}x",
    }, title="Aggregate Acceptance Report")

    cli.print("")
    cli.success("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
