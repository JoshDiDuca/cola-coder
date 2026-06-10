"""Evaluate the semantic router's domain classification accuracy.

Loads a trained router checkpoint, classifies the built-in labeled test set,
and reports accuracy, per-domain precision/recall/F1, and the confusion
matrix.

Usage:
    python scripts/evaluate_router.py --router-checkpoint checkpoints/router
    python scripts/evaluate_router.py --router-checkpoint checkpoints/router --domains typescript
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli

# Domain scopes for --domains. Test-set labels outside DEFAULT_DOMAINS are
# normalized (python / general_ts → general).
_DOMAIN_SCOPES = {
    "all": None,  # No filtering
    "typescript": {"react", "nextjs", "general"},
    "backend": {"graphql", "prisma"},
}

_LABEL_NORMALIZE = {"python": "general", "general_ts": "general"}


def _resolve_checkpoint_dir(path_str: str) -> Path:
    """Resolve a router checkpoint dir, following a 'latest' pointer file."""
    path = Path(path_str)
    if path.name == "latest" and path.is_file():
        return Path(path.read_text(encoding="utf-8").strip())
    if path.name == "latest" and not path.exists():
        # checkpoints/router/latest with no pointer → use the parent dir
        return path.parent
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic router classification accuracy.",
    )
    parser.add_argument(
        "--router-checkpoint", required=True,
        help="Router checkpoint directory (contains best_router.pt / router_final.pt)",
    )
    parser.add_argument(
        "--domains", default="all", choices=sorted(_DOMAIN_SCOPES),
        help="Domain scope to evaluate (default: all)",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    args = parser.parse_args()

    cli.header("Cola-Coder", "Router Evaluation")

    import torch

    from cola_coder.features.router_evaluation import (
        RouterEvaluator,
        create_test_dataset,
    )
    from cola_coder.features.router_model import (
        DEFAULT_DOMAINS,
        RouterConfig,
        create_router,
    )
    from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

    # ── Locate weights ────────────────────────────────────────────────
    ckpt_dir = _resolve_checkpoint_dir(args.router_checkpoint)
    weights = None
    for name in ("best_router.pt", "router_final.pt"):
        candidate = ckpt_dir / name
        if candidate.exists():
            weights = candidate
            break
    if weights is None:
        # Search one level of subdirectories (e.g. checkpoints/router/<run>/)
        matches = sorted(ckpt_dir.glob("*/best_router.pt")) + sorted(
            ckpt_dir.glob("*/router_final.pt")
        )
        if matches:
            weights = matches[0]
            ckpt_dir = weights.parent
    if weights is None:
        cli.fatal(
            f"No router weights found under {ckpt_dir}",
            hint="Train one first: Training → Alignment → Train Semantic Router",
        )

    # ── Build model from saved config ─────────────────────────────────
    config_path = ckpt_dir / "router_config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        known = {f for f in RouterConfig.__dataclass_fields__}
        router_config = RouterConfig(**{k: v for k, v in raw.items() if k in known})
    else:
        router_config = RouterConfig()
        cli.warn("router_config.json not found — using default RouterConfig")

    model = create_router(router_config, architecture=router_config.architecture)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    cli.info("Router", str(weights))
    cli.info("Architecture", router_config.architecture)

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        try:
            from cola_coder.data.dataset_resolver import DatasetResolver

            if DatasetResolver.tokenizer_exists():
                tokenizer_path = str(DatasetResolver.get_tokenizer_path())
        except Exception:
            pass
    if tokenizer_path is None or not Path(tokenizer_path).exists():
        cli.fatal(
            "Tokenizer not found",
            hint="Pass --tokenizer path/to/tokenizer.json",
        )
    tokenizer = CodeTokenizer(tokenizer_path)

    # ── Evaluate ──────────────────────────────────────────────────────
    scope = _DOMAIN_SCOPES[args.domains]
    samples = [
        s for s in create_test_dataset()
        if scope is None
        or _LABEL_NORMALIZE.get(s.expected_domain, s.expected_domain) in scope
    ]
    cli.info("Test samples", f"{len(samples)} ({args.domains} scope)")

    evaluator = RouterEvaluator()
    with torch.no_grad():
        for sample in samples:
            ids = tokenizer.encode(sample.prompt, add_bos=False)
            ids = ids[: router_config.max_seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long)
            domain_idx, confidence = model.predict(input_ids)
            predicted = DEFAULT_DOMAINS[int(domain_idx.item())]
            actual = _LABEL_NORMALIZE.get(
                sample.expected_domain, sample.expected_domain
            )
            evaluator.add_result(predicted, actual, float(confidence.item()))

    cli.rule("Results")
    cli.info("Accuracy", f"{evaluator.accuracy():.1%}")

    per_domain = evaluator.per_domain_metrics()
    if per_domain:
        cli.kv_table(
            {
                domain: (
                    f"P {m['precision']:.2f}  R {m['recall']:.2f}  "
                    f"F1 {m['f1']:.2f}  n={int(m['support'])}"
                )
                for domain, m in sorted(per_domain.items())
            },
            title="Per-Domain Metrics",
        )

    cm = evaluator.confusion_matrix()
    if cm:
        cli.rule("Confusion Matrix (actual → predicted)")
        for actual, row in cm.items():
            predicted_counts = {p: c for p, c in row.items() if c}
            cli.dim(f"  {actual:<12} {predicted_counts}")

    cli.done("Router evaluation finished")


if __name__ == "__main__":
    main()
