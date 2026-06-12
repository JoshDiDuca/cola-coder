"""Multi-source data collection and mixing.

Reads configs/data_sources.yaml and downloads code, text, and math data
from HuggingFace. Tokenizes each source into separate .npy files, then
combines them with weighted mixing per Qwen2.5-Coder ratios (70/20/10).

Usage:
    .venv/Scripts/python scripts/collect_data.py --config configs/small.yaml
    .venv/Scripts/python scripts/collect_data.py --config configs/4080_max.yaml --sources code,text
    .venv/Scripts/python scripts/collect_data.py --config configs/tiny.yaml --max-samples 1000
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cola_coder.cli import cli
from cola_coder.data.combine import DatasetCombiner, DatasetInput
from cola_coder.data.dataset_resolver import DatasetResolver
from cola_coder.data.download import stream_code_data
from cola_coder.data.preprocess import tokenize_and_chunk
from cola_coder.model.config import Config
from cola_coder.security.scanner import CompositeMalwareScanner
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer


# ── Malware scanning ─────────────────────────────────────────────────────

def _load_scan_config() -> dict:
    """Load the scoring.security section from configs/scoring.yaml."""
    scoring_path = Path("configs/scoring.yaml")
    if not scoring_path.exists():
        return {}
    with open(scoring_path, encoding="utf-8") as f:
        scoring_cfg = yaml.safe_load(f) or {}
    return scoring_cfg.get("scoring", {}).get("security", {})


def _scan_text_stream(iterator, label: str, stats: dict):
    """Yield text records, dropping any whose CONTENT matches threat rules.

    This must run BEFORE tokenization: the post-collection directory scan
    only sees tokenized .npy output, where pattern scanners can match
    nothing. Uses the YARA scanner's in-memory text API (regex fallback
    when yara-python is missing), so it works on streamed HF records
    without writing them to disk.
    """
    from cola_coder.security.yara_scanner import YaraScanner

    scanner = YaraScanner()
    for i, text in enumerate(iterator):
        threats = scanner.scan_text(text, identifier=f"{label}#{i}")
        if threats:
            stats["dropped"] = stats.get("dropped", 0) + 1
            for t in threats:
                logger.warning(
                    "MALWARE pattern in streamed %s record %d [%s/%s]: %s — record dropped",
                    label, i, t.scanner, t.severity, t.name,
                )
            continue
        stats["clean"] = stats.get("clean", 0) + 1
        yield text


def _maybe_scan_stream(iterator, label: str, scan_config: dict, stats: dict):
    """Wrap *iterator* with in-stream threat scanning unless disabled.

    Disabling is allowed but NEVER silent: scraped/streamed content is
    untrusted, so turning off scanning is a security downgrade and must be
    visible in the logs (SEC-001).
    """
    malware_cfg = scan_config.get("malware_scan", {})
    if not malware_cfg.get("enabled", True):
        cli.warn(
            f"SECURITY: malware scanning is DISABLED (malware_scan.enabled=false "
            f"in configs/scoring.yaml) — streamed {label} content is NOT scanned "
            f"before tokenization. Re-enable it unless you fully trust this source."
        )
        return iterator
    if not malware_cfg.get("in_stream", True):
        cli.warn(
            f"SECURITY: in-stream malware scanning is DISABLED "
            f"(malware_scan.in_stream=false) — streamed {label} content is NOT "
            f"pattern-scanned. Tokenized .npy output cannot be pattern-scanned "
            f"afterwards, so threats in streamed records would pass through."
        )
        return iterator
    return _scan_text_stream(iterator, label, stats)


def _quarantine_dest(quarantine_dir: Path, src: Path) -> Path:
    """Collision-free quarantine destination for a threatening file.

    The old code used ``quarantine_dir / src.name``, so two threats sharing a
    basename (e.g. the ubiquitous ``index.js`` / ``__init__.py``) collided —
    ``rename`` SILENTLY OVERWROTE the first, destroying quarantined-malware
    evidence. Prefix with a short hash of the FULL source path: unique per
    distinct path (idempotent for the same path), while keeping the readable
    basename for forensic triage.
    """
    digest = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:8]
    return quarantine_dir / f"{digest}_{src.name}"


def _maybe_quality_filter(iterator, filter_mode: str, languages: list[str], workers=None):
    """Wrap a CODE content stream with the quality filter (or pass through).

    filter_mode: "conservative" (default — only rejects clearly-bad code:
    minified, auto-generated, data dumps, broken syntax), "strict" (tighter +
    style/structure checks), or "off" (no filtering). Language-aware so e.g. a
    TypeScript run isn't rejected by the Python AST check. Mirrors prepare_data.
    ``workers`` defaults to CPU cores (capped at 12); pass 1 for sequential.
    """
    if filter_mode == "off":
        cli.warn("  Quality filter OFF for code — raw (unfiltered) code will be tokenized.")
        return iterator

    from cola_coder.data.quality_filter import (
        FilterMode,
        FilterStats,
        parallel_filtered_stream,
    )

    mode = FilterMode.STRICT if filter_mode == "strict" else FilterMode.CONSERVATIVE
    if workers is None:
        workers = max(1, min(os.cpu_count() or 4, 12))
    cli.info("  Quality filter", f"{mode.value} ({workers} workers)")
    return parallel_filtered_stream(
        iterator, mode=mode, stats=FilterStats(),
        num_workers=workers, languages=languages,
    )


def _maybe_dedup(npy_path: str, mode: str, tokenizer=None) -> None:
    """Deduplicate a tokenized .npy in place (raw corpora are 25-40%% duplicates).

    'exact' (SHA-256, default) drops byte-identical chunks; 'minhash' also removes
    near-dups (falls back to exact if datasketch is absent); 'none' is a no-op.
    Mirrors prepare_data's dedup step, which collect_data previously skipped — so
    the multi-source path was tokenizing all those duplicates into the training set.
    """
    if mode == "none":
        return
    from cola_coder.data.dedup import dedup_npy_file

    result = dedup_npy_file(npy_path, mode=mode, tokenizer=tokenizer)
    if result.removed > 0:
        pct = 100.0 * result.removed / max(result.before, 1)
        cli.info(
            "  Dedup",
            f"{result.before:,} → {result.after:,} chunks "
            f"({result.removed:,} removed, {pct:.1f}%, {result.mode})",
        )
    else:
        cli.dim(f"  Dedup: no duplicates found ({result.mode})")


def _maybe_score(npy_path: str, enabled: bool, tokenizer) -> bool:
    """Score a tokenized .npy → aligned <stem>.weights.npy for weighted training.

    Runs AFTER dedup so weights align with the surviving chunks (prepare_data's
    dedup-before-score invariant). Uses the shared `score_npy_to_weights` helper
    (same scorer/semantics as prepare_data --score). Returns True iff a weights
    sidecar was written.

    Only the CODE source is scored: the `code_scorer` feature judges CODE quality
    (syntax/structure), so running it on prose (text) or math corpora would
    mis-weight them. Unscored sources carry neutral weight 1.0 in the combine.
    """
    if not enabled:
        return False
    from cola_coder.data.weight_scoring import score_npy_to_weights

    wpath, weights = score_npy_to_weights(npy_path, tokenizer, progress=True)
    if wpath is None:
        cli.warn(
            "  --score requested but code_scorer is disabled/unavailable — "
            "no weights written (this source trains with neutral weight 1.0). "
            "Enable it in configs/features.yaml."
        )
        return False
    cli.info("  Quality weights", f"mean {float(weights.mean()):.3f} → {Path(wpath).name}")
    return True


def _scan_downloaded_data(
    raw_dir: Path,
    config: dict,
    step_num: int,
    total_steps: int,
) -> bool:
    """Scan downloaded data for malware. Returns True if clean or user continues."""
    scanner = CompositeMalwareScanner.from_config(config.get("malware_scan", {}))
    if not scanner.available_scanners:
        return True  # No scanners available, continue

    cli.step(step_num, total_steps, f"Scanning {raw_dir.name} for malware...")
    cli.dim(f"  Active scanners: {', '.join(scanner.available_scanners)}")

    result = scanner.scan_directory(raw_dir)

    if result.is_clean:
        cli.success(f"  Clean: {result.files_scanned} files scanned ({result.scan_duration_ms:.0f}ms)")
        return True

    # Threats found
    for t in result.threats:
        logger.warning(
            "MALWARE DETECTED in downloaded data [%s/%s]: %s in %s",
            t.scanner, t.severity, t.name, t.file_path,
        )
    cli.warn(f"  {len(result.threats)} threat(s) found in {result.files_scanned} files")
    for t in result.threats:
        cli.error(f"    [{t.severity.upper()}] {t.name}: {t.file_path}")
        if t.details:
            cli.dim(f"      {t.details}")

    on_threat = config.get("malware_scan", {}).get("on_threat", "warn")
    if on_threat == "abort":
        cli.error("  Aborting (on_threat=abort)")
        return False
    elif on_threat == "quarantine":
        # Move threatening files to quarantine dir
        quarantine_dir = raw_dir.parent / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        for t in result.threats:
            src = Path(t.file_path)
            if src.exists():
                dst = _quarantine_dest(quarantine_dir, src)
                src.rename(dst)
                cli.dim(f"    Quarantined: {src.name} -> {dst.name}")
        return True
    else:  # warn
        # Fail-CLOSED: cli.confirm returns its default on EOF/no-TTY, so a
        # non-interactive collection run would otherwise silently continue with
        # malware in the data. default=False makes the safe choice the default.
        return cli.confirm("  Continue despite threats? (default No = abort)", default=False)


# ── Generic HF text streaming ────────────────────────────────────────────

def stream_hf_text(
    dataset_name: str,
    *,
    split: str = "train",
    text_field: str = "text",
    min_length: int = 50,
    max_length: int = 50_000,
    max_samples: int | None = None,
):
    """Stream text from a HuggingFace dataset (non-code, no language dirs).

    Works for FineWeb-Edu, OpenWebMath, C4, OpenWebText2, etc.
    """
    from datasets import load_dataset

    cli.dim(f"  Streaming from {dataset_name} (field: {text_field})...")
    ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=False)

    count = 0
    for sample in ds:
        content = sample.get(text_field, "")
        if not content:
            continue
        if len(content) < min_length or len(content) > max_length:
            continue
        yield content
        count += 1
        if count % 10_000 == 0:
            cli.dim(f"    {count:,} samples streamed...")
        if max_samples is not None and count >= max_samples:
            cli.dim(f"  Reached sample limit: {max_samples:,}")
            return

    cli.dim(f"  Total: {count:,} text samples yielded from {dataset_name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source data collection (code + text + math)",
    )
    parser.add_argument("--config", required=True, help="Model config YAML path")
    parser.add_argument(
        "--data-sources", default="configs/data_sources.yaml",
        help="Data sources config (default: configs/data_sources.yaml)",
    )
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated sources to collect: code,text,math (default: all enabled)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per source (for testing)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npy files (default: auto-resolved from DatasetResolver)",
    )
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path override")
    parser.add_argument("--no-combine", action="store_true", help="Skip combining step")
    parser.add_argument(
        "--filter", choices=["conservative", "strict", "off"], default=None,
        help="Quality filter for the CODE source before tokenization "
             "(rejects minified/auto-generated/data-dump files). Default: read "
             "data_sources.yaml code.filter, else 'conservative'. Text/math are "
             "not code-filtered.",
    )
    parser.add_argument(
        "--dedup", choices=["none", "exact", "minhash"], default="exact",
        help="Per-source chunk dedup after tokenization (raw corpora are 25-40%% "
             "duplicates). 'exact' (default, SHA-256) drops identical chunks; "
             "'minhash' also removes near-dups (needs datasketch); 'none' keeps all.",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Score the CODE source's chunks (code_scorer) into an aligned "
             ".weights.npy and carry per-chunk quality weights through the mix, "
             "so the combined dataset trains quality-weighted (like "
             "prepare_data --score). Text/math keep neutral weight 1.0.",
    )
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────
    config = Config.from_yaml(args.config)

    ds_path = Path(args.data_sources)
    if not ds_path.exists():
        cli.error("Data sources config not found", str(ds_path))
        sys.exit(1)

    with open(ds_path, encoding="utf-8") as f:
        ds_config = yaml.safe_load(f)

    sources_config = ds_config.get("sources", {})
    requested = set(args.sources.split(",")) if args.sources else None

    # ── Load tokenizer ────────────────────────────────────────────────
    tok_path = args.tokenizer or str(DatasetResolver.get_tokenizer_path(ds_path, config_path=args.config))
    if not Path(tok_path).exists():
        cli.error("Tokenizer not found", tok_path)
        cli.dim(f"  Run: .venv/Scripts/python scripts/train_tokenizer.py --config {args.config}")
        sys.exit(1)

    tokenizer = CodeTokenizer(tok_path)
    seq_len = config.model.max_seq_len
    output_dir = args.output_dir or str(DatasetResolver.get_dataset_dir(ds_path, config_path=args.config))

    cli.header("Multi-Source Data Collection", f"Config: {args.config}")

    # Security config loaded up front: in-stream scanning happens DURING
    # collection (content-level), not just on the tokenized output after.
    scan_config = _load_scan_config()
    scan_stats: dict[str, int] = {}

    collected: list[DatasetInput] = []

    # ── Code ──────────────────────────────────────────────────────────
    code_cfg = sources_config.get("code", {})
    if code_cfg.get("enabled", True) and (requested is None or "code" in requested):
        dataset = code_cfg.get("dataset", "bigcode/starcoderdata")
        # Model config data.languages takes precedence over data_sources.yaml
        from cola_coder.data.dataset_resolver import _read_config_languages
        _cfg_langs = _read_config_languages(args.config)
        languages = _cfg_langs if _cfg_langs is not None else code_cfg.get("languages", ["python", "typescript", "javascript"])
        weight = code_cfg.get("weight", 0.7)

        cli.step(1, 3, f"Collecting code from {dataset}")
        cli.info("Languages", ", ".join(languages))

        code_iter = stream_code_data(
            dataset, languages=languages, max_samples=args.max_samples,
        )
        code_iter = _maybe_scan_stream(code_iter, "code", scan_config, scan_stats)
        # Quality filter the CODE source (the multi-source path previously
        # tokenized raw code — minified/auto-generated/data-dump files included —
        # despite tokenize_and_chunk documenting "already quality-filtered" input.
        # text/math are prose and intentionally NOT run through the code filter.
        filter_mode = args.filter or code_cfg.get("filter", "conservative")
        code_iter = _maybe_quality_filter(code_iter, filter_mode, languages)
        output_path = tokenize_and_chunk(
            code_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="code_data",
        )
        _maybe_dedup(output_path, args.dedup, tokenizer=tokenizer)
        _maybe_score(output_path, args.score, tokenizer)  # after dedup → weights align
        collected.append(DatasetInput(path=output_path, weight=weight, name="code"))
        cli.success(f"Code data saved: {output_path}")

    # ── Text ──────────────────────────────────────────────────────────
    text_cfg = sources_config.get("text", {})
    if text_cfg.get("enabled", True) and (requested is None or "text" in requested):
        dataset = text_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
        weight = text_cfg.get("weight", 0.2)
        min_len = text_cfg.get("min_length", 100)
        max_len = text_cfg.get("max_length", 50_000)

        cli.step(2, 3, f"Collecting text from {dataset}")

        text_iter = stream_hf_text(
            dataset, min_length=min_len, max_length=max_len,
            max_samples=args.max_samples,
        )
        text_iter = _maybe_scan_stream(text_iter, "text", scan_config, scan_stats)
        output_path = tokenize_and_chunk(
            text_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="text_data",
        )
        _maybe_dedup(output_path, args.dedup, tokenizer=tokenizer)
        collected.append(DatasetInput(path=output_path, weight=weight, name="text"))
        cli.success(f"Text data saved: {output_path}")

    # ── Math ──────────────────────────────────────────────────────────
    math_cfg = sources_config.get("math", {})
    if math_cfg.get("enabled", True) and (requested is None or "math" in requested):
        dataset = math_cfg.get("dataset", "open-web-math/open-web-math")
        weight = math_cfg.get("weight", 0.1)
        min_len = math_cfg.get("min_length", 50)
        max_len = math_cfg.get("max_length", 30_000)

        cli.step(3, 3, f"Collecting math from {dataset}")

        math_iter = stream_hf_text(
            dataset, min_length=min_len, max_length=max_len,
            max_samples=args.max_samples,
        )
        math_iter = _maybe_scan_stream(math_iter, "math", scan_config, scan_stats)
        output_path = tokenize_and_chunk(
            math_iter, tokenizer, chunk_size=seq_len,
            output_dir=output_dir, output_name="math_data",
        )
        _maybe_dedup(output_path, args.dedup, tokenizer=tokenizer)
        collected.append(DatasetInput(path=output_path, weight=weight, name="math"))
        cli.success(f"Math data saved: {output_path}")

    # ── Malware scan summary ─────────────────────────────────────────
    if collected:
        # In-stream scan stats (content-level, ran during collection above)
        if scan_stats:
            dropped = scan_stats.get("dropped", 0)
            clean = scan_stats.get("clean", 0)
            if dropped:
                cli.warn(
                    f"In-stream scan: dropped {dropped} record(s) matching "
                    f"threat patterns ({clean} clean)"
                )
            else:
                cli.success(f"In-stream scan: {clean} records clean")

        # Backstop directory scan for any real files in the output dir
        # (the .npy token arrays themselves can't carry textual patterns)
        scan_ok = _scan_downloaded_data(
            Path(output_dir), scan_config, step_num=4, total_steps=4,
        )
        if not scan_ok:
            cli.error("Data collection aborted due to malware scan results.")
            sys.exit(1)

    # ── Combine ───────────────────────────────────────────────────────
    if len(collected) > 1 and not args.no_combine:
        cli.header("Combining Datasets", "Weighted interleaving per data_sources.yaml")

        for ds_input in collected:
            cli.info(ds_input.name, f"weight={ds_input.weight:.0%}, path={ds_input.path}")

        combiner = DatasetCombiner()
        combined_path = str(Path(output_dir) / "mixed_train_data.npy")
        result = combiner.combine(
            collected, strategy="interleave", output_path=combined_path,
            carry_weights=args.score,  # carry code's .weights.npy; text/math → 1.0
        )
        cli.success(f"Combined dataset: {result.output_path}")
        cli.info("Total chunks", f"{result.total_chunks:,}")
        if result.weights_path:
            cli.info("Quality weights", Path(result.weights_path).name)
    elif len(collected) == 1:
        cli.info("Single source", "No combining needed")
    else:
        cli.warn("No data collected. Check data_sources.yaml settings.")

    cli.done("Data collection complete.")


if __name__ == "__main__":
    main()
