"""Interactive CLI for generating instruction-tuning data via SelfCodeAlign.

Walks through three steps:
  1. Choose a source of raw code (prepared dataset, local directory, etc.)
  2. Choose a generation mode (template, LLM API, self-instruct)
  3. Set output options (count, file path)

Then generates instruction-solution pairs and saves them as JSONL.

Usage:
    python scripts/generate_instructions.py
    python scripts/generate_instructions.py --non-interactive \\
        --source local --paths ./my-code --mode template --count 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


# ---------------------------------------------------------------------------
# Menu helpers (simple numbered input — no arrow keys needed)
# ---------------------------------------------------------------------------

def _ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    """Ask the user to pick from numbered options. Returns 1-based index."""
    cli.print()
    cli.print(f"  [bold]{prompt}[/bold]")
    for i, opt in enumerate(options, 1):
        marker = " [yellow](default)[/yellow]" if i == default else ""
        cli.print(f"    [cyan][{i}][/cyan] {opt}{marker}")
    cli.print()

    while True:
        try:
            raw = input(f"  Choose [1-{len(options)}] (default {default}): ").strip()
            if not raw:
                return default
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except (ValueError, EOFError):
            pass
        cli.print(f"  [red]Please enter a number 1-{len(options)}[/red]")


def _ask_input(prompt: str, default: str = "") -> str:
    """Ask for free-form text input with a default."""
    suffix = f" (default: {default})" if default else ""
    try:
        raw = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _ask_int(prompt: str, default: int = 1000) -> int:
    """Ask for an integer with a default."""
    while True:
        raw = _ask_input(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            cli.print("  [red]Please enter a valid number.[/red]")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    cli.header("SelfCodeAlign", "Instruction Data Generator")


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    """Run the interactive menu flow."""
    storage = get_storage_config()
    _print_banner()

    # --- Step 1: Source ---
    cli.print()
    cli.print("  [bold]Step 1/3 - Source Code[/bold]")
    source_choice = _ask_choice(
        "Where should raw code come from?",
        [
            "From local directory  -- Use local code files",
            "From prepared dataset -- Use existing .npy or JSONL file",
            "From inline examples  -- Small built-in demo set",
        ],
        default=1,
    )

    source_paths: list[str] = []
    source_dataset: str | None = None
    source_type: str | None = None

    if source_choice == 1:
        path = _ask_input("Directory path", str(Path(storage.data_dir) / "raw"))
        source_paths = [path]
        source_type = "local"
    elif source_choice == 2:
        source_dataset = _ask_input("Dataset path or HF name", "bigcode/starcoderdata")
        source_type = "huggingface"
    else:
        source_type = None  # Will use inline demo

    # --- Step 2: Generation Mode ---
    cli.print()
    cli.print("  [bold]Step 2/3 - Generation Mode[/bold]")
    mode_choice = _ask_choice(
        "How should instructions be generated?",
        [
            "Template      -- Use templates (free, fast, decent quality)",
            "LLM API       -- Use Claude/GPT (best quality, needs API key)",
            "Self-Instruct -- Use trained model (needs base model checkpoint)",
        ],
        default=1,
    )
    mode = ["template", "llm", "self"][mode_choice - 1]

    # --- Step 3: Output ---
    cli.print()
    cli.print("  [bold]Step 3/3 - Output[/bold]")
    count = _ask_int("How many examples?", 1000)
    language = _ask_input("Language", "typescript")
    min_quality = float(_ask_input("Minimum quality score (0.0-1.0)", "0.5"))
    output_path = _ask_input(
        "Output file",
        str(Path(storage.data_dir) / "processed" / f"instructions_{language}_{count}.jsonl"),
    )

    # --- Run ---
    _run_pipeline(
        source_type=source_type,
        source_paths=source_paths,
        source_dataset=source_dataset,
        mode=mode,
        count=count,
        languages=[language],
        min_quality=min_quality,
        output_path=output_path,
    )


def _run_pipeline(
    source_type: str | None,
    source_paths: list[str],
    source_dataset: str | None,
    mode: str,
    count: int,
    languages: list[str],
    min_quality: float,
    output_path: str,
) -> None:
    """Run the generation pipeline and save results."""
    from cola_coder.data.sources.self_align import SelfAlignPipeline

    cli.print()
    cli.print("[bold]Generating instruction data...[/bold]")
    cli.print(f"  Mode: {mode}")
    cli.print(f"  Target: {count} examples")
    cli.print(f"  Languages: {', '.join(languages)}")
    cli.print(f"  Min quality: {min_quality}")
    cli.print()

    start = time.time()
    examples: list = []

    if source_type == "huggingface" and source_dataset:
        # Run a separate pipeline per language — seed extraction and instruction
        # templates are language-specific, so mixing languages in one pipeline
        # produces wrong results (e.g. Python extractor on TypeScript code).
        try:
            from cola_coder.data.sources.huggingface import HuggingFaceSource
        except ImportError:
            cli.fatal("HuggingFace source requires datasets package.")

        per_lang = max(1, count // len(languages))
        for lang in languages:
            cli.dim(f"  [{lang}] target {per_lang} examples...")
            src = HuggingFaceSource(dataset=source_dataset, languages=[lang])
            pl = SelfAlignPipeline(
                source=src, mode=mode, min_quality=min_quality, language=lang,
            )
            examples.extend(pl.generate(max_examples=per_lang))
        examples = examples[:count]

    elif source_type == "local" and source_paths:
        from cola_coder.data.sources.local import LocalFileSource
        ext_map = {
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "python": [".py"],
        }
        extensions: list[str] = []
        for lang in languages:
            extensions.extend(ext_map.get(lang, []))
        # Deduplicate while preserving order
        extensions = list(dict.fromkeys(extensions))
        inner_source = LocalFileSource(paths=source_paths, extensions=extensions)
        # Seed extraction uses the first language as the primary heuristic
        pipeline = SelfAlignPipeline(
            source=inner_source, mode=mode, min_quality=min_quality, language=languages[0],
        )
        examples = pipeline.generate(max_examples=count)

    else:
        # Demo mode: only 6 built-in examples — warn loudly if a source was requested
        if source_type is not None:
            cli.warn(
                f"Source '{source_type}' did not produce a data reader "
                "(missing --dataset or --paths?). Falling back to 6 demo examples."
            )
        pipeline = SelfAlignPipeline(mode=mode, min_quality=min_quality, language=languages[0])
        examples = _generate_demo_examples(pipeline, count)

    elapsed = time.time() - start

    cli.print()
    cli.success(f"Generated {len(examples)} examples in {elapsed:.1f}s")

    if not examples:
        cli.warn("No examples generated. Check your source path and settings.")
        return

    # Show quality distribution
    scores = [ex.quality_score for ex in examples]
    avg_score = sum(scores) / len(scores) if scores else 0
    cli.info("Average quality", f"{avg_score:.2f}")
    cli.info("Quality range", f"{min(scores):.2f} - {max(scores):.2f}")

    # Show a sample
    cli.print()
    cli.print("[bold]Sample example:[/bold]")
    sample = examples[0]
    cli.print(f"  [cyan]Instruction:[/cyan] {sample.instruction[:120]}...")
    cli.print(f"  [cyan]Output:[/cyan] {sample.output[:120]}...")
    cli.print(f"  [cyan]Quality:[/cyan] {sample.quality_score:.2f}")

    # Save in ChatML format expected by train_sft.py.
    # Each example is converted from {instruction, output, quality_score} to
    # {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} so
    # that Stage 6 (SFT instruction tuning) can consume it directly.
    _save_chatml_jsonl(examples, output_path)
    cli.print()
    cli.success(f"Saved to {output_path} (ChatML format)")
    cli.info("Size", f"{len(examples)} examples, {Path(output_path).stat().st_size / 1024:.1f} KB")


def _save_chatml_jsonl(examples: list, output_path: str) -> None:
    """Save instruction examples as ChatML JSONL for train_sft.py.

    Converts each InstructionExample to the ``{"messages": [...]}`` format
    that ``train_sft.py`` and ``SFTDataset`` expect.  Each example becomes a
    two-turn conversation: a user turn with the instruction and an assistant
    turn with the solution.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "messages": [
                    {"role": "user", "content": ex.instruction},
                    {"role": "assistant", "content": ex.output},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _generate_demo_examples(
    pipeline: object, count: int
) -> list:
    """Generate examples from built-in demo code snippets."""
    demo_code = [
        # TypeScript function examples
        '''\
export function debounce<T extends (...args: any[]) => void>(
    fn: T,
    delay: number
): (...args: Parameters<T>) => void {
    let timer: ReturnType<typeof setTimeout> | null = null;
    return (...args: Parameters<T>) => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}''',
        '''\
async function fetchWithRetry(
    url: string,
    retries: number = 3
): Promise<Response> {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url);
            if (res.ok) return res;
        } catch (err) {
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
        }
    }
    throw new Error("All retries exhausted");
}''',
        '''\
function groupBy<T, K extends string>(
    items: T[],
    keyFn: (item: T) => K
): Record<K, T[]> {
    return items.reduce((acc, item) => {
        const key = keyFn(item);
        (acc[key] ??= []).push(item);
        return acc;
    }, {} as Record<K, T[]>);
}''',
        '''\
class EventEmitter<Events extends Record<string, any[]>> {
    private listeners = new Map<keyof Events, Set<Function>>();

    on<E extends keyof Events>(event: E, listener: (...args: Events[E]) => void): void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(listener);
    }

    emit<E extends keyof Events>(event: E, ...args: Events[E]): void {
        const handlers = this.listeners.get(event);
        if (handlers) {
            for (const handler of handlers) {
                handler(...args);
            }
        }
    }

    off<E extends keyof Events>(event: E, listener: (...args: Events[E]) => void): void {
        this.listeners.get(event)?.delete(listener);
    }
}''',
        '''\
interface PaginatedResponse<T> {
    data: T[];
    total: number;
    page: number;
    pageSize: number;
    hasNext: boolean;
    hasPrevious: boolean;
}''',
        '''\
function memoize<Args extends any[], R>(
    fn: (...args: Args) => R,
    keyFn?: (...args: Args) => string
): (...args: Args) => R {
    const cache = new Map<string, R>();
    return (...args: Args) => {
        const key = keyFn ? keyFn(...args) : JSON.stringify(args);
        if (cache.has(key)) return cache.get(key)!;
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}''',
    ]

    examples = []
    for code in demo_code:
        if len(examples) >= count:
            break
        new_ex = pipeline.generate_from_code(code)
        examples.extend(new_ex)

    return examples[:count]


# ---------------------------------------------------------------------------
# Non-interactive (CLI flags) mode
# ---------------------------------------------------------------------------

def run_cli(args: argparse.Namespace) -> None:
    """Run in non-interactive mode from CLI arguments."""
    _run_pipeline(
        source_type=args.source,
        source_paths=args.paths or [],
        source_dataset=args.dataset,
        mode=args.mode,
        count=args.count,
        languages=args.languages,
        min_quality=args.min_quality,
        output_path=args.output,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Generate instruction-tuning data from raw code (SelfCodeAlign)."
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Run without interactive menus (use CLI flags instead).",
    )
    parser.add_argument("--source", choices=["local", "huggingface", "demo"], default=None)
    parser.add_argument("--paths", nargs="*", help="Local directory paths (for --source local).")
    parser.add_argument("--dataset", default=None, help="HF dataset name (for --source huggingface).")
    parser.add_argument("--mode", choices=["template", "llm", "self"], default="template")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument(
        "--languages", nargs="+", default=["typescript"],
        help="Programming language(s). Example: --languages typescript python",
    )
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--output", default=str(Path(storage.data_dir) / "processed" / "instructions.jsonl"))

    args = parser.parse_args()

    if args.non_interactive:
        if args.source is None:
            args.source = "demo"
        run_cli(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
