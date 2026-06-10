"""Interactive code generation CLI.

Loads a trained model from a checkpoint and provides a REPL for generating
code interactively. Enter a prompt, press Enter twice to submit, and the
model will generate a continuation.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml
    python scripts/generate.py --checkpoint ./checkpoints/step_00010000 --config configs/small.yaml --temperature 0.5
"""

import argparse
from pathlib import Path

from cola_coder.cli import cli
from cola_coder.model.config import get_storage_config


def main():
    storage = get_storage_config()

    parser = argparse.ArgumentParser(
        description="Interactive code generation using a trained cola-coder model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (required).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (required).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json (auto-resolved from data sources config if omitted).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature: 0 = greedy, higher = more random (default: 0.8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (default: 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold (default: 50).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository directory for context-aware generation (optional).",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generate N candidates, verify each in a sandbox (tsc/syntax), "
             "return the best (default: 1 = off).",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["auto", "python", "typescript"],
        default="auto",
        help="Verifier language for --best-of (default: auto-detect from prompt).",
    )
    args = parser.parse_args()

    if args.tokenizer is None:
        # Priority 1: checkpoint metadata (most reliable — saved during training)
        meta_path = Path(args.checkpoint) / "metadata.json"
        if meta_path.exists():
            try:
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                saved_tok = meta.get("tokenizer_path", "")
                if saved_tok and Path(saved_tok).exists():
                    args.tokenizer = saved_tok
            except (json.JSONDecodeError, OSError):
                pass

        # Priority 2: DatasetResolver with config (matches training tokenizer)
        if args.tokenizer is None:
            try:
                from cola_coder.data.dataset_resolver import DatasetResolver
                args.tokenizer = (
                    str(DatasetResolver.get_tokenizer_path(config_path=args.config))
                    if DatasetResolver.tokenizer_exists(config_path=args.config)
                    else storage.tokenizer_path
                )
            except Exception:
                args.tokenizer = storage.tokenizer_path

    cli.header("Cola-Coder", "Code Generation")

    # ---- Validate inputs ----
    if not Path(args.checkpoint).exists():
        cli.fatal(f"Checkpoint not found: {args.checkpoint}", hint="Check the path")

    if not Path(args.config).exists():
        cli.fatal(f"Config file not found: {args.config}", hint="Check the path")

    if not Path(args.tokenizer).exists():
        cli.fatal(f"Tokenizer file not found: {args.tokenizer}", hint="Check the path")

    # ---- Determine device ----
    device = cli.gpu_info()

    # ---- Load model ----
    cli.print("Loading model...")

    try:
        from cola_coder.model.config import Config
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError:
        cli.fatal(
            "Could not import cola_coder. Make sure the package is installed.",
            hint="Try: pip install -e .",
        )

    try:
        config = Config.from_yaml(args.config)

        # Auto-detect vocab size from checkpoint — SFT checkpoints have extra ChatML tokens
        ckpt_vocab_size = config.model.vocab_size
        try:
            from safetensors import safe_open
            with safe_open(
                str(Path(args.checkpoint) / "model.safetensors"),
                framework="pt",
            ) as f:
                if "tok_emb.weight" in f.keys():
                    ckpt_vocab_size = f.get_tensor("tok_emb.weight").shape[0]
        except Exception:
            pass

        if ckpt_vocab_size != config.model.vocab_size:
            cli.dim(f"  Checkpoint vocab: {ckpt_vocab_size} (config: {config.model.vocab_size})")
            config.model.vocab_size = ckpt_vocab_size

        cli.info("Model", f"{config.model.total_params_human} parameters")

        tokenizer = CodeTokenizer(args.tokenizer)
        cli.info("Tokenizer", f"{tokenizer.vocab_size} tokens")

        model = Transformer(config.model).to(device)
        load_model_only(args.checkpoint, model, device=device)
        cli.info("Checkpoint", args.checkpoint)
        cli.info("Device", device)

        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)

        # Optional repo context: scan once, prepend a context block per prompt.
        # The base CodeGenerator does the generating (ContextAwareGenerator's
        # generate() needs a per-call target file_path, which the REPL collects
        # interactively below) — this also lets --repo compose with --best-of.
        scanner = None
        if args.repo:
            from cola_coder.inference.repo_context import RepoScanner

            repo_root = Path(args.repo)
            if not repo_root.is_dir():
                cli.fatal(f"Repo directory not found: {repo_root}")
            scanner = RepoScanner(repo_root)
            scanner.scan()
            cli.info("Repo context", str(repo_root))
    except Exception as e:
        cli.fatal(f"Loading model: {e}")

    # ---- REPL loop ----
    cli.success("Code generation ready!")
    cli.info("Temperature", args.temperature)
    cli.info("Max tokens", args.max_tokens)
    cli.info("Top-p", f"{args.top_p}, Top-k: {args.top_k}")
    if args.best_of > 1:
        cli.info("Best-of", f"{args.best_of} candidates, verifier language: {args.language}")
    cli.print("\nEnter a prompt, then press Enter on an empty line to submit.")
    cli.print("Press Ctrl+C to exit.\n")

    while True:
        try:
            # Read multiline input
            lines = []
            print(">>> ", end="", flush=True)
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)

            if not lines:
                continue

            prompt = "\n".join(lines)

            # Repo mode: ask which file is being completed so the context
            # block can pull its imports + similar files (Enter to skip).
            if scanner is not None:
                target = input("Target file for context (Enter to skip): ").strip()
                context_str = scanner.get_context_for_file(target) if target else ""
                if context_str:
                    cli.dim(f"  [repo context: {len(context_str)} chars]")
                prompt = context_str + prompt

            # Generate
            cli.rule("Generating")
            if args.best_of > 1:
                from cola_coder.inference.best_of_n import generate_best_of_n

                bon = generate_best_of_n(
                    generator,
                    prompt,
                    num_candidates=args.best_of,
                    language=args.language,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(bon.best.text)
                cli.rule("Candidates")
                for i, cand in enumerate(bon.candidates):
                    status = "PASS" if cand.verified else "fail"
                    cli.dim(
                        f"  [{i + 1}] {status}  score {cand.score:.3f}  "
                        f"({bon.verifier}, {bon.language})"
                    )
            else:
                result = generator.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(result)
            cli.rule("End")
            print()

        except KeyboardInterrupt:
            cli.done("Session ended.")
            break
        except EOFError:
            cli.done("Session ended.")
            break
        except Exception as e:
            cli.error(f"Generation error: {e}")
            print()


if __name__ == "__main__":
    main()
