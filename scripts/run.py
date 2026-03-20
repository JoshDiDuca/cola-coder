"""Cola-Coder: Dead-simple interactive code generation entry point.

This is THE script you run after training. No flags required — it auto-detects
your latest checkpoint, config, and tokenizer, then drops you into a REPL.

Usage (simplest possible):
    python scripts/run.py

Usage (with overrides):
    python scripts/run.py --preset creative
    python scripts/run.py --checkpoint checkpoints/tiny/step_00017000
    python scripts/run.py --preset precise --max-tokens 512

REPL commands:
    /clear           Reset conversation history
    /info            Show loaded model info
    /preset <name>   Switch sampling preset (precise | balanced | creative)
    /history         Show all (prompt, output) pairs from this session
    /save <file>     Save session history to a text file
    /smoke           Re-run the smoke test suite on demand
    /quit            Exit

For a TypeScript developer: this is like the `next dev` of cola-coder.
You don't need to know the internals — it just runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Sampling presets
#
# Think of these like Prettier config profiles:
#   - precise  = strict, low creativity  (good for completing known patterns)
#   - balanced = middle ground            (good general-purpose default)
#   - creative = high variance            (good for exploring novel code)
#
# Each field controls a different aspect of the "randomness" during generation:
#   temperature      : scales the raw probability distribution. 0 = always pick
#                      the most likely token (greedy). 1.0 = use raw probs.
#                      >1.0 = amplify unlikely tokens. Think of it as "boldness".
#   top_p            : nucleus sampling. Only sample from the smallest set of
#                      tokens whose combined probability exceeds this threshold.
#                      0.9 = ignore the long tail of unlikely options.
#   top_k            : hard cap — only consider the K most likely tokens.
#                      Prevents extremely unlikely tokens even if top_p lets them in.
#   repetition_penalty: multiply the logit of already-generated tokens by 1/penalty,
#                      making them less likely to repeat. 1.0 = no penalty.
# ──────────────────────────────────────────────────────────────────────────────
PRESETS: dict[str, dict[str, float | int]] = {
    "precise": {
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.2,
    },
    "balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "creative": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 100,
        "repetition_penalty": 1.05,
    },
}

PRESET_NAMES = list(PRESETS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detection helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    """Walk up from this script to find the project root (contains pyproject.toml)."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fall back to the script's parent directory
    return here.parent


def auto_detect_checkpoint(checkpoints_dir: Path) -> tuple[str, dict] | None:
    """Scan checkpoints/ for the most recent checkpoint across all model sizes.

    Strategy:
    1. Look for `latest` pointer files (text files written by save_checkpoint).
       These are the most reliable — the trainer always keeps them up to date.
    2. Fall back to scanning step_* directories directly if no pointer files exist.

    Returns (checkpoint_dir_path, metadata_dict) or None if nothing found.

    The metadata dict comes from metadata.json and contains:
        {"step": int, "loss": float, "config": {...}}
    """
    if not checkpoints_dir.exists():
        return None

    # detect_latest_checkpoint already implements exactly this logic — use it
    from cola_coder.training.checkpoint import detect_latest_checkpoint
    result = detect_latest_checkpoint(str(checkpoints_dir))
    if result is None:
        return None
    ckpt_path, metadata = result
    # The path stored in `latest` may be a relative Windows path (e.g.
    # "checkpoints\tiny\step_00017000"). Resolve it relative to the project root.
    resolved = Path(ckpt_path)
    if not resolved.is_absolute():
        resolved = checkpoints_dir.parent / resolved
    return str(resolved), metadata


def auto_detect_config(checkpoint_dir: str, metadata: dict, project_root: Path) -> str | None:
    """Find the best YAML config for a given checkpoint.

    Priority:
    1. Read metadata.json — if it has a "config" key, reconstruct a temporary
       Config object from it (avoids needing the YAML file at all).
       But for file-based loading we still need the YAML path.
    2. Match the checkpoint's grandparent dir name to configs/<name>.yaml.
       e.g. checkpoints/tiny/step_00017000 -> configs/tiny.yaml

    Returns path to a YAML config file, or None if not found.
    """
    ckpt_path = Path(checkpoint_dir)
    # grandparent of step_XXXXXXXX is the size dir (tiny/small/medium/large)
    size_name = ckpt_path.parent.name
    yaml_candidate = project_root / "configs" / f"{size_name}.yaml"
    if yaml_candidate.exists():
        return str(yaml_candidate)
    # Scan configs/ for any YAML file as last resort
    configs_dir = project_root / "configs"
    if configs_dir.exists():
        yamls = sorted(configs_dir.glob("*.yaml"))
        if yamls:
            return str(yamls[0])
    return None


def build_config_from_metadata(metadata: dict):
    """Reconstruct a Config object from the config dict embedded in metadata.json.

    metadata.json["config"] contains the model + training config that was active
    when the checkpoint was saved. This lets us load the model even if the YAML
    file has been deleted or renamed.

    Returns a Config object, or None on failure.
    """
    try:
        from cola_coder.model.config import Config, ModelConfig, TrainingConfig

        raw = metadata.get("config", {})
        model_cfg = ModelConfig(**raw.get("model", {}))
        # DataConfig and CheckpointConfig are not needed for inference
        from cola_coder.model.config import DataConfig, CheckpointConfig
        return Config(
            model=model_cfg,
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(),
            checkpoint=CheckpointConfig(),
        )
    except Exception:
        return None


def get_vram_usage_mb() -> float | None:
    """Return current GPU VRAM usage in MB, or None if not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 2)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Model info display
# ──────────────────────────────────────────────────────────────────────────────

def show_model_info(
    cli,
    *,
    size_name: str,
    params_human: str,
    step: int,
    loss: float,
    device: str,
    preset_name: str,
    checkpoint_dir: str,
) -> None:
    """Print the startup model info table."""
    vram = get_vram_usage_mb()
    vram_str = f"{vram:.0f} MB" if vram is not None else "N/A"

    cli.kv_table(
        {
            "Model size": size_name,
            "Parameters": params_human,
            "Checkpoint step": f"{step:,}",
            "Loss at checkpoint": f"{loss:.4f}",
            "Device": device.upper(),
            "VRAM (after load)": vram_str,
            "Sampling preset": preset_name,
        },
        title="Model Info",
    )


# ──────────────────────────────────────────────────────────────────────────────
# REPL helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_multiline_prompt() -> str | None:
    """Read a multi-line prompt from stdin.

    The user types their prompt and submits by pressing Enter on a blank line.
    First blank line when no input yet = skip (keep looping).

    Returns the prompt string, or None on EOF/Ctrl+C.
    """
    lines: list[str] = []
    print(">>> ", end="", flush=True)
    try:
        while True:
            line = input()
            # Blank line while we already have content = submit
            if line == "" and lines:
                break
            # Blank line on first input = ignore, keep waiting
            if line == "" and not lines:
                print(">>> ", end="", flush=True)
                continue
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        return None

    return "\n".join(lines)


def handle_slash_command(
    command: str,
    *,
    cli,
    generator,
    history: list[tuple[str, str]],
    active_preset: dict[str, float | int],
    preset_name_ref: list[str],   # mutable single-element list for pass-by-ref
    model_info_kwargs: dict,
) -> bool:
    """Handle a /command entered in the REPL.

    Returns True if the session should continue, False to quit.
    The preset_name_ref is a [str] wrapper so we can mutate the caller's name.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
        return False  # Signal to exit the REPL

    elif cmd == "/clear":
        # Reset multi-turn conversation context
        # In multi-turn mode the context is built by concatenating prompts + outputs.
        # The generator itself is stateless — "context" is just what we feed as prompt.
        # So clearing means we reset our accumulated history string in the REPL.
        history.clear()
        # Signal the caller to also clear its context_so_far string
        active_preset["__clear_context"] = True
        cli.success("Conversation history cleared.")

    elif cmd == "/info":
        show_model_info(cli, **model_info_kwargs)

    elif cmd == "/preset":
        if arg in PRESETS:
            new_preset = PRESETS[arg]
            active_preset.update(new_preset)
            preset_name_ref[0] = arg
            # Update the info dict so /info shows the current preset
            model_info_kwargs["preset_name"] = arg
            cli.success(f"Switched to preset: {arg}")
            cli.kv_table(
                {k: str(v) for k, v in new_preset.items()},
                title=f"Preset: {arg}",
            )
        else:
            cli.warn(f"Unknown preset '{arg}'. Choose from: {', '.join(PRESET_NAMES)}")

    elif cmd == "/history":
        if not history:
            cli.warn("No generation history yet.")
        else:
            cli.rule("Generation History")
            for i, (prompt, output) in enumerate(history, 1):
                cli.print(f"\n[bold cyan]#{i} Prompt:[/bold cyan]")
                cli.print(prompt)
                cli.print(f"[bold cyan]#{i} Output:[/bold cyan]")
                cli.print(output)
            cli.rule()

    elif cmd == "/save":
        if not arg:
            cli.warn("Usage: /save <filename>")
        elif not history:
            cli.warn("Nothing to save yet.")
        else:
            try:
                out_path = Path(arg)
                lines: list[str] = []
                for i, (prompt, output) in enumerate(history, 1):
                    lines.append(f"=== #{i} PROMPT ===")
                    lines.append(prompt)
                    lines.append(f"=== #{i} OUTPUT ===")
                    lines.append(output)
                    lines.append("")
                out_path.write_text("\n".join(lines), encoding="utf-8")
                cli.success(f"History saved to: {out_path}")
            except Exception as e:
                cli.error(f"Could not save history: {e}")

    elif cmd == "/smoke":
        # Re-run the smoke test on demand — useful for debugging generation quality
        # mid-session without restarting the REPL.
        try:
            from cola_coder.features.smoke_test import SmokeTest
            SmokeTest().run(generator)
        except Exception as e:
            cli.warn(f"Smoke test failed: {e}")

    else:
        cli.warn(f"Unknown command '{cmd}'. Available: /clear, /info, /preset, /history, /save, /smoke, /quit")

    return True  # Continue the REPL


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from cola_coder.cli import cli

    # ── Argument parsing ──────────────────────────────────────────────────────
    #
    # All flags are OPTIONAL. The defaults auto-detect everything.
    # We only require user input if auto-detection fails.

    parser = argparse.ArgumentParser(
        description=(
            "Cola-Coder interactive code generation — no flags needed.\n"
            "Auto-detects the latest checkpoint, config, and tokenizer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Auto-detectable args (all optional)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory. "
            "Default: auto-detect the latest checkpoint in checkpoints/."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config file. "
            "Default: read from checkpoint metadata, then match configs/<size>.yaml."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json. Default: tokenizer.json in project root.",
    )

    # Sampling preset
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="balanced",
        help=(
            "Sampling preset: precise (conservative), balanced (default), creative (wild). "
            "Individual --temperature/--top-p/--top-k flags override the preset."
        ),
    )

    # Individual sampling overrides (all optional — override preset values)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature override (overrides --preset value).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        help="Top-p sampling threshold override (overrides --preset value).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Top-k sampling threshold override (overrides --preset value).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per turn (default: 256).",
    )

    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    cli.header("Cola-Coder", "Code Generation")

    # ── Project root and paths ────────────────────────────────────────────────
    project_root = find_project_root()
    checkpoints_dir = project_root / "checkpoints"

    # ── Auto-detect checkpoint ────────────────────────────────────────────────
    checkpoint_dir: str
    metadata: dict

    if args.checkpoint is not None:
        # User provided an explicit path — resolve it, read its metadata
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            cli.fatal(
                f"Checkpoint not found: {args.checkpoint}",
                hint="Check the path or omit --checkpoint to auto-detect.",
            )
        checkpoint_dir = str(ckpt_path.resolve())
        meta_path = ckpt_path / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        cli.info("Checkpoint", checkpoint_dir)
    else:
        result = auto_detect_checkpoint(checkpoints_dir)
        if result is None:
            cli.error(
                "No checkpoints found.",
                hint=(
                    f"Train a model first:\n"
                    f"  python scripts/train.py --config configs/tiny.yaml\n"
                    f"\nExpected structure: {checkpoints_dir}/<size>/step_XXXXXXXX/"
                ),
            )
            sys.exit(1)
        checkpoint_dir, metadata = result
        size_name = Path(checkpoint_dir).parent.name
        step = metadata.get("step", "?")
        cli.success(f"Auto-detected checkpoint: {size_name} step {step:,}" if isinstance(step, int) else f"Auto-detected checkpoint: {checkpoint_dir}")

    # ── Auto-detect config ────────────────────────────────────────────────────
    #
    # For inference we only need the model architecture (ModelConfig), not training
    # or data config. We try two strategies in order:
    #   1. Reconstruct from embedded config in metadata.json (most reliable)
    #   2. Load from a YAML config file (fallback if metadata config is missing)

    config = None  # will be a Config object

    if args.config is not None:
        # Explicit YAML path provided
        from cola_coder.model.config import Config
        config_path = Path(args.config)
        if not config_path.exists():
            cli.fatal(f"Config file not found: {args.config}")
        config = Config.from_yaml(str(config_path))
        cli.info("Config", str(config_path))
    else:
        # Try reading from metadata.json first (no file needed)
        config = build_config_from_metadata(metadata)
        if config is not None:
            cli.info("Config", "loaded from checkpoint metadata.json")
        else:
            # Fall back to matching a YAML file by size name
            yaml_path = auto_detect_config(checkpoint_dir, metadata, project_root)
            if yaml_path is None:
                cli.fatal(
                    "Could not auto-detect a config file.",
                    hint="Provide --config configs/<size>.yaml explicitly.",
                )
            from cola_coder.model.config import Config
            config = Config.from_yaml(yaml_path)
            cli.info("Config", yaml_path)

    # ── Auto-detect tokenizer ─────────────────────────────────────────────────
    if args.tokenizer is not None:
        tokenizer_path = str(Path(args.tokenizer).resolve())
    else:
        tokenizer_path = str(project_root / "tokenizer.json")

    if not Path(tokenizer_path).exists():
        cli.fatal(
            f"Tokenizer not found: {tokenizer_path}",
            hint=(
                "Train one first: python scripts/train_tokenizer.py\n"
                "Or provide --tokenizer <path>"
            ),
        )

    # ── Device ────────────────────────────────────────────────────────────────
    device = cli.gpu_info()

    # ── Load model ────────────────────────────────────────────────────────────
    #
    # Loading sequence (TypeScript analogy):
    #   1. Build an "empty" model object with the right architecture shape
    #      (like `new Transformer(config)`)
    #   2. Fill it with the saved weights from the checkpoint
    #      (like `Object.assign(model, savedWeights)`)
    #   3. Move it to GPU memory
    #      (like moving data from disk to a GPU buffer)

    cli.print("\nLoading model...")

    try:
        import torch
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
    except ImportError as e:
        cli.fatal(
            f"Import error: {e}",
            hint="Make sure the package is installed: pip install -e .",
        )

    try:
        # Build skeleton model (random weights, correct shape)
        model = Transformer(config.model).to(device)

        # Fill in the actual trained weights from the checkpoint
        load_model_only(checkpoint_dir, model, device=device)

        # Load tokenizer (maps text <-> token IDs)
        tokenizer = CodeTokenizer(tokenizer_path)

        # Wrap model + tokenizer in the CodeGenerator which handles KV-caching
        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)

    except torch.cuda.OutOfMemoryError:
        cli.fatal(
            "CUDA out of memory while loading the model.",
            hint=(
                "Try:\n"
                "  - Close other GPU applications (check nvidia-smi)\n"
                "  - Use a smaller model size (configs/tiny.yaml)\n"
                "  - Reduce --max-tokens to lower activation memory"
            ),
        )
    except Exception as e:
        cli.fatal(f"Failed to load model: {e}")

    # ── Build sampling preset ─────────────────────────────────────────────────
    #
    # Start from the preset, then apply any individual overrides the user passed.
    # This dict is mutable — /preset <name> will update it in-place.
    active_preset: dict[str, float | int] = dict(PRESETS[args.preset])
    preset_name_ref = [args.preset]  # mutable wrapper so REPL can update it

    if args.temperature is not None:
        active_preset["temperature"] = args.temperature
    if args.top_p is not None:
        active_preset["top_p"] = args.top_p
    if args.top_k is not None:
        active_preset["top_k"] = args.top_k

    # ── Show startup info ─────────────────────────────────────────────────────
    ckpt_step = metadata.get("step", 0)
    ckpt_loss = metadata.get("loss", float("nan"))
    size_name = Path(checkpoint_dir).parent.name

    model_info_kwargs = dict(
        size_name=size_name,
        params_human=config.model.total_params_human,
        step=ckpt_step,
        loss=ckpt_loss,
        device=device,
        preset_name=preset_name_ref[0],
        checkpoint_dir=checkpoint_dir,
    )
    show_model_info(cli, **model_info_kwargs)

    # ── Smoke test (optional) ─────────────────────────────────────────────────
    #
    # Quick sanity-check that the model actually generates tokens before we hand
    # control to the user. Runs the full SmokeTest suite (5 TypeScript prompts)
    # only when the feature flag is on. Any failure is caught and printed as a
    # warning — it must never prevent the REPL from starting.

    try:
        from cola_coder.features.smoke_test import is_enabled as smoke_test_enabled, SmokeTest
        if smoke_test_enabled():
            cli.dim("Running smoke test (set FEATURE_ENABLED=False in smoke_test.py to skip)...")
            SmokeTest().run(generator)
    except Exception as e:
        cli.warn(f"Smoke test skipped: {e}")

    cli.success("Ready! Enter a prompt and press Enter on a blank line to submit.")
    cli.print("Commands: [dim]/clear  /info  /preset <name>  /history  /save <file>  /smoke  /quit[/dim]")
    cli.print()

    # ── REPL ──────────────────────────────────────────────────────────────────
    #
    # Multi-turn mode: we accumulate a "context" string by appending each
    # (prompt + output) pair with a newline separator. On the next turn we
    # feed the full context as the prompt, so the model "remembers" prior turns.
    #
    # This is conceptually like a chat history: each new message is appended
    # to the thread, and the model sees the whole conversation every time.
    # The downside is the context grows until it hits max_seq_len — use /clear
    # to reset when you start a new topic.

    history: list[tuple[str, str]] = []  # (user_prompt, model_output)
    context_so_far: str = ""             # accumulated conversation string

    while True:
        try:
            user_prompt = read_multiline_prompt()
        except KeyboardInterrupt:
            cli.done("Session ended.")
            break

        if user_prompt is None:
            # EOF or Ctrl+C
            cli.done("Session ended.")
            break

        if not user_prompt.strip():
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_prompt.strip().startswith("/"):
            should_continue = handle_slash_command(
                user_prompt.strip(),
                cli=cli,
                generator=generator,
                history=history,
                active_preset=active_preset,
                preset_name_ref=preset_name_ref,
                model_info_kwargs=model_info_kwargs,
            )
            # /clear sets a sentinel to also reset our context string
            if active_preset.pop("__clear_context", False):
                context_so_far = ""
            if not should_continue:
                cli.done("Session ended.")
                break
            continue

        # ── Build full context for this turn ─────────────────────────────────
        #
        # Multi-turn: prepend everything we've seen so far.
        # First turn: context_so_far is empty, so we just use the user's prompt.
        full_prompt = (context_so_far + "\n" + user_prompt) if context_so_far else user_prompt

        # ── Generate ─────────────────────────────────────────────────────────
        cli.rule("Generating")
        try:
            # Extract sampling params from the active preset dict.
            # The __clear_context sentinel might be absent — use .get() safely.
            output = generator.generate(
                prompt=full_prompt,
                max_new_tokens=args.max_tokens,
                temperature=float(active_preset["temperature"]),
                top_p=float(active_preset["top_p"]),
                top_k=int(active_preset["top_k"]),
                repetition_penalty=float(active_preset["repetition_penalty"]),
            )

        except torch.cuda.OutOfMemoryError:
            cli.error(
                "CUDA out of memory during generation.",
                hint=f"Try: --max-tokens {args.max_tokens // 2}  or  /clear to reset context",
            )
            cli.rule()
            print()
            continue
        except Exception as e:
            cli.error(f"Generation error: {e}")
            cli.rule()
            print()
            continue

        # The generator returns prompt + generated text. Strip the prompt portion
        # so we only print and store the new content.
        new_text = output[len(full_prompt):] if output.startswith(full_prompt) else output
        print(new_text)
        cli.rule("End")
        print()

        # ── Update history and context ────────────────────────────────────────
        history.append((user_prompt, new_text))

        # Accumulate context for multi-turn: append "prompt\noutput" to the thread
        if context_so_far:
            context_so_far = context_so_far + "\n" + user_prompt + "\n" + new_text
        else:
            context_so_far = user_prompt + "\n" + new_text


if __name__ == "__main__":
    main()
