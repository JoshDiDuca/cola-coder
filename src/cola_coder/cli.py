"""Shared CLI styling for all cola-coder scripts.

Provides consistent rich-formatted output across all entry points.
Falls back to plain print() if rich is not installed.

Usage in scripts:
    from cola_coder.cli import cli

    cli.header("Cola-Coder", "Training")
    cli.step(1, 3, "Loading tokenizer")
    cli.info("Vocabulary size", "32,768")
    cli.success("Training complete!")
    cli.error("Config file not found", hint="Check the path")
    cli.warn("No GPU detected")
    cli.done("Output saved to ./data/processed/train_data.npy")
"""

from __future__ import annotations

import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    # Force UTF-8 on Windows to avoid cp1252 encoding errors with symbols
    import sys
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        if hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore


class CLI:
    """Consistent CLI output for cola-coder scripts."""

    # ── Branding ──────────────────────────────────────────────────────────

    def header(self, title: str, subtitle: str = "") -> None:
        """Print the app header banner."""
        if _HAS_RICH:
            text = Text()
            text.append(f" {title}", style="bold cyan")
            if subtitle:
                text.append(f"  {subtitle}", style="bold white")
            _console.print(Panel(
                text, box=box.HEAVY, style="cyan", padding=(0, 1),
            ))
        else:
            line = f"═══ {title}"
            if subtitle:
                line += f" — {subtitle}"
            line += " ═══"
            print(line)

    # ── Steps & Progress ──────────────────────────────────────────────────

    def step(self, current: int, total: int, message: str) -> None:
        """Print a step indicator: Step 1/3 · Loading tokenizer"""
        if _HAS_RICH:
            _console.print(
                f"\n[bold cyan]Step {current}/{total}[/bold cyan]"
                f" [dim]·[/dim] [bold]{message}[/bold]"
            )
        else:
            print(f"\nStep {current}/{total} · {message}")

    def substep(self, message: str) -> None:
        """Print an indented sub-step."""
        if _HAS_RICH:
            _console.print(f"  [bold cyan]·[/bold cyan] {message}")
        else:
            print(f"  · {message}")

    # ── Key/value info ────────────────────────────────────────────────────

    def info(self, key: str, value: str | int | float) -> None:
        """Print a key: value pair."""
        if _HAS_RICH:
            _console.print(f"  [cyan]{key}:[/cyan] {value}")
        else:
            print(f"  {key}: {value}")

    def kv_table(self, items: dict[str, str], title: str = "") -> None:
        """Print a formatted key-value table."""
        if _HAS_RICH:
            table = Table(
                show_header=False, box=box.SIMPLE_HEAVY,
                padding=(0, 2), title=f"[bold]{title}[/bold]" if title else None,
                title_style="bold white",
            )
            table.add_column("Key", style="cyan", width=20)
            table.add_column("Value", style="white")
            for k, v in items.items():
                table.add_row(k, str(v))
            _console.print(table)
        else:
            if title:
                print(f"\n{title}")
                print("─" * 40)
            for k, v in items.items():
                print(f"  {k}: {v}")

    # ── Status messages ───────────────────────────────────────────────────

    def success(self, message: str) -> None:
        """Print a success message."""
        if _HAS_RICH:
            _console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def error(self, message: str, hint: str = "") -> None:
        """Print an error message and optional hint."""
        if _HAS_RICH:
            _console.print(f"[bold red]✗ Error:[/bold red] {message}")
            if hint:
                _console.print(f"  [dim]{hint}[/dim]")
        else:
            print(f"✗ Error: {message}")
            if hint:
                print(f"  {hint}")

    def warn(self, message: str) -> None:
        """Print a warning message."""
        if _HAS_RICH:
            _console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        else:
            print(f"⚠ {message}")

    def dim(self, message: str) -> None:
        """Print a dimmed/secondary message."""
        if _HAS_RICH:
            _console.print(f"  [dim]{message}[/dim]")
        else:
            print(f"  {message}")

    # ── Completion ────────────────────────────────────────────────────────

    def done(self, message: str, extras: dict[str, str] | None = None) -> None:
        """Print a completion panel with optional extra info."""
        if _HAS_RICH:
            body = f"[bold green]✓ {message}[/bold green]"
            if extras:
                body += "\n"
                for k, v in extras.items():
                    body += f"\n  [cyan]{k}:[/cyan] {v}"
            _console.print(Panel(
                body, box=box.ROUNDED, padding=(1, 2),
                title="[bold]Complete[/bold]",
            ))
        else:
            print(f"\n✓ {message}")
            if extras:
                for k, v in extras.items():
                    print(f"  {k}: {v}")

    # ── GPU info ──────────────────────────────────────────────────────────

    def gpu_info(self) -> str:
        """Print GPU info and return the device string."""
        try:
            import torch
        except ImportError:
            self.warn("PyTorch not installed")
            return "cpu"

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = (getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)) / 1e9
            if _HAS_RICH:
                _console.print(
                    f"  [cyan]GPU:[/cyan] {name} "
                    f"[dim]({vram:.1f} GB VRAM)[/dim]"
                )
            else:
                print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
        else:
            self.warn("No CUDA GPU detected — running on CPU (slow)")
            return "cpu"

    # ── Utilities ─────────────────────────────────────────────────────────

    def fatal(self, message: str, hint: str = "") -> None:
        """Print error and exit."""
        self.error(message, hint)
        sys.exit(1)

    def rule(self, title: str = "") -> None:
        """Print a horizontal rule."""
        if _HAS_RICH:
            _console.rule(title, style="dim")
        else:
            if title:
                print(f"─── {title} ───")
            else:
                print("─" * 40)

    def print(self, *args, **kwargs) -> None:
        """Pass-through to rich console or plain print."""
        if _HAS_RICH:
            _console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    # ── Interactive menus ──────────────────────────────────────────────────

    def choose(
        self,
        prompt: str,
        options: list[dict[str, str]],
        *,
        allow_cancel: bool = False,
    ) -> int | None:
        """Show a numbered menu and return the selected index.

        Args:
            prompt: Title shown above the options.
            options: List of dicts with 'label' and optional 'detail' keys.
                     Each option is rendered as a numbered row.
            allow_cancel: If True, adds a "Cancel" option and returns None
                          if selected.

        Returns:
            Index of selected option, or None if cancelled.
        """
        if _HAS_RICH:
            _console.print()
            table = Table(
                box=box.ROUNDED, show_header=True, header_style="bold cyan",
                padding=(0, 1), title=f"[bold]{prompt}[/bold]",
                title_style="bold white",
            )
            table.add_column("#", style="bold cyan", width=4, justify="right")
            table.add_column("Option", style="bold white")
            table.add_column("Details", style="dim")

            for i, opt in enumerate(options):
                table.add_row(
                    str(i + 1),
                    opt.get("label", ""),
                    opt.get("detail", ""),
                )
            if allow_cancel:
                table.add_row(
                    str(len(options) + 1),
                    "[dim]Cancel[/dim]",
                    "",
                )
            _console.print(table)
            _console.print()
        else:
            print(f"\n{prompt}")
            print("─" * 40)
            for i, opt in enumerate(options):
                detail = f"  ({opt['detail']})" if opt.get("detail") else ""
                print(f"  {i + 1}) {opt.get('label', '')}{detail}")
            if allow_cancel:
                print(f"  {len(options) + 1}) Cancel")
            print()

        max_choice = len(options) + (1 if allow_cancel else 0)
        while True:
            try:
                raw = input("  Select [1-{}]: ".format(max_choice)).strip()
                choice = int(raw)
                if 1 <= choice <= max_choice:
                    if allow_cancel and choice == max_choice:
                        return None
                    return choice - 1
            except (ValueError, EOFError):
                pass
            if _HAS_RICH:
                _console.print(f"  [red]Please enter a number 1-{max_choice}[/red]")
            else:
                print(f"  Please enter a number 1-{max_choice}")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Ask a yes/no question and return the answer.

        Args:
            prompt: The question to ask.
            default: Default value if user just presses Enter.

        Returns:
            True for yes, False for no.
        """
        suffix = "[Y/n]" if default else "[y/N]"
        if _HAS_RICH:
            _console.print(f"\n  [bold]{prompt}[/bold] [dim]{suffix}[/dim] ", end="")
        else:
            print(f"\n  {prompt} {suffix} ", end="")

        try:
            raw = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default

        if raw in ("y", "yes"):
            return True
        elif raw in ("n", "no"):
            return False
        return default

    def file_table(
        self,
        title: str,
        files: list[dict[str, str]],
    ) -> None:
        """Show a table of files with metadata.

        Args:
            title: Table title.
            files: List of dicts with keys: name, size, date, detail (all str).
        """
        if _HAS_RICH:
            table = Table(
                box=box.ROUNDED, show_header=True, header_style="bold cyan",
                padding=(0, 1), title=f"[bold]{title}[/bold]",
                title_style="bold white",
            )
            table.add_column("#", style="bold cyan", width=4, justify="right")
            table.add_column("Dataset", style="bold white")
            table.add_column("Size", style="green", justify="right")
            table.add_column("Created", style="dim")
            table.add_column("Details", style="dim")

            for i, f in enumerate(files):
                table.add_row(
                    str(i + 1),
                    f.get("name", ""),
                    f.get("size", ""),
                    f.get("date", ""),
                    f.get("detail", ""),
                )
            _console.print()
            _console.print(table)
            _console.print()
        else:
            print(f"\n{title}")
            print("─" * 60)
            for i, f in enumerate(files):
                print(f"  {i + 1}) {f.get('name', '')}  "
                      f"({f.get('size', '')}  {f.get('date', '')})")
            print()


# Singleton — import and use directly
cli = CLI()
