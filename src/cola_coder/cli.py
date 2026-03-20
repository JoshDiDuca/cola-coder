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


# Singleton — import and use directly
cli = CLI()
