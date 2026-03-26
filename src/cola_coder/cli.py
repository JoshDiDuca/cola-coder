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

import re
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

# ── Language constants ─────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = [
    {"label": "TypeScript",       "detail": "Core TS — .ts files, Node.js, libraries",    "slug": "typescript"},
    {"label": "TypeScript React", "detail": "React/Next.js — .tsx files with JSX + hooks", "slug": "typescript-react"},
    {"label": "JavaScript",       "detail": "ES6+ — .js/.mjs/.cjs, browser + Node.js",    "slug": "javascript"},
    {"label": "JavaScript React", "detail": "React — .jsx files with JSX components",      "slug": "javascript-react"},
    {"label": "Python",           "detail": "General purpose + ML/data science",            "slug": "python"},
    {"label": "Go",               "detail": "Systems, cloud, CLI tools",                    "slug": "go"},
    {"label": "Java",             "detail": "Enterprise, Android, Spring",                  "slug": "java"},
    {"label": "Rust",             "detail": "Systems programming, safety-first",            "slug": "rust"},
    {"label": "C++",              "detail": "Performance-critical, games, systems",         "slug": "cpp"},
    {"label": "C",                "detail": "Low-level systems, embedded",                  "slug": "c"},
]

EXTENSION_TO_LANG: dict[str, str] = {
    ".ts": "typescript", ".tsx": "typescript-react",
    ".js": "javascript", ".jsx": "javascript-react",
    ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python", ".pyw": "python",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".c": "c", ".h": "c",
}

LANG_TO_EXTENSIONS: dict[str, list[str]] = {
    "typescript": [".ts"],
    "typescript-react": [".tsx"],
    "javascript": [".js", ".mjs", ".cjs"],
    "javascript-react": [".jsx"],
    "python": [".py", ".pyw"],
    "go": [".go"],
    "java": [".java"],
    "rust": [".rs"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp"],
    "c": [".c", ".h"],
}

# HF datasets group tsx under "typescript" dir — we post-filter
HF_LANG_MAP: dict[str, list[str]] = {
    "typescript": ["typescript", "typescript-react"],
    "javascript": ["javascript", "javascript-react"],
    "python": ["python"],
    "go": ["go"],
    "java": ["java"],
    "rust": ["rust"],
    "cpp": ["cpp"],
    "c": ["c"],
}

# ── Framework detection ────────────────────────────────────────────────────────

_REACT_IMPORT_RE = re.compile(
    r"""(?:from\s+['"]react['"]|import\s+React|from\s+['"]next/|"""
    r"""require\s*\(\s*['"]react['"])""",
    re.MULTILINE,
)


def detect_framework_language(content: str, file_path: str = "") -> str:
    """Detect framework-language from extension + content.

    .tsx -> typescript-react, .jsx -> javascript-react.
    .ts/.js with React imports -> upgrade to -react variant.
    """
    ext = ""
    if file_path:
        from pathlib import Path

        ext = Path(file_path).suffix.lower()

    # Extension-based detection
    base_lang = EXTENSION_TO_LANG.get(ext, "")
    if base_lang:
        # tsx/jsx already map to -react variants
        if base_lang in ("typescript-react", "javascript-react"):
            return base_lang
        # For .ts/.js, check content for React imports
        if base_lang in ("typescript", "javascript") and _REACT_IMPORT_RE.search(content):
            return f"{base_lang}-react"
        return base_lang

    return ""


def _read_key() -> str:
    """Read a single keypress. Returns arrow names or characters.

    Platform-agnostic: uses msvcrt on Windows, tty/termios on Unix.
    """
    if sys.platform == "win32":
        import msvcrt
        key = msvcrt.getwch()
        if key in ("\xe0", "\x00"):  # Arrow/special key prefix
            key2 = msvcrt.getwch()
            return {"H": "up", "P": "down", "M": "right", "K": "left"}.get(key2, "")
        if key == "\r":
            return "enter"
        if key == "\x1b":
            return "escape"
        if key == " ":
            return "space"
        return key
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {
                        "A": "up", "B": "down", "C": "right", "D": "left",
                    }.get(ch3, "")
            if ch in ("\r", "\n"):
                return "enter"
            if ch == " ":
                return "space"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _strip_rich_markup(text: str) -> str:
    """Remove Rich markup tags like [bold cyan] from a string."""
    return re.sub(r"\[/?[a-z_ ]+\]", "", text)


class CLI:
    """Consistent CLI output for cola-coder scripts."""

    def _log(self, text: str) -> None:
        """Write plain text to the session log file (if active)."""
        from cola_coder.session_log import get_session_log
        session = get_session_log()
        if session is not None:
            plain = _strip_rich_markup(text)
            session.write(plain)

    # ── Branding ──────────────────────────────────────────────────────────

    def header(self, title: str, subtitle: str = "") -> None:
        """Print the app header banner."""
        line = f"═══ {title}"
        if subtitle:
            line += f" — {subtitle}"
        line += " ═══"
        self._log(line)
        if _HAS_RICH:
            text = Text()
            text.append(f" {title}", style="bold cyan")
            if subtitle:
                text.append(f"  {subtitle}", style="bold white")
            _console.print(Panel(
                text, box=box.HEAVY, style="cyan", padding=(0, 1),
            ))
        else:
            print(line)

    # ── Steps & Progress ──────────────────────────────────────────────────

    def step(self, current: int, total: int, message: str) -> None:
        """Print a step indicator: Step 1/3 · Loading tokenizer"""
        self._log(f"Step {current}/{total} · {message}")
        if _HAS_RICH:
            _console.print(
                f"\n[bold cyan]Step {current}/{total}[/bold cyan]"
                f" [dim]·[/dim] [bold]{message}[/bold]"
            )
        else:
            print(f"\nStep {current}/{total} · {message}")

    def substep(self, message: str) -> None:
        """Print an indented sub-step."""
        self._log(f"  · {message}")
        if _HAS_RICH:
            _console.print(f"  [bold cyan]·[/bold cyan] {message}")
        else:
            print(f"  · {message}")

    # ── Key/value info ────────────────────────────────────────────────────

    def info(self, key: str, value: str | int | float) -> None:
        """Print a key: value pair."""
        self._log(f"  {key}: {value}")
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
        self._log(f"✓ {message}")
        if _HAS_RICH:
            _console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def error(self, message: str, hint: str = "") -> None:
        """Print an error message and optional hint."""
        self._log(f"✗ Error: {message}")
        if hint:
            self._log(f"  {hint}")
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
        self._log(f"⚠ {message}")
        if _HAS_RICH:
            _console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        else:
            print(f"⚠ {message}")

    def dim(self, message: str) -> None:
        """Print a dimmed/secondary message."""
        self._log(f"  {message}")
        if _HAS_RICH:
            _console.print(f"  [dim]{message}[/dim]")
        else:
            print(f"  {message}")

    # ── Completion ────────────────────────────────────────────────────────

    def done(self, message: str, extras: dict[str, str] | None = None) -> None:
        """Print a completion panel with optional extra info."""
        self._log(f"✓ {message}")
        if extras:
            for k, v in extras.items():
                self._log(f"  {k}: {v}")
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
        """Show an arrow-key navigable menu and return the selected index.

        Uses questionary for arrow-key navigation when available, falls back
        to a numbered input menu otherwise.

        Args:
            prompt: Title shown above the options.
            options: List of dicts with 'label' and optional 'detail' keys.
            allow_cancel: If True, adds a "Cancel" option and returns None
                          if selected.

        Returns:
            Index of selected option, or None if cancelled.
        """
        try:
            import questionary
            from questionary import Style

            custom_style = Style([
                ('qmark', 'fg:cyan bold'),
                ('question', 'bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
            ])

            _CANCEL = -1  # sentinel — questionary ignores value=None

            choices = []
            for i, opt in enumerate(options):
                label = opt.get("label", "")
                detail = opt.get("detail", "")
                display = f"{label}  ({detail})" if detail else label
                choices.append(questionary.Choice(title=display, value=i))

            if allow_cancel:
                choices.append(questionary.Choice(title="Cancel", value=_CANCEL))

            result = questionary.select(
                prompt,
                choices=choices,
                style=custom_style,
                use_shortcuts=False,
                use_arrow_keys=True,
            ).ask()

            # .ask() returns None on Ctrl-C; sentinel on Cancel selection
            if result is None or result == _CANCEL:
                return None
            return result

        except ImportError:
            pass

        # ── Fallback: numbered menu (questionary not installed) ──────────────
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

        Uses questionary for a styled prompt when available, falls back to
        plain input otherwise.

        Args:
            prompt: The question to ask.
            default: Default value if user just presses Enter.

        Returns:
            True for yes, False for no.
        """
        try:
            import questionary
            from questionary import Style

            custom_style = Style([
                ('qmark', 'fg:cyan bold'),
                ('question', 'bold'),
            ])

            result = questionary.confirm(
                prompt,
                default=default,
                style=custom_style,
            ).ask()

            # .ask() returns None on Ctrl-C / EOF — fall back to default
            return result if result is not None else default

        except ImportError:
            pass

        # ── Fallback: plain input (questionary not installed) ────────────────
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

    def multi_select(
        self,
        prompt: str,
        options: list[dict[str, str]],
        *,
        preselected: list[int] | None = None,
    ) -> list[int]:
        """Show a multi-select menu and return selected indices.

        Uses questionary checkbox when available, falls back to sequential
        confirm() calls otherwise.

        Args:
            prompt: Title shown above the options.
            options: List of dicts with 'label' and optional 'detail' keys.
            preselected: Indices to pre-check (default: all selected).

        Returns:
            Sorted list of selected indices (0-based). Empty if all
            deselected or cancelled.
        """
        if preselected is None:
            preselected = list(range(len(options)))

        try:
            import questionary
            from questionary import Style

            custom_style = Style([
                ('qmark', 'fg:cyan bold'),
                ('question', 'bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
                ('checked', 'fg:green bold'),
            ])

            choices = []
            for i, opt in enumerate(options):
                label = opt.get("label", "")
                detail = opt.get("detail", "")
                display = f"{label}  ({detail})" if detail else label
                choices.append(
                    questionary.Choice(
                        title=display, value=i, checked=i in preselected,
                    )
                )

            result = questionary.checkbox(
                prompt,
                choices=choices,
                style=custom_style,
            ).ask()

            if result is None:
                return []
            return sorted(result)

        except ImportError:
            pass

        # Fallback: sequential confirm
        self.print(f"\n  [bold]{prompt}[/bold]" if _HAS_RICH else f"\n  {prompt}")
        selected = []
        for i, opt in enumerate(options):
            label = opt.get("label", "")
            default = i in preselected
            if self.confirm(f"  Include {label}?", default=default):
                selected.append(i)
        return sorted(selected)

    def weight_editor(
        self,
        items: list[dict[str, str]],
        *,
        title: str = "Set weights for each dataset",
    ) -> list[float]:
        """Interactive weight editor with left/right arrow adjustment.

        Args:
            items: List of dicts with 'label' key (and optional 'detail').
            title: Header text shown above the editor.

        Returns:
            List of normalized weights summing to 1.0.
        """
        import os

        n = len(items)
        weights = [1.0 / n] * n
        cursor = 0
        increment = 0.05

        def _clear():
            if _HAS_RICH:
                _console.clear()
            else:
                os.system("cls" if sys.platform == "win32" else "clear")

        while True:
            _clear()
            self.header("Cola-Coder", "Weight Editor")
            self.print()
            self.print(f"  [bold]{title}[/bold]" if _HAS_RICH else f"  {title}")
            self.dim(
                "  Use up/down to navigate, left/right to adjust, Enter to confirm"
            )
            self.print()

            for i, item in enumerate(items):
                is_cursor = i == cursor
                w = weights[i]
                bar_len = int(w * 30)
                bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
                label = item.get("label", f"Item {i + 1}")

                if _HAS_RICH:
                    prefix = "  [bold green]>[/bold green] " if is_cursor else "    "
                    if is_cursor:
                        self.print(
                            f"{prefix}[bold white on blue] {label:<30} "
                            f"[/bold white on blue]  [{w:.2f}] {bar}"
                        )
                    else:
                        self.print(
                            f"{prefix}[white]{label:<30}[/white]"
                            f"  [{w:.2f}] [dim]{bar}[/dim]"
                        )
                else:
                    prefix = "  > " if is_cursor else "    "
                    print(f"{prefix}{label:<30}  [{w:.2f}] {bar}")

            self.print()
            total = sum(weights)
            self.dim(f"  Total: {total:.2f} (will be normalized to 1.0)")

            key = _read_key()
            if key == "up":
                cursor = (cursor - 1) % n
            elif key == "down":
                cursor = (cursor + 1) % n
            elif key == "right":
                weights[cursor] = min(1.0, weights[cursor] + increment)
            elif key == "left":
                weights[cursor] = max(0.05, weights[cursor] - increment)
            elif key == "enter":
                total = sum(weights)
                return [w / total for w in weights]
            elif key in ("escape", "\x03"):
                self.warn("Cancelled.")
                sys.exit(0)

    def pick_languages(
        self,
        prompt: str = "Select languages:",
        presets: bool = True,
    ) -> list[str] | None:
        """Interactive language/framework selection with presets.

        Returns list of language slugs or None on cancel.
        Uses choose() for presets and multi_select() for custom selection.
        """
        if presets:
            preset_options = [
                {"label": "TypeScript (all)",
                 "detail": "typescript + typescript-react"},
                {"label": "TypeScript React only",
                 "detail": ".tsx files with React/Next.js"},
                {"label": "TypeScript + JavaScript",
                 "detail": "Full JS/TS ecosystem incl. React variants"},
                {"label": "Python only",
                 "detail": "Single language — python"},
                {"label": "All languages",
                 "detail": "All 10 supported languages/frameworks"},
                {"label": "Custom selection...",
                 "detail": "Pick individual languages/frameworks"},
            ]
            choice = self.choose(prompt, preset_options, allow_cancel=True)
            if choice is None:
                return None

            preset_map = {
                0: ["typescript", "typescript-react"],
                1: ["typescript-react"],
                2: ["typescript", "typescript-react", "javascript", "javascript-react"],
                3: ["python"],
                4: [lang["slug"] for lang in SUPPORTED_LANGUAGES],
            }

            if choice in preset_map:
                return preset_map[choice]
            # choice == 5 -> fall through to custom selection

        # Custom selection via multi_select
        lang_options = [
            {"label": lang["label"], "detail": lang["detail"]}
            for lang in SUPPORTED_LANGUAGES
        ]
        # Pre-select TypeScript + TypeScript React (indices 0, 1)
        selected = self.multi_select(
            "Select languages/frameworks:",
            lang_options,
            preselected=[0, 1],
        )

        if not selected:
            return None

        return [SUPPORTED_LANGUAGES[i]["slug"] for i in selected]

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
