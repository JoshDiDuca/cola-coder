"""Unified master menu for Cola-Coder.

Single entry point for all Cola-Coder operations. Replaces 12 separate
PowerShell scripts with one interactive, keyboard-driven menu.
"""

import sys
import subprocess
from pathlib import Path
from cola_coder.cli import cli

# Feature toggle - this feature is OPTIONAL
FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Check if this feature is enabled."""
    return FEATURE_ENABLED


class MasterMenu:
    """Unified CLI menu for all Cola-Coder operations."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.venv_python = self.project_root / ".venv" / "Scripts" / "python"
        if not self.venv_python.exists():
            # Linux/Mac fallback
            self.venv_python = self.project_root / ".venv" / "bin" / "python"

    def _run_script(self, script: str, args: list[str] | None = None):
        """Run a Python script from the scripts/ directory."""
        cmd = [str(self.venv_python), f"scripts/{script}"]
        if args:
            cmd.extend(args)
        cli.dim(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode != 0:
                cli.error(f"Script exited with code {result.returncode}")
        except KeyboardInterrupt:
            cli.warn("Interrupted.")
        except Exception as e:
            cli.error(str(e))

    def _detect_pipeline_status(self) -> dict[str, str]:
        """Detect current pipeline state: what's been completed."""
        status = {}

        # Tokenizer
        tokenizer_path = self.project_root / "tokenizer.json"
        status["tokenizer"] = "ready" if tokenizer_path.exists() else "missing"

        # Data
        data_dir = self.project_root / "data" / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []
        status["data"] = f"{len(npy_files)} dataset(s)" if npy_files else "missing"

        # Checkpoints
        ckpt_dir = self.project_root / "checkpoints"
        if ckpt_dir.exists():
            ckpt_dirs = [d for d in ckpt_dir.rglob("model.safetensors")]
            status["checkpoints"] = f"{len(ckpt_dirs)} checkpoint(s)" if ckpt_dirs else "none"
        else:
            status["checkpoints"] = "none"

        return status

    def show_status_bar(self):
        """Show pipeline status at the top of the menu."""
        status = self._detect_pipeline_status()
        parts = []
        for key, val in status.items():
            if "missing" in val or val == "none":
                parts.append(f"[red]{key}: {val}[/red]")
            else:
                parts.append(f"[green]{key}: {val}[/green]")
        cli.print(f"  Pipeline: {' | '.join(parts)}")

    def main_menu(self):
        """Show the main menu and handle selection."""
        while True:
            cli.header("Cola-Coder", "Master Menu")
            self.show_status_bar()

            options = [
                {"label": "Train Model", "detail": "Train tiny/small/medium/large models"},
                {"label": "Prepare Data", "detail": "Download, filter, tokenize training data"},
                {"label": "Generate Code", "detail": "Interactive code generation"},
                {"label": "Evaluate Model", "detail": "Run HumanEval benchmark"},
                {"label": "Serve API", "detail": "Start FastAPI inference server"},
                {"label": "Train Tokenizer", "detail": "Train BPE tokenizer from scratch"},
                {"label": "Train Reasoning", "detail": "GRPO fine-tuning with thinking tokens"},
                {"label": "Tools", "detail": "Tests, linting, data inspection"},
            ]

            choice = cli.choose("What would you like to do?", options, allow_cancel=True)

            if choice is None:
                cli.dim("Goodbye!")
                break

            handlers = [
                self.train_menu,
                self.prepare_menu,
                self.generate_menu,
                self.evaluate_menu,
                self.serve_menu,
                self.tokenizer_menu,
                self.reasoning_menu,
                self.tools_menu,
            ]

            handlers[choice]()

    def train_menu(self):
        """Training sub-menu."""
        cli.header("Cola-Coder", "Train Model")

        options = [
            {"label": "Tiny (50M)", "detail": "~3.6GB VRAM, ~4 hours"},
            {"label": "Small (125M)", "detail": "~6.5GB VRAM, ~2 days"},
            {"label": "Medium (350M)", "detail": "~8.2GB VRAM, ~7 days"},
            {"label": "Large (1B+)", "detail": "Cloud only, ~24GB VRAM"},
        ]

        choice = cli.choose("Select model size:", options, allow_cancel=True)
        if choice is None:
            return

        sizes = ["tiny", "small", "medium", "large"]
        size = sizes[choice]

        # Check for existing checkpoints to resume
        ckpt_dir = self.project_root / "checkpoints" / size
        if ckpt_dir.exists():
            latest = ckpt_dir / "latest"
            if latest.exists() or any(ckpt_dir.glob("step_*")):
                if cli.confirm(f"Found existing {size} checkpoints. Resume training?"):
                    self._run_script("train.py", [
                        "--config", f"configs/{size}.yaml",
                        "--resume", str(ckpt_dir / "latest"),
                    ])
                    return

        # Check for wandb
        use_wandb = cli.confirm("Enable Weights & Biases logging?", default=False)
        args = ["--config", f"configs/{size}.yaml"]
        if use_wandb:
            args.append("--wandb")

        self._run_script("train.py", args)

    def prepare_menu(self):
        """Data preparation sub-menu."""
        cli.header("Cola-Coder", "Prepare Data")

        options = [
            {"label": "Interactive Mode", "detail": "Full menu-driven data preparation"},
            {"label": "Quick Tiny Dataset", "detail": "Small dataset for testing (~5 min)"},
            {"label": "Standard Preparation", "detail": "Full data pipeline with defaults"},
            {"label": "Test/Validation Data", "detail": "Prepare test split"},
        ]

        choice = cli.choose("Data preparation mode:", options, allow_cancel=True)
        if choice is None:
            return

        scripts = [
            ("prepare_data_interactive.py", []),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json",
                                 "--max-tokens", "500000"]),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json"]),
            ("prepare_data.py", ["--config", "configs/tiny.yaml", "--tokenizer", "tokenizer.json",
                                 "--split", "test"]),
        ]

        script, args = scripts[choice]
        self._run_script(script, args)

    def generate_menu(self):
        """Code generation."""
        cli.header("Cola-Coder", "Generate Code")

        # Find available checkpoints
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        checkpoints.append({
                            "label": f"{size_dir.name}/latest",
                            "detail": str(latest),
                            "path": str(latest),
                        })
                    for step_dir in sorted(size_dir.glob("step_*")):
                        checkpoints.append({
                            "label": f"{size_dir.name}/{step_dir.name}",
                            "detail": str(step_dir),
                            "path": str(step_dir),
                        })

        if not checkpoints:
            cli.error("No checkpoints found. Train a model first.")
            return

        choice = cli.choose("Select checkpoint:",
                            [{"label": c["label"], "detail": ""} for c in checkpoints],
                            allow_cancel=True)
        if choice is None:
            return

        self._run_script("generate.py", ["--checkpoint", checkpoints[choice]["path"]])

    def evaluate_menu(self):
        """Evaluation menu."""
        cli.header("Cola-Coder", "Evaluate Model")

        # Find checkpoints (same pattern as generate)
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        checkpoints.append({"label": f"{size_dir.name}/latest", "path": str(latest)})

        if not checkpoints:
            cli.error("No checkpoints found. Train a model first.")
            return

        choice = cli.choose("Select checkpoint to evaluate:",
                            [{"label": c["label"], "detail": ""} for c in checkpoints],
                            allow_cancel=True)
        if choice is None:
            return

        self._run_script("evaluate.py", ["--checkpoint", checkpoints[choice]["path"]])

    def serve_menu(self):
        """Start API server."""
        cli.header("Cola-Coder", "Serve API")
        ckpt_dir = self.project_root / "checkpoints"
        checkpoints = []
        if ckpt_dir.exists():
            for size_dir in sorted(ckpt_dir.iterdir()):
                if size_dir.is_dir():
                    latest = size_dir / "latest"
                    if latest.exists():
                        checkpoints.append({"label": f"{size_dir.name}/latest", "path": str(latest)})

        if not checkpoints:
            cli.error("No checkpoints found.")
            return

        choice = cli.choose("Select checkpoint to serve:",
                            [{"label": c["label"], "detail": ""} for c in checkpoints],
                            allow_cancel=True)
        if choice is None:
            return

        self._run_script("serve.py", ["--checkpoint", checkpoints[choice]["path"]])

    def tokenizer_menu(self):
        """Train tokenizer."""
        cli.header("Cola-Coder", "Train Tokenizer")

        tokenizer_path = self.project_root / "tokenizer.json"
        if tokenizer_path.exists():
            if not cli.confirm("tokenizer.json already exists. Retrain?", default=False):
                return

        self._run_script("train_tokenizer.py", [])

    def reasoning_menu(self):
        """Reasoning/GRPO training."""
        cli.header("Cola-Coder", "Reasoning Training")
        cli.info("Mode", "GRPO with thinking tokens")

        if cli.confirm("Start reasoning training?"):
            self._run_script("train_reasoning.py", [])

    def tools_menu(self):
        """Developer tools sub-menu."""
        cli.header("Cola-Coder", "Tools")

        options = [
            {"label": "Run Tests", "detail": "pytest tests/ -v"},
            {"label": "Run Linter", "detail": "ruff check src/ scripts/ tests/"},
            {"label": "GPU Status", "detail": "Show GPU info and VRAM usage"},
            {"label": "Dataset Inspector", "detail": "Browse training data samples"},
            {"label": "Feature Toggles", "detail": "Enable/disable optional features"},
        ]

        choice = cli.choose("Select tool:", options, allow_cancel=True)
        if choice is None:
            return

        if choice == 0:
            subprocess.run([str(self.venv_python), "-m", "pytest", "tests/", "-v"],
                           cwd=str(self.project_root))
        elif choice == 1:
            subprocess.run([str(self.venv_python), "-m", "ruff", "check", "src/", "scripts/",
                            "tests/"],
                           cwd=str(self.project_root))
        elif choice == 2:
            cli.gpu_info()
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    cli.print(result.stdout)
            except FileNotFoundError:
                cli.warn("nvidia-smi not found")
        elif choice == 3:
            self._inspect_dataset()
        elif choice == 4:
            self._feature_toggles()

        input("\nPress Enter to continue...")

    def _inspect_dataset(self):
        """Browse random samples from training data."""
        import numpy as np
        data_dir = self.project_root / "data" / "processed"
        npy_files = list(data_dir.glob("*.npy")) if data_dir.exists() else []

        if not npy_files:
            cli.error("No datasets found.")
            return

        # Pick a dataset
        if len(npy_files) == 1:
            npy_path = npy_files[0]
        else:
            options = [{"label": f.stem, "detail": ""} for f in npy_files]
            choice = cli.choose("Select dataset:", options, allow_cancel=True)
            if choice is None:
                return
            npy_path = npy_files[choice]

        data = np.load(str(npy_path), mmap_mode="r")
        cli.info("Shape", f"{data.shape[0]:,} chunks x {data.shape[1]} tokens")
        cli.info("Total tokens", f"{data.shape[0] * data.shape[1]:,}")

        # Show random samples
        try:
            from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer
            tokenizer_path = self.project_root / "tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = CodeTokenizer(str(tokenizer_path))
                indices = np.random.choice(data.shape[0], size=min(3, data.shape[0]), replace=False)
                for idx in indices:
                    cli.rule(f"Sample #{idx}")
                    tokens = data[idx].tolist()
                    text = tokenizer.decode(tokens)
                    # Truncate for display
                    display_text = text[:500] + ("..." if len(text) > 500 else "")
                    cli.print(display_text)
        except Exception as e:
            cli.warn(f"Could not decode samples: {e}")

    def _feature_toggles(self):
        """Show and toggle optional features."""
        cli.header("Cola-Coder", "Feature Toggles")
        cli.dim("Feature toggles are managed via configs/features.yaml")
        cli.dim("Edit that file to enable/disable optional features.")
        # This will be expanded as features are added


def run_master_menu():
    """Entry point for the master menu."""
    if not is_enabled():
        cli.error("Master menu feature is disabled.")
        return

    # Find project root (look for pyproject.toml or configs/)
    cwd = Path.cwd()
    if (cwd / "configs").exists():
        root = cwd
    elif (cwd / "cola-coder" / "configs").exists():
        root = cwd / "cola-coder"
    else:
        root = cwd

    menu = MasterMenu(project_root=root)
    menu.main_menu()
