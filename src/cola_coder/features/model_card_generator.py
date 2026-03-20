"""
Model card generator for trained Cola-Coder models.

Generates markdown documentation cards that describe a model's architecture,
training details, performance metrics, limitations, and usage examples.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return whether the model card generator feature is enabled."""
    return FEATURE_ENABLED


@dataclass
class ModelInfo:
    """Basic descriptive information about a model."""

    name: str
    version: str
    architecture: str
    parameters: int
    languages: List[str]
    license: str


@dataclass
class TrainingInfo:
    """Information about how a model was trained."""

    dataset: str
    epochs: int
    learning_rate: float
    batch_size: int
    hardware: str
    training_time: str


@dataclass
class _Metric:
    name: str
    value: float
    dataset: str


@dataclass
class _Example:
    prompt: str
    output: str


class ModelCardGenerator:
    """Generates model cards (markdown documentation) for trained models."""

    def __init__(
        self,
        model_info: ModelInfo,
        training_info: Optional[TrainingInfo] = None,
    ) -> None:
        self.model_info = model_info
        self.training_info = training_info
        self._metrics: List[_Metric] = []
        self._limitations: List[str] = []
        self._examples: List[_Example] = []

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_metric(self, name: str, value: float, dataset: str = "") -> None:
        """Record an evaluation metric."""
        self._metrics.append(_Metric(name=name, value=value, dataset=dataset))

    def add_limitation(self, text: str) -> None:
        """Record a known limitation of the model."""
        self._limitations.append(text)

    def add_example(self, prompt: str, output: str) -> None:
        """Add a usage example (prompt -> output pair)."""
        self._examples.append(_Example(prompt=prompt, output=output))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """Generate the full model card as a markdown string."""
        sections: List[str] = []

        sections.append(self._render_header())
        sections.append(self._render_model_info())

        if self.training_info is not None:
            sections.append(self._render_training_info())

        if self._metrics:
            sections.append(self._render_metrics())

        if self._limitations:
            sections.append(self._render_limitations())

        if self._examples:
            sections.append(self._render_examples())

        sections.append(self._render_footer())

        return "\n\n".join(sections)

    def save(self, path: str) -> None:
        """Save the generated model card to *path*."""
        card = self.generate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(card)

    # ------------------------------------------------------------------
    # Class method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "ModelCardGenerator":
        """
        Attempt to build a ModelCardGenerator from metadata files stored
        inside *checkpoint_dir*.

        Looks for ``model_card_meta.json`` which should contain keys
        ``model_info`` and (optionally) ``training_info``.  If the file is
        absent a minimal stub generator is returned instead.
        """
        meta_path = os.path.join(checkpoint_dir, "model_card_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

            mi_data = meta.get("model_info", {})
            model_info = ModelInfo(
                name=mi_data.get("name", os.path.basename(checkpoint_dir)),
                version=mi_data.get("version", "unknown"),
                architecture=mi_data.get("architecture", "unknown"),
                parameters=int(mi_data.get("parameters", 0)),
                languages=list(mi_data.get("languages", [])),
                license=mi_data.get("license", "unknown"),
            )

            training_info: Optional[TrainingInfo] = None
            ti_data = meta.get("training_info")
            if ti_data:
                training_info = TrainingInfo(
                    dataset=ti_data.get("dataset", "unknown"),
                    epochs=int(ti_data.get("epochs", 0)),
                    learning_rate=float(ti_data.get("learning_rate", 0.0)),
                    batch_size=int(ti_data.get("batch_size", 0)),
                    hardware=ti_data.get("hardware", "unknown"),
                    training_time=ti_data.get("training_time", "unknown"),
                )
        else:
            # Fallback: create a stub with the directory name as model name.
            model_info = ModelInfo(
                name=os.path.basename(os.path.abspath(checkpoint_dir)),
                version="unknown",
                architecture="unknown",
                parameters=0,
                languages=[],
                license="unknown",
            )
            training_info = None

        return cls(model_info, training_info)

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _format_parameters(self) -> str:
        p = self.model_info.parameters
        if p >= 1_000_000_000:
            return f"{p / 1_000_000_000:.1f}B"
        if p >= 1_000_000:
            return f"{p / 1_000_000:.0f}M"
        if p >= 1_000:
            return f"{p / 1_000:.0f}K"
        return str(p)

    def _render_header(self) -> str:
        mi = self.model_info
        return (
            f"# {mi.name}\n\n"
            f"**Version:** {mi.version}  \n"
            f"**License:** {mi.license}  \n"
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )

    def _render_model_info(self) -> str:
        mi = self.model_info
        languages = ", ".join(mi.languages) if mi.languages else "N/A"
        lines = [
            "## Model Information",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Architecture | {mi.architecture} |",
            f"| Parameters | {self._format_parameters()} |",
            f"| Languages | {languages} |",
            f"| License | {mi.license} |",
        ]
        return "\n".join(lines)

    def _render_training_info(self) -> str:
        ti = self.training_info
        lines = [
            "## Training Details",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Dataset | {ti.dataset} |",
            f"| Epochs | {ti.epochs} |",
            f"| Learning Rate | {ti.learning_rate} |",
            f"| Batch Size | {ti.batch_size} |",
            f"| Hardware | {ti.hardware} |",
            f"| Training Time | {ti.training_time} |",
        ]
        return "\n".join(lines)

    def _render_metrics(self) -> str:
        lines = [
            "## Performance",
            "",
            "| Metric | Value | Dataset |",
            "|--------|-------|---------|",
        ]
        for m in self._metrics:
            dataset_cell = m.dataset if m.dataset else "—"
            lines.append(f"| {m.name} | {m.value} | {dataset_cell} |")
        return "\n".join(lines)

    def _render_limitations(self) -> str:
        lines = ["## Limitations", ""]
        for limitation in self._limitations:
            lines.append(f"- {limitation}")
        return "\n".join(lines)

    def _render_examples(self) -> str:
        parts = ["## Usage Examples", ""]
        for i, ex in enumerate(self._examples, start=1):
            parts.append(f"### Example {i}")
            parts.append("")
            parts.append("**Prompt:**")
            parts.append("")
            parts.append(f"```python\n{ex.prompt}\n```")
            parts.append("")
            parts.append("**Output:**")
            parts.append("")
            parts.append(f"```python\n{ex.output}\n```")
            parts.append("")
        return "\n".join(parts)

    def _render_footer(self) -> str:
        return (
            "---\n\n"
            "*This model card was generated automatically by Cola-Coder's "
            "model card generator.*"
        )
