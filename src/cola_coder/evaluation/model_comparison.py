"""Model comparison utilities for cola-coder.

Compare multiple checkpoints side-by-side on the same prompts. Loads one
model at a time to conserve VRAM, then collects outputs and metrics.

For a TS dev: like a benchmark suite where you run the same test against
multiple builds of your app and compare the results in a table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .quality_report import STANDARD_PROMPTS, _count_params, _human_params, _read_metadata


# ── Default comparison prompts ────────────────────────────────────────────────

DEFAULT_COMPARISON_PROMPTS = STANDARD_PROMPTS[:3]


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Results from comparing multiple model checkpoints."""

    models: list[dict]          # [{name, checkpoint, params, params_human, step, loss}]
    prompts: list[str]
    outputs: list[list[str]]    # outputs[model_idx][prompt_idx] = output text
    metrics: list[dict]         # Per-model: [{tokens_per_sec, avg_output_len, ...}]

    def to_markdown(self) -> str:
        """Side-by-side markdown comparison."""
        lines: list[str] = []
        lines.append("# Cola-Coder Model Comparison")
        lines.append("")

        # Model summary table
        lines.append("## Models")
        lines.append("| # | Checkpoint | Params | Step | Loss |")
        lines.append("|---|-----------|--------|------|------|")
        for i, m in enumerate(self.models):
            name = m.get("name", m.get("checkpoint", "?"))
            params = m.get("params_human", _human_params(m.get("params", 0)))
            step = m.get("step", "?")
            step_str = f"{step:,}" if isinstance(step, int) else str(step)
            loss = m.get("loss", float("nan"))
            loss_str = f"{loss:.4f}" if isinstance(loss, float) and loss == loss else "N/A"
            lines.append(f"| {i + 1} | {name} | {params} | {step_str} | {loss_str} |")
        lines.append("")

        # Per-prompt outputs
        lines.append("## Outputs")
        for p_idx, prompt in enumerate(self.prompts):
            short_prompt = prompt.split("\n")[0][:60]
            lines.append(f"### Prompt {p_idx + 1}: `{short_prompt}`")
            lines.append("")
            for m_idx, model_info in enumerate(self.models):
                name = model_info.get("name", f"Model {m_idx + 1}")
                output = self.outputs[m_idx][p_idx] if m_idx < len(self.outputs) else ""
                lines.append(f"**{name}:**")
                lines.append("```")
                lines.append(output)
                lines.append("```")
                lines.append("")

        # Metrics table
        if self.metrics:
            lines.append("## Performance Metrics")
            lines.append("| Model | Tokens/s | Avg Output Len | Step | Loss |")
            lines.append("|-------|----------|---------------|------|------|")
            for m_idx, (model_info, m) in enumerate(zip(self.models, self.metrics)):
                name = model_info.get("name", f"Model {m_idx + 1}")
                tps = m.get("tokens_per_sec", 0)
                avg_len = m.get("avg_output_len", 0)
                step = model_info.get("step", "?")
                step_str = f"{step:,}" if isinstance(step, int) else str(step)
                loss = model_info.get("loss", float("nan"))
                loss_str = f"{loss:.4f}" if isinstance(loss, float) and loss == loss else "N/A"
                lines.append(
                    f"| {name} | {tps:.1f} | {avg_len:.0f} | {step_str} | {loss_str} |"
                )

        return "\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────────────


class ModelComparator:
    """Compare multiple model checkpoints side-by-side.

    Loads models one at a time to avoid running out of VRAM when comparing
    several large checkpoints. All models must share the same config / tokenizer.
    """

    def __init__(
        self,
        checkpoints: list[str],
        configs: list[str],
        device: str = "auto",
    ):
        """
        Args:
            checkpoints: List of checkpoint directory paths to compare.
            configs: List of YAML config paths. If a single path is given it's
                     reused for all checkpoints. Must have same length as
                     checkpoints or length 1.
            device: "cuda", "cpu", or "auto".
        """
        if not checkpoints:
            raise ValueError("Need at least one checkpoint to compare")
        if len(configs) == 1:
            configs = configs * len(checkpoints)
        if len(configs) != len(checkpoints):
            raise ValueError(
                f"configs must have length 1 or match checkpoints length "
                f"({len(checkpoints)}), got {len(configs)}"
            )
        self.checkpoints = checkpoints
        self.configs = configs
        self.device = device
        self._resolved_device: str | None = None

    def _get_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device != "auto":
            self._resolved_device = self.device
            return self._resolved_device
        try:
            import torch
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._resolved_device = "cpu"
        return self._resolved_device

    def _find_tokenizer(self, checkpoint_path: str, config_path: str) -> str:
        """Try to find tokenizer.json near the checkpoint or project root."""
        candidates = [
            Path(checkpoint_path) / "tokenizer.json",
            Path(checkpoint_path).parent / "tokenizer.json",
            Path(checkpoint_path).parent.parent / "tokenizer.json",
            Path("tokenizer.json"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        try:
            from cola_coder.model.config import get_storage_config
            storage = get_storage_config()
            p = Path(storage.tokenizer_path)
            if p.exists():
                return str(p)
        except Exception:
            pass
        return "tokenizer.json"

    def _load_generator(self, checkpoint_path: str, config_path: str) -> tuple[Any, Any]:
        """Load and return (generator, tokenizer) for a checkpoint."""
        from cola_coder.model.config import Config, ModelConfig
        from cola_coder.model.transformer import Transformer
        from cola_coder.training.checkpoint import load_model_only
        from cola_coder.inference.generator import CodeGenerator
        from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer

        device = self._get_device()

        # Prefer model config from checkpoint metadata
        metadata = _read_metadata(checkpoint_path)
        meta_config = metadata.get("config", {})
        model_cfg_raw = meta_config.get("model", {}) or {}

        if model_cfg_raw:
            valid_fields = ModelConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in model_cfg_raw.items() if k in valid_fields}
            model_cfg = ModelConfig(**filtered)
        else:
            config_obj = Config.from_yaml(config_path)
            model_cfg = config_obj.model

        tokenizer_path = self._find_tokenizer(checkpoint_path, config_path)
        tokenizer = CodeTokenizer(tokenizer_path)
        model = Transformer(model_cfg).to(device)
        load_model_only(checkpoint_path, model, device=device)
        model.eval()
        generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
        return generator, tokenizer

    def compare(
        self,
        prompts: list[str] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 256,
    ) -> ComparisonResult:
        """Generate outputs from all models for the same prompts.

        Loads each model one at a time to save VRAM.

        Args:
            prompts: Prompts to use. Defaults to DEFAULT_COMPARISON_PROMPTS.
            temperature: Sampling temperature.
            max_tokens: Max new tokens per generation.

        Returns:
            ComparisonResult with outputs and metrics for all models.
        """
        if prompts is None:
            prompts = DEFAULT_COMPARISON_PROMPTS

        all_outputs: list[list[str]] = []   # [model_idx][prompt_idx]
        all_metrics: list[dict] = []
        model_infos: list[dict] = []

        for i, (ckpt, cfg) in enumerate(zip(self.checkpoints, self.configs)):
            ckpt_path = Path(ckpt)
            metadata = _read_metadata(ckpt)
            step = int(metadata.get("step", 0))
            loss = float(metadata.get("loss", float("nan")))
            name = ckpt_path.name

            try:
                generator, tokenizer = self._load_generator(ckpt, cfg)
                params = _count_params(generator.model)
            except Exception as exc:
                # Can't load model — record error outputs
                model_infos.append({
                    "name": name,
                    "checkpoint": ckpt,
                    "params": 0,
                    "params_human": "?",
                    "step": step,
                    "loss": loss,
                    "error": str(exc),
                })
                all_outputs.append([f"[LOAD ERROR: {exc}]"] * len(prompts))
                all_metrics.append({"tokens_per_sec": 0.0, "avg_output_len": 0.0})
                continue

            model_infos.append({
                "name": name,
                "checkpoint": ckpt,
                "params": params,
                "params_human": _human_params(params),
                "step": step,
                "loss": loss,
            })

            # Generate outputs for each prompt
            model_outputs: list[str] = []
            total_tokens = 0
            total_time_s = 0.0

            for prompt in prompts:
                t0 = time.perf_counter()
                try:
                    output = generator.generate(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.9,
                    )
                except Exception as exc:
                    output = f"[GENERATION ERROR: {exc}]"

                elapsed = time.perf_counter() - t0
                total_time_s += elapsed

                # Count new tokens approximately
                try:
                    out_ids = len(tokenizer.encode(output, add_bos=False))
                    in_ids = len(tokenizer.encode(prompt, add_bos=False))
                    new_toks = max(0, out_ids - in_ids)
                except Exception:
                    new_toks = 0
                total_tokens += new_toks
                model_outputs.append(output)

            all_outputs.append(model_outputs)
            avg_len = sum(len(o) for o in model_outputs) / max(len(model_outputs), 1)
            tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
            all_metrics.append({
                "tokens_per_sec": tps,
                "avg_output_len": avg_len,
            })

            # Delete model to free VRAM before loading the next one
            try:
                import torch
                del generator
                torch.cuda.empty_cache()
            except Exception:
                pass

        return ComparisonResult(
            models=model_infos,
            prompts=prompts,
            outputs=all_outputs,
            metrics=all_metrics,
        )

    def compare_quick(self) -> ComparisonResult:
        """Quick comparison with 3 standard prompts."""
        return self.compare(
            prompts=DEFAULT_COMPARISON_PROMPTS,
            temperature=0.8,
            max_tokens=128,
        )
