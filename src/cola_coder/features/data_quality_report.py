"""Training Data Quality Report: summarize quality metrics for a dataset.

Analyzes a training data file (numpy memmap or plain text) to produce a
comprehensive report including:
- Length distribution (tokens and characters)
- Language mix (Python / TypeScript / JavaScript)
- Quality score distribution (if scores available)
- Sample statistics (mean, median, std, percentiles)

Output: Markdown report via to_markdown(), or structured dataclass.

For a TS dev: like a code coverage report but for training data quality —
tells you what you're actually feeding the model.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if the data quality report feature is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LengthStats:
    """Descriptive statistics for a length distribution."""

    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0


@dataclass
class LanguageMix:
    """Estimated language distribution (fraction per language)."""

    python: float = 0.0
    typescript: float = 0.0
    javascript: float = 0.0
    other: float = 0.0


@dataclass
class QualityBuckets:
    """Distribution of quality scores across the standard tiers."""

    excellent: float = 0.0  # 0.8+
    good: float = 0.0  # 0.6-0.8
    average: float = 0.0  # 0.4-0.6
    poor: float = 0.0  # 0.2-0.4
    reject: float = 0.0  # < 0.2


@dataclass
class Report:
    """Full data quality report."""

    source_path: str = ""
    tokenizer_path: str = ""
    total_samples: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    length_stats: LengthStats = field(default_factory=LengthStats)
    char_length_stats: LengthStats = field(default_factory=LengthStats)
    language_mix: LanguageMix = field(default_factory=LanguageMix)
    quality_buckets: QualityBuckets | None = None
    quality_mean: float | None = None
    quality_std: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as a Markdown string."""
        lines: list[str] = []
        lines.append("# Training Data Quality Report")
        lines.append("")
        lines.append(f"**Source:** `{self.source_path}`")
        if self.tokenizer_path:
            lines.append(f"**Tokenizer:** `{self.tokenizer_path}`")
        lines.append("")

        lines.append("## Overview")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total samples | {self.total_samples:,} |")
        lines.append(f"| Total tokens | {self.total_tokens:,} |")
        lines.append(f"| Total chars | {self.total_chars:,} |")
        lines.append("")

        lines.append("## Token Length Distribution")
        lines.append("")
        lines += _stats_table(self.length_stats, "Tokens")
        lines.append("")

        lines.append("## Character Length Distribution")
        lines.append("")
        lines += _stats_table(self.char_length_stats, "Characters")
        lines.append("")

        lines.append("## Language Mix")
        lines.append("")
        lines.append("| Language | Fraction |")
        lines.append("|----------|----------|")
        lm = self.language_mix
        for lang, frac in [
            ("Python", lm.python),
            ("TypeScript", lm.typescript),
            ("JavaScript", lm.javascript),
            ("Other", lm.other),
        ]:
            lines.append(f"| {lang} | {frac:.1%} |")
        lines.append("")

        if self.quality_buckets is not None:
            lines.append("## Quality Score Distribution")
            lines.append("")
            qb = self.quality_buckets
            lines.append("| Tier | Range | Fraction |")
            lines.append("|------|-------|----------|")
            for tier, rng, frac in [
                ("Excellent", "0.8–1.0", qb.excellent),
                ("Good", "0.6–0.8", qb.good),
                ("Average", "0.4–0.6", qb.average),
                ("Poor", "0.2–0.4", qb.poor),
                ("Reject", "0.0–0.2", qb.reject),
            ]:
                lines.append(f"| {tier} | {rng} | {frac:.1%} |")
            if self.quality_mean is not None:
                lines.append("")
                lines.append(
                    f"**Mean quality:** {self.quality_mean:.3f}  "
                    f"**Std:** {self.quality_std or 0.0:.3f}"
                )
            lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize the report to a JSON string."""
        d = asdict(self)
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class DataQualityReport:
    """Generate a quality report for a training data corpus.

    Supports:
    - ``.npy`` / ``.npz`` numpy arrays (token IDs)
    - Plain-text files (one sample per line)
    - Directories of ``.txt`` / ``.py`` / ``.ts`` files

    Usage::

        dqr = DataQualityReport()
        report = dqr.generate("data/processed/train_data.npy")
        print(report.to_markdown())
    """

    def __init__(self, sample_limit: int = 50_000) -> None:
        self.sample_limit = sample_limit

    def generate(
        self,
        data_path: str | Path,
        tokenizer_path: str | Path | None = None,
    ) -> Report:
        """Analyze training data and return a Report.

        Args:
            data_path: Path to training data file or directory.
            tokenizer_path: Optional path to tokenizer (for decoding tokens).

        Returns:
            Report with all computed metrics.
        """
        path = Path(data_path)
        report = Report(
            source_path=str(path),
            tokenizer_path=str(tokenizer_path) if tokenizer_path else "",
        )

        # Load samples
        samples = self._load_samples(path, tokenizer_path)
        if not samples:
            report.notes.append("No samples loaded — file may be empty or unsupported.")
            return report

        samples = samples[: self.sample_limit]
        report.total_samples = len(samples)

        # Compute lengths
        token_lengths = [s.get("token_len", 0) for s in samples]
        char_lengths = [len(s.get("text", "")) for s in samples]
        total_tokens = sum(token_lengths)
        total_chars = sum(char_lengths)

        report.total_tokens = total_tokens
        report.total_chars = total_chars
        report.length_stats = _compute_stats(token_lengths)
        report.char_length_stats = _compute_stats(char_lengths)

        # Language detection
        texts = [s.get("text", "") for s in samples if s.get("text")]
        report.language_mix = self._detect_language_mix(texts)

        # Quality scores
        scores = [s["quality"] for s in samples if "quality" in s]
        if scores:
            report.quality_buckets = _quality_buckets(scores)
            report.quality_mean = sum(scores) / len(scores)
            variance = sum((s - report.quality_mean) ** 2 for s in scores) / len(scores)
            report.quality_std = math.sqrt(variance)

        return report

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_samples(
        self,
        path: Path,
        tokenizer_path: Any,
    ) -> list[dict[str, Any]]:
        """Load samples from various data formats."""
        samples: list[dict[str, Any]] = []

        if not path.exists():
            return samples

        if path.is_dir():
            return self._load_directory(path)

        suffix = path.suffix.lower()

        if suffix == ".npy":
            return self._load_npy(path, tokenizer_path)
        if suffix == ".npz":
            return self._load_npz(path, tokenizer_path)
        if suffix in (".txt", ".py", ".ts", ".js"):
            return self._load_text_file(path)
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        # Default: try text
        return self._load_text_file(path)

    @staticmethod
    def _load_npy(path: Path, tokenizer_path: Any) -> list[dict[str, Any]]:
        """Load token IDs from a numpy memmap / npy file."""
        try:
            import numpy as np  # type: ignore[import-not-found]

            data = np.load(str(path), mmap_mode="r")
            # Check for companion weights file
            weights_path = path.with_suffix(".weights.npy")
            weights = np.load(str(weights_path)) if weights_path.exists() else None

            samples: list[dict[str, Any]] = []
            # data shape: (N,) flat or (N, seq_len)
            if data.ndim == 1:
                chunk = 512
                count = len(data) // chunk
                for i in range(min(count, 50_000)):
                    token_ids = data[i * chunk : (i + 1) * chunk].tolist()
                    entry: dict[str, Any] = {"token_len": len(token_ids)}
                    if weights is not None and i < len(weights):
                        entry["quality"] = float(weights[i])
                    samples.append(entry)
            else:
                for i in range(min(len(data), 50_000)):
                    token_ids = data[i].tolist()
                    entry = {"token_len": len(token_ids)}
                    if weights is not None and i < len(weights):
                        entry["quality"] = float(weights[i])
                    samples.append(entry)
            return samples
        except Exception:
            return []

    @staticmethod
    def _load_npz(path: Path, tokenizer_path: Any) -> list[dict[str, Any]]:
        """Load from numpy .npz archive."""
        try:
            import numpy as np

            archive = np.load(str(path))
            key = list(archive.keys())[0]
            data = archive[key]
            samples: list[dict[str, Any]] = []
            for i in range(min(len(data), 50_000)):
                samples.append({"token_len": int(len(data[i]) if data.ndim > 1 else 512)})
            return samples
        except Exception:
            return []

    @staticmethod
    def _load_text_file(path: Path) -> list[dict[str, Any]]:
        """Load a plain-text file as one sample per line (or the whole file)."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return []
            # If few long lines: treat as multi-line code samples
            if len(lines) < 100 and any(len(ln) > 200 for ln in lines):
                return [{"text": text, "token_len": len(text.split())}]
            return [{"text": ln, "token_len": len(ln.split())} for ln in lines]
        except Exception:
            return []

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load JSONL file with optional 'text', 'tokens', 'quality' fields."""
        samples: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry: dict[str, Any] = {}
                    if "text" in obj:
                        entry["text"] = obj["text"]
                        entry["token_len"] = len(obj["text"].split())
                    if "tokens" in obj:
                        entry["token_len"] = len(obj["tokens"])
                    if "quality" in obj:
                        entry["quality"] = float(obj["quality"])
                    if "score" in obj:
                        entry["quality"] = float(obj["score"])
                    if entry:
                        samples.append(entry)
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        return samples

    def _load_directory(self, path: Path) -> list[dict[str, Any]]:
        """Recursively load code files from a directory."""
        samples: list[dict[str, Any]] = []
        for ext in ("*.py", "*.ts", "*.js", "*.txt"):
            for fpath in sorted(path.rglob(ext))[: self.sample_limit]:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    samples.append({"text": text, "token_len": len(text.split())})
                except Exception:
                    pass
                if len(samples) >= self.sample_limit:
                    break
        return samples

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language_mix(texts: list[str]) -> LanguageMix:
        """Heuristically classify each text as Python/TS/JS/other."""
        counts = {"python": 0, "typescript": 0, "javascript": 0, "other": 0}
        for text in texts:
            lang = _detect_language(text)
            counts[lang] = counts.get(lang, 0) + 1
        total = len(texts) or 1
        return LanguageMix(
            python=counts["python"] / total,
            typescript=counts["typescript"] / total,
            javascript=counts["javascript"] / total,
            other=counts["other"] / total,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Simple heuristic language detector."""
    py_score = len(re.findall(r"\bdef\b|\bclass\b.*:|\bimport\b|\bself\b", text))
    ts_score = len(re.findall(r"\binterface\b|\btype\b\s+\w+\s*=|\bconst\b|\blet\b", text))
    js_score = len(re.findall(r"\bfunction\b|\bvar\b|\brequire\(", text))
    if py_score >= ts_score and py_score >= js_score and py_score > 0:
        return "python"
    if ts_score > js_score and ts_score > 0:
        return "typescript"
    if js_score > 0:
        return "javascript"
    return "other"


def _compute_stats(values: list[int | float]) -> LengthStats:
    """Compute descriptive statistics for a list of numeric values."""
    if not values:
        return LengthStats()
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n

    def pct(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    return LengthStats(
        count=n,
        mean=mean,
        median=pct(50),
        std=math.sqrt(variance),
        p10=pct(10),
        p25=pct(25),
        p75=pct(75),
        p90=pct(90),
        p99=pct(99),
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
    )


def _quality_buckets(scores: list[float]) -> QualityBuckets:
    n = len(scores) or 1
    excellent = sum(1 for s in scores if s >= 0.8) / n
    good = sum(1 for s in scores if 0.6 <= s < 0.8) / n
    average = sum(1 for s in scores if 0.4 <= s < 0.6) / n
    poor = sum(1 for s in scores if 0.2 <= s < 0.4) / n
    reject = sum(1 for s in scores if s < 0.2) / n
    return QualityBuckets(
        excellent=excellent, good=good, average=average, poor=poor, reject=reject
    )


def _stats_table(stats: LengthStats, label: str) -> list[str]:
    return [
        f"| Metric | {label} |",
        "|--------|---------|",
        f"| Count | {stats.count:,} |",
        f"| Mean | {stats.mean:.1f} |",
        f"| Median | {stats.median:.1f} |",
        f"| Std | {stats.std:.1f} |",
        f"| Min | {stats.min_val:.0f} |",
        f"| P10 | {stats.p10:.0f} |",
        f"| P25 | {stats.p25:.0f} |",
        f"| P75 | {stats.p75:.0f} |",
        f"| P90 | {stats.p90:.0f} |",
        f"| P99 | {stats.p99:.0f} |",
        f"| Max | {stats.max_val:.0f} |",
    ]
