"""Training Data Balancer (improvement #66).

Analyzes and rebalances training data distribution by:
  - programming language
  - difficulty (proxy: cyclomatic complexity / file size)
  - file size (lines / bytes)
  - topic/category (inferred from keywords or labels)

Generates per-sample sampling weights so the training loop sees a
balanced distribution.

TypeScript analogy: like a weighted random sampler in a DataLoader —
analogous to using class_weight='balanced' in sklearn but with finer
control over the dimensions we want to balance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Feature toggle (project convention)
# ---------------------------------------------------------------------------

FEATURE_ENABLED = True


def is_enabled() -> bool:
    """Return True if data balancing is active."""
    return FEATURE_ENABLED


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SampleMetadata:
    """Metadata for a single training sample."""

    idx: int
    language: str = "unknown"
    difficulty: float = 0.5        # 0.0 (easy) – 1.0 (hard)
    size_bytes: int = 0
    topic: str = "general"


@dataclass
class BalancerReport:
    """Report from the data balancer."""

    num_samples: int
    language_counts: Dict[str, int] = field(default_factory=dict)
    topic_counts: Dict[str, int] = field(default_factory=dict)
    difficulty_histogram: Dict[str, int] = field(default_factory=dict)
    size_histogram: Dict[str, int] = field(default_factory=dict)
    weights: List[float] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def effective_samples(self) -> float:
        """Kish effective sample size (inverse participation ratio)."""
        if not self.weights:
            return float(self.num_samples)
        s = sum(self.weights)
        s2 = sum(w**2 for w in self.weights)
        return s**2 / max(s2, 1e-10)


# ---------------------------------------------------------------------------
# Language detection (simple heuristics)
# ---------------------------------------------------------------------------

_LANG_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
}

_LANG_SHEBANGS = {
    "python": "python",
    "node": "javascript",
    "ruby": "ruby",
}


def detect_language(source: str, filename: str = "") -> str:
    """Detect programming language from filename or source heuristics."""
    if filename:
        for ext, lang in _LANG_EXTENSIONS.items():
            if filename.endswith(ext):
                return lang
    # Shebang
    first_line = source.split("\n", 1)[0]
    if first_line.startswith("#!"):
        for key, lang in _LANG_SHEBANGS.items():
            if key in first_line:
                return lang
    # Python heuristics
    if "def " in source and "import " in source:
        return "python"
    if "function " in source or "const " in source or "=>" in source:
        return "javascript"
    return "unknown"


# ---------------------------------------------------------------------------
# Difficulty estimator
# ---------------------------------------------------------------------------


def estimate_difficulty(source: str) -> float:
    """Estimate code difficulty as a 0-1 score.

    Uses proxy metrics: nesting depth, unique tokens, line count.
    """
    lines = source.splitlines()
    if not lines:
        return 0.0
    # Nesting depth proxy: max leading spaces / 4
    max_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
    depth_score = min(max_indent / 32, 1.0)  # normalise to [0,1]

    # Unique token ratio
    tokens = source.split()
    unique_ratio = min(len(set(tokens)) / max(len(tokens), 1), 1.0)

    # Line count factor
    size_factor = min(len(lines) / 500, 1.0)

    return (depth_score * 0.4 + unique_ratio * 0.4 + size_factor * 0.2)


# ---------------------------------------------------------------------------
# Balancer
# ---------------------------------------------------------------------------


class DataBalancer:
    """Compute per-sample weights to balance training data distribution.

    Parameters
    ----------
    balance_language:
        Weight to give language balancing (0–1).
    balance_difficulty:
        Weight to give difficulty balancing.
    balance_size:
        Weight to give file-size balancing.
    balance_topic:
        Weight to give topic balancing.
    min_weight:
        Floor weight so rare samples don't get zero weight.
    max_weight:
        Cap weight so common samples don't dominate.
    """

    def __init__(
        self,
        balance_language: float = 0.4,
        balance_difficulty: float = 0.2,
        balance_size: float = 0.2,
        balance_topic: float = 0.2,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
    ) -> None:
        self.balance_language = balance_language
        self.balance_difficulty = balance_difficulty
        self.balance_size = balance_size
        self.balance_topic = balance_topic
        self.min_weight = min_weight
        self.max_weight = max_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, samples: Sequence[SampleMetadata]) -> BalancerReport:
        """Analyze distribution and compute sampling weights."""
        n = len(samples)
        if n == 0:
            return BalancerReport(num_samples=0, weights=[])

        # Count distributions
        lang_counts: Counter[str] = Counter(s.language for s in samples)
        topic_counts: Counter[str] = Counter(s.topic for s in samples)
        difficulties = [s.difficulty for s in samples]
        sizes = [s.size_bytes for s in samples]

        diff_hist = self._histogram(difficulties, bins=5, labels=["trivial", "easy", "medium", "hard", "expert"])
        size_hist = self._size_histogram(sizes)

        # Compute inverse-frequency weights per dimension
        lang_weights = self._inverse_freq_weights(lang_counts, n)
        topic_weights = self._inverse_freq_weights(topic_counts, n)
        diff_weights = self._difficulty_weights(difficulties)
        size_weights = self._size_weights(sizes)

        # Combine
        weights: List[float] = []
        for s, dw, sw in zip(samples, diff_weights, size_weights):
            w = (
                self.balance_language * lang_weights.get(s.language, 1.0)
                + self.balance_difficulty * dw
                + self.balance_size * sw
                + self.balance_topic * topic_weights.get(s.topic, 1.0)
            )
            w = max(self.min_weight, min(self.max_weight, w))
            weights.append(w)

        issues = self._detect_issues(lang_counts, topic_counts, n)

        return BalancerReport(
            num_samples=n,
            language_counts=dict(lang_counts),
            topic_counts=dict(topic_counts),
            difficulty_histogram=diff_hist,
            size_histogram=size_hist,
            weights=weights,
            issues=issues,
        )

    def from_sources(
        self,
        sources: Sequence[str],
        filenames: Optional[Sequence[str]] = None,
        topics: Optional[Sequence[str]] = None,
    ) -> BalancerReport:
        """Build metadata from raw source strings and analyze."""
        fns = list(filenames) if filenames else [""] * len(sources)
        tps = list(topics) if topics else ["general"] * len(sources)
        meta = [
            SampleMetadata(
                idx=i,
                language=detect_language(src, fn),
                difficulty=estimate_difficulty(src),
                size_bytes=len(src.encode("utf-8")),
                topic=tp,
            )
            for i, (src, fn, tp) in enumerate(zip(sources, fns, tps))
        ]
        return self.analyze(meta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse_freq_weights(counts: Counter, total: int) -> Dict[str, float]:
        """Return {category: weight} using inverse frequency."""
        n_classes = len(counts)
        if n_classes == 0:
            return {}
        ideal = total / n_classes
        return {k: ideal / max(v, 1) for k, v in counts.items()}

    @staticmethod
    def _difficulty_weights(difficulties: List[float]) -> List[float]:
        """Weight hard samples slightly more to encourage challenge."""
        if not difficulties:
            return []
        # Bin into 5 buckets, apply inverse frequency
        bins = [0, 0, 0, 0, 0]
        for d in difficulties:
            idx = min(int(d * 5), 4)
            bins[idx] += 1
        total = len(difficulties)
        weights: List[float] = []
        for d in difficulties:
            idx = min(int(d * 5), 4)
            w = (total / 5) / max(bins[idx], 1)
            weights.append(w)
        return weights

    @staticmethod
    def _size_weights(sizes: List[int]) -> List[float]:
        """Penalise very large files slightly."""
        if not sizes:
            return []
        max_size = max(sizes) or 1
        return [1.0 - 0.3 * (s / max_size) for s in sizes]

    @staticmethod
    def _histogram(
        values: List[float], bins: int, labels: List[str]
    ) -> Dict[str, int]:
        hist: Dict[str, int] = defaultdict(int)
        for v in values:
            idx = min(int(v * bins), bins - 1)
            hist[labels[idx]] += 1
        return dict(hist)

    @staticmethod
    def _size_histogram(sizes: List[int]) -> Dict[str, int]:
        hist: Dict[str, int] = defaultdict(int)
        for s in sizes:
            if s < 1_000:
                hist["<1KB"] += 1
            elif s < 10_000:
                hist["1-10KB"] += 1
            elif s < 100_000:
                hist["10-100KB"] += 1
            else:
                hist[">100KB"] += 1
        return dict(hist)

    @staticmethod
    def _detect_issues(
        lang_counts: Counter, topic_counts: Counter, total: int
    ) -> List[str]:
        issues: List[str] = []
        # Check for extreme imbalance
        if lang_counts:
            most_common_count = lang_counts.most_common(1)[0][1]
            if most_common_count / total > 0.8:
                lang = lang_counts.most_common(1)[0][0]
                issues.append(
                    f"Language imbalance: '{lang}' comprises "
                    f"{most_common_count/total:.0%} of samples"
                )
        if topic_counts:
            most_common_count = topic_counts.most_common(1)[0][1]
            if most_common_count / total > 0.7:
                topic = topic_counts.most_common(1)[0][0]
                issues.append(
                    f"Topic imbalance: '{topic}' comprises "
                    f"{most_common_count/total:.0%} of samples"
                )
        if total < 100:
            issues.append(f"Small dataset: only {total} samples")
        return issues


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def compute_weights(
    samples: Sequence[Any],
    language_attr: str = "language",
    topic_attr: str = "topic",
) -> List[float]:
    """Compute sampling weights from a list of objects with language/topic attrs."""
    meta = [
        SampleMetadata(
            idx=i,
            language=getattr(s, language_attr, "unknown"),
            topic=getattr(s, topic_attr, "general"),
        )
        for i, s in enumerate(samples)
    ]
    report = DataBalancer().analyze(meta)
    return report.weights
