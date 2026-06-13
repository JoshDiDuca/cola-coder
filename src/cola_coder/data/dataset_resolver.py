"""DatasetResolver: single source of truth for per-dataset storage path resolution.

Derives a stable folder name from configs/data_sources.yaml and provides
paths for the tokenizer and dataset directory.

When a model config path is given, the config's ``data.languages`` list takes
precedence over the languages listed in data_sources.yaml.  This lets each
model config produce its own isolated dataset folder (e.g. a typescript-only
config produces ``typescript-text-math/`` even if data_sources.yaml also lists
python and javascript).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cola_coder.model.config import get_storage_config


def _read_config_languages(config_path: str | Path | None) -> list[str] | None:
    """Return data.languages from a model config YAML, or None if unavailable."""
    if config_path is None:
        return None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        langs = cfg.get("data", {}).get("languages")
        if isinstance(langs, list) and langs:
            return [str(lang) for lang in langs]
    except Exception:
        pass
    return None


class DatasetResolver:
    @staticmethod
    def get_dataset_name(
        data_sources_path: str | Path = "configs/data_sources.yaml",
        config_path: str | Path | None = None,
    ) -> str:
        """Derive stable folder name from active sources in data_sources.yaml.

        Algorithm:
        1. Load the YAML file at data_sources_path
        2. Get code languages: config.data.languages (if config_path given) else
           data_sources.yaml code.languages — sorted alphabetically
        3. Get enabled non-code source names (text, math, etc.) in definition order
        4. Join with hyphens: e.g. "typescript-text-math"
        5. If data_sources.yaml not found or parse error: return "default"

        Args:
            data_sources_path: Path to data_sources.yaml.
            config_path: Optional model config YAML.  When present, its
                ``data.languages`` list overrides the code languages from
                data_sources.yaml so that each model config gets its own
                isolated dataset folder.
        """
        try:
            path = Path(data_sources_path)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except (FileNotFoundError, OSError, yaml.YAMLError):
            return "default"

        sources: dict = raw.get("sources", {})
        if not isinstance(sources, dict):
            return "default"

        # Languages from model config take precedence over data_sources.yaml
        config_languages = _read_config_languages(config_path)

        parts: list[str] = []

        # Enabled code languages — sorted alphabetically
        code_source = sources.get("code", {})
        if isinstance(code_source, dict) and code_source.get("enabled", False):
            languages: list[str] = (
                config_languages
                if config_languages is not None
                else code_source.get("languages", [])
            )
            if isinstance(languages, list):
                parts.extend(
                    re.sub(r"[^\w-]", "_", str(lang))
                    for lang in sorted(str(lang) for lang in languages)
                )

        # Enabled non-code sources — in definition order
        for name, source_cfg in sources.items():
            if name == "code":
                continue
            if isinstance(source_cfg, dict) and source_cfg.get("enabled", False):
                parts.append(re.sub(r"[^\w-]", "_", str(name)))

        if not parts:
            return "default"

        return "-".join(parts)

    @staticmethod
    def get_dataset_dir(
        data_sources_path: str | Path = "configs/data_sources.yaml",
        config_path: str | Path | None = None,
    ) -> Path:
        """Get per-dataset directory under storage.data_dir.

        Returns: storage.data_dir / get_dataset_name(data_sources_path, config_path)
        Creates the directory (mkdir parents=True, exist_ok=True) before returning.
        """
        storage = get_storage_config()
        base_dir = Path(storage.data_dir)
        dataset_name = DatasetResolver.get_dataset_name(data_sources_path, config_path)
        dataset_dir = base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    @staticmethod
    def get_tokenizer_path(
        data_sources_path: str | Path = "configs/data_sources.yaml",
        config_path: str | Path | None = None,
    ) -> Path:
        """Get tokenizer.json path inside the dataset directory."""
        return DatasetResolver.get_dataset_dir(data_sources_path, config_path) / "tokenizer.json"

    @staticmethod
    def tokenizer_exists(
        data_sources_path: str | Path = "configs/data_sources.yaml",
        config_path: str | Path | None = None,
    ) -> bool:
        """Return True if tokenizer.json exists in the dataset directory."""
        return DatasetResolver.get_tokenizer_path(data_sources_path, config_path).exists()

    @staticmethod
    def find_dataset_npys(
        data_sources_path: str | Path = "configs/data_sources.yaml",
        config_path: str | Path | None = None,
    ) -> list[Path]:
        """Real training-data ``.npy`` files for the active dataset, sorted.

        Single source of truth for "where are the prepared datasets" — used by
        the menu status panel, dataset inspector, combiner, and router-data
        source so they all agree (BUG-117/DATA-048). Scans the per-dataset dir
        first, falling back to the legacy ``storage.data_dir/processed/`` for
        older setups. EXCLUDES ``.weights``/``.scores`` sidecars (they are not
        datasets). Returns an empty list when nothing is prepared yet.
        """
        def _real(d: Path) -> list[Path]:
            if not d.exists():
                return []
            return sorted(
                f for f in d.glob("*.npy")
                if ".weights" not in f.name and ".scores" not in f.name
            )

        found = _real(DatasetResolver.get_dataset_dir(data_sources_path, config_path))
        if found:
            return found
        return _real(Path(get_storage_config().data_dir) / "processed")

    @staticmethod
    def save_tokenizer_meta(
        tokenizer_path: Path,
        vocab_size: int,
        sources: list[str],
        num_samples: int,
    ) -> None:
        """Write tokenizer_meta.json alongside tokenizer.json.

        Content: {"vocab_size": N, "sources": [...], "num_samples": N, "trained_at": "ISO timestamp"}
        File: tokenizer_path.parent / "tokenizer_meta.json"
        """
        meta: dict[str, object] = {
            "vocab_size": vocab_size,
            "sources": sources,
            "num_samples": num_samples,
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        meta_path = tokenizer_path.parent / "tokenizer_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def get_tokenizer_meta(tokenizer_path: Path) -> dict[str, object]:
        """Read tokenizer_meta.json. Returns {} if missing or parse error."""
        meta_path = tokenizer_path.parent / "tokenizer_meta.json"
        try:
            with open(meta_path) as f:
                result = json.load(f)
            if isinstance(result, dict):
                return result
            return {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
