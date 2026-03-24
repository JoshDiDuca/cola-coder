"""DatasetResolver: single source of truth for per-dataset storage path resolution.

Derives a stable folder name from configs/data_sources.yaml and provides
paths for the tokenizer and dataset directory.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cola_coder.model.config import get_storage_config


class DatasetResolver:
    @staticmethod
    def get_dataset_name(
        data_sources_path: str | Path = "configs/data_sources.yaml",
    ) -> str:
        """Derive stable folder name from active sources in data_sources.yaml.

        Algorithm:
        1. Load the YAML file at data_sources_path
        2. Get enabled code languages (sorted alphabetically)
        3. Get enabled non-code source names (text, math, etc. in definition order)
        4. Join with hyphens: "javascript-typescript-text-math"
        5. If data_sources.yaml not found or parse error: return "default"
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

        parts: list[str] = []

        # Enabled code languages — sorted alphabetically
        code_source = sources.get("code", {})
        if isinstance(code_source, dict) and code_source.get("enabled", False):
            languages: list[str] = code_source.get("languages", [])
            if isinstance(languages, list):
                parts.extend(
                    re.sub(r"[^\w-]", "_", str(lang))
                    for lang in sorted(str(lang) for lang in languages)
                )

        # Enabled non-code sources — in definition order
        for name, config in sources.items():
            if name == "code":
                continue
            if isinstance(config, dict) and config.get("enabled", False):
                parts.append(re.sub(r"[^\w-]", "_", str(name)))

        if not parts:
            return "default"

        return "-".join(parts)

    @staticmethod
    def get_dataset_dir(
        data_sources_path: str | Path = "configs/data_sources.yaml",
    ) -> Path:
        """Get per-dataset directory under storage.data_dir.

        Returns: storage.data_dir / get_dataset_name(data_sources_path)
        Creates the directory (mkdir parents=True, exist_ok=True) before returning.
        """
        storage = get_storage_config()
        base_dir = Path(storage.data_dir)
        dataset_name = DatasetResolver.get_dataset_name(data_sources_path)
        dataset_dir = base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    @staticmethod
    def get_tokenizer_path(
        data_sources_path: str | Path = "configs/data_sources.yaml",
    ) -> Path:
        """Get tokenizer.json path inside the dataset directory."""
        return DatasetResolver.get_dataset_dir(data_sources_path) / "tokenizer.json"

    @staticmethod
    def tokenizer_exists(
        data_sources_path: str | Path = "configs/data_sources.yaml",
    ) -> bool:
        """Return True if tokenizer.json exists in the dataset directory."""
        return DatasetResolver.get_tokenizer_path(data_sources_path).exists()

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
