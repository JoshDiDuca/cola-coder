"""Quality filter plugin — wraps the existing quality_filter.py.

Does NOT modify quality_filter.py. Just adapts it to the FilterPlugin
interface so it can be used in pipeline YAML configs.
"""

from __future__ import annotations

from typing import Any

from cola_coder.data.pipeline import DataRecord, FilterPlugin
from cola_coder.data.registry import register_filter


@register_filter("quality")
class QualityFilterPlugin(FilterPlugin):
    """Wraps the existing quality_filter.filter_code() as a pipeline plugin.

    Config options (via setup() or YAML):
        mode: "conservative" (default) or "strict"
        languages: list of languages for language-aware checks
    """

    def __init__(self) -> None:
        self._mode_str: str = "conservative"
        self._languages: list[str] | None = None

    def name(self) -> str:
        return f"quality({self._mode_str})"

    def setup(self, config: dict[str, Any]) -> None:
        self._mode_str = config.get("mode", "conservative")
        self._languages = config.get("languages", None)

    def check(self, record: DataRecord) -> tuple[bool, str]:
        from cola_coder.data.quality_filter import FilterMode, filter_code

        mode = FilterMode(self._mode_str)
        # Prefer the pipeline-config languages; otherwise fall back to the
        # record's OWN language. Sources set the canonical SINGULAR "language"
        # key (DATA-007/008); the previous fallback read a plural "languages"
        # key that no source sets, so it was always None and the language-aware
        # quality checks never engaged. filter_code expects a list.
        languages = self._languages
        if languages is None:
            rec_lang = record.metadata.get("language")
            languages = [rec_lang] if rec_lang else None
        return filter_code(record.content, mode=mode, languages=languages)
