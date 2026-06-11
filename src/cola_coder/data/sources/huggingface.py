"""HuggingFace dataset source plugin.

Wraps the existing download.py logic as a DataSource plugin.
Does NOT modify download.py — just calls its functions.
"""

from __future__ import annotations

from typing import Iterator

from cola_coder.data.pipeline import DataRecord, DataSource
from cola_coder.data.registry import register_source


@register_source("huggingface")
class HuggingFaceSource(DataSource):
    """Stream code files from a HuggingFace dataset.

    Wraps the existing stream_code_data() function from download.py,
    converting raw content strings into DataRecord objects.

    Args:
        dataset: HuggingFace dataset identifier (e.g. "bigcode/starcoderdata").
        languages: List of programming languages to include.
        split: Dataset split (default: "train").
        max_samples: Stop after this many samples (for testing).
        streaming: Force slow HTTP streaming mode.
    """

    def __init__(
        self,
        dataset: str = "bigcode/starcoderdata",
        languages: list[str] | None = None,
        split: str = "train",
        max_samples: int | None = None,
        streaming: bool = False,
    ):
        self._dataset = dataset
        self._languages = languages or ["python"]
        self._split = split
        self._max_samples = max_samples
        self._streaming = streaming

    def name(self) -> str:
        langs = ", ".join(self._languages)
        return f"huggingface({self._dataset}, [{langs}])"

    def stream(self) -> Iterator[DataRecord]:
        """Yield DataRecords by wrapping the existing download.py stream."""
        from cola_coder.data.download import stream_code_data

        # When a single language was requested, every yielded record IS that
        # language — tag it so downstream language-aware scorers/filters don't
        # have to guess from content. (stream_code_data yields only content
        # strings, so per-record language isn't recoverable for multi-language
        # sources; the project convention builds one source per language anyway.)
        single_lang = self._languages[0] if len(self._languages) == 1 else None

        for content in stream_code_data(
            dataset_name=self._dataset,
            languages=self._languages,
            split=self._split,
            max_samples=self._max_samples,
            streaming=self._streaming,
        ):
            metadata = {
                "source": "huggingface",
                "dataset": self._dataset,
            }
            if single_lang is not None:
                metadata["language"] = single_lang
            yield DataRecord(content=content, metadata=metadata)

    def estimate_size(self) -> int | None:
        return self._max_samples
