"""Download code data from HuggingFace for training.

We use the `datasets` library to stream data from HuggingFace Hub.
Streaming means we don't need to download the entire dataset upfront —
we process it chunk by chunk and save the tokenized result.

The primary dataset is `bigcode/starcoderdata` — a curated, deduplicated
collection of code from GitHub. It includes code in 80+ languages and
is the same data used to train StarCoder.
"""

from pathlib import Path
from typing import Iterator

from datasets import load_dataset


def stream_code_data(
    dataset_name: str = "bigcode/starcoderdata",
    languages: list[str] | None = None,
    split: str = "train",
    max_samples: int | None = None,
) -> Iterator[str]:
    """Stream code files from a HuggingFace dataset.

    This is a Python generator (like a TS generator/async iterator).
    It yields one code file at a time without loading everything into memory.

    Args:
        dataset_name: HuggingFace dataset identifier.
        languages: Filter to these programming languages.
                   None = use all languages.
        split: Dataset split ("train" for training data).
        max_samples: Stop after this many samples (for testing/debugging).

    Yields:
        Code file contents as strings.
    """
    if languages is None:
        languages = ["python"]

    count = 0
    for lang in languages:
        print(f"Streaming {lang} data from {dataset_name}...")
        try:
            # streaming=True means we don't download the whole thing
            ds = load_dataset(
                dataset_name,
                data_dir=lang,
                split=split,
                streaming=True,
                trust_remote_code=False,  # Security: don't run arbitrary code
            )

            for sample in ds:
                content = sample.get("content", "")
                if not content or len(content) < 50:
                    continue  # Skip very short files (likely not useful)

                yield content
                count += 1

                if max_samples is not None and count >= max_samples:
                    return

        except Exception as e:
            print(f"Warning: Could not load {lang} data: {e}")
            continue

    print(f"Streamed {count} code files total")


def download_sample_data(
    output_dir: str = "./data/raw",
    languages: list[str] | None = None,
    num_samples: int = 10000,
) -> list[str]:
    """Download a sample of code data and save as text files.

    Useful for tokenizer training (needs files on disk) and for quick
    experimentation without waiting for the full streaming pipeline.

    Args:
        output_dir: Where to save the downloaded files.
        languages: Languages to download.
        num_samples: How many code files to download.

    Returns:
        List of file paths that were created.
    """
    if languages is None:
        languages = ["python", "typescript", "javascript"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_paths = []
    samples_per_lang = num_samples // len(languages)

    for lang in languages:
        lang_dir = out_path / lang
        lang_dir.mkdir(exist_ok=True)

        print(f"Downloading {samples_per_lang} {lang} files...")
        for i, content in enumerate(
            stream_code_data(languages=[lang], max_samples=samples_per_lang)
        ):
            file_path = lang_dir / f"{i:06d}.txt"
            file_path.write_text(content, encoding="utf-8")
            file_paths.append(str(file_path))

    print(f"Downloaded {len(file_paths)} files to {output_dir}")
    return file_paths
