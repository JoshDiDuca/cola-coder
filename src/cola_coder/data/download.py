"""Load code data for training.

Two loading strategies:

  LOCAL PARQUET (default): Reads parquet files directly from the HuggingFace
  cache on disk. This is ~1000x faster than streaming because there's zero
  HTTP overhead — it's just reading local files. Requires that the dataset
  has been downloaded at least once (the HF streaming mode downloads and
  caches the parquet files automatically).

  HF STREAMING (--stream): Fetches rows one at a time over HTTP via the
  HuggingFace datasets API. Extremely slow (~200-400 rows/sec) because
  each row is a separate HTTP round-trip. Only use this for the initial
  download or if you have no local cache.

Performance comparison (same data, same machine):
  - Streaming: ~400 files/sec  (7+ hours for 10M files)
  - Local parquet: ~400,000 files/sec  (25 seconds for 10M files)

For a TS dev: streaming is like calling fetch() in a loop with await.
Local parquet is like reading a JSON file from disk with fs.readFileSync().
"""

import os
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Cache path detection
# ---------------------------------------------------------------------------

def _find_hf_hub_dir() -> Path:
    """Return the HuggingFace hub cache directory, respecting StorageConfig.

    Resolution order:
    1. StorageConfig.hf_cache_dir (from configs/storage.yaml)
    2. HF_HOME / HUGGINGFACE_HUB_CACHE env vars
    3. Default: ~/.cache/huggingface/hub
    """
    try:
        from cola_coder.model.config import get_storage_config
        storage = get_storage_config()
        if storage.hf_cache_dir:
            p = Path(storage.hf_cache_dir)
            hub = p / "hub" if not str(p).endswith("hub") else p
            return hub
    except Exception:
        pass

    # Env vars
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        val = os.environ.get(var)
        if val:
            p = Path(val)
            hub = p / "hub" if not str(p).endswith("hub") else p
            if hub.exists():
                return hub

    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_cache_dir(dataset_name: str = "bigcode/starcoderdata") -> Path | None:
    """Find the HuggingFace cache directory for a dataset.

    HF stores downloaded datasets in:
      <hf_hub>/datasets--{org}--{name}/snapshots/{hash}/

    Returns the snapshot directory, or None if not found.
    """
    # Convert "bigcode/starcoderdata" -> "datasets--bigcode--starcoderdata"
    safe_name = f"datasets--{dataset_name.replace('/', '--')}"

    hf_hub = _find_hf_hub_dir()
    dataset_dir = hf_hub / safe_name / "snapshots"

    if not dataset_dir.exists():
        return None

    # Get the latest snapshot (usually just one)
    snapshots = sorted(dataset_dir.iterdir())
    if not snapshots:
        return None

    return snapshots[-1]


# ---------------------------------------------------------------------------
# Auto-download (runs once, then all subsequent runs use local cache)
# ---------------------------------------------------------------------------

def _download_dataset(
    dataset_name: str,
    languages: list[str],
) -> None:
    """Download dataset parquet files from HuggingFace Hub.

    Uses huggingface_hub to download the actual parquet files, which
    get cached in ~/.cache/huggingface/hub/ for fast local access.
    This saturates your bandwidth — much faster than streaming row-by-row.
    """
    from huggingface_hub import snapshot_download

    # Convert language names to the file patterns HF expects
    # starcoderdata stores each language in its own directory
    allow_patterns = []
    for lang in languages:
        allow_patterns.append(f"{lang}/*.parquet")
        allow_patterns.append(f"{lang}/**/*.parquet")

    print(f"  Downloading parquet files for: {', '.join(languages)}")
    print("  This will saturate your bandwidth. First run only — cached after this.")

    # Use configured cache dir if set
    cache_kwargs = {}
    try:
        from cola_coder.model.config import get_storage_config
        storage = get_storage_config()
        if storage.hf_cache_dir:
            cache_kwargs["cache_dir"] = str(Path(storage.hf_cache_dir).resolve())
            print(f"  HF cache dir: {cache_kwargs['cache_dir']}")
    except Exception:
        pass

    try:
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            **cache_kwargs,
        )
        print("  Download complete!")
    except Exception as e:
        print(f"  Download error: {e}")
        print("  If this is an auth error, run: huggingface-cli login")
        raise


# ---------------------------------------------------------------------------
# Local parquet loading (fast path)
# ---------------------------------------------------------------------------

def _iter_parquet_files(
    cache_dir: Path,
    lang: str,
) -> Iterator[str]:
    """Read all parquet files for a language and yield content strings.

    Uses PyArrow to read parquet files directly — no HuggingFace API,
    no HTTP, no overhead. Just local disk reads.
    """
    lang_dir = cache_dir / lang
    if not lang_dir.exists():
        print(f"  Warning: No cached data for {lang} at {lang_dir}")
        return

    parquet_files = sorted(
        f for f in lang_dir.iterdir() if f.suffix == ".parquet"
    )

    if not parquet_files:
        print(f"  Warning: No parquet files found in {lang_dir}")
        return

    print(f"  {lang}: {len(parquet_files)} parquet files on disk")

    for pf in parquet_files:
        table = pq.read_table(str(pf), columns=["content"])
        col = table.column("content")

        for i in range(len(col)):
            content = col[i].as_py()
            if content and len(content) >= 50:
                yield content


# ---------------------------------------------------------------------------
# HF streaming (slow fallback)
# ---------------------------------------------------------------------------

def _iter_hf_streaming(
    dataset_name: str,
    lang: str,
    split: str,
) -> Iterator[str]:
    """Stream from HuggingFace API. Slow but works without local cache."""
    from datasets import load_dataset

    print(f"  Streaming {lang} from {dataset_name} (slow HTTP mode)...")
    ds = load_dataset(
        dataset_name,
        data_dir=lang,
        split=split,
        streaming=True,
        trust_remote_code=False,
    )

    for sample in ds:
        content = sample.get("content", "")
        if content and len(content) >= 50:
            yield content


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stream_code_data(
    dataset_name: str = "bigcode/starcoderdata",
    languages: list[str] | None = None,
    split: str = "train",
    max_samples: int | None = None,
    streaming: bool = False,
    **kwargs,
) -> Iterator[str]:
    """Load code files from a dataset.

    By default, reads directly from the local HuggingFace cache (parquet
    files on disk). Falls back to HF streaming if no cache is found.

    Args:
        dataset_name: HuggingFace dataset identifier.
        languages: Filter to these programming languages.
                   None = use all languages.
        split: Dataset split (only used for streaming fallback).
        max_samples: Stop after this many samples total (for testing/debugging).
        streaming: Force slow HTTP streaming mode. Default: use local cache.

    Yields:
        Code file contents as strings.
    """
    if languages is None:
        languages = ["python"]

    # Try local parquet first (unless streaming forced)
    cache_dir = None if streaming else _find_cache_dir(dataset_name)

    if cache_dir and not streaming:
        # Verify the languages we need are actually cached
        missing = [lang for lang in languages if not (cache_dir / lang).exists()
                   or not any((cache_dir / lang).glob("*.parquet"))]
        if missing:
            print(f"Missing cached data for: {', '.join(missing)}")
            print("Downloading missing languages...")
            _download_dataset(dataset_name, missing)
            # Re-detect cache after download
            cache_dir = _find_cache_dir(dataset_name)

        print(f"Reading from local cache: {cache_dir}")
        print("  (This is ~1000x faster than HTTP streaming)")
    elif not streaming:
        print(f"No local cache found. Downloading {dataset_name}...")
        print(f"  Languages: {', '.join(languages)}")
        _download_dataset(dataset_name, languages)
        cache_dir = _find_cache_dir(dataset_name)
        if not cache_dir:
            print("Download failed. Falling back to HTTP streaming (slow).")
            streaming = True
        else:
            print("Download complete. Reading from local cache.")

    count = 0
    for lang in languages:
        try:
            if streaming:
                source = _iter_hf_streaming(dataset_name, lang, split)
            else:
                source = _iter_parquet_files(cache_dir, lang)

            for content in source:
                yield content
                count += 1

                if max_samples is not None and count >= max_samples:
                    print(f"  Reached sample limit: {max_samples:,}")
                    return

        except Exception as e:
            print(f"  Warning: Error loading {lang}: {e}")
            continue

    print(f"Total: {count:,} code files yielded")


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
