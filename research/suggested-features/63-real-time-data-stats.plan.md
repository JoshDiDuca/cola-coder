# 63 - Real-Time Data Stats

## Overview

During data preparation (`prepare.py`), display live statistics in the terminal using Rich's `Live` display. Show file processing progress, rejection rates, token distribution, processing speed, and estimated completion time—all updating in real time without blocking the preparation pipeline.

**Feature flag:** `--live-stats` / `--no-live-stats` (enabled by default when TTY detected)

---

## Motivation

Currently `prepare.py` produces little or no output until it completes, making it a black box. For large corpora (50k+ files), a 10-minute data prep run with no feedback is painful. You can't tell if it's stuck, if the filter is too aggressive, or if the dataset will have enough tokens.

Live stats serve several purposes:
- **Early termination**: if rejection rate hits 90% in first 1000 files, kill the run and fix the filter config before wasting 9 more minutes
- **Quality feedback**: watch the token distribution shift as different repo types are processed
- **Debugging**: see which rejection reason dominates (too_short vs duplicate vs low_quality) and tune thresholds accordingly
- **Progress confidence**: know how long the run will take based on files/sec rate

---

## Architecture / Design

### Display Layout

```
╔═══════════════════════════════════════════════════════════╗
║  Cola-Coder Data Preparation                               ║
╠══════════════════╦════════════════════════════════════════╣
║ Progress         ║ Processing Speed                       ║
║ ████████░░ 82%   ║ 847 files/sec  |  1.2M tokens/sec     ║
║ 41,234 / 50,000  ║ ETA: 0:01:23                           ║
╠══════════════════╬════════════════════════════════════════╣
║ Rejection Rate   ║ Token Distribution                     ║
║ ████░░░░░░ 38%   ║  0-128  ██████████████  45%            ║
║ 15,669 rejected  ║128-256  ████████████    38%            ║
║ 25,565 accepted  ║256-512  ████            14%            ║
╠══════════════════╬512-1024 █                3%            ║
║ Top Rejections   ║                                        ║
║ too_short  52%   ║                                        ║
║ duplicate  31%   ║                                        ║
║ low_qual   17%   ║                                        ║
╚══════════════════╩════════════════════════════════════════╝
```

### Update Strategy

- Update display every **100 files** to avoid Rich overhead dominating the pipeline
- All counters are simple integers incremented in the processing loop; zero thread contention
- Non-blocking: the display thread reads counters; the processing thread writes them
- On non-TTY (piped output, CI), fall back to periodic line-based progress logs every 5000 files

---

## Implementation Steps

### Step 1: Stats Accumulator (`data/stats_tracker.py`)

```python
import time
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock

@dataclass
class DataPrepStats:
    """Thread-safe statistics accumulator for data preparation."""
    total_files: int = 0          # total files to process (known upfront)
    files_processed: int = 0
    files_accepted: int = 0
    files_rejected: int = 0
    total_tokens: int = 0

    rejection_reasons: dict = field(default_factory=lambda: defaultdict(int))

    # Token length buckets: [0,128), [128,256), [256,512), [512,1024), [1024+)
    token_buckets: list = field(default_factory=lambda: [0, 0, 0, 0, 0])
    bucket_edges: tuple = (128, 256, 512, 1024)

    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    last_update_files: int = 0
    last_update_tokens: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_accepted(self, token_count: int):
        with self._lock:
            self.files_processed += 1
            self.files_accepted += 1
            self.total_tokens += token_count
            bucket = sum(1 for edge in self.bucket_edges if token_count >= edge)
            self.token_buckets[bucket] += 1

    def record_rejected(self, reason: str):
        with self._lock:
            self.files_processed += 1
            self.files_rejected += 1
            self.rejection_reasons[reason] += 1

    def snapshot(self) -> dict:
        """Return a consistent read of all stats."""
        with self._lock:
            now = time.time()
            elapsed = now - self.start_time
            interval = now - self.last_update_time
            files_delta = self.files_processed - self.last_update_files
            tokens_delta = self.total_tokens - self.last_update_tokens

            files_per_sec = files_delta / interval if interval > 0 else 0
            tokens_per_sec = tokens_delta / interval if interval > 0 else 0

            remaining = self.total_files - self.files_processed
            eta_sec = remaining / files_per_sec if files_per_sec > 0 else float("inf")

            rejection_rate = (
                self.files_rejected / self.files_processed
                if self.files_processed > 0 else 0.0
            )

            # Compute top rejection reasons
            top_reasons = sorted(
                self.rejection_reasons.items(),
                key=lambda x: -x[1]
            )[:5]

            self.last_update_time = now
            self.last_update_files = self.files_processed
            self.last_update_tokens = self.total_tokens

            return {
                "total_files": self.total_files,
                "files_processed": self.files_processed,
                "files_accepted": self.files_accepted,
                "files_rejected": self.files_rejected,
                "total_tokens": self.total_tokens,
                "rejection_rate": rejection_rate,
                "files_per_sec": files_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "eta_sec": eta_sec,
                "elapsed_sec": elapsed,
                "top_reasons": top_reasons,
                "token_buckets": list(self.token_buckets),
            }
```

### Step 2: Rich Live Display (`data/live_display.py`)

```python
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich import box
import time

def _format_eta(seconds: float) -> str:
    if seconds == float("inf"):
        return "calculating..."
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def _mini_bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for inline use."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def build_display(stats: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="progress", size=6),
        Layout(name="rejection", size=6),
        Layout(name="top_reasons"),
    )
    layout["right"].split_column(
        Layout(name="speed", size=6),
        Layout(name="tokens"),
    )

    # Header
    layout["header"].update(Panel(
        "[bold cyan]Cola-Coder Data Preparation[/] — "
        f"[dim]elapsed {_format_eta(stats['elapsed_sec'])}[/]",
        box=box.HORIZONTALS
    ))

    # Progress panel
    pct = stats["files_processed"] / max(stats["total_files"], 1)
    progress_text = Text()
    progress_text.append(f"{_mini_bar(pct, 24)} {pct*100:.0f}%\n", style="green")
    progress_text.append(
        f"{stats['files_processed']:,} / {stats['total_files']:,} files"
    )
    layout["progress"].update(Panel(progress_text, title="Progress"))

    # Rejection panel
    rr = stats["rejection_rate"]
    rejection_text = Text()
    rejection_text.append(f"{_mini_bar(rr, 24)} {rr*100:.0f}%\n",
                          style="red" if rr > 0.5 else "yellow" if rr > 0.3 else "green")
    rejection_text.append(
        f"{stats['files_rejected']:,} rejected  {stats['files_accepted']:,} accepted"
    )
    layout["rejection"].update(Panel(rejection_text, title="Rejection Rate"))

    # Top rejection reasons
    reasons_table = Table(box=None, show_header=False, padding=(0, 1))
    reasons_table.add_column(style="dim")
    reasons_table.add_column(justify="right")
    reasons_table.add_column()
    total_rejected = max(stats["files_rejected"], 1)
    for reason, count in stats["top_reasons"]:
        pct_r = count / total_rejected
        reasons_table.add_row(
            reason, str(count), f"[dim]{pct_r*100:.0f}%[/]"
        )
    layout["top_reasons"].update(Panel(reasons_table, title="Top Rejections"))

    # Speed panel
    speed_text = Text()
    speed_text.append(f"{stats['files_per_sec']:.0f} files/sec\n", style="cyan")
    speed_text.append(f"{stats['tokens_per_sec']/1000:.1f}K tokens/sec\n", style="cyan")
    speed_text.append(f"ETA: {_format_eta(stats['eta_sec'])}", style="bold")
    layout["speed"].update(Panel(speed_text, title="Processing Speed"))

    # Token distribution histogram
    buckets = stats["token_buckets"]
    total_accepted = max(stats["files_accepted"], 1)
    bucket_labels = ["  0-128", "128-256", "256-512", "512-1024", "1024+  "]
    hist_table = Table(box=None, show_header=False, padding=(0, 0))
    hist_table.add_column(style="dim", width=8)
    hist_table.add_column()
    hist_table.add_column(justify="right", width=5)
    for label, count in zip(bucket_labels, buckets):
        frac = count / total_accepted
        bar = "█" * int(frac * 30)
        hist_table.add_row(
            label, f"[blue]{bar}[/]", f"[dim]{frac*100:.0f}%[/]"
        )
    layout["tokens"].update(Panel(hist_table, title="Token Distribution"))

    return layout

class DataPrepLiveDisplay:
    def __init__(self, stats: "DataPrepStats", refresh_every_n_files: int = 100):
        self.stats = stats
        self.refresh_every = refresh_every_n_files
        self._console = Console()
        self._live = None
        self._last_refresh = 0

    def __enter__(self):
        self._live = Live(
            build_display(self.stats.snapshot()),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        # Final update with complete stats
        self._live.update(build_display(self.stats.snapshot()))
        self._live.__exit__(*args)
        self._print_summary()

    def maybe_refresh(self):
        """Call after each file. Refreshes every N files."""
        if self.stats.files_processed - self._last_refresh >= self.refresh_every:
            self._live.update(build_display(self.stats.snapshot()))
            self._last_refresh = self.stats.files_processed

    def _print_summary(self):
        snap = self.stats.snapshot()
        self._console.rule("[bold green]Preparation Complete[/]")
        self._console.print(
            f"  Files accepted:   [green]{snap['files_accepted']:,}[/]\n"
            f"  Files rejected:   [red]{snap['files_rejected']:,}[/] "
            f"({snap['rejection_rate']*100:.1f}%)\n"
            f"  Total tokens:     [cyan]{snap['total_tokens']:,}[/]\n"
            f"  Time elapsed:     {_format_eta(snap['elapsed_sec'])}"
        )
```

### Step 3: Integration in `prepare.py`

```python
from data.stats_tracker import DataPrepStats
from data.live_display import DataPrepLiveDisplay
import sys

def prepare_data(config, args):
    all_files = list(scan_files(config.source_paths))
    stats = DataPrepStats(total_files=len(all_files))
    use_live = sys.stdout.isatty() and not args.no_live_stats

    ctx = DataPrepLiveDisplay(stats) if use_live else nullcontext()

    with ctx as display:
        for file_path in all_files:
            tokens = tokenize_file(file_path)
            if tokens is None:
                stats.record_rejected("parse_error")
            elif len(tokens) < config.min_tokens:
                stats.record_rejected("too_short")
            elif len(tokens) > config.max_tokens:
                stats.record_rejected("too_long")
            elif is_duplicate(tokens, dedup_index):
                stats.record_rejected("duplicate")
            else:
                stats.record_accepted(len(tokens))
                output_buffer.append(tokens)

            if use_live:
                display.maybe_refresh()
```

---

## Key Files to Modify

- `data/prepare.py` - Integrate `DataPrepStats` and `DataPrepLiveDisplay`
- `data/stats_tracker.py` - New file: stats accumulator
- `data/live_display.py` - New file: Rich Live display
- `cli/prepare_cmd.py` - Add `--no-live-stats` flag
- `config/data.yaml` - Add `live_stats.refresh_interval` setting

---

## Testing Strategy

1. **Stats accumulator unit tests**: call `record_accepted` / `record_rejected` N times, assert all counters correct.
2. **Thread safety test**: run 4 threads each calling `record_accepted` 1000 times concurrently, assert `files_processed == 4000`.
3. **Display snapshot test**: verify `snapshot()` returns consistent data (no partial updates visible).
4. **ETA accuracy test**: simulate 1000 files at known speed, verify ETA is within 10% of true remaining time.
5. **Non-TTY fallback test**: redirect stdout to `/dev/null`, verify `DataPrepLiveDisplay` does not crash and falls back to plain logging.
6. **Full integration test**: run `prepare.py` on a 100-file fixture corpus with live stats enabled, assert final summary matches manual count.

---

## Performance Considerations

- Updating every 100 files means at most 500 Rich renders for a 50k-file corpus. Negligible overhead.
- `DataPrepStats` uses a `Lock` for thread safety. If preparation runs single-threaded, replace with a plain counter to eliminate lock contention.
- The display layout is rebuilt from scratch on each refresh. Pre-allocate `Table` and `Text` objects and mutate them instead, reducing GC pressure for very frequent refreshes.
- On Windows (where Rich rendering can be slower), increase `refresh_every_n_files` to 500.

---

## Dependencies

No new dependencies. Requires `rich>=13.0` (already a Cola-Coder dependency).

---

## Estimated Complexity

**Low.** This is a pure UI feature layered on top of existing data preparation logic. The stats accumulator is simple, the Rich layout is self-contained. No changes to core data pipeline logic required. Estimated implementation time: 1-2 days.

---

## 2026 Best Practices

- **Non-blocking display**: never let the display update block the data pipeline. Rich's `Live` handles this internally; call `live.update()` rather than re-rendering to a new console.
- **Graceful TTY detection**: always check `sys.stdout.isatty()` before enabling live display. In CI/CD or piped contexts, fall back to timestamped log lines.
- **Rejection reason taxonomy**: define a fixed vocabulary of rejection reasons (enum or constants) to avoid typos creating phantom reason categories in stats.
- **Update interval as config**: expose `refresh_every_n_files` in config YAML so users on slower machines can reduce update frequency without code changes.
- **Final summary always printed**: even if the live display is disabled, always print a one-line completion summary with accepted/rejected counts and total tokens. This is the most important output.
