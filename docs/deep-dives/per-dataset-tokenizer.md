# Per-Dataset Tokenizers: One Vocabulary Per Config

Most code model tutorials tell you to train a single tokenizer and use it
everywhere. That works fine when you have one language and one dataset. But
the moment you train a TypeScript-focused model and a Python-focused model
from the same codebase, you hit a problem: the tokenizer trained on
TypeScript+JavaScript+Python wastes vocabulary on Python keywords that your
TypeScript model never sees, and vice versa.

cola-coder solves this by giving each dataset configuration its own
tokenizer. The tokenizer lives *inside* the dataset directory, alongside
the training data it was built from. When you change languages or data
sources in your config, the system derives a new dataset name, creates a
new directory, and trains a fresh tokenizer for that exact combination.

This guide covers how it works, why it matters, and how to use it.

---

## Table of Contents

1. [Why Per-Dataset Tokenizers Matter](#1-why-per-dataset-tokenizers-matter)
2. [Dataset Naming: From Config to Folder Name](#2-dataset-naming-from-config-to-folder-name)
3. [Directory Structure](#3-directory-structure)
4. [The DatasetResolver Class](#4-the-datasetresolver-class)
5. [Model Config Language Override](#5-model-config-language-override)
6. [Tokenizer Training: BPE Under the Hood](#6-tokenizer-training-bpe-under-the-hood)
7. [Pipeline Integration](#7-pipeline-integration)
8. [When to Share vs Separate Tokenizers](#8-when-to-share-vs-separate-tokenizers)
9. [Configuration: data_sources.yaml + Model Config](#9-configuration-data_sourcesyaml--model-config)
10. [Tokenizer Comparison and Debugging](#10-tokenizer-comparison-and-debugging)

---

## 1. Why Per-Dataset Tokenizers Matter

A tokenizer converts text into numbers. Specifically, it maintains a
vocabulary of "tokens" -- subword units like `function`, `const`, ` {`,
`\n    ` -- and maps each to an integer ID. The model sees only these
integers during training.

The vocabulary is a fixed-size table (32,768 entries by default in
cola-coder). Every slot in that table costs model parameters -- the
embedding layer has `vocab_size * model_dim` weights. Wasting vocabulary
slots on tokens the model never sees means wasting parameters.

### Domain-specific vocabulary

Consider the difference between TypeScript-heavy and Python-heavy code:

| TypeScript tokens | Python tokens |
|-------------------|---------------|
| `interface` | `def` |
| `readonly` | `self` |
| `: string` | `-> str` |
| `?.` (optional chaining) | `__init__` |
| `<T>` (generics) | `@property` |
| `async () =>` | `async def` |

A tokenizer trained on TypeScript will learn to encode `interface` as a
single token. A tokenizer trained on Python will split it into
`inter` + `face` because it never saw the word often enough to merge it.
That means TypeScript code tokenized with a Python tokenizer produces
longer sequences -- more tokens per line of code -- which means:

1. **Slower training.** More tokens = more forward/backward passes per
   file.
2. **Shorter effective context.** If your model sees 2,048 tokens at a
   time and each line takes 10 tokens instead of 6, you see 40% fewer
   lines of code in each training window.
3. **Worse compression ratio.** The model has to spend capacity learning
   to reassemble sub-tokens that should have been single units.

### Real-world impact

Here is a concrete measurement. Take this TypeScript snippet:

```typescript
export interface UserConfig {
  readonly apiKey: string;
  timeout?: number;
}
```

| Tokenizer trained on... | Token count | Compression ratio |
|-------------------------|-------------|-------------------|
| TypeScript + JS only | 18 tokens | 3.8 chars/token |
| TypeScript + JS + Python | 22 tokens | 3.1 chars/token |
| Python only | 29 tokens | 2.4 chars/token |

That is a 60% increase in token count from the Python tokenizer. Over
millions of training examples, this compounds into meaningful differences
in model quality.

### The cola-coder solution

Instead of training one tokenizer for all possible configurations,
cola-coder trains a separate tokenizer for each unique combination of:

- Programming languages (from `data_sources.yaml` or model config)
- Enabled data sources (code, text, math)

This way, a TypeScript-only model gets a TypeScript-optimized tokenizer,
and a multi-language model gets a broader tokenizer. Each lives in its
own directory, named after the configuration that produced it.

---

## 2. Dataset Naming: From Config to Folder Name

The dataset name is a stable, human-readable string derived from your
configuration. It determines where the tokenizer and training data live
on disk.

### The algorithm

```
1. Read data_sources.yaml
2. Get code languages (sorted alphabetically)
3. Get enabled non-code source names (in definition order)
4. Join everything with hyphens
```

### Examples

Given this `data_sources.yaml`:

```yaml
sources:
  code:
    enabled: true
    languages: [typescript, javascript, python]
  text:
    enabled: true
    dataset: "HuggingFaceFW/fineweb-edu"
  math:
    enabled: true
    dataset: "open-web-math/open-web-math"
```

The dataset name is: **`javascript-python-typescript-text-math`**

Why that order?
- Code languages are sorted alphabetically: `javascript`, `python`, `typescript`
- Non-code sources appear in definition order: `text`, `math`
- Everything is joined with hyphens

More examples:

| Languages | Sources | Dataset name |
|-----------|---------|-------------|
| `[typescript]` | code + text + math | `typescript-text-math` |
| `[python, typescript]` | code only | `python-typescript` |
| `[go, rust, typescript]` | code + text | `go-rust-typescript-text` |
| (none / parse error) | (any) | `default` |

### Why alphabetical sort?

So the name is deterministic regardless of how you order languages in your
config. Whether you write `languages: [typescript, python]` or
`languages: [python, typescript]`, you get the same folder name:
`python-typescript`.

### Special character handling

Language names are sanitized with `re.sub(r"[^\w-]", "_", name)`. So
`C++` becomes `C__` and `C#` becomes `C_`. In practice, the languages
in `data_sources.yaml` are lowercase identifiers (`python`, `typescript`)
so this rarely applies.

---

## 3. Directory Structure

Each dataset configuration produces an isolated directory under
`storage.data_dir`:

```
data/                                    (storage.data_dir)
  typescript-text-math/                  (dataset directory)
    tokenizer.json                       (trained BPE tokenizer)
    tokenizer_meta.json                  (metadata: vocab size, sources, date)
    code_data.npy                        (tokenized code chunks)
    text_data.npy                        (tokenized text chunks)
    math_data.npy                        (tokenized math chunks)
    mixed_train_data.npy                 (combined weighted data)
  python-typescript/                     (different config → different dir)
    tokenizer.json
    tokenizer_meta.json
    code_data.npy
    ...
```

The key insight: **the tokenizer and data live together.** You never
accidentally use a Python tokenizer on TypeScript data because the
tokenizer that trained on TypeScript+text+math is physically located in
the `typescript-text-math/` directory, right next to the data it will
tokenize.

### tokenizer_meta.json

Alongside `tokenizer.json`, the training script saves metadata:

```json
{
  "vocab_size": 32768,
  "sources": ["code", "text", "math"],
  "num_samples": 0,
  "trained_at": "2026-03-15T14:30:00+00:00"
}
```

This tells you at a glance what went into the tokenizer without having to
re-parse the vocabulary. The `sources` field records which data source
iterators were used during training; `trained_at` is an ISO timestamp so
you can tell if the tokenizer is stale relative to your config changes.

---

## 4. The DatasetResolver Class

`DatasetResolver` is the single source of truth for all path resolution.
Every script that needs to find the tokenizer or data directory goes
through this class.

```python
# src/cola_coder/data/dataset_resolver.py

class DatasetResolver:
    @staticmethod
    def get_dataset_name(
        data_sources_path="configs/data_sources.yaml",
        config_path=None,
    ) -> str: ...

    @staticmethod
    def get_dataset_dir(
        data_sources_path="configs/data_sources.yaml",
        config_path=None,
    ) -> Path: ...

    @staticmethod
    def get_tokenizer_path(
        data_sources_path="configs/data_sources.yaml",
        config_path=None,
    ) -> Path: ...

    @staticmethod
    def tokenizer_exists(
        data_sources_path="configs/data_sources.yaml",
        config_path=None,
    ) -> bool: ...

    @staticmethod
    def save_tokenizer_meta(tokenizer_path, vocab_size, sources, num_samples) -> None: ...

    @staticmethod
    def get_tokenizer_meta(tokenizer_path) -> dict: ...
```

### Usage pattern

Every pipeline script follows the same pattern:

```python
from cola_coder.data.dataset_resolver import DatasetResolver

# Get the tokenizer path for this config combination
tok_path = DatasetResolver.get_tokenizer_path(
    data_sources_path="configs/data_sources.yaml",
    config_path="configs/small.yaml",
)
# → data/typescript-text-math/tokenizer.json

# Check if we need to train
if not tok_path.exists():
    print("Train a tokenizer first!")

# Get the data directory
data_dir = DatasetResolver.get_dataset_dir(
    data_sources_path="configs/data_sources.yaml",
    config_path="configs/small.yaml",
)
# → data/typescript-text-math/
```

If you are a TypeScript developer, think of `DatasetResolver` as a
utility module with pure functions:

```typescript
// TypeScript mental model
namespace DatasetResolver {
  function getDatasetName(dataSourcesPath: string, configPath?: string): string;
  function getDatasetDir(dataSourcesPath: string, configPath?: string): string;
  function getTokenizerPath(dataSourcesPath: string, configPath?: string): string;
  function tokenizerExists(dataSourcesPath: string, configPath?: string): boolean;
}
```

### All methods are static

There is no instance state. `DatasetResolver` is a namespace for related
functions, not a stateful object. This makes it safe to call from
anywhere without worrying about initialization order.

### Directory auto-creation

`get_dataset_dir()` calls `mkdir(parents=True, exist_ok=True)` before
returning. This means you never have to manually create directories -- the
first time any script asks for the data directory, it is created.

### Fallback to "default"

If `data_sources.yaml` cannot be found, cannot be parsed, or has no
enabled sources, the dataset name falls back to `"default"`. This
prevents crashes when running scripts without a complete config setup.

---

## 5. Model Config Language Override

Here is where per-dataset tokenizers get interesting. The model config's
`data.languages` list **overrides** the languages from `data_sources.yaml`.

### The problem this solves

Say your `data_sources.yaml` lists three languages:

```yaml
# configs/data_sources.yaml
sources:
  code:
    enabled: true
    languages: [typescript, javascript, python]
```

But your `configs/small.yaml` model config says:

```yaml
# configs/small.yaml
data:
  languages: ["typescript"]
```

You want the small model to train on TypeScript only, with a TypeScript-
optimized tokenizer. But you also want a larger model config that uses all
three languages.

### How the override works

When `config_path` is provided to any `DatasetResolver` method, it reads
the model config's `data.languages` and uses that *instead of* the
`data_sources.yaml` code languages:

```python
# Inside DatasetResolver.get_dataset_name():
config_languages = _read_config_languages(config_path)

# Model config languages take precedence:
languages = (
    config_languages          # ← from configs/small.yaml
    if config_languages is not None
    else code_source.get("languages", [])  # ← from data_sources.yaml
)
```

### The result

```
DatasetResolver.get_dataset_name(config_path="configs/small.yaml")
→ "typescript-text-math"

DatasetResolver.get_dataset_name(config_path="configs/medium.yaml")
→ "javascript-python-typescript-text-math"

DatasetResolver.get_dataset_name(config_path=None)
→ "javascript-python-typescript-text-math"  (uses data_sources.yaml languages)
```

Each model config gets its own dataset directory with its own tokenizer:

```
data/
  typescript-text-math/
    tokenizer.json              ← trained on TS + text + math
  javascript-python-typescript-text-math/
    tokenizer.json              ← trained on JS + Py + TS + text + math
```

### The same override flows through the entire pipeline

It is not just the tokenizer that respects this override. The data
collection script (`collect_data.py`) also reads the model config
languages:

```python
# In collect_data.py:
_cfg_langs = _read_config_languages(args.config)
languages = _cfg_langs if _cfg_langs is not None else code_cfg.get("languages", [...])
```

And the tokenizer training script (`train_tokenizer.py`) does the same:

```python
# In train_tokenizer.py:
config_languages = None
cl = model_cfg.get("data", {}).get("languages")
if isinstance(cl, list) and cl:
    config_languages = [str(lang) for lang in cl]
```

This means the entire chain -- tokenizer training, data collection, data
preparation -- is driven by the same language list, and that list can be
overridden per model config.

---

## 6. Tokenizer Training: BPE Under the Hood

The tokenizer is a Byte Pair Encoding (BPE) tokenizer, the same
algorithm used by GPT, LLaMA, StarCoder, and most modern language models.

### How BPE works

Start with individual bytes as your vocabulary. Then iteratively:

1. Count every adjacent pair of tokens in your training data
2. Find the most frequent pair (e.g., `t` + `h` appears 1 million times)
3. Merge that pair into a new token (`th`)
4. Repeat until you have `vocab_size` tokens (default: 32,768)

```
Step 0: vocabulary = {a, b, c, d, ..., z, 0, 1, ..., 9, space, ...}
Step 1: merge "t"+"h" → "th"     (most common pair)
Step 2: merge "th"+"e" → "the"   (now most common)
Step 3: merge " "+"t" → " t"     (space-t is very common)
...
Step 32,768: done!
```

The result is a vocabulary where common patterns are single tokens:

```
"function"     → 1 token (seen 500K times in training data)
"createElement" → 1 token (very common in React code)
"xyzzy"        → 3 tokens: "x" + "yz" + "zy" (rare word, split up)
```

### The training setup in cola-coder

```python
# src/cola_coder/tokenizer/train_tokenizer.py

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,         # 32,768
    special_tokens=SPECIAL_TOKENS,  # <|pad|>, <|bos|>, <|eos|>, etc.
    min_frequency=2,               # A pair must appear 2+ times
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
```

**ByteLevel pre-tokenizer:** Converts all bytes to visible Unicode
characters before BPE learning. This means the tokenizer can handle any
encoding (UTF-8, binary, whatever) without crashing.

**min_frequency=2:** A pair must appear at least twice to be merged. This
prevents one-off patterns from consuming vocabulary slots.

### Special tokens

Seven special tokens are reserved at the start of the vocabulary:

| Token | ID | Purpose |
|-------|----|---------|
| `<\|pad\|>` | 0 | Padding (fills unused positions in a batch) |
| `<\|bos\|>` | 1 | Beginning of sequence |
| `<\|eos\|>` | 2 | End of sequence |
| `<\|unk\|>` | 3 | Unknown token (fallback) |
| `<\|fim_prefix\|>` | 4 | Fill-in-the-middle: text before the gap |
| `<\|fim_middle\|>` | 5 | Fill-in-the-middle: the gap (model generates this) |
| `<\|fim_suffix\|>` | 6 | Fill-in-the-middle: text after the gap |

Additional special tokens (context tokens, ChatML tokens) can be added
later without retraining the tokenizer -- they are appended to the
vocabulary and the model's embedding layer is resized.

### Multi-source training

When you run `train_tokenizer.py --config configs/small.yaml`, the
script trains on data from all enabled sources in `data_sources.yaml`:

```
1. Stream code samples from bigcode/starcoderdata (TypeScript only, per config)
2. Stream text samples from HuggingFaceFW/fineweb-edu
3. Stream math samples from open-web-math/open-web-math
4. Chain all iterators into one combined stream
5. Train BPE on the combined stream
6. Save tokenizer.json + tokenizer_meta.json
```

The `--tok-samples` flag controls how many samples per source are used
for training (default: 100,000). This keeps training fast while ensuring
the vocabulary covers all domains.

```bash
# Train with default sample count
python scripts/train_tokenizer.py --config configs/small.yaml

# Train with more samples for a richer vocabulary
python scripts/train_tokenizer.py --config configs/small.yaml --tok-samples 500000

# Train with unlimited code samples (may OOM on large caches)
python scripts/train_tokenizer.py --config configs/small.yaml --tok-samples 0
```

---

## 7. Pipeline Integration

The per-dataset tokenizer is woven into the full pipeline. Here is how
each script interacts with it.

### train_tokenizer.py

The entry point for tokenizer training. When `--config` is provided, it
uses `DatasetResolver` to determine the output path:

```bash
python scripts/train_tokenizer.py --config configs/small.yaml
# Output: data/typescript-text-math/tokenizer.json

python scripts/train_tokenizer.py --config configs/medium.yaml
# Output: data/javascript-python-typescript-text-math/tokenizer.json
```

If `--config` is *not* provided, it falls back to the legacy behavior
(single tokenizer at `storage.tokenizer_path`).

### collect_data.py

Loads the tokenizer from the per-dataset directory:

```python
tok_path = DatasetResolver.get_tokenizer_path(ds_path, config_path=args.config)
if not Path(tok_path).exists():
    cli.error("Tokenizer not found", tok_path)
    cli.dim(f"  Run: python scripts/train_tokenizer.py --config {args.config}")
    sys.exit(1)
```

If you forget to train the tokenizer first, the script tells you exactly
what command to run.

### run_pipeline.py

The full pipeline runs stages in order:
`tokenizer -> data_prep -> training -> smoke_test -> evaluation -> export`

The `tokenizer` stage is first. If a tokenizer already exists for the
current config, it is reused. If not, it is trained automatically.

### What happens when you change data_sources.yaml

Say you trained a tokenizer for `typescript-text-math` and then add
`python` to your `data_sources.yaml` languages:

```yaml
sources:
  code:
    languages: [typescript, javascript, python]  # added python
```

The next time you run any pipeline script *without* a model config
override, the dataset name becomes
`javascript-python-typescript-text-math` -- a new directory. The old
`typescript-text-math/` directory and tokenizer are untouched.

The script will detect that no tokenizer exists in the new directory and
prompt you to train one. This is by design: changing the data mix should
produce a fresh tokenizer trained on the new mix.

### What happens when you change model config languages

Same behavior. If `configs/small.yaml` changes from
`languages: ["typescript"]` to `languages: ["typescript", "python"]`, the
dataset name changes and a new tokenizer is needed.

---

## 8. When to Share vs Separate Tokenizers

Not every config change needs a new tokenizer. Here are guidelines:

### Use separate tokenizers when...

- **Different programming languages.** A TypeScript tokenizer and a
  Python tokenizer will have meaningfully different vocabularies. The
  compression ratio difference is real and measurable.
- **Different domain mixes.** A code-only dataset vs a code+text+math
  dataset will produce different token frequency distributions.
- **Different vocab sizes.** This is obvious but worth stating: you cannot
  share a 16K-vocab tokenizer with a 32K-vocab model without retraining.

### Share tokenizers when...

- **Same languages, different model sizes.** A tiny (30M) model and a
  small (125M) model training on the same TypeScript+text+math mix should
  use the same tokenizer. The vocabulary is determined by data, not model
  architecture.
- **Minor config tweaks.** Changing `batch_size`, `learning_rate`, or
  `max_seq_len` does not affect the tokenizer.
- **Adding more training data of the same type.** If you add more
  TypeScript repos to your training data but do not change the language
  mix, the existing tokenizer is fine.

### The built-in behavior matches these guidelines

`DatasetResolver.get_dataset_name()` only considers:
- Code languages (sorted alphabetically)
- Enabled non-code source names

It does *not* consider model architecture params, training hyperparameters,
or data volume. So `configs/tiny.yaml` and `configs/small.yaml` with the
same `data.languages` list will resolve to the same dataset directory and
share a tokenizer automatically.

```
configs/tiny.yaml    data.languages: ["typescript"]  →  typescript-text-math/
configs/small.yaml   data.languages: ["typescript"]  →  typescript-text-math/  (same!)
configs/medium.yaml  data.languages: ["typescript", "python"]  →  python-typescript-text-math/  (different)
```

---

## 9. Configuration: data_sources.yaml + Model Config

Two configuration files interact to determine the dataset name and
tokenizer:

### data_sources.yaml

The base configuration. Defines what data sources exist and their
properties:

```yaml
# configs/data_sources.yaml
sources:
  code:
    dataset: "bigcode/starcoderdata"
    weight: 0.7
    enabled: true
    languages:
      - typescript
      - javascript
      - python

  text:
    dataset: "HuggingFaceFW/fineweb-edu"
    weight: 0.2
    enabled: true
    min_length: 100
    max_length: 50000

  math:
    dataset: "open-web-math/open-web-math"
    weight: 0.1
    enabled: true
    min_length: 50
    max_length: 30000

mix_temperature: 0.3
```

**What affects the dataset name:**
- `sources.code.languages` (if no model config override)
- Which sources have `enabled: true`
- Source names (`code`, `text`, `math`)

**What does NOT affect the dataset name:**
- `weight` values (affect training mix, not folder naming)
- `dataset` values (the HF dataset identifier)
- `min_length`, `max_length` (filtering params)
- `mix_temperature`

### Model config YAML

The per-model configuration. Only the `data.languages` field matters for
dataset naming:

```yaml
# configs/small.yaml
model:
  vocab_size: 32768
  dim: 768
  # ... model architecture

training:
  batch_size: 12
  # ... training hyperparameters

data:
  dataset: "bigcode/starcoderdata"
  languages: ["typescript"]    # ← THIS overrides data_sources.yaml
  max_tokens_per_file: 4096
  data_dir: "./data"
```

### Precedence rules

```
Dataset name components:
  1. Code languages:
     model_config.data.languages > data_sources.yaml.sources.code.languages
  2. Non-code sources:
     Always from data_sources.yaml (text, math, etc.)
  3. Fallback:
     "default" if nothing can be resolved
```

### Adding a new data source

To add a new data source (say, documentation):

```yaml
sources:
  code:
    # ...
  text:
    # ...
  math:
    # ...
  docs:                              # New source
    dataset: "your-org/docs-dataset"
    weight: 0.05
    enabled: true
    min_length: 200
    max_length: 100000
```

This changes the dataset name from `typescript-text-math` to
`typescript-text-math-docs` (non-code sources are appended in definition
order). A new tokenizer will be needed.

### Disabling a source

```yaml
sources:
  code:
    enabled: true
  text:
    enabled: false      # ← disabled
  math:
    enabled: true
```

Dataset name changes from `typescript-text-math` to `typescript-math`.
New tokenizer needed.

---

## 10. Tokenizer Comparison and Debugging

When you have multiple tokenizers, you will eventually want to compare
them or debug tokenization issues.

### Quick comparison: encode the same snippet

```python
from tokenizers import Tokenizer

tok_ts = Tokenizer.from_file("data/typescript-text-math/tokenizer.json")
tok_multi = Tokenizer.from_file("data/javascript-python-typescript-text-math/tokenizer.json")

code = "export const handler: RequestHandler = async (req, res) => {"

ts_tokens = tok_ts.encode(code)
multi_tokens = tok_multi.encode(code)

print(f"TS tokenizer:    {len(ts_tokens.ids)} tokens")
print(f"Multi tokenizer: {len(multi_tokens.ids)} tokens")
print(f"Difference:      {len(multi_tokens.ids) - len(ts_tokens.ids)}")
```

If the TS tokenizer produces fewer tokens for TypeScript code, it has a
better compression ratio for that domain.

### Vocabulary overlap

```python
def vocab_overlap(tok_a: Tokenizer, tok_b: Tokenizer) -> float:
    """What percentage of tokens appear in both vocabularies?"""
    vocab_a = set(tok_a.get_vocab().keys())
    vocab_b = set(tok_b.get_vocab().keys())
    overlap = vocab_a & vocab_b
    return len(overlap) / min(len(vocab_a), len(vocab_b))

# Typical results:
# Same-language tokenizers:   85-95% overlap
# Different-language:         50-70% overlap
# Code vs pure-text:          40-60% overlap
```

High overlap means the tokenizers are interchangeable for most content.
Low overlap means they are specialized for different domains.

### Compression ratio

The gold standard metric for tokenizer quality is the compression ratio:
bytes per token on a representative sample.

```python
def compression_ratio(tokenizer: Tokenizer, text: str) -> float:
    """Bytes per token -- higher is better (more compressed)."""
    tokens = tokenizer.encode(text)
    return len(text.encode("utf-8")) / len(tokens.ids)
```

Good compression ratios for code:
- **3.5-4.5 chars/token:** Excellent (domain-matched tokenizer)
- **2.5-3.5 chars/token:** Average (general-purpose tokenizer on code)
- **< 2.5 chars/token:** Poor (wrong domain or very small vocab)

### The tokenizer health script

cola-coder includes `scripts/tokenizer_health.py` for automated tokenizer
diagnostics. It checks:

- Vocabulary size matches config
- Special tokens are present and correctly numbered
- Compression ratio on sample code
- Round-trip encoding (`encode → decode → encode` produces same IDs)
- No unknown tokens on representative samples

### Debugging unexpected tokenization

If a model produces garbage output, the tokenizer is often the first
suspect. Common issues:

**Problem:** Model outputs random characters or garbled text.
**Check:** Are you using the right tokenizer for this model? A model
trained with `typescript-text-math/tokenizer.json` must use that exact
tokenizer at inference time.

**Problem:** Certain keywords are always misspelled.
**Check:** Encode the keyword and see how it splits:

```python
enc = tokenizer.encode("interface")
print([tokenizer.id_to_token(id) for id in enc.ids])
# Good: ["interface"]  (single token)
# Bad:  ["inter", "face"]  (two tokens -- tokenizer wasn't trained on TS)
```

**Problem:** Very long outputs for short inputs.
**Check:** Compression ratio is too low. The tokenizer needs retraining
with more data from the target domain.

### When to retrain

Retrain your tokenizer when:

1. You add or remove a programming language from your config
2. You enable or disable a data source (text, math)
3. You significantly change the data distribution (new datasets)
4. You change vocab_size

Do NOT retrain when:
1. You change model architecture params
2. You change training hyperparameters
3. You add more data of the same type
4. You change quality filter thresholds

A good rule of thumb: if `DatasetResolver.get_dataset_name()` returns a
different string, you need a new tokenizer. If it returns the same string,
your existing tokenizer is fine.
