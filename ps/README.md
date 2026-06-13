# PowerShell convenience scripts (Windows)

One-line wrappers around `scripts/*.py` so you can drive the project from
anywhere without typing `.venv\Scripts\python scripts\…`. Each script resolves
the project root automatically (the parent of this `ps\` folder), activates the
venv's Python, and forwards any extra `@args` straight through to the underlying
Python script.

> **Tip:** `.\cola-menu.ps1` opens the interactive master menu, which can reach
> **every** feature. The scripts below are shortcuts for the common workflows.

Run scripts from the `ps\` folder, e.g. `cd ps; .\cola-env-check.ps1`.

## First-time setup
| Script | What it does |
|--------|--------------|
| `cola-setup.ps1` | Create `.venv` and install the project (`-e ".[dev,logging]"`) |
| `cola-env-check.ps1` | Validate Python / PyTorch / CUDA / GPU / deps (`--no-internet` to skip the net probe) |
| `cola-menu.ps1` | Interactive master menu — single entry point to everything |

## End-to-end pipeline
| Script | What it does |
|--------|--------------|
| `cola-pipeline.ps1` | Full Auto Pipeline: detect hardware → pick config → run all stages. `-Smoke`, `-DryRun`, `-Yes`, `-BaseConfig` |

## Data
| Script | What it does |
|--------|--------------|
| `cola-collect.ps1` | Multi-source collection (code/text/math). `-Config`, `-Score` |
| `cola-prepare.ps1` | Single-source download + filter + tokenize. `-Config`, `-MaxTokens`, `-Workers` |
| `cola-prepare-menu.ps1` | Guided interactive data prep |
| `cola-prepare-tiny.ps1` | Prepare data for the `tiny` config (50M tokens default) |
| `cola-prepare-test.ps1` | Throughput probe + full-run time estimate |
| `cola-tokenizer.ps1` | Train the BPE tokenizer. `-VocabSize`, `-NumSamples`, `-Config`, `-Output` |

## Training
| Script | What it does |
|--------|--------------|
| `cola-train.ps1` | Pretrain a model: `tiny`/`small`/`medium`/`4080_max`/`large`/`reasoning` |
| `cola-sft.ps1` | Instruction tuning (stage 6). `-Checkpoint`, `-Config`, `-Data`, `-Epochs`, `-Lr` |

## Inference & serving
| Script | What it does |
|--------|--------------|
| `cola-run.ps1` | Interactive code-generation REPL |
| `cola-generate.ps1` | One-shot code generation CLI |
| `cola-chat.ps1` | Multi-turn chat REPL |
| `cola-serve.ps1` | FastAPI server (add `--cors` for the VS Code extension) |

## Evaluation & quality
| Script | What it does |
|--------|--------------|
| `cola-smoke.ps1` | Fast 8-check validation of a checkpoint (~30s) |
| `cola-evaluate.ps1` | HumanEval pass@k |
| `cola-eval-suite.ps1` | Run the full evaluation suite in sequence |
| `cola-benchmark.ps1` | Throughput (tok/s) benchmark |
| `cola-quality.ps1` | Auto quality report (syntax, types, tokens) |
| `cola-safety.ps1` | Safety probes on generated code (secrets, dangerous patterns) |
| `cola-compare.ps1` | Side-by-side comparison of two checkpoints |
| `cola-lint.ps1` | `ruff check src\ scripts\ tests\` |
| `cola-test.ps1` | `pytest tests\ -v` |

## Tuning & specialists
| Script | What it does |
|--------|--------------|
| `cola-vram.ps1` | Estimate VRAM for a config before training |
| `cola-find-lr.ps1` | Learning-rate range finder |
| `cola-router.ps1` | Train the semantic domain router (stage 8) |

## Export & docs
| Script | What it does |
|--------|--------------|
| `cola-export.ps1` | Export GGUF / Ollama / quantized (`--action`) |
| `cola-model-card.ps1` | Generate a HuggingFace model card |

## Notes
- Tokenizer paths auto-resolve from `configs/storage.yaml` + the dataset dir — you
  rarely need to pass `-Tokenizer`/`--tokenizer`.
- Any flag the underlying Python script accepts can be appended; it's forwarded
  verbatim (e.g. `.\cola-evaluate.ps1 --num-samples 10`).
