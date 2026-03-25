# Scripts Reference (55 total)

## Pipeline & Orchestration
| Script | Purpose |
|--------|---------|
| `menu.py` | Master arrow-key menu |
| `full_pipeline.py` | Run all 10 pipeline stages end-to-end |
| `run_pipeline.py` | Flexible multi-stage pipeline runner |

## Training & Tokenizer
| Script | Purpose |
|--------|---------|
| `train_tokenizer.py` | Train BPE tokenizer from scratch |
| `train.py` | Main training loop (pretraining + context extension stage) |
| `train_sft.py` | Supervised fine-tuning on instruction pairs (stage 6) — args: --data, --config, --checkpoint, --epochs, --lr |
| `train_reasoning.py` | SFT warmup + GRPO reasoning — args: --config, --sft-warmup, --reward {python_exec,typescript,combined}, --problems {builtin,all,curriculum} |
| `train_quality_classifier.py` | Train ML-based quality scorer |
| `train_router.py` | Train semantic domain router (MLP or Transformer) |
| `find_lr.py` | LR range finder |
| `background_train.py` | Overnight training with GPU throttling |
| `upcycle_to_moe.py` | Convert dense checkpoint to Mixture of Experts (stage 7) |

## Data Collection & Preparation
| Script | Purpose |
|--------|---------|
| `collect_data.py` | Multi-source collection: code 70% + text 20% + math 10% — reads data_sources.yaml |
| `prepare_data.py` | Download, filter, tokenize code training data |
| `prepare_data_interactive.py` | Guided interactive data prep |
| `prepare_fim_data.py` | FIM-formatted training data |
| `prepare_docs_data.py` | Tokenize framework docs (React, Next.js, Zod, etc.) |
| `prepare_repo_context_data.py` | Create repo context training pairs |
| `combine_datasets.py` | Merge datasets with weighted mixing |
| `scrape_github.py` | GitHub API scraper (stars, language, license, owner filters) |
| `scrape_docs.py` | Framework documentation scraper |

## Inference & Generation
| Script | Purpose |
|--------|---------|
| `run.py` | Interactive REPL |
| `generate.py` | One-shot code generation |
| `generate_instructions.py` | Instruction pairs from code (template/LLM/self-instruct modes) |
| `generate_sft_data.py` | Generate instruction-following pairs |
| `generate_router_data.py` | Router domain classifier training data |
| `serve.py` | FastAPI server (OpenAI-compatible + FIM + SSE) |

## Evaluation
| Script | Purpose |
|--------|---------|
| `evaluate.py` | HumanEval pass@k (62 problems) |
| `benchmark.py` | tok/s throughput benchmark |
| `nano_benchmark.py` | Fast generation speed test |
| `inference_benchmark.py` | Detailed latency profiling |
| `smoke_test.py` | 8-check quick validation (~30s) |
| `ts_benchmark.py` | TypeScript benchmark (tsc --strict) |
| `regression_test.py` | Quality regression tracking |
| `quality_report.py` | Auto quality report (syntax, types, tokens) |
| `compare_models.py` | Side-by-side model comparison |
| `run_eval_suite.py` | Run all evaluations in sequence |
| `test_type_reward.py` | Test GRPO TypeScript reward function |

## Checkpoint Management
| Script | Purpose |
|--------|---------|
| `compare_checkpoints.py` | Side-by-side checkpoint comparison |
| `checkpoint_diff.py` | Weight-by-weight checkpoint diff |
| `checkpoint_info.py` | Checkpoint metadata and architecture |
| `average_checkpoints.py` | Checkpoint averaging (model soups) |

## Monitoring & Tools
| Script | Purpose |
|--------|---------|
| `training_status.py` | Training progress (CPU-only readout) |
| `training_dashboard.py` | Live metrics dashboard |
| `training_eval_history.py` | Auto-eval snapshots over training — requires --checkpoint-dir |
| `vram_estimate.py` | VRAM estimator |
| `data_stats.py` | Training data statistics |
| `tokenizer_health.py` | Tokenizer health check |
| `env_check.py` | Environment validation |
| `project_health.py` | Project health score |

## Export & Utilities
| Script | Purpose |
|--------|---------|
| `export_model.py` | Export GGUF/Ollama/quantized |
| `model_card.py` | HuggingFace model card |
| `migrate_storage.py` | Storage migration |
| `score_repos.py` | Rank repos by quality |
