# Scripts Reference (45 total)

## Training & Data
| Script | Purpose |
|--------|---------|
| `menu.py` | Master arrow-key menu |
| `train_tokenizer.py` | Train BPE tokenizer |
| `prepare_data.py` | Download, filter, tokenize training data |
| `prepare_data_interactive.py` | Guided interactive data prep |
| `prepare_fim_data.py` | FIM-formatted training data |
| `train.py` | Main training loop |
| `train_reasoning.py` | SFT warmup + GRPO reasoning |
| `train_quality_classifier.py` | Train quality scorer |
| `train_router.py` | Train domain router |
| `find_lr.py` | LR range finder |
| `combine_datasets.py` | Merge datasets |

## Inference
| Script | Purpose |
|--------|---------|
| `run.py` | Interactive REPL |
| `generate.py` | One-shot generation |
| `generate_instructions.py` | Instruction pairs from code |
| `generate_router_data.py` | Router training data |
| `serve.py` | FastAPI server |

## Evaluation
| Script | Purpose |
|--------|---------|
| `evaluate.py` | HumanEval pass@k |
| `benchmark.py` | tok/s benchmark |
| `nano_benchmark.py` | Fast gen speed test |
| `inference_benchmark.py` | Detailed inference profiling |
| `smoke_test.py` | 8-check quick validation |
| `ts_benchmark.py` | TypeScript benchmark |
| `regression_test.py` | Quality tracking |
| `quality_report.py` | Auto quality report |
| `compare_models.py` | Side-by-side comparison |
| `run_eval_suite.py` | Run all evaluations |
| `test_type_reward.py` | Test GRPO rewards |

## Tools
| Script | Purpose |
|--------|---------|
| `training_status.py` | Training progress (CPU-only) |
| `training_dashboard.py` | Live metrics dashboard |
| `training_eval_history.py` | Auto-eval history |
| `compare_checkpoints.py` | Checkpoint comparison |
| `checkpoint_diff.py` | Checkpoint diff |
| `checkpoint_info.py` | Checkpoint metadata |
| `average_checkpoints.py` | Checkpoint averaging |
| `model_card.py` | HuggingFace model card |
| `vram_estimate.py` | VRAM estimator |
| `export_model.py` | Export GGUF/Ollama/quantized |
| `run_pipeline.py` | Multi-stage pipeline |
| `migrate_storage.py` | Storage migration |
| `scrape_github.py` | GitHub API data collection |
| `score_repos.py` | Rank repos by quality |
| `data_stats.py` | Training data statistics |
| `tokenizer_health.py` | Tokenizer health check |
| `env_check.py` | Environment validation |
| `project_health.py` | Project health score |
