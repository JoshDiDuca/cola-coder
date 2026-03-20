# Skill: Data Pipeline

## Data Source
bigcode/starcoderdata (gated, needs HF_TOKEN)
Languages: Python, TypeScript, JavaScript

## Prepare Data
```bash
python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json
# Add --score to compute quality weights for weighted training
python scripts/prepare_data.py --config configs/tiny.yaml --tokenizer tokenizer.json --score
```

## Quality Filter (binary pass/fail)
`src/cola_coder/data/quality_filter.py`
- CONSERVATIVE mode: ~48% rejection (default)
- STRICT mode: ~65% rejection
- 15+ individual checks (length, syntax, naming, etc.)
- `filter_code(content, mode)` -> (keep, reason)
- NEW: `score_code(content, mode)` -> (score, breakdown)

## Quality Scorer (continuous 0.0-1.0)
`src/cola_coder/features/code_scorer.py`
- 13 weighted signals: structure, naming, docs, complexity, syntax, modernness, etc.
- Tiers: excellent (0.8+), good (0.6-0.8), average (0.4-0.6), poor (0.2-0.4), reject (<0.2)
- `score_to_weight()` converts to training weights (0.0-2.0 range)

## Quality Classifier (ML-based)
`src/cola_coder/data/filters/quality_classifier.py`
- HeuristicQualityScorer — 8 weighted sub-scores, always available
- CodeQualityClassifier — DistilBERT/CodeBERTa neural scorer (optional)
- QualityAnnotator — LLM-based scoring with Claude API (expensive)

## Ollama Integration (optional, free/local)
`src/cola_coder/features/ollama_improver.py`
- Local AI code improvement using Ollama (codellama model)
- Adds comments, improves naming, adds type hints
- `OllamaScorer` for AI-powered quality scoring

## Quality-Weighted Training
When `--score` is used during data prep, a `.weights.npy` sidecar file is created.
The trainer auto-detects this file and uses per-example quality weights.
