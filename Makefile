.PHONY: setup train-tiny train-small train-medium prepare tokenizer evaluate serve generate test lint clean

# First-time setup: create venv and install dependencies
setup:
	python -m venv .venv
	.venv/Scripts/pip install -e ".[dev,logging]"

# Train the BPE tokenizer on downloaded code data
tokenizer:
	python scripts/train_tokenizer.py

# Download and preprocess training data
prepare:
	python scripts/prepare_data.py

# Training runs for each model size
train-tiny:
	python scripts/train.py --config configs/tiny.yaml

train-small:
	python scripts/train.py --config configs/small.yaml

train-medium:
	python scripts/train.py --config configs/medium.yaml

train-large:
	python scripts/train.py --config configs/large.yaml

# Run evaluation (HumanEval pass@k)
evaluate:
	python scripts/evaluate.py

# Interactive code generation CLI
generate:
	python scripts/generate.py

# Start the inference HTTP server
serve:
	python scripts/serve.py

# Train with reasoning (GRPO)
train-reasoning:
	python scripts/train_reasoning.py

# Run tests
test:
	pytest tests/ -v

# Lint
lint:
	ruff check src/ scripts/ tests/

# Clean generated files
clean:
	rm -rf checkpoints/ data/processed/ __pycache__ .pytest_cache
