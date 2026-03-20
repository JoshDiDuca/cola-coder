# Skill: Linting and Testing

## Run Tests
```bash
.venv/Scripts/pytest tests/ -v
# Expected: 437+ tests, 0 failures
```

## Lint
```bash
.venv/Scripts/ruff check src/ scripts/ tests/
.venv/Scripts/ruff check --fix src/ scripts/ tests/  # auto-fix
```

## Config
pyproject.toml — ruff config (line-length: 100)

## Common Lint Issues
- E741: Ambiguous variable name `l` — use `line` instead
- F541: f-strings without placeholders — remove `f` prefix
- F401: Unused imports — remove or add noqa comment
- F841: Unused variables — remove assignment

## Feature Validation
```python
from cola_coder.features import list_features
print(len(list_features()))  # Expected: 83+
```

## __pycache__ Issues on Windows
After editing files, old .pyc cache may cause stale behavior:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

## Pre-commit Checks
1. `ruff check src/ scripts/ tests/` — zero errors
2. `pytest tests/ -x -q` — all pass
3. `python -c "from cola_coder.features import list_features; print(len(list_features()))"` — 83+
