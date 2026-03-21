# Code Style Rules

- Use ruff for linting. Line length: 100 (configured in pyproject.toml)
- Use pytest for tests. 122 test files, ~2600 tests
- Type hints used but not strictly enforced
- Use `from cola_coder.cli import cli` for ALL CLI output — never import Rich directly
- CLI methods: `cli.header()`, `cli.step()`, `cli.info()`, `cli.success()`, `cli.error()`, `cli.warn()`, `cli.done()`, `cli.kv_table()`, `cli.choose()`, `cli.confirm()`
- Features toggled via `configs/features.yaml` with `FEATURE_ENABLED` / `is_enabled()` pattern
- All menus use questionary arrow-key navigation via `cli.choose()`
- Checkpoints use safetensors format, never pickle
