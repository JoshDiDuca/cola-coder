# Skill: Feature System

## Pattern
Every feature module in `src/cola_coder/features/` follows:
```python
FEATURE_ENABLED = True

def is_enabled() -> bool:
    return FEATURE_ENABLED
```
Features are optional — the system works without any features enabled.

## Config
- `configs/features.yaml` — master toggle for all 83 features
- `src/cola_coder/features/__init__.py` — config loader/saver
- Runtime toggle: `set_feature_enabled(name, enabled)`

## Categories (7)
Training, Generation, Evaluation, Infrastructure, Routing & Specialists, Code Analysis, UI & Dashboard

## Adding a New Feature
1. Create `src/cola_coder/features/your_feature.py`
2. Add `FEATURE_ENABLED = True` flag + `is_enabled()` function
3. Add to `configs/features.yaml` under appropriate category
4. Import with try/except in consuming code

## Notable Features
- `code_scorer` — Continuous quality scoring (0.0-1.0) for training weights
- `ollama_improver` — Local AI code improvement via Ollama (DISABLED by default, needs Ollama running)
- `master_menu` — Arrow-key CLI menu with ESC navigation, all 22 scripts accessible

## Key API
```python
from cola_coder.features import list_features, get_feature_status, set_feature_enabled
list_features()  # returns list of module names
get_feature_status()  # returns dict of name -> enabled
set_feature_enabled("my_feature", False)  # runtime toggle
```
