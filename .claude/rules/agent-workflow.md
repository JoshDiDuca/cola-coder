# Agent Workflow Rules

## Planning
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Write detailed specs upfront to reduce ambiguity

## Subagents
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One task per subagent for focused execution

## Self-Improvement
- After ANY correction from user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake repeating
- Review lessons at session start

## Verification
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Ask: "Would a staff engineer approve this?"

## Task Management
1. Write plan to `tasks/todo.md` with checkable items
2. Check in before starting implementation
3. Mark items complete as you go
4. Update `tasks/lessons.md` after corrections

## Menu Integration
- Menus are split into sub-modules under `src/cola_coder/features/menus/`:
  - `data_menu.py` — data collection, modification, scoring, inspection, preparation
  - `training_menu.py` — 6 sub-groups: Pipeline Manager, Foundation (1-2), Pre-Training (3),
    Post-Training (4-7), Alignment & Reasoning (8-9), Monitoring & Tools
  - `eval_menu.py` — HumanEval, benchmarks, comparisons, quality reports
  - `tools_menu.py` — tests, linting, GPU, features, settings, export
  - `pipeline_menu.py` — named pipeline runs, resume, stage override, state persistence
- `master_menu.py` is the thin coordinator with shared helpers only
- When adding a new feature, add it to the appropriate sub-module menu
- Follow existing patterns: label + detail dict, dispatch by choice index
- Data sources use `cli.choose()`, `cli.confirm()`, `cli.kv_table()` — never raw Rich
- If a new config size is added, include it in `training_menu._train_size_menu` sizes
- Every new script must be wired into the appropriate sub-menu before marking the task complete
- Language-aware features must use `cli.pick_languages()` — never hardcode language lists
- New feature modules must be assigned to a `_FEATURE_CATEGORIES` group, not left in "Other"

## Pipeline Manager Rules
- Pipeline stage scripts must use `_run_stage_script()` (not `_master._run_script()`)
  - `_run_script()` silently ignores non-zero exit codes; only suitable for non-critical menu actions
  - `_run_stage_script()` raises RuntimeError on failure so `_execute_stage` marks the stage failed
  - Exception: purely informational scripts (evaluate.py, quality_report.py) may use `_run_script()`
  - Quality gates like smoke_test.py MUST use `_run_stage_script()`
- Stage 6 (instruction tuning) uses `train_sft.py`, never `train.py`
- Stage 4 (context extension) auto-skips when `rope_scaling.type == "none"` or `factor <= 1.0`
- Stage 7 (MoE upcycling) auto-skips when `model.moe.enabled` is False
- Always pass `--tokenizer` from `DatasetResolver.get_tokenizer_path(...)` to data scripts
- Pipeline run state saved to `pipeline_runs/{name}.json` (never in project source)
- Artifact resolution order: stage override → previous stage artifact → filesystem auto-detect

## Pipeline Audit Checklist — ALL Classes, Every Stage

When reviewing the pipeline ("review everything," "check all stages," etc.), audit EVERY stage against EVERY class below. Do not stop after finding bugs in one class.

1. **Missing args** — every required script arg passed? (dataset, languages, tokenizer, checkpoint, config)
2. **Hardcoded values** — anything that should come from config? (counts, epochs, paths, hyperparams)
3. **Wrong runner method** — quality gates use `_run_stage_script()`, informational use `_master._run_script()`
4. **Wrong data distribution** — does data match the model's expected distribution? (language, domain, source)
5. **Stale references** — any path/pointer that could be deleted? (latest pointer, artifact chains)
6. **Scaling** — hyperparameters scale with model size? (use `_model_scale(config)`)

Reward functions, problem sets, and CoT examples must all derive from `config.data.languages`:
- single `typescript` → `--reward typescript`, TypeScript problems, TypeScript CoT examples
- single `python` → `--reward python_exec`, Python problems, Python CoT examples
- multi-language → `--reward combined`, problems per language

## Pipeline Stage Args — Read from Config, Never Hardcode
Every stage arg that relates to the model or data MUST come from `config` and `run`, not be hardcoded.
Hardcoded values silently diverge from what the user configured and produce wrong results with no error.

- **Dataset**: `getattr(config.data, "dataset", "bigcode/starcoderdata")`
- **Languages**: `getattr(config.data, "languages", ["typescript"])` — pass ALL languages, not just `[0]`
- **Data path**: `DatasetResolver.get_dataset_dir(_DATA_SOURCES_PATH, config_path=run.config_path)`
- **Hyperparameters that scale with model size**: use `_model_scale(config)` (keyed on `max_steps`):
  - tiny (20K steps) → 5K SFT examples, 3 SFT epochs, GRPO group_size=4
  - small (100K) → 25K examples, 2 epochs, group_size=8
  - medium (150K) → 37.5K examples, 2 epochs, group_size=16
  - 4080_max (200K) → 50K examples, 2 epochs, group_size=16

## Data Source Language Filtering — Never Omit
`HuggingFaceSource` defaults to `languages=["python"]` when `languages` is not passed.
Always construct as `HuggingFaceSource(dataset=dataset, languages=[lang])`.

`SelfAlignPipeline` is language-specific: seed extraction (TS vs Python AST patterns) and instruction
templates differ per language. Never feed multi-language data into a single pipeline instance.

For multi-language configs (medium, 4080_max):
- Loop over each language
- Create `HuggingFaceSource(languages=[lang])` + `SelfAlignPipeline(language=lang)` per language
- Generate `count // len(languages)` examples each, then combine

In `generate_instructions.py` non-interactive mode: `--languages` (nargs="+"), not `--language`.
In `pipeline_menu.py` stage 5: pass `"--languages", *languages` with the full config language list.

## Principles
- **Simplicity First**: make every change as simple as possible
- **No Laziness**: find root causes, no temporary fixes, senior developer standards
- **Autonomous Bug Fixing**: just fix it, don't ask for hand-holding
- **Demand Elegance**: for non-trivial changes, pause and ask "is there a more elegant way?"
