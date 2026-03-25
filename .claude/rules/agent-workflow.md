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
- Stage 6 (instruction tuning) uses `train_sft.py`, never `train.py`
- Stage 4 (context extension) auto-skips when `rope_scaling.type == "none"` or `factor <= 1.0`
- Stage 7 (MoE upcycling) auto-skips when `model.moe.enabled` is False
- Always pass `--tokenizer` from `self._master.storage.tokenizer_path` to data scripts
- Pipeline run state saved to `pipeline_runs/{name}.json` (never in project source)
- Artifact resolution order: stage override → previous stage artifact → filesystem auto-detect

## Principles
- **Simplicity First**: make every change as simple as possible
- **No Laziness**: find root causes, no temporary fixes, senior developer standards
- **Autonomous Bug Fixing**: just fix it, don't ask for hand-holding
- **Demand Elegance**: for non-trivial changes, pause and ask "is there a more elegant way?"
