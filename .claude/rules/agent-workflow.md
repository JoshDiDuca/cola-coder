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
  - `training_menu.py` — model training, tokenizer, reasoning, VRAM, LR finder
  - `eval_menu.py` — HumanEval, benchmarks, comparisons, quality reports
  - `tools_menu.py` — tests, linting, GPU, features, settings, export
- `master_menu.py` is the thin coordinator with shared helpers only
- When adding a new feature, add it to the appropriate sub-module menu
- Follow existing patterns: label + detail dict, dispatch by choice index
- Data sources use `cli.choose()`, `cli.confirm()`, `cli.kv_table()` — never raw Rich
- If a new config size is added, include it in `training_menu._train_size_menu` sizes

## Principles
- **Simplicity First**: make every change as simple as possible
- **No Laziness**: find root causes, no temporary fixes, senior developer standards
- **Autonomous Bug Fixing**: just fix it, don't ask for hand-holding
- **Demand Elegance**: for non-trivial changes, pause and ask "is there a more elegant way?"
