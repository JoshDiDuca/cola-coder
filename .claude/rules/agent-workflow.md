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
- When adding a new user-facing feature or script, add it to the master menu
  (`src/cola_coder/features/master_menu.py`) in the appropriate section
- Follow existing patterns: label + detail dict, dispatch by choice index
- If a new config size is added, include it in `_resume_training_menu` sizes list

## Principles
- **Simplicity First**: make every change as simple as possible
- **No Laziness**: find root causes, no temporary fixes, senior developer standards
- **Autonomous Bug Fixing**: just fix it, don't ask for hand-holding
- **Demand Elegance**: for non-trivial changes, pause and ask "is there a more elegant way?"
