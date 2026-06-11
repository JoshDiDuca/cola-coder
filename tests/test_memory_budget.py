"""MEM-001: memory context must respect MemoryConfig.max_context_tokens.

The memory module had no tests, and max_context_tokens (documented as "Max
tokens of memory to inject into prompts") was never enforced — get_relevant_
memories concatenated project.md + chunks with no cap, so a large memory store
could overflow the model's context window. These lock the budget enforcement.
"""

from cola_coder.memory.config import MemoryConfig
from cola_coder.memory.manager import MemoryManager


def _manager(tmp_path, max_tokens):
    cfg = MemoryConfig(max_context_tokens=max_tokens)
    mm = MemoryManager(project_root=tmp_path, config=cfg)
    mm.init_project(description="A test project.", tech_stack={"lang": "TypeScript"})
    return mm


class TestTokenEstimate:
    def test_estimate_roughly_chars_over_four(self):
        assert MemoryManager._estimate_tokens("") == 0
        assert MemoryManager._estimate_tokens("a" * 4) == 1
        assert MemoryManager._estimate_tokens("a" * 4000) == 1000


class TestBudgetEnforcement:
    def test_large_project_context_truncated_to_budget(self, tmp_path):
        mm = _manager(tmp_path, max_tokens=50)  # ~200 char budget
        # Overwrite project.md with a huge document.
        mm._write_file("project", "X " * 5000)  # ~10k chars
        out = mm.get_relevant_memories(query="anything")
        # Output must be within ~budget tokens (allow truncation marker slack).
        assert MemoryManager._estimate_tokens(out) <= 50 + 10
        assert "truncated" in out

    def test_small_memory_not_truncated(self, tmp_path):
        mm = _manager(tmp_path, max_tokens=1024)
        out = mm.get_relevant_memories(query="project")
        assert "truncated" not in out
        assert "test project" in out.lower()

    def test_total_context_within_budget_with_chunks(self, tmp_path):
        mm = _manager(tmp_path, max_tokens=60)  # ~240 chars
        # Add several large memory entries that, unbounded, would blow the budget.
        for i in range(6):
            mm.add_pattern(f"pattern {i} about auth handling", "code " * 40)
            mm.add_error(f"error {i} about auth handling", "fix " * 40)
        out = mm.get_relevant_memories(query="auth handling pattern error")
        # Even with many relevant chunks, total stays within budget (+ marker slack).
        assert MemoryManager._estimate_tokens(out) <= 60 + 10

    def test_returns_empty_when_uninitialized(self, tmp_path):
        mm = MemoryManager(project_root=tmp_path / "nope", config=MemoryConfig())
        assert mm.get_relevant_memories(query="x") == ""
