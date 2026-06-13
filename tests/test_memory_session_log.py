"""MEM-002: session log rolling window must actually bound the log.

`_trim_session_log` used a DOTALL regex (`^# .+\n\n.*?\n\n`) to isolate the
file preamble. Because `.` matched newlines, the non-greedy `.*?` still swallowed
the first "## " section bodies into the "header" string. The kept sections were
then appended *after* that bloated header, so:

  1. old entries were never actually dropped (they hid inside the "header"), and
  2. each trim re-appended the kept entries, duplicating them.

Net effect: the session log grew without bound instead of shrinking to
`session_log_max_entries`. These tests lock the rolling-window contract.
"""

from cola_coder.memory.config import MemoryConfig
from cola_coder.memory.manager import MemoryManager, _iter_sections


def _manager(tmp_path, max_entries):
    cfg = MemoryConfig(session_log_max_entries=max_entries)
    mm = MemoryManager(project_root=tmp_path, config=cfg)
    mm.init_project(description="A test project.")
    return mm


class TestSessionLogRollingWindow:
    def test_log_trimmed_to_max_entries(self, tmp_path):
        mm = _manager(tmp_path, max_entries=2)
        for i in range(5):
            mm.log_session(f"summary number {i} doing things", domain="react")
        sections = list(_iter_sections(mm._read_file("session_log")))
        assert len(sections) == 2

    def test_keeps_most_recent_entries(self, tmp_path):
        mm = _manager(tmp_path, max_entries=2)
        for i in range(5):
            mm.log_session(f"unique-summary-{i}", domain="react")
        content = mm._read_file("session_log")
        # The two newest survive; older ones are gone.
        assert "unique-summary-4" in content
        assert "unique-summary-3" in content
        assert "unique-summary-2" not in content
        assert "unique-summary-0" not in content

    def test_no_duplicate_entries_after_repeated_trims(self, tmp_path):
        mm = _manager(tmp_path, max_entries=3)
        for i in range(10):
            mm.log_session(f"unique-summary-{i}", domain="react")
        content = mm._read_file("session_log")
        # The surviving newest entry must appear exactly once (no re-appending).
        assert content.count("unique-summary-9") == 1

    def test_preamble_preserved(self, tmp_path):
        mm = _manager(tmp_path, max_entries=2)
        for i in range(4):
            mm.log_session(f"summary {i}", domain="react")
        content = mm._read_file("session_log")
        assert content.startswith("# Session Log")
        assert "Recent interaction summaries." in content

    def test_under_limit_not_trimmed(self, tmp_path):
        mm = _manager(tmp_path, max_entries=5)
        for i in range(3):
            mm.log_session(f"summary {i}", domain="react")
        sections = list(_iter_sections(mm._read_file("session_log")))
        assert len(sections) == 3

    def test_stats_chunk_count_matches_window(self, tmp_path):
        mm = _manager(tmp_path, max_entries=4)
        for i in range(20):
            mm.log_session(f"summary number {i}", domain="react")
        assert mm.stats()["session_log"]["chunks"] == 4
