"""Tests for the distillation data generation loop (MODEL-028).

No network / no Docker: a fake teacher supplies completions, and verification is an
injected predicate (in production it's the SANDBOXED verifier — the loop itself
never executes untrusted code, SEC-014).
"""

from cola_coder.distillation import generate_distillation_dataset
from cola_coder.distillation.teacher import TeacherError


class _FakeTeacher:
    name = "fake"

    def __init__(self, replies):
        # replies: list popped in order, or a callable(messages)->str
        self._replies = replies
        self.seen = []

    def complete(self, messages, *, max_tokens=512, temperature=0.7, stop=None):
        self.seen.append(messages)
        r = self._replies(messages) if callable(self._replies) else self._replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_basic_chatml_output_and_stats():
    teacher = _FakeTeacher(["def add(a,b): return a+b", "const x = 1"])
    records, stats = generate_distillation_dataset(teacher, ["add", "declare x"])
    assert len(records) == 2
    assert records[0]["messages"][-1] == {
        "role": "assistant", "content": "def add(a,b): return a+b"
    }
    assert records[0]["messages"][0] == {"role": "user", "content": "add"}
    assert stats["kept"] == 2 and stats["teacher_ok"] == 2


def test_system_prompt_prepended():
    teacher = _FakeTeacher(["ok"])
    records, _ = generate_distillation_dataset(teacher, ["hi"], system="You are a coder.")
    roles = [m["role"] for m in records[0]["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_chatml_prompt_passthrough_no_duplicate_system():
    teacher = _FakeTeacher(["ok"])
    prompt = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    records, _ = generate_distillation_dataset(teacher, [prompt], system="OTHER")
    roles = [m["role"] for m in records[0]["messages"]]
    assert roles == ["system", "user", "assistant"]  # not two systems
    assert records[0]["messages"][0]["content"] == "S"


def test_rejection_sampling_drops_unverified():
    teacher = _FakeTeacher(["good", "bad", "good"])
    # verify accepts only "good"
    records, stats = generate_distillation_dataset(
        teacher, ["p1", "p2", "p3"], verify=lambda c: c == "good",
    )
    assert len(records) == 2
    assert all(r["messages"][-1]["content"] == "good" for r in records)
    assert stats["verified"] == 2 and stats["rejected"] == 1 and stats["kept"] == 2


def test_keep_unverified_when_flag_off():
    teacher = _FakeTeacher(["good", "bad"])
    records, stats = generate_distillation_dataset(
        teacher, ["p1", "p2"], verify=lambda c: c == "good", keep_only_verified=False,
    )
    assert len(records) == 2  # both kept
    assert stats["rejected"] == 1 and stats["kept"] == 2


def test_teacher_errors_skipped_not_fatal():
    teacher = _FakeTeacher(["ok", TeacherError("down"), "ok2"])
    records, stats = generate_distillation_dataset(teacher, ["a", "b", "c"])
    assert len(records) == 2
    assert stats["teacher_errors"] == 1 and stats["kept"] == 2


def test_empty_completion_skipped():
    teacher = _FakeTeacher(["", "   ", "real"])
    records, stats = generate_distillation_dataset(teacher, ["a", "b", "c"])
    assert len(records) == 1
    assert stats["teacher_errors"] == 2 and stats["kept"] == 1


def test_loop_never_executes_completion():
    # The completion is "malicious" code; with no verifier the loop must NOT run it.
    teacher = _FakeTeacher(["import os; os.system('rm -rf /')"])
    records, _ = generate_distillation_dataset(teacher, ["x"])  # verify=None
    assert records[0]["messages"][-1]["content"] == "import os; os.system('rm -rf /')"
    # (nothing executed — it's just stored as text)
