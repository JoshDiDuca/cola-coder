"""SEC-016: tsc must FAIL CLOSED when the sandbox didn't run.

When SandboxedRunner returns a negative returncode (timeout -1 / error -2 /
unavailable -3), tsc never executed — its output is empty, which previously parsed
to 0 errors and a PERFECT quality score for UNVERIFIED code (a corpus-poisoning
fail-open). The fix returns a SANDBOX_UNAVAILABLE sentinel so no caller mistakes
unverified code for clean.
"""

import subprocess

from cola_coder.data.scorers.tsc_scorer import TscScorer
from cola_coder.reasoning.rewards.tsc_runner import (
    SANDBOX_UNAVAILABLE_CODE,
    TscRunner,
)

_TS = "interface Foo { x: number }\nconst y: string = 'a';\n"
_META = {"file_path": "x.ts", "path": "x.ts", "extension": ".ts"}


class _FakeRunner:
    """Stand-in SandboxedRunner returning a fixed CompletedProcess."""

    def __init__(self, returncode, stdout="", stderr=""):
        self._rc, self._out, self._err = returncode, stdout, stderr
        self.calls = 0

    def run(self, cmd, cwd=None, label=None, file_hash=None):
        self.calls += 1
        return subprocess.CompletedProcess(cmd, self._rc, self._out, self._err)


class TestTscRunnerFailClosed:
    def test_negative_returncode_returns_sentinel_not_empty(self):
        r = TscRunner(runner=_FakeRunner(-3, "", "Sandbox unavailable, fail closed."))
        errs = r.check(_TS)
        assert len(errs) == 1
        assert errs[0].code == SANDBOX_UNAVAILABLE_CODE
        assert errs[0].severity == "error"

    def test_sentinel_is_not_cached(self):
        runner = _FakeRunner(-3, "", "unavailable")
        r = TscRunner(runner=runner)
        r.check(_TS)
        r.check(_TS)
        assert runner.calls == 2  # re-ran, not served from cache

    def test_clean_run_returns_empty(self):
        r = TscRunner(runner=_FakeRunner(0, "", ""))
        assert r.check(_TS) == []

    def test_real_tsc_errors_still_parsed(self):
        out = "check.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'."
        r = TscRunner(runner=_FakeRunner(1, out, ""))  # tsc exits 1 when it finds errors
        errs = r.check(_TS)
        assert len(errs) == 1 and errs[0].code == "TS2322"

    def test_batch_negative_returncode_marks_all_unverified(self):
        r = TscRunner(runner=_FakeRunner(-2, "", "error"))
        res = r.check_batch([_TS, _TS])
        assert set(res) == {0, 1}
        assert all(e.code == SANDBOX_UNAVAILABLE_CODE for errs in res.values() for e in errs)


class TestTscScorerFailClosed:
    def test_unavailable_scores_not_verified_not_perfect(self):
        s = TscScorer(runner=_FakeRunner(-3, "", "Sandbox unavailable"))
        res = s.score(_TS, _META)
        assert res.details.get("not_verified") is True
        assert res.score == 0.0  # NOT the old false-perfect 1.0

    def test_clean_code_still_scores_perfect(self):
        s = TscScorer(runner=_FakeRunner(0, "", ""))
        res = s.score(_TS, _META)
        assert not res.details.get("not_verified")
        assert res.score == 1.0  # verified, 0 errors -> unchanged behavior

    def test_batch_unavailable_marks_not_verified(self):
        s = TscScorer(runner=_FakeRunner(-3, "", "unavailable"))
        results = s.score_batch([(_TS, _META), (_TS, _META)])
        assert all(r.details.get("not_verified") is True for r in results)
        assert all(r.score == 0.0 for r in results)
