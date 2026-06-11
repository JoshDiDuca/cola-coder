"""TOOL-004: TSBenchmark's tsc tier must route through the shared sandboxed
TscRunner, not an ad-hoc ``subprocess.run(["tsc", ...])``.

The project mandates that ALL tsc execution and ALL execution-based scoring of
model-generated code go through ``TscRunner`` (hardened tsconfig: plugins/types/
typeRoots disabled, executed via SandboxedRunner). ``_tsc_check`` previously
shelled out to ``tsc`` directly on generated TypeScript in the default temp dir,
bypassing that isolation and duplicating tsc-invocation logic.
"""

import cola_coder.evaluation.ts_benchmark as tsb
from cola_coder.evaluation.ts_benchmark import TSBenchmark, _tsc_check
from cola_coder.reasoning.rewards.tsc_runner import TscError, TscRunner


class _FakeRunner:
    def __init__(self, errors=None, raises=False):
        self._errors = errors or []
        self._raises = raises

    def check(self, code):
        if self._raises:
            raise RuntimeError("tsc failed to launch")
        return self._errors


def _err(severity="error"):
    return TscError(file="check.ts", line=1, col=1, severity=severity, code="TS2322", message="x")


def _problem():
    return TSBenchmark().get_problems()[0]


class TestNoAdHocTsc:
    def test_subprocess_helpers_removed(self):
        # The ad-hoc path imported subprocess/tempfile/shutil at module level;
        # routing through TscRunner removes them. Guards against reintroduction.
        for name in ("subprocess", "tempfile", "shutil"):
            assert not hasattr(tsb, name), f"{name} should no longer be imported"


class TestTscCheckRouting:
    def test_none_when_tsc_unavailable(self, monkeypatch):
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: False))
        assert _tsc_check("const x: number = 1;", _problem()) is None

    def test_true_when_no_errors(self, monkeypatch):
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(errors=[]))
        assert _tsc_check("ok", _problem()) is True

    def test_false_when_type_error(self, monkeypatch):
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(errors=[_err("error")]))
        assert _tsc_check("bad", _problem()) is False

    def test_warnings_do_not_fail(self, monkeypatch):
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(errors=[_err("warning")]))
        assert _tsc_check("warn", _problem()) is True

    def test_none_when_runner_raises(self, monkeypatch):
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(raises=True))
        assert _tsc_check("boom", _problem()) is None


class TestEvaluateSolutionIntegration:
    def test_canonical_passes_with_tsc_clean(self, monkeypatch):
        # Tier 4 (tsc) clean → canonical solution still passes overall.
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(errors=[]))
        bench = TSBenchmark()
        prob = bench.get_problems()[0]
        assert bench.evaluate_solution(prob, prob.canonical_solution) is True

    def test_tsc_error_fails_otherwise_valid_solution(self, monkeypatch):
        # A solution that passes the static tiers is still failed by a tsc error.
        monkeypatch.setattr(TscRunner, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(tsb, "_get_tsc_runner", lambda: _FakeRunner(errors=[_err("error")]))
        bench = TSBenchmark()
        prob = bench.get_problems()[0]
        assert bench.evaluate_solution(prob, prob.canonical_solution) is False
