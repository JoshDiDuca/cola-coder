"""Tests for HyperparamLogger (features/hyperparam_logger.py)."""

from __future__ import annotations


import pytest

from cola_coder.features.hyperparam_logger import (
    FEATURE_ENABLED,
    HyperparamDiff,
    HyperparamLogger,
    RunRecord,
    SuggestionReport,
    is_enabled,
    log_and_diff,
)

# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# log_run
# ---------------------------------------------------------------------------


class TestLogRun:
    def test_returns_run_id(self):
        logger = HyperparamLogger()
        run_id = logger.log_run({"lr": 1e-4})
        assert isinstance(run_id, str)

    def test_custom_run_id(self):
        logger = HyperparamLogger()
        run_id = logger.log_run({"lr": 1e-4}, run_id="my_run")
        assert run_id == "my_run"

    def test_get_run_returns_record(self):
        logger = HyperparamLogger()
        run_id = logger.log_run({"lr": 1e-4, "batch_size": 32})
        rec = logger.get_run(run_id)
        assert isinstance(rec, RunRecord)
        assert rec.params["lr"] == pytest.approx(1e-4)

    def test_get_run_missing_returns_none(self):
        logger = HyperparamLogger()
        assert logger.get_run("nonexistent") is None

    def test_all_runs_length(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4})
        logger.log_run({"lr": 5e-5})
        assert len(logger.all_runs()) == 2


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_diff_returns_hyperparam_diff(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4, "bs": 32})
        b = logger.log_run({"lr": 5e-5, "bs": 32})
        diff = logger.diff(a, b)
        assert isinstance(diff, HyperparamDiff)

    def test_changed_param_detected(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4})
        b = logger.log_run({"lr": 5e-5})
        diff = logger.diff(a, b)
        assert "lr" in diff.changed
        assert diff.changed["lr"] == (1e-4, 5e-5)

    def test_added_param_detected(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4})
        b = logger.log_run({"lr": 1e-4, "warmup": 100})
        diff = logger.diff(a, b)
        assert "warmup" in diff.added

    def test_removed_param_detected(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4, "dropout": 0.1})
        b = logger.log_run({"lr": 1e-4})
        diff = logger.diff(a, b)
        assert "dropout" in diff.removed

    def test_no_changes(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4})
        b = logger.log_run({"lr": 1e-4})
        diff = logger.diff(a, b)
        assert not diff.has_changes

    def test_diff_latest_returns_last_two(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4})
        logger.log_run({"lr": 5e-5})
        diff = logger.diff_latest()
        assert diff is not None
        assert "lr" in diff.changed

    def test_diff_latest_none_if_less_than_two(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4})
        assert logger.diff_latest() is None

    def test_diff_missing_run_raises(self):
        logger = HyperparamLogger()
        a = logger.log_run({"lr": 1e-4})
        with pytest.raises(KeyError):
            logger.diff(a, "ghost_run")


# ---------------------------------------------------------------------------
# best_run + suggest
# ---------------------------------------------------------------------------


class TestBestRunAndSuggest:
    def test_best_run_returns_lowest_loss(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4}, final_loss=2.5)
        logger.log_run({"lr": 5e-5}, final_loss=2.1)
        logger.log_run({"lr": 1e-3}, final_loss=3.0)
        best = logger.best_run()
        assert best is not None
        assert best.final_loss == pytest.approx(2.1)

    def test_best_run_none_if_no_losses(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4})
        assert logger.best_run() is None

    def test_suggest_returns_report(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4}, final_loss=2.5)
        s = logger.suggest()
        assert isinstance(s, SuggestionReport)
        assert "lr" in s.suggestions

    def test_suggest_none_if_no_runs(self):
        logger = HyperparamLogger()
        assert logger.suggest() is None


# ---------------------------------------------------------------------------
# history_for_param
# ---------------------------------------------------------------------------


class TestHistoryForParam:
    def test_returns_history(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4}, final_loss=2.5)
        logger.log_run({"lr": 5e-5}, final_loss=2.1)
        hist = logger.history_for_param("lr")
        assert len(hist) == 2
        values = [v for _, v, _ in hist]
        assert 1e-4 in values
        assert 5e-5 in values

    def test_missing_param_excluded(self):
        logger = HyperparamLogger()
        logger.log_run({"lr": 1e-4})
        logger.log_run({"bs": 32})
        hist = logger.history_for_param("lr")
        assert len(hist) == 1


# ---------------------------------------------------------------------------
# Persistence (tmp_path)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        log_file = tmp_path / "runs.json"
        logger = HyperparamLogger(log_path=log_file)
        logger.log_run({"lr": 1e-4}, final_loss=2.5)
        assert log_file.exists()

        logger2 = HyperparamLogger(log_path=log_file)
        assert len(logger2.all_runs()) == 1
        assert logger2.all_runs()[0].params["lr"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# log_and_diff convenience
# ---------------------------------------------------------------------------


class TestLogAndDiff:
    def test_returns_logger_and_diff(self):
        logger, diff = log_and_diff(
            [{"lr": 1e-4, "bs": 32}, {"lr": 5e-5, "bs": 32}],
            losses=[2.5, 2.1],
        )
        assert isinstance(logger, HyperparamLogger)
        assert diff is not None
        assert "lr" in diff.changed

    def test_single_run_diff_is_none(self):
        _, diff = log_and_diff([{"lr": 1e-4}])
        assert diff is None
