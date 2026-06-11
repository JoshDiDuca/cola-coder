"""EVAL-008: the trainer's during-training auto-eval must be crash-safe.

AutoEvaluator was implemented and integrated into the Trainer but never wired
into train.py, so the eval path had never run in production. Now that train.py
exposes it via opt-in --auto-eval, the trainer's call must guarantee that a
failure in the (generation/sandbox) eval path NEVER crashes a long training run,
and that the model is always returned to train mode (evaluate() switches to eval
mode and only restores at its end).

`_run_auto_eval` is tested directly via a lightweight stub (no full Trainer
construction needed) — it only touches self.auto_evaluator / self.model /
self.device.
"""

from unittest.mock import MagicMock

from cola_coder.training.trainer import Trainer


class _Stub:
    pass


def _stub(auto_evaluator):
    s = _Stub()
    s.auto_evaluator = auto_evaluator
    s.model = MagicMock()
    s.device = "cpu"
    return s


def test_failure_is_swallowed_and_train_mode_restored():
    ae = MagicMock()
    ae.should_eval.return_value = True
    ae.evaluate.side_effect = RuntimeError("generator blew up")
    s = _stub(ae)

    # Must NOT raise — a failed eval cannot kill training.
    Trainer._run_auto_eval(s, step=100, tokenizer=object())

    ae.evaluate.assert_called_once()
    s.model.train.assert_called_once()  # restored via finally even on failure


def test_success_runs_report_and_regression_check():
    ae = MagicMock()
    ae.should_eval.return_value = True
    ae.format_report.return_value = "report"
    ae.history = [MagicMock(pass_at_1=0.3)]
    ae.check_regression.return_value = False
    s = _stub(ae)

    Trainer._run_auto_eval(s, step=100, tokenizer=object())

    ae.evaluate.assert_called_once()
    ae.format_report.assert_called_once()
    ae.check_regression.assert_called_once()
    s.model.train.assert_called_once()


def test_skipped_when_no_evaluator():
    s = _stub(auto_evaluator=None)
    Trainer._run_auto_eval(s, step=100, tokenizer=object())
    s.model.train.assert_not_called()


def test_skipped_when_no_tokenizer():
    ae = MagicMock()
    ae.should_eval.return_value = True
    s = _stub(ae)
    Trainer._run_auto_eval(s, step=100, tokenizer=None)
    ae.evaluate.assert_not_called()
    s.model.train.assert_not_called()


def test_skipped_when_should_eval_false():
    ae = MagicMock()
    ae.should_eval.return_value = False
    s = _stub(ae)
    Trainer._run_auto_eval(s, step=101, tokenizer=object())
    ae.evaluate.assert_not_called()
    s.model.train.assert_not_called()
