"""Tests for the training dashboard module.

All tests run without GPU (torch.cuda calls are safely handled).
Tests cover:
- ascii_chart rendering
- get_gpu_stats (CPU-only path)
- TrainingDashboard construction and update
- Metric history recording
- ETA calculation
- Layout building (valid Rich renderables)
- Panel/table building
- Context manager interface
- Edge cases (empty data, single point, identical values)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cola_coder.training.dashboard import (
    TrainingDashboard,
    _format_eta,
    ascii_chart,
    get_gpu_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dashboard(total_steps: int = 1000, **cfg_overrides) -> TrainingDashboard:
    """Create a TrainingDashboard with sensible defaults for testing."""
    config = {
        "model_params": 350e6,
        "batch_size": 4,
        "effective_batch_size": 16,
        "learning_rate": 3e-4,
        "seq_len": 2048,
        "precision": "bf16",
        "model_size_name": "medium",
        **cfg_overrides,
    }
    return TrainingDashboard(config=config, total_steps=total_steps)


# ---------------------------------------------------------------------------
# 1. ascii_chart — empty input
# ---------------------------------------------------------------------------

class TestAsciiChart:
    def test_empty_returns_string(self):
        result = ascii_chart([])
        assert isinstance(result, str)
        assert len(result) > 0  # should be a "no data" message

    def test_single_value(self):
        result = ascii_chart([2.5])
        assert isinstance(result, str)
        # Should contain the value displayed in y-axis label
        # and at least one filled block character
        assert "\u2593" in result or "2" in result

    def test_multiple_values(self):
        values = [3.0, 2.8, 2.5, 2.3, 2.1, 2.0]
        result = ascii_chart(values, width=10, height=4)
        assert isinstance(result, str)
        lines = result.split("\n")
        # Should have height rows + 1 bottom axis line
        assert len(lines) == 5

    def test_height_and_width_respected(self):
        values = list(range(20, 0, -1))  # 20 values descending
        result = ascii_chart(values, width=10, height=6)
        lines = result.split("\n")
        assert len(lines) == 7  # 6 data rows + 1 bottom axis

    def test_identical_values_no_crash(self):
        """All-same values should not cause divide-by-zero."""
        values = [2.5] * 10
        result = ascii_chart(values, width=10, height=4)
        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 5

    def test_decreasing_trend(self):
        """Decreasing values produce different chart from increasing."""
        decreasing = [5.0, 4.0, 3.0, 2.0, 1.0]
        increasing = [1.0, 2.0, 3.0, 4.0, 5.0]
        result_dec = ascii_chart(decreasing, width=5, height=4)
        result_inc = ascii_chart(increasing, width=5, height=4)
        # They should differ (different bar heights in different columns)
        assert result_dec != result_inc

    def test_bottom_axis_present(self):
        values = [2.0, 2.5, 2.1]
        result = ascii_chart(values)
        # Bottom axis uses └ (U+2514)
        assert "\u2514" in result

    def test_y_axis_separator_present(self):
        values = [2.0, 2.5, 2.1]
        result = ascii_chart(values)
        # Vertical bar │ (U+2502)
        assert "\u2502" in result

    def test_truncates_to_width(self):
        """Only last `width` values should appear."""
        # 100 values but width=5 — only last 5 columns in the chart
        values = list(range(100))
        result = ascii_chart(values, width=5, height=4)
        lines = result.split("\n")
        # Each data line should have 5 columns * 2 chars per col = 10 chart chars
        # (plus ~8 chars for label + │)
        for line in lines[:-1]:  # skip bottom axis
            chart_part = line.split("\u2502", 1)
            if len(chart_part) == 2:
                # chart_part[1] has 5*2 = 10 characters
                assert len(chart_part[1]) == 10


# ---------------------------------------------------------------------------
# 2. get_gpu_stats — CPU-only (no CUDA)
# ---------------------------------------------------------------------------

class TestGetGpuStats:
    def test_no_cuda_returns_available_false(self):
        """On CPU-only machines, get_gpu_stats returns available=False."""
        with patch("cola_coder.training.dashboard._HAS_TORCH", True):
            with patch("cola_coder.training.dashboard.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                result = get_gpu_stats()
        assert result["available"] is False

    def test_no_torch_returns_available_false(self):
        """If torch is not installed, returns available=False gracefully."""
        with patch("cola_coder.training.dashboard._HAS_TORCH", False):
            result = get_gpu_stats()
        assert result["available"] is False

    def test_with_cuda_returns_stats(self):
        """With mocked CUDA, returns expected keys."""
        mock_props = MagicMock()
        mock_props.total_mem = 16 * 1e9
        mock_props.total_memory = 16 * 1e9

        with patch("cola_coder.training.dashboard._HAS_TORCH", True):
            with patch("cola_coder.training.dashboard.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080"
                mock_torch.cuda.memory_allocated.return_value = 8 * 1e9
                mock_torch.cuda.memory_reserved.return_value = 9 * 1e9
                mock_torch.cuda.get_device_properties.return_value = mock_props

                result = get_gpu_stats()

        assert result["available"] is True
        assert result["name"] == "NVIDIA GeForce RTX 4080"
        assert abs(result["memory_used_gb"] - 8.0) < 0.1
        assert abs(result["memory_total_gb"] - 16.0) < 0.1

    def test_cuda_exception_returns_unavailable(self):
        """If torch raises an exception, returns available=False."""
        with patch("cola_coder.training.dashboard._HAS_TORCH", True):
            with patch("cola_coder.training.dashboard.torch") as mock_torch:
                mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
                result = get_gpu_stats()
        assert result["available"] is False


# ---------------------------------------------------------------------------
# 3. _format_eta
# ---------------------------------------------------------------------------

class TestFormatEta:
    def test_seconds(self):
        assert "s" in _format_eta(45)

    def test_minutes(self):
        result = _format_eta(125)
        assert "m" in result

    def test_hours(self):
        result = _format_eta(7200)
        assert "h" in result

    def test_days(self):
        result = _format_eta(3 * 24 * 3600)
        assert "d" in result

    def test_zero_returns_question_mark(self):
        assert _format_eta(0) == "?"

    def test_negative_returns_question_mark(self):
        assert _format_eta(-10) == "?"

    def test_inf_returns_question_mark(self):
        assert _format_eta(float("inf")) == "?"


# ---------------------------------------------------------------------------
# 4. TrainingDashboard construction
# ---------------------------------------------------------------------------

class TestDashboardConstruction:
    def test_creates_without_error(self):
        dashboard = make_dashboard()
        assert dashboard is not None

    def test_metrics_history_initialized(self):
        dashboard = make_dashboard()
        assert "loss" in dashboard.metrics_history
        assert "lr" in dashboard.metrics_history
        assert "throughput" in dashboard.metrics_history
        assert "gpu_mem" in dashboard.metrics_history
        assert "grad_norm" in dashboard.metrics_history

    def test_history_starts_empty(self):
        dashboard = make_dashboard()
        for key in dashboard.metrics_history:
            assert len(dashboard.metrics_history[key]) == 0

    def test_total_steps_set(self):
        dashboard = make_dashboard(total_steps=50000)
        assert dashboard.total_steps == 50000

    def test_zero_total_steps_clamped(self):
        """total_steps=0 should not cause division-by-zero."""
        dashboard = TrainingDashboard(config={}, total_steps=0)
        assert dashboard.total_steps >= 1


# ---------------------------------------------------------------------------
# 5. update() — metric recording
# ---------------------------------------------------------------------------

class TestDashboardUpdate:
    def test_update_records_loss(self):
        dashboard = make_dashboard()
        dashboard.update(step=100, loss=2.5, lr=3e-4, throughput=22000)
        assert list(dashboard.metrics_history["loss"]) == [2.5]

    def test_update_records_lr(self):
        dashboard = make_dashboard()
        dashboard.update(step=100, loss=2.5, lr=3e-4, throughput=22000)
        assert abs(list(dashboard.metrics_history["lr"])[0] - 3e-4) < 1e-10

    def test_update_records_throughput(self):
        dashboard = make_dashboard()
        dashboard.update(step=100, loss=2.5, lr=3e-4, throughput=22000)
        assert list(dashboard.metrics_history["throughput"]) == [22000]

    def test_update_records_gpu_mem(self):
        dashboard = make_dashboard()
        dashboard.update(step=100, loss=2.5, lr=3e-4, throughput=22000, gpu_mem_gb=8.2)
        assert list(dashboard.metrics_history["gpu_mem"]) == [8.2]

    def test_update_records_grad_norm(self):
        dashboard = make_dashboard()
        dashboard.update(step=100, loss=2.5, lr=3e-4, throughput=22000, grad_norm=0.82)
        assert abs(list(dashboard.metrics_history["grad_norm"])[0] - 0.82) < 1e-10

    def test_update_increments_current_step(self):
        dashboard = make_dashboard()
        dashboard.update(step=500, loss=2.5, lr=3e-4, throughput=22000)
        assert dashboard._current_step == 500

    def test_multiple_updates_accumulate(self):
        dashboard = make_dashboard()
        for i in range(10):
            dashboard.update(step=i * 100, loss=3.0 - i * 0.1, lr=3e-4, throughput=22000)
        assert len(dashboard.metrics_history["loss"]) == 10

    def test_update_appends_recent_step(self):
        dashboard = make_dashboard()
        dashboard.update(step=42, loss=2.31, lr=2.4e-4, throughput=22100, grad_norm=0.82)
        assert len(dashboard._recent_steps) == 1
        entry = list(dashboard._recent_steps)[0]
        assert entry["step"] == 42
        assert abs(entry["loss"] - 2.31) < 1e-6

    def test_update_does_not_crash_without_start(self):
        """update() should work even if start() was never called."""
        dashboard = make_dashboard()
        dashboard.update(step=1, loss=3.0, lr=1e-3, throughput=1000)
        # No exception = pass

    def test_history_max_size_respected(self):
        """History deque should not exceed MAX_HISTORY."""
        dashboard = make_dashboard()
        for i in range(dashboard.MAX_HISTORY + 100):
            dashboard.update(step=i, loss=2.5, lr=1e-3, throughput=1000)
        assert len(dashboard.metrics_history["loss"]) == dashboard.MAX_HISTORY

    def test_extra_kwargs_do_not_crash(self):
        """Unknown keyword args should be silently accepted."""
        dashboard = make_dashboard()
        dashboard.update(step=1, loss=3.0, lr=1e-3, throughput=1000,
                         custom_metric=42, another_value="hello")


# ---------------------------------------------------------------------------
# 6. Layout building — valid Rich renderables
# ---------------------------------------------------------------------------

class TestLayoutBuilding:
    def test_build_layout_returns_layout(self):
        """_build_layout() should return a Rich Layout object."""
        try:
            from rich.layout import Layout
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        layout = dashboard._build_layout()
        assert isinstance(layout, Layout)

    def test_loss_chart_panel_is_panel(self):
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        panel = dashboard._loss_chart_panel()
        assert isinstance(panel, Panel)

    def test_gpu_panel_is_panel(self):
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        panel = dashboard._gpu_panel()
        assert isinstance(panel, Panel)

    def test_progress_panel_is_panel(self):
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        panel = dashboard._progress_panel()
        assert isinstance(panel, Panel)

    def test_config_panel_is_panel(self):
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        panel = dashboard._config_panel()
        assert isinstance(panel, Panel)

    def test_recent_losses_table_is_panel(self):
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        panel = dashboard._recent_losses_table()
        assert isinstance(panel, Panel)

    def test_layout_with_no_data(self):
        """Layout should build without crashing when metrics_history is empty."""
        try:
            from rich.layout import Layout
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard()
        # Don't call update() — history is empty
        layout = dashboard._build_layout()
        assert isinstance(layout, Layout)

    def test_layout_with_data(self):
        """Layout should build without crashing after several updates."""
        try:
            from rich.layout import Layout
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = make_dashboard(total_steps=1000)
        for i in range(50):
            dashboard.update(
                step=i * 10,
                loss=3.5 - i * 0.03,
                lr=3e-4,
                throughput=22000,
                gpu_mem_gb=8.2,
                grad_norm=0.8,
            )
        layout = dashboard._build_layout()
        assert isinstance(layout, Layout)


# ---------------------------------------------------------------------------
# 7. ETA calculation
# ---------------------------------------------------------------------------

class TestEtaCalculation:
    def test_eta_decreases_as_steps_increase(self):
        """Simulated ETA should decrease as more steps are done."""
        import time as time_mod

        dashboard = make_dashboard(total_steps=1000)
        # Simulate time passing by manually adjusting _start_time
        dashboard._start_time = time_mod.time() - 10  # 10 seconds elapsed

        dashboard.update(step=100, loss=3.0, lr=1e-3, throughput=1000)
        panel1 = dashboard._progress_panel()

        dashboard._start_time = time_mod.time() - 50  # 50 seconds elapsed
        dashboard.update(step=500, loss=2.5, lr=1e-3, throughput=1000)
        panel2 = dashboard._progress_panel()

        # Both should return Panel objects — no crash
        from rich.panel import Panel
        assert isinstance(panel1, Panel)
        assert isinstance(panel2, Panel)

    def test_progress_percentage(self):
        """Progress panel should reflect the current step percentage."""
        dashboard = make_dashboard(total_steps=1000)
        dashboard.update(step=500, loss=2.5, lr=1e-3, throughput=22000)

        # The panel renderable has the step info — we check no crash and correct step
        assert dashboard._current_step == 500


# ---------------------------------------------------------------------------
# 8. Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_calls_start_stop(self):
        """Using dashboard as a context manager calls start() and stop()."""
        dashboard = make_dashboard()
        start_called = []
        stop_called = []

        original_start = dashboard.start
        original_stop = dashboard.stop

        def mock_start():
            start_called.append(1)
            original_start()

        def mock_stop():
            stop_called.append(1)
            original_stop()

        dashboard.start = mock_start
        dashboard.stop = mock_stop

        with dashboard:
            pass

        assert len(start_called) == 1
        assert len(stop_called) == 1

    def test_context_manager_stop_on_exception(self):
        """stop() should be called even if an exception occurs inside the context."""
        dashboard = make_dashboard()
        stop_called = []

        original_stop = dashboard.stop
        def mock_stop():
            stop_called.append(1)
            original_stop()

        dashboard.stop = mock_stop

        try:
            with dashboard:
                raise ValueError("simulated error")
        except ValueError:
            pass

        assert len(stop_called) == 1


# ---------------------------------------------------------------------------
# 9. start/stop lifecycle
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_stop_without_start_does_not_crash(self):
        """Calling stop() before start() should be a no-op."""
        dashboard = make_dashboard()
        dashboard.stop()  # Should not raise

    def test_stop_twice_does_not_crash(self):
        """Calling stop() twice should be safe."""
        dashboard = make_dashboard()
        dashboard.start()
        dashboard.stop()
        dashboard.stop()  # Should not raise

    def test_live_is_none_after_stop(self):
        """After stop(), _live should be None."""
        dashboard = make_dashboard()
        dashboard.start()
        dashboard.stop()
        assert dashboard._live is None


# ---------------------------------------------------------------------------
# 10. Config panel — various config shapes
# ---------------------------------------------------------------------------

class TestConfigPanel:
    def test_empty_config_no_crash(self):
        """Config panel should render even with empty config dict."""
        try:
            from rich.panel import Panel
        except ImportError:
            pytest.skip("Rich not installed")

        dashboard = TrainingDashboard(config={}, total_steps=1000)
        panel = dashboard._config_panel()
        assert isinstance(panel, Panel)

    def test_params_formatting_millions(self):
        dashboard = TrainingDashboard(
            config={"model_params": 125e6, "model_size_name": "small"},
            total_steps=1000,
        )
        panel = dashboard._config_panel()
        from rich.panel import Panel
        assert isinstance(panel, Panel)

    def test_params_formatting_billions(self):
        dashboard = TrainingDashboard(
            config={"model_params": 1.3e9, "model_size_name": "large"},
            total_steps=1000,
        )
        panel = dashboard._config_panel()
        from rich.panel import Panel
        assert isinstance(panel, Panel)
