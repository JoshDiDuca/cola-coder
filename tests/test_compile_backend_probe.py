"""Tests for the torch.compile backend readiness probe (BUG-119).

torch.compile's default (inductor) backend needs Triton on CUDA. When Triton is
absent (common on Windows), compilation crashes LAZILY on the first forward pass
with ``InductorError: TritonMissing`` — past the try/except around the compile
call — killing training at step 0. ``_torch_compile_backend_ready`` probes for
Triton up front so the trainer can stay in eager mode instead of crashing.
"""

import importlib.util

from cola_coder.training.trainer import _torch_compile_backend_ready


def test_backend_not_ready_on_cpu():
    # No compile backend on CPU regardless of Triton — always eager.
    assert _torch_compile_backend_ready("cpu") is False


def test_backend_not_ready_when_triton_missing(monkeypatch):
    # Simulate Triton not installed: find_spec returns None.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _torch_compile_backend_ready("cuda") is False


def test_backend_ready_when_triton_present(monkeypatch):
    # Simulate Triton installed: find_spec returns a spec object.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert _torch_compile_backend_ready("cuda") is True


def test_backend_probe_swallows_find_spec_errors(monkeypatch):
    # find_spec can raise (ImportError/ValueError) on broken namespace pkgs;
    # the probe must degrade to "not ready" rather than propagate.
    def _boom(name):
        raise ValueError("broken namespace package")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert _torch_compile_backend_ready("cuda") is False
