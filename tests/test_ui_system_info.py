"""Hermetic tests for cola_coder.ui.system_info.

These tests must not require a GPU and must not import torch. They monkeypatch
``subprocess.run`` to simulate both an nvidia-smi reading and its absence.
"""

from __future__ import annotations

import subprocess
import sys

from cola_coder.ui import system_info as si

_EXPECTED_PACKAGE_KEYS = {
    "torch",
    "transformers",
    "tokenizers",
    "fastapi",
    "numpy",
    "safetensors",
}


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_returns_dict_with_all_top_level_keys(tmp_path):
    info = si.system_info(str(tmp_path))
    assert isinstance(info, dict)
    for key in ("python_version", "platform", "packages", "gpus", "disk"):
        assert key in info


def test_python_version_and_platform_are_nonempty_strings(tmp_path):
    info = si.system_info(str(tmp_path))
    assert isinstance(info["python_version"], str) and info["python_version"]
    assert isinstance(info["platform"], str) and info["platform"]


def test_packages_is_dict_with_expected_keys(tmp_path):
    info = si.system_info(str(tmp_path))
    packages = info["packages"]
    assert isinstance(packages, dict)
    assert set(packages.keys()) == _EXPECTED_PACKAGE_KEYS
    for value in packages.values():
        assert value is None or isinstance(value, str)


def test_gpus_is_a_list(tmp_path):
    info = si.system_info(str(tmp_path))
    assert isinstance(info["gpus"], list)


def test_disk_has_int_fields_for_real_path(tmp_path):
    info = si.system_info(str(tmp_path))
    disk = info["disk"]
    assert disk["path"] == str(tmp_path)
    assert isinstance(disk["total_bytes"], int) and disk["total_bytes"] > 0
    assert isinstance(disk["free_bytes"], int)
    assert isinstance(disk["used_bytes"], int)


def test_call_never_raises_default_root():
    # Should not raise even with the default "." root.
    info = si.system_info()
    assert isinstance(info, dict)


def test_module_does_not_import_torch():
    # The whole point: importing/using this module must not pull in torch.
    si.system_info()
    assert "torch" not in sys.modules


def test_nvidia_smi_present_is_parsed(monkeypatch, tmp_path):
    fake_out = (
        "NVIDIA GeForce RTX 4080 SUPER, 16376, 2048, 37\n"
        "NVIDIA GeForce RTX 3080, 10240, 512, 5\n"
    )

    def fake_run(*args, **kwargs):
        return _FakeProc(fake_out, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = si.system_info(str(tmp_path))
    gpus = info["gpus"]
    assert len(gpus) == 2
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 4080 SUPER"
    assert gpus[0]["mem_total_mb"] == 16376
    assert gpus[0]["mem_used_mb"] == 2048
    assert gpus[0]["util_pct"] == 37
    assert gpus[1]["name"] == "NVIDIA GeForce RTX 3080"


def test_nvidia_smi_absent_yields_empty_gpus(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = si.system_info(str(tmp_path))
    assert info["gpus"] == []
    # Other fields remain intact despite GPU failure.
    assert info["python_version"]
    assert isinstance(info["disk"]["total_bytes"], int)


def test_nvidia_smi_nonzero_returncode_yields_empty_gpus(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return _FakeProc("", returncode=9)

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = si.system_info(str(tmp_path))
    assert info["gpus"] == []


def test_nvidia_smi_malformed_line_skipped(monkeypatch, tmp_path):
    fake_out = "only-one-field\nNVIDIA Test GPU, 8192, 1024, 50\n"

    def fake_run(*args, **kwargs):
        return _FakeProc(fake_out, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = si.system_info(str(tmp_path))
    gpus = info["gpus"]
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA Test GPU"
    assert gpus[0]["mem_total_mb"] == 8192


def test_disk_failure_leaves_none_fields(monkeypatch):
    import shutil as _shutil

    def fake_disk_usage(path):
        raise OSError("no such path")

    monkeypatch.setattr(_shutil, "disk_usage", fake_disk_usage)
    info = si.system_info("/definitely/not/a/real/path")
    disk = info["disk"]
    assert disk["total_bytes"] is None
    assert disk["free_bytes"] is None
    assert disk["used_bytes"] is None
    # Call still succeeds overall.
    assert isinstance(info, dict)


def test_result_is_json_serializable(tmp_path):
    import json

    info = si.system_info(str(tmp_path))
    # Should round-trip without error.
    json.loads(json.dumps(info))
