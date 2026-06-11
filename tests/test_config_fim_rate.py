"""DATA-012 discoverability: shipped configs expose data.fim_rate, off by default.

Dynamic train-time FIM is wired via DataConfig.fim_rate. The config templates now
document the knob so it's discoverable. These lock that (a) every main config
parses with the field and (b) it defaults to 0.0 (off) — no surprise FIM, and a
future config that flips it on must be deliberate.
"""

import pytest

from cola_coder.model.config import Config

_CONFIGS = ["tiny", "small", "medium", "4080_max", "large"]


@pytest.mark.parametrize("name", _CONFIGS)
def test_config_has_fim_rate_off_by_default(name):
    cfg = Config.from_yaml(f"configs/{name}.yaml")
    assert hasattr(cfg.data, "fim_rate"), f"{name} missing data.fim_rate"
    assert cfg.data.fim_rate == 0.0, f"{name} should default fim_rate to 0.0 (off)"
    # psm rate is a valid float in [0, 1]
    assert 0.0 <= float(getattr(cfg.data, "fim_psm_rate", 0.5)) <= 1.0


def test_config_yaml_documents_fim_rate():
    # The knob must appear (commented or not) in the data templates so users can
    # discover it — not just live in the dataclass default.
    from pathlib import Path

    for name in _CONFIGS:
        text = Path(f"configs/{name}.yaml").read_text(encoding="utf-8")
        assert "fim_rate" in text, f"{name}.yaml does not surface fim_rate"
