"""Regression tests: configs/features.yaml must stay honest.

History: features.yaml accumulated 8 phantom keys (typescript_rewards,
sft_warmup ×2 — a duplicate that YAML silently collapses — parallel_generation,
expanded_problems, doc_scraper, doc_fetcher, context_training, repo_context)
that mapped to no module in src/cola_coder/features/, so toggling them did
nothing. The loader only patches FEATURE_ENABLED on modules in that package.

The reverse (a module missing from the yaml) is fine by design: the yaml is a
sparse override file and unlisted modules keep their source default.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
FEATURES_YAML = ROOT / "configs" / "features.yaml"
FEATURES_PKG = ROOT / "src" / "cola_coder" / "features"


def _feature_modules() -> set[str]:
    return {p.stem for p in FEATURES_PKG.glob("*.py")} - {"__init__"}


def test_every_features_yaml_key_has_a_module():
    cfg = yaml.safe_load(FEATURES_YAML.read_text(encoding="utf-8"))
    orphans = set(cfg["features"]) - _feature_modules()
    assert not orphans, (
        f"features.yaml keys with no module in src/cola_coder/features/: "
        f"{sorted(orphans)} — toggling them does nothing. Either create the "
        "module or delete the key."
    )


def test_no_duplicate_keys_in_features_yaml():
    # yaml.safe_load silently keeps only the LAST duplicate — scan raw text.
    text = FEATURES_YAML.read_text(encoding="utf-8")
    keys = re.findall(r"^  (\w+):", text, flags=re.MULTILINE)
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"duplicate keys in features.yaml: {sorted(dupes)}"


def test_features_yaml_parses_and_is_all_bools():
    cfg = yaml.safe_load(FEATURES_YAML.read_text(encoding="utf-8"))
    non_bool = {k: v for k, v in cfg["features"].items() if not isinstance(v, bool)}
    assert not non_bool, f"non-boolean feature values: {non_bool}"
