"""DATA-014: the router evaluation taxonomy must match the model/training one.

router_evaluation.KNOWN_DOMAINS (and create_test_dataset's labels) had diverged
to an 8-domain set including "python"/"general_ts", while the router model
(DEFAULT_DOMAINS), router_data_generator, and domain_detector all use the
canonical 7 ending in "general". A router can only ever output its trained
labels, so eval samples labelled "python"/"general_ts" were unroutable by
design — capping accuracy at 80% and adding phantom confusion-matrix rows.

These lock the taxonomy in sync and assert every built-in eval label is routable.
"""

from cola_coder.features.router_evaluation import (
    KNOWN_DOMAINS,
    create_test_dataset,
    RouterEvaluator,
)


class TestTaxonomyConsistency:
    def test_known_domains_match_default_domains(self):
        # DEFAULT_DOMAINS lives in router_model (torch); import lazily.
        from cola_coder.features.router_model import DEFAULT_DOMAINS

        assert KNOWN_DOMAINS == DEFAULT_DOMAINS

    def test_known_domains_match_data_generator_and_detector(self):
        # The training-data domains and the heuristic detector's domains
        # (its DOMAINS keys + the "general" fallback) must equal KNOWN_DOMAINS.
        from cola_coder.features.domain_detector import DOMAINS as DETECTOR_DOMAINS

        detector_set = set(DETECTOR_DOMAINS.keys()) | {"general"}
        assert detector_set == set(KNOWN_DOMAINS)

    def test_no_python_or_general_ts_labels(self):
        # The stale labels must be gone.
        assert "python" not in KNOWN_DOMAINS
        assert "general_ts" not in KNOWN_DOMAINS


class TestEvalDatasetRoutable:
    def test_all_sample_labels_are_known_domains(self):
        # Every expected_domain must be something the router can actually emit,
        # otherwise the sample is unwinnable and silently deflates accuracy.
        for sample in create_test_dataset():
            assert sample.expected_domain in KNOWN_DOMAINS, (
                f"unroutable label {sample.expected_domain!r}"
            )

    def test_perfect_router_scores_100_percent(self):
        # A hypothetical perfect router (pred == expected for every sample) must
        # be able to reach 1.0 accuracy — impossible before the fix because the
        # python/general_ts samples could never match a routable prediction.
        ev = RouterEvaluator()
        for sample in create_test_dataset():
            ev.add_result(sample.expected_domain, sample.expected_domain, 1.0)
        assert ev.accuracy() == 1.0
