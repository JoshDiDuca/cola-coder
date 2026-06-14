"""DATA-065: DecontaminationFilter — drop training samples that overlap eval
benchmarks. Reuses the DataLeakageDetector shingling/containment via the standard
FilterPlugin contract.
"""

from cola_coder.data.registry import get_filter

_BENCH = (
    "def has_close_elements(numbers, threshold):\n"
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)


class R:
    def __init__(self, content: str):
        self.content = content
        self.metadata: dict = {}


def _filter(**cfg):
    f = get_filter("decontamination")()
    f.setup(cfg)
    return f


class TestRegistration:
    def test_registered_and_constructible(self):
        import cola_coder.data.filters  # noqa: F401 — fire @register_filter
        from cola_coder.data.registry import list_filters

        assert "decontamination" in set(list_filters())
        f = get_filter("decontamination")()
        assert f.name()
        f.setup({})  # empty config must not raise


class TestDecontamination:
    def test_drops_record_containing_benchmark(self):
        f = _filter(eval_texts=[_BENCH], threshold=0.8)
        # A scraped file that embeds the benchmark solution verbatim.
        rec = R("# my utils\n" + _BENCH + "\nprint('ok')\n")
        keep, reason = f.check(rec)
        assert keep is False
        assert "contamination" in reason

    def test_keeps_unrelated_code(self):
        f = _filter(eval_texts=[_BENCH], threshold=0.8)
        keep, _ = f.check(R("export const add = (a: number, b: number) => a + b;"))
        assert keep is True

    def test_no_eval_texts_is_noop(self):
        f = _filter()  # no refs configured
        keep, _ = f.check(R(_BENCH))
        assert keep is True

    def test_threshold_controls_sensitivity(self):
        # A record that only PARTIALLY overlaps the benchmark.
        partial = R("def has_close_elements(numbers, threshold):\n    pass\n")
        # Lenient threshold keeps it; strict (low) threshold drops it.
        assert _filter(eval_texts=[_BENCH], threshold=0.95).check(partial)[0] is True
        assert _filter(eval_texts=[_BENCH], threshold=0.05).check(partial)[0] is False

    def test_empty_record_kept(self):
        f = _filter(eval_texts=[_BENCH])
        assert f.check(R(""))[0] is True
