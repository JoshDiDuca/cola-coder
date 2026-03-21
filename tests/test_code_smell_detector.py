"""Tests for CodeSmellDetector (features/code_smell_detector.py)."""

from __future__ import annotations


from cola_coder.features.code_smell_detector import (
    FEATURE_ENABLED,
    CodeSmellDetector,
    SmellInstance,
    SmellReport,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Sample code
# ---------------------------------------------------------------------------

CLEAN_CODE = """\
def add(a: int, b: int) -> int:
    return a + b


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""

LONG_METHOD_CODE = """\
def very_long_method():
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    h = 11
    i = 12
    j = 13
    k = 14
    l = 15
    m = 16
    n = 17
    o = 18
    p = 19
    q = 20
    r = 21
    s = 22
    return s
"""

DEEPLY_NESTED_CODE = """\
def deeply_nested(x):
    if x > 0:
        for i in range(x):
            while i > 0:
                if i % 2 == 0:
                    for j in range(i):
                        pass
"""

DATA_CLUMP_CODE = """\
def create_user(name, email, age, role):
    pass

def update_user(name, email, age, role):
    pass

def validate_user(name, email, age, role):
    pass
"""

FEATURE_ENVY_CODE = """\
class Order:
    def __init__(self):
        self.customer = None
        self.product = None

    def compute_price(self, customer, product, discount):
        base = product.price
        tax = product.tax_rate
        loyalty = customer.loyalty_discount
        tier = customer.tier_discount
        region = customer.region_rate
        return base * (1 + tax) * (1 - loyalty) * (1 - tier) * (1 - region) * (1 - discount)
"""


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------

class TestIsEnabled:
    def test_feature_enabled_constant(self):
        assert FEATURE_ENABLED is True

    def test_is_enabled_function(self):
        assert is_enabled() is True


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestBasic:
    def test_returns_smell_report(self):
        detector = CodeSmellDetector()
        result = detector.detect(CLEAN_CODE)
        assert isinstance(result, SmellReport)

    def test_clean_code_no_smells(self):
        detector = CodeSmellDetector()
        result = detector.detect(CLEAN_CODE)
        assert len(result.smells) == 0

    def test_invalid_syntax_no_crash(self):
        detector = CodeSmellDetector()
        result = detector.detect("def broken(:\n    pass\n")
        assert isinstance(result, SmellReport)

    def test_empty_code(self):
        detector = CodeSmellDetector()
        result = detector.detect("")
        assert len(result.smells) == 0


# ---------------------------------------------------------------------------
# Long method
# ---------------------------------------------------------------------------

class TestLongMethod:
    def test_detects_long_method(self):
        detector = CodeSmellDetector(max_method_statements=10)
        result = detector.detect(LONG_METHOD_CODE)
        assert result.has_smell("long_method")

    def test_severity_in_range(self):
        detector = CodeSmellDetector(max_method_statements=10)
        result = detector.detect(LONG_METHOD_CODE)
        for smell in result.smells:
            assert 0.0 <= smell.severity <= 1.0

    def test_no_long_method_on_short(self):
        detector = CodeSmellDetector(max_method_statements=100)
        result = detector.detect(LONG_METHOD_CODE)
        assert not result.has_smell("long_method")


# ---------------------------------------------------------------------------
# Deep nesting
# ---------------------------------------------------------------------------

class TestDeepNesting:
    def test_detects_deep_nesting(self):
        detector = CodeSmellDetector(max_nesting_depth=2)
        result = detector.detect(DEEPLY_NESTED_CODE)
        assert result.has_smell("deep_nesting")

    def test_no_nesting_smell_when_threshold_high(self):
        detector = CodeSmellDetector(max_nesting_depth=10)
        result = detector.detect(DEEPLY_NESTED_CODE)
        assert not result.has_smell("deep_nesting")


# ---------------------------------------------------------------------------
# Data clumps
# ---------------------------------------------------------------------------

class TestDataClumps:
    def test_detects_data_clump(self):
        detector = CodeSmellDetector(data_clump_min_size=3, data_clump_min_repeat=2)
        result = detector.detect(DATA_CLUMP_CODE)
        assert result.has_smell("data_clump")

    def test_suggestion_mentions_dataclass(self):
        detector = CodeSmellDetector(data_clump_min_size=3, data_clump_min_repeat=2)
        result = detector.detect(DATA_CLUMP_CODE)
        clump_smells = [s for s in result.smells if s.smell_type == "data_clump"]
        assert any("dataclass" in s.suggestion.lower() for s in clump_smells)


# ---------------------------------------------------------------------------
# Smell report utilities
# ---------------------------------------------------------------------------

class TestSmellReport:
    def test_average_severity_zero_no_smells(self):
        report = SmellReport()
        assert report.average_severity == 0.0

    def test_add_smell(self):
        report = SmellReport()
        smell = SmellInstance("long_method", "foo", 0.5, "details", "suggestion")
        report.add(smell)
        assert len(report.smells) == 1
        assert report.average_severity == 0.5

    def test_summary_no_smells(self):
        report = SmellReport()
        assert "No smells" in report.summary()

    def test_summary_with_smells(self):
        detector = CodeSmellDetector(max_method_statements=10)
        result = detector.detect(LONG_METHOD_CODE)
        summary = result.summary()
        assert "smell" in summary.lower()

    def test_smell_counts_tracked(self):
        detector = CodeSmellDetector(max_method_statements=10)
        result = detector.detect(LONG_METHOD_CODE)
        assert "long_method" in result.smell_counts
