"""Tests for api_extractor.py."""

from __future__ import annotations


from cola_coder.features.api_extractor import (
    FEATURE_ENABLED,
    APIExtractor,
    ModuleAPI,
    ParamInfo,
    is_enabled,
)

# ---------------------------------------------------------------------------
# Sample code
# ---------------------------------------------------------------------------

SIMPLE_MODULE = '''\
"""A simple module."""

MAX_VALUE: int = 100
DEFAULT_NAME = "world"


def greet(name: str, times: int = 1) -> str:
    """Return a greeting."""
    return f"Hello, {name}!" * times


async def fetch(url: str) -> bytes:
    """Fetch data from a URL."""
    ...


class Point:
    """A 2D point."""

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance(self) -> float:
        """Compute distance from origin."""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def _private_helper(self) -> None:
        pass

    @property
    def magnitude(self) -> float:
        return self.distance()

    @classmethod
    def origin(cls) -> "Point":
        return cls(0.0, 0.0)

    @staticmethod
    def from_tuple(t: tuple) -> "Point":
        return Point(t[0], t[1])
'''


class TestIsEnabled:
    def test_constant(self):
        assert FEATURE_ENABLED is True

    def test_function(self):
        assert is_enabled() is True


class TestBasicExtraction:
    def test_returns_module_api(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        assert isinstance(result, ModuleAPI)

    def test_module_docstring(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        assert result.docstring is not None
        assert "simple" in result.docstring

    def test_extracts_functions(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        func_names = [f.name for f in result.functions]
        assert "greet" in func_names

    def test_extracts_async_function(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        funcs = {f.name: f for f in result.functions}
        assert "fetch" in funcs
        assert funcs["fetch"].is_async is True

    def test_extracts_classes(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        class_names = [c.name for c in result.classes]
        assert "Point" in class_names

    def test_extracts_constants(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        const_names = [c.name for c in result.constants]
        assert "DEFAULT_NAME" in const_names

    def test_invalid_syntax_no_crash(self):
        extractor = APIExtractor()
        result = extractor.extract("def broken(:\n    pass\n")
        assert isinstance(result, ModuleAPI)


class TestFunctionInfo:
    def test_function_params(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        funcs = {f.name: f for f in result.functions}
        greet = funcs["greet"]
        param_names = [p.name for p in greet.params]
        assert "name" in param_names
        assert "times" in param_names

    def test_function_return_annotation(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        funcs = {f.name: f for f in result.functions}
        assert funcs["greet"].return_annotation == "str"

    def test_function_default_value(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        funcs = {f.name: f for f in result.functions}
        times_param = next(p for p in funcs["greet"].params if p.name == "times")
        assert times_param.default == "1"

    def test_param_stub_with_annotation(self):
        p = ParamInfo(name="x", annotation="int", default=None)
        assert p.to_stub() == "x: int"

    def test_param_stub_with_default(self):
        p = ParamInfo(name="x", annotation="int", default="0")
        assert p.to_stub() == "x: int = 0"


class TestClassInfo:
    def test_class_methods(self):
        extractor = APIExtractor(include_dunder=True)
        result = extractor.extract(SIMPLE_MODULE)
        classes = {c.name: c for c in result.classes}
        point = classes["Point"]
        method_names = [m.name for m in point.methods]
        assert "distance" in method_names
        assert "__init__" in method_names

    def test_class_property_detected(self):
        extractor = APIExtractor(include_dunder=True)
        result = extractor.extract(SIMPLE_MODULE)
        classes = {c.name: c for c in result.classes}
        methods = {m.name: m for m in classes["Point"].methods}
        assert methods["magnitude"].is_property is True

    def test_class_classmethod_detected(self):
        extractor = APIExtractor(include_dunder=True)
        result = extractor.extract(SIMPLE_MODULE)
        classes = {c.name: c for c in result.classes}
        methods = {m.name: m for m in classes["Point"].methods}
        assert methods["origin"].is_classmethod is True

    def test_private_excluded_by_default(self):
        extractor = APIExtractor(include_private=False, include_dunder=True)
        result = extractor.extract(SIMPLE_MODULE)
        classes = {c.name: c for c in result.classes}
        public = classes["Point"].public_methods()
        names = [m.name for m in public]
        assert "_private_helper" not in names

    def test_class_stub_output(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        classes = {c.name: c for c in result.classes}
        stub = classes["Point"].to_stub()
        assert "class Point" in stub


class TestStubGeneration:
    def test_to_stub_contains_function(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        stub = result.to_stub()
        assert "def greet" in stub

    def test_to_markdown_contains_headings(self):
        extractor = APIExtractor()
        result = extractor.extract(SIMPLE_MODULE)
        md = result.to_markdown()
        assert "## Classes" in md or "## Functions" in md
