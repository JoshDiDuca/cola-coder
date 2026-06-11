"""Plugin registry for data pipeline components.

Provides decorators to register DataSource, FilterPlugin, and Transform
classes by name, so they can be looked up from YAML config strings.

Think of it like a DI container / service registry:
  @register_source("huggingface")
  class HuggingFaceSource(DataSource): ...

  # Later, from YAML config:
  source_cls = get_source("huggingface")
  source = source_cls(dataset="bigcode/starcoderdata", ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cola_coder.data.pipeline import DataSource, FilterPlugin, Transform

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_SOURCE_REGISTRY: dict[str, type[DataSource]] = {}
_FILTER_REGISTRY: dict[str, type[FilterPlugin]] = {}
_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

def register_source(name: str):
    """Decorator to register a DataSource class.

    Usage:
        @register_source("huggingface")
        class HuggingFaceSource(DataSource): ...
    """
    def decorator(cls: type[DataSource]) -> type[DataSource]:
        _SOURCE_REGISTRY[name] = cls
        return cls
    return decorator


def register_filter(name: str):
    """Decorator to register a FilterPlugin class.

    Usage:
        @register_filter("quality")
        class QualityFilterPlugin(FilterPlugin): ...
    """
    def decorator(cls: type[FilterPlugin]) -> type[FilterPlugin]:
        _FILTER_REGISTRY[name] = cls
        return cls
    return decorator


def register_transform(name: str):
    """Decorator to register a Transform class.

    Usage:
        @register_transform("normalize_whitespace")
        class NormalizeWhitespace(Transform): ...
    """
    def decorator(cls: type[Transform]) -> type[Transform]:
        _TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def get_source(name: str) -> type[DataSource]:
    """Look up a registered DataSource class by name."""
    if name not in _SOURCE_REGISTRY:
        available = ", ".join(sorted(_SOURCE_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown data source: {name!r}. Available: {available}"
        )
    return _SOURCE_REGISTRY[name]


def get_filter(name: str) -> type[FilterPlugin]:
    """Look up a registered FilterPlugin class by name."""
    if name not in _FILTER_REGISTRY:
        available = ", ".join(sorted(_FILTER_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown filter: {name!r}. Available: {available}"
        )
    return _FILTER_REGISTRY[name]


def get_transform(name: str) -> type[Transform]:
    """Look up a registered Transform class by name."""
    if name not in _TRANSFORM_REGISTRY:
        available = ", ".join(sorted(_TRANSFORM_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown transform: {name!r}. Available: {available}"
        )
    return _TRANSFORM_REGISTRY[name]


def list_sources() -> list[str]:
    """Return names of all registered sources."""
    return sorted(_SOURCE_REGISTRY.keys())


def list_filters() -> list[str]:
    """Return names of all registered filters."""
    return sorted(_FILTER_REGISTRY.keys())


def list_transforms() -> list[str]:
    """Return names of all registered transforms."""
    return sorted(_TRANSFORM_REGISTRY.keys())
