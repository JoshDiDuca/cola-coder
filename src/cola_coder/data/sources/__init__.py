"""Built-in data source plugins.

Import all sources here so their @register_source decorators fire
when the package is imported.
"""

from __future__ import annotations

# Import sources that exist — others are being built in parallel
_all_sources: list[str] = []

try:
    from cola_coder.data.sources.huggingface import HuggingFaceSource
    _all_sources.append("HuggingFaceSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.local import LocalFileSource
    _all_sources.append("LocalFileSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.mixed import MixedSource
    _all_sources.append("MixedSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.github import GitHubSource
    _all_sources.append("GitHubSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.self_align import SelfAlignSource
    _all_sources.append("SelfAlignSource")
except ImportError:
    pass

__all__ = _all_sources
