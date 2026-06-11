"""Built-in data source plugins.

Import all sources here so their @register_source decorators fire
when the package is imported.
"""

from __future__ import annotations

# Import sources that exist — others are being built in parallel
_all_sources: list[str] = []

# noqa: F401 on each — the import fires the source's @register_source decorator
# (a side effect) and probes availability; the bound name itself is unused here.
try:
    from cola_coder.data.sources.huggingface import HuggingFaceSource  # noqa: F401
    _all_sources.append("HuggingFaceSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.local import LocalFileSource  # noqa: F401
    _all_sources.append("LocalFileSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.mixed import MixedSource  # noqa: F401
    _all_sources.append("MixedSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.github import GitHubSource  # noqa: F401
    _all_sources.append("GitHubSource")
except ImportError:
    pass

try:
    from cola_coder.data.sources.self_align import SelfAlignSource  # noqa: F401
    _all_sources.append("SelfAlignSource")
except ImportError:
    pass

__all__ = _all_sources
