"""Built-in transform plugins.

Import all transforms here so their @register_transform decorators fire
when the package is imported.
"""

from __future__ import annotations

from cola_coder.data.transforms.metadata import AddMetadata
from cola_coder.data.transforms.whitespace import NormalizeWhitespace

__all__ = ["NormalizeWhitespace", "AddMetadata"]
