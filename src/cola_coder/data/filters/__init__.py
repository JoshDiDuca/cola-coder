"""Data filter plugins for code quality filtering.

Provides composable filter classes that each implement:
    - name() -> str
    - check(record) -> tuple[bool, str]
    - setup(config: dict) -> None  (optional)

Records are duck-typed: any object with .content (str) and .metadata (dict).
"""

from cola_coder.data.filters.content import ContentFilter
from cola_coder.data.filters.dedup import DeduplicationFilter
from cola_coder.data.filters.license_filter import LicenseFilter
from cola_coder.data.filters.pii import PIIFilter
from cola_coder.data.filters.syntax import SyntaxFilter

__all__ = [
    "ContentFilter",
    "DeduplicationFilter",
    "LicenseFilter",
    "PIIFilter",
    "SyntaxFilter",
]

# Import pipeline-registered filter plugins so their @register_filter decorators fire
try:
    from cola_coder.data.filters.quality import QualityFilterPlugin  # noqa: F401
    __all__.append("QualityFilterPlugin")
except ImportError:
    pass

try:
    from cola_coder.data.filters.length import LengthFilter  # noqa: F401
    __all__.append("LengthFilter")
except ImportError:
    pass

try:
    from cola_coder.data.filters.quality_classifier import QualityClassifierFilter  # noqa: F401
    __all__.append("QualityClassifierFilter")
except ImportError:
    pass

try:
    from cola_coder.data.filters.injection import InjectionFilter  # noqa: F401
    __all__.append("InjectionFilter")
except ImportError:
    pass

try:
    from cola_coder.data.filters.repetition import RepetitionFilter  # noqa: F401
    __all__.append("RepetitionFilter")
except ImportError:
    pass

try:
    from cola_coder.data.filters.decontamination import DecontaminationFilter  # noqa: F401
    __all__.append("DecontaminationFilter")
except ImportError:
    pass

# Try to import from registry if it exists (parallel agent may create it)
try:
    from cola_coder.data.filters.registry import FilterRegistry  # noqa: F401
    __all__.append("FilterRegistry")
except ImportError:
    pass
