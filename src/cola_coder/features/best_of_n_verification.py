"""Best-of-N verified generation — feature toggle and entry point.

Generate N candidate completions for the same prompt, verify each one with
real sandboxed tools (tsc for TypeScript, sandboxed execution for Python
tests, compile() for Python syntax), and return the best candidate.

This is inference-time compute scaling: trade a few seconds of extra GPU and
sandbox time per request for a candidate that's been proven to compile or
pass tests, instead of hoping the first sample is good.

The core implementation lives in `cola_coder.inference.best_of_n` (it's an
inference-layer capability used by scripts/generate.py and the FastAPI
server). This module is the feature-registry face: the toggle plus a lazy
passthrough so scanning feature modules never imports torch.

Usage:
    scripts/generate.py --best-of 4 --language typescript
    POST /v1/completions {"prompt": ..., "best_of": 4, "verify_language": "auto"}
"""

FEATURE_ENABLED = True


def is_enabled() -> bool:
    return FEATURE_ENABLED


def generate_best_of_n(*args, **kwargs):
    """Lazy passthrough to cola_coder.inference.best_of_n.generate_best_of_n."""
    from ..inference.best_of_n import generate_best_of_n as _impl

    return _impl(*args, **kwargs)
