"""The bounded Zhang route for morphology.skeletonize."""

from . import _core
from ._compat import fallback


def skeletonize(image, *, method=None):
    """Reduce foreground components to a skeleton while preserving topology."""
    result = _core.skeletonize_zhang(image, method)
    if result is None:
        return fallback('morphology', 'skeletonize', image, method=method)
    return result
