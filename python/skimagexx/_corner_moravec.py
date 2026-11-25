"""Bounded native Moravec responses backed by NumPy-owned storage."""

from . import _core
from ._compat import fallback


def corner_moravec(image, window_size=1):
    """Compute a Moravec corner response for a two-dimensional image."""
    owner = _core.corner_moravec_u8(image, window_size)
    if owner is None:
        return fallback('feature', 'corner_moravec', image, window_size)
    return owner
