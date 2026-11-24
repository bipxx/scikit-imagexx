"""Bounded native FAST corner response with a NumPy-backed memoryview."""

import numpy as np

from . import _core
from ._compat import fallback


def corner_fast(image, n=12, threshold=0.15):
    """Extract FAST corners for a given image."""
    owner = _core.corner_fast_apply(image, n, threshold)
    if owner is None:
        return fallback('feature', 'corner_fast', image, n, threshold)
    return np.asarray(memoryview(owner))
