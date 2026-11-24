"""Bounded native corner orientations with a NumPy-backed memoryview."""

import numpy as np

from . import _core
from ._compat import fallback


def corner_orientations(image, corners, mask):
    """Compute the orientation of corners."""
    owner = _core.corner_orientations_apply(image, corners, mask)
    if owner is None:
        return fallback('feature', 'corner_orientations', image, corners, mask)
    return np.asarray(memoryview(owner))
