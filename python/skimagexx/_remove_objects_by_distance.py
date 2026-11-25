"""Greedy distance filtering of integer-labeled objects."""

import numpy as np

from . import _core
from ._compat import fallback


def remove_objects_by_distance(label_image, min_distance, *, priority=None, p_norm=2, spacing=None, out=None):
    """Remove lower-priority objects until the survivors are farther apart than min_distance."""
    if (type(label_image) is np.ndarray and out is None and
            type(min_distance) in (int, float) and 0 <= min_distance <= 1048576 and
            type(p_norm) in (int, float) and p_norm in (1, 2, float('inf'))):
        rank = label_image.ndim
        if spacing is None:
            factors = (1.0,) * rank
        elif (type(spacing) in (tuple, list) and len(spacing) == rank and
              all(type(value) in (int, float) and 1 / 16 <= value <= 16 for value in spacing)):
            factors = tuple(float(value) for value in spacing)
        else:
            factors = None
        if factors is not None:
            result = _core.remove_objects_by_distance_integer(
                label_image, float(min_distance), float(p_norm), factors, priority)
            if result is not None:
                return result
    return fallback('morphology', 'remove_objects_by_distance', label_image, min_distance,
                    priority=priority, p_norm=p_norm, spacing=spacing, out=out)
