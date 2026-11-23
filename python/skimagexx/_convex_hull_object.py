"""Union of planar hulls of bounded connected binary objects."""

import numpy as np

from . import _core
from ._compat import fallback


def convex_hull_object(image, *, connectivity=2):
    """Return the Boolean union of the individual objects' convex hulls."""
    if type(image) is np.ndarray:
        if image.ndim > 2:
            raise ValueError("Input must be a 2D image")
        if type(connectivity) is int:
            if connectivity not in (1, 2):
                raise ValueError("`connectivity` must be either 1 or 2.")
            if image.ndim == 2:
                result = _core.convex_hull_object_binary(image, connectivity)
                if result is not None:
                    return result
    return fallback("morphology", "convex_hull_object", image, connectivity=connectivity)
