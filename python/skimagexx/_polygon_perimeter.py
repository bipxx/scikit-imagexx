"""Rasterize an ordered polygon perimeter."""

from . import _core
from ._compat import fallback


def polygon_perimeter(r, c, shape=None, clip=False):
    if type(clip) is bool:
        if clip and shape is None:
            raise ValueError("Must specify clipping shape")
        result = _core.polygon_perimeter(r, c, shape, clip)
        if result is not None:
            return result
    return fallback("draw", "polygon_perimeter", r, c, shape=shape, clip=clip)
