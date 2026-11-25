"""Rasterize the perimeter outside a rectangle slice."""

from . import _core
from ._compat import fallback


def rectangle_perimeter(start, end=None, extent=None, shape=None, clip=False):
    if end is None and extent is None:
        raise ValueError("Either `end` or `extent` must be given.")
    if end is not None and extent is not None:
        raise ValueError("Cannot provide both `end` and `extent`.")
    if type(clip) is bool:
        result = _core.rectangle_perimeter(start, extent if end is None else end, end is None, shape, clip)
        if result is not None:
            return result
    return fallback("draw", "rectangle_perimeter", start, end=end, extent=extent, shape=shape, clip=clip)
