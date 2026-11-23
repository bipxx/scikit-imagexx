"""Bounded planar hull on a doubled pixel lattice."""

from . import _core
from ._compat import fallback


def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10, include_borders=True):
    """Return the Boolean mask of a planar convex hull."""
    if (
        type(offset_coordinates) is bool and type(include_borders) is bool
        and type(tolerance) is float and tolerance == 1e-10
    ):
        result = _core.convex_hull_image_binary(image, offset_coordinates, include_borders)
        if result is not None:
            return result
    return fallback(
        "morphology", "convex_hull_image", image, offset_coordinates=offset_coordinates,
        tolerance=tolerance, include_borders=include_borders,
    )
