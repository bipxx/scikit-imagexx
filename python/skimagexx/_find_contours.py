"""Bounded native marching squares with original contour orientation semantics."""

from . import _core
from ._compat import fallback


def find_contours(image, level=None, fully_connected='low', positive_orientation='low', *, mask=None):
    """Return the ordered paths where a two-dimensional image crosses a level."""
    contours = _core.find_contours_finite(image, level, fully_connected, positive_orientation, mask)
    if contours is None:
        return fallback('measure', 'find_contours', image, level, fully_connected,
                        positive_orientation, mask=mask)
    if positive_orientation == 'high':
        return [contour[::-1] for contour in contours]
    return contours
