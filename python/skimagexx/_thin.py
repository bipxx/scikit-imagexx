"""Bounded two-subiteration morphological thinning."""

from . import _core
from ._compat import fallback


def thin(image, max_num_iter=None):
    """Thin a binary image until convergence or the inclusive iteration cap."""
    if max_num_iter is None or type(max_num_iter) is int and 0 <= max_num_iter <= 512:
        result = _core.thin_binary(image, 0 if max_num_iter is None else max_num_iter)
        if result is not None:
            return result
    return fallback("morphology", "thin", image, max_num_iter=max_num_iter)
