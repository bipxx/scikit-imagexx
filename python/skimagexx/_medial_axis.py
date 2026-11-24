"""Medial-axis topology with explicit SciPy EDT and NumPy RNG providers."""

import numpy as np

from . import _core
from ._compat import fallback


def _distance_provider():
    # The base package requires NumPy. SciPy is an optional provider here.
    try:
        from scipy import ndimage
        from scipy.ndimage import _morphology
    except ModuleNotFoundError as error:
        if error.name != 'scipy':
            raise
        return None
    provider = ndimage.distance_transform_edt
    return provider if provider is _morphology.distance_transform_edt else None


def medial_axis(image, mask=None, return_distance=False, *, rng=None):
    """Compute the medial axis, optionally with the Euclidean distance field."""
    seeded = (
        (type(rng) is int and rng >= 0)
        or type(rng) is np.random.PCG64
        or (type(rng) is np.random.Generator and type(rng.bit_generator) is np.random.PCG64)
    )
    if type(return_distance) is bool and seeded:
        prepared = _core.medial_axis_prepare(image, mask)
        if prepared is not None:
            edt = _distance_provider()
            default_rng = np.random.default_rng
            if edt is not None and default_rng is np.random._generator.default_rng:
                foreground, count = prepared
                distance = edt(foreground)
                returned_distance = distance.copy() if return_distance else None
                priority = default_rng(rng).permutation(np.arange(count))
                skeleton = _core.medial_axis_ordered(foreground, distance, priority)
                if mask is not None:
                    outside = ~mask
                    skeleton[outside] = image[outside]
                if return_distance:
                    return skeleton, returned_distance
                return skeleton
    return fallback('morphology', 'medial_axis', image, mask, return_distance, rng=rng)
