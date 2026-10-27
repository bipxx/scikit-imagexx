"""Source-ordered uint8 preparation and owned perimeter arithmetic."""
from math import sqrt
import numpy as np
from . import _core

_STREL_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
_STREL_8 = np.ones((3, 3), dtype=np.uint8)


def _weights():
    weights = np.zeros(50, dtype=np.float64)
    weights[[5, 7, 15, 17, 25, 27]] = 1
    weights[[21, 33]] = sqrt(2)
    weights[[13, 23]] = (1 + sqrt(2)) / 2
    return weights


def perimeter(image, neighborhood=4):
    """Estimate perimeter with literal uint8 casting and owned byte arithmetic.

    The cast, including its warnings and protocols, remains a NumPy provider.
    Ordinary cast arrays use owned erosion, wrapped stencil counts and weights.
    Unassessed ndarray-subclass preparation keeps local primitive operations.
    """
    if image.ndim != 2:
        raise NotImplementedError('`perimeter` supports 2D images only')
    if neighborhood == 4:
        cross = True
    else:
        cross = False
    if type(image) is np.ndarray:
        _core.fft_check(image)
    image = image.astype(np.uint8)
    workers = _core.get_num_threads()
    if type(image) is np.ndarray:
        # Fixed coefficients have no input protocols; construct them before
        # the fused erosion/stencil stage, preserving pre-histogram ordering.
        perimeter_weights = _weights()
        perimeter_histogram = _core.perimeter_counts(image, cross, workers)
        if perimeter_histogram.size != 50:
            return perimeter_histogram @ perimeter_weights
        return np.float64(_core.perimeter_weighted(perimeter_histogram, perimeter_weights, workers))
    from scipy import ndimage as ndi
    strel = _STREL_4 if cross else _STREL_8
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image
    perimeter_weights = _weights()
    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]]),
                                   mode='constant', cval=0)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    return perimeter_histogram @ perimeter_weights
