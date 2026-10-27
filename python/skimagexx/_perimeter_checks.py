"""Untimed perimeter verification with the frozen exact-rational allowance."""
from fractions import Fraction
import math
import numpy as np


def default_error(image, actual, expected):
    if image is None:
        raise ValueError('Perimeter verification requires the original input')
    # Generic dtype-metadata checks project scalar returns to zero-dimensional arrays.
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape == ()
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert np.isfinite(actual) and np.isfinite(expected)
    from scipy import ndimage as ndi
    from ._perimeter import _STREL_4, _weights
    converted = image.astype(np.uint8)
    eroded = ndi.binary_erosion(converted, _STREL_4, border_value=0)
    border = converted - eroded
    codes = ndi.convolve(border, np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]]),
                         mode='constant', cval=0)
    histogram = np.bincount(codes.ravel(), minlength=50)
    assert histogram.shape == (50,) and np.all(histogram <= 2**53)
    scale = sum((abs(Fraction.from_float(float(c))) * int(n)
                 for c, n in zip(_weights(), histogram)), Fraction(0))
    allowance = math.nextafter(float(2 * Fraction(100, 2**53 - 100) * scale), math.inf) if scale else 0.
    difference = abs(float(actual) - float(expected))
    assert difference <= allowance, 'Perimeter exceeds the frozen weighted-sum allowance'
    if expected == 0:
        assert actual == 0 and np.signbit(actual) == np.signbit(expected)
    return difference
