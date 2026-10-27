"""Untimed default Crofton check with the frozen exact-rational allowance."""
from fractions import Fraction
import math
import numpy as np


def default_error(image, actual, expected):
    if image is None:
        raise ValueError('Crofton verification requires the original input')
    assert type(actual) is type(expected) is np.float64
    assert np.isfinite(actual) and np.isfinite(expected)
    from scipy import ndimage as ndi
    from ._perimeter_crofton import _coefficients
    binary = np.pad((image > 0).astype(np.uint8), 1, mode='constant')
    codes = ndi.convolve(binary, np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
                         mode='constant', cval=0)
    histogram = np.bincount(codes.ravel(), minlength=16)
    scale = sum((abs(Fraction.from_float(float(c))) * int(n)
                 for c, n in zip(_coefficients(4), histogram)), Fraction(0))
    allowance = math.nextafter(float(2 * Fraction(32, 2**53 - 32) * scale), math.inf) if scale else 0.
    difference = abs(float(actual) - float(expected))
    assert difference <= allowance, 'Crofton exceeds the frozen weighted-sum allowance'
    if expected == 0:
        assert actual == 0 and np.signbit(actual) == np.signbit(expected)
    return difference
