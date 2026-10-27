"""Untimed exact Euler checker in the frozen scaled-LUT domain."""
import numpy as np


def default_error(image, actual, expected):
    if image is None:
        raise ValueError('Euler verification requires the actual original call input')
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape == ()
    assert actual.dtype == expected.dtype and expected.dtype.kind == 'i'
    from scipy import ndimage as ndi
    from ._euler_number import _configuration
    binary = np.pad((image > 0).astype(int), pad_width=1, mode='constant')
    config, coefs, bins = _configuration(binary.ndim, binary.ndim)
    codes = ndi.convolve(binary, config, mode='constant', cval=0)
    h = np.bincount(codes.ravel(), minlength=bins)
    terms = [int(c) * int(n) for c, n in zip(coefs, h)]
    assert sum(abs(t) for t in terms) <= 2**53, 'unassessed Euler weighted precision domain'
    value = sum(terms)
    if binary.ndim != 2:
        value = value // 8 if value >= 0 else -((-value) // 8)
    assert int(expected) == value and int(actual) == value
    return 0.
