"""Exact default Blur output words from the prospective original-first contract."""
import numpy as np


def default_error(image, actual, expected):
    # Generic dtype-metadata callers project scalar results to zero-dimensional
    # arrays; dedicated public tests retain the exact NumPy scalar type.
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape == (), 'blur result must be scalar'
    assert a.dtype == e.dtype and a.dtype.metadata == e.dtype.metadata
    assert a.tobytes() == e.tobytes(), 'blur result words differ'
    return 0.0
