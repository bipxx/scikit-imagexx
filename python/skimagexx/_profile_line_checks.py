"""Exact profile output policy frozen before the first candidate output."""
import numpy as np


def default_error(image, actual, expected):
    assert type(actual) is type(expected) is np.ndarray, 'profile result type differs'
    a, e = np.asarray(actual), np.asarray(expected)
    def layout(value):
        return (type(value), value.dtype.str, value.dtype.metadata,
                value.shape, value.strides, value.flags.c_contiguous,
                value.flags.f_contiguous, value.flags.owndata,
                value.flags.writeable, type(value.base))
    assert layout(a) == layout(e), 'profile result metadata differs'
    assert a.tobytes(order='C') == e.tobytes(order='C'), 'profile result words differ'
    return 0.0
