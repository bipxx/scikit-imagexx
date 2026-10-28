"""Exact default image-comparison storage and words; no numerical allowance."""
import numpy as np


def default_error(image, actual, expected):
    assert type(actual) is type(expected), 'comparison public types differ'
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape and a.dtype == e.dtype and a.dtype.metadata == e.dtype.metadata
    assert (a.strides, a.flags.owndata, a.flags.writeable, type(a.base)) == (
        e.strides, e.flags.owndata, e.flags.writeable, type(e.base))
    assert a.tobytes() == e.tobytes(), 'comparison words differ'
    return 0.0
