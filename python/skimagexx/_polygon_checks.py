"""Exact polygon output policy, including source noncanonical bool words."""
import numpy as np


def default_error(image, actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    def layout(x):
        return (type(x), x.dtype, x.dtype.metadata, x.shape, x.strides,
                x.flags.owndata, x.flags.writeable, type(x.base))
    assert layout(a)==layout(e), 'polygon result metadata differs'
    assert a.dtype.kind in 'bu' and e.dtype.kind in 'bu'
    assert a.tobytes(order='C')==e.tobytes(order='C'), 'polygon classification bytes differ'
    return 0.0
