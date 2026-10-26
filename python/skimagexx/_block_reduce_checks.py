"""Untimed default block-sum validation using the frozen per-cell policy."""
import math
import numpy as np


def error_limits(blocked, expected):
    if expected.dtype.kind != 'f':
        return np.zeros(expected.shape, dtype=np.float64)
    rank = expected.ndim
    axes = tuple(range(rank, blocked.ndim))
    m = math.prod(blocked.shape[rank:])
    component = np.float32 if expected.dtype.itemsize <= 4 else np.float64
    u = float(np.finfo(component).eps) / 2
    assert m * u < 1
    gamma = m * u / (1 - m * u)
    S = np.sum(np.abs(np.asarray(blocked, dtype=np.float64)), axis=axes)
    bound = 2 * gamma * S
    if expected.dtype.itemsize == 2:
        eps = float(np.finfo(np.float16).eps)
        bound += eps * (np.abs(np.asarray(expected, dtype=np.float64)) + gamma * S) / (1 - eps / 2)
    finite = np.isfinite(expected)
    assert np.all(np.isfinite(bound[finite])) and np.all(bound[finite] >= 0)
    return bound


def default_error(image, actual, expected):
    """Default block_size=2/cval=0 only; no candidate-dependent allowance."""
    if image is None:
        raise ValueError('Block sum verification requires the original input')
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape and a.dtype == e.dtype
    assert e.dtype.kind in 'biuf', 'Default block verification requires real output'
    if e.dtype.kind not in 'fc':
        np.testing.assert_array_equal(a, e)
        return 0.
    from skimage.util import view_as_blocks
    factors = (2,) * image.ndim
    padding = [(0, (-size) % 2) for size in image.shape]
    padded = np.pad(image, padding, mode='constant') if any(p[1] for p in padding) else image
    blocked = view_as_blocks(padded, factors)
    bound = error_limits(blocked, e)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(a), mask(e))
    finite = np.isfinite(e)
    difference = np.abs(np.asarray(a[finite], dtype=np.float64) - np.asarray(e[finite], dtype=np.float64))
    assert np.all(difference <= bound[finite]), 'Block sum exceeds the frozen per-block allowance'
    zeros = finite & (e == 0)
    np.testing.assert_array_equal(np.signbit(a[zeros]), np.signbit(e[zeros]))
    return float(difference.max(initial=0))
