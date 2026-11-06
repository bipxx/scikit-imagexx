"""Native integral geometry, explicit NumPy protocol/as_strided providers.

Ordered argument boundary follows scikit-image 0.26 shape.py (BSD-3-Clause;
third_party/scikit-image-LICENSE.txt). No original API imports/forwarding.
"""
import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from . import _core


def view_as_blocks(arr_in, block_shape):
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')
    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")
    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length " "as 'arr_in.shape'")
    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")
    if (type(arr_in) is np.ndarray and block_shape.ndim == 1
            and block_shape.dtype.kind in 'biu'
            and all(int(b) <= np.iinfo(np.intp).max and
                    -np.iinfo(np.intp).max-1 <= int(s)*int(b) <= np.iinfo(np.intp).max
                    for b,s in zip(block_shape,arr_in.strides))):
        new_shape, new_strides = _core.util_view_as_blocks(arr_in, block_shape.tolist())
    else:
        # Non-array buffers, object/float parameter arithmetic and protocols
        # retain NumPy computation/errors; never credited as native geometry.
        new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
        new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides
    return as_strided(arr_in, shape=tuple(new_shape), strides=tuple(new_strides))


def view_as_windows(arr_in, window_shape, step=1):
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")
    ndim = arr_in.ndim
    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")
    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")
    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)
    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")
    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)
    indexing_strides = arr_in[slices].strides
    if (type(arr_in) is np.ndarray and window_shape.ndim == 1
            and window_shape.dtype.kind in 'biu'
            and all(type(st) is int and 0 < st <= np.iinfo(np.intp).max
                    and -np.iinfo(np.intp).max-1 <= int(s)*st <= np.iinfo(np.intp).max
                    for st,s in zip(step,arr_in.strides))):
        new_shape, strides = _core.util_view_as_windows(arr_in, window_shape.tolist(), step)
    else:
        # Unusual step/parameter domains retain exact NumPy promotion, division
        # warnings and shape errors after the source-ordered basic slice.
        win_indices_shape = ((np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)) + 1
        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(window_strides))
    return as_strided(arr_in, shape=tuple(new_shape), strides=tuple(strides))
