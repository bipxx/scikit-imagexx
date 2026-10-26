"""Source-ordered block preparation and owned ordinary default sums."""
import numpy as np
from . import _core

_SUM = np.sum


def _view_as_blocks(arr_in, block_shape):
    # Literal 0.26 metadata validation/restriding; no upstream algorithm call.
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')
    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")
    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length as 'arr_in.shape'")
    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides
    return np.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=new_strides)


def block_reduce(image, block_size=2, func=np.sum, cval=0, func_kwargs=None):
    """Reduce local blocks, with NumPy padding and explicit reducer providers.

    Ordinary default real sums use owned typed block arithmetic. Custom funcs,
    reducer kwargs, complex padding and exceptional/type-frontier reductions
    retain literal primitive calls. They do not acquire owned-sum credit.
    Half sums additionally require the original padded storage to be native,
    aligned and C-contiguous, with block volume no larger than np.getbufsize().
    Half F/reversed/swapped/unaligned or larger buffered reductions remain
    literal providers; native support for those source orders is open work.
    """
    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError('`block_size` must be a scalar or have the same length as `image.shape`')
    if func_kwargs is None:
        func_kwargs = {}
    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError('Down-sampling factors must be >= 1. Use `skimage.transform.resize` to up-sample an image.')
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    if np.any(np.asarray(pad_width)):
        # Padding is the first external pixel read. Its allocated result is a
        # new boundary, admitted independently by the native binding if used.
        if type(image) is np.ndarray:
            _core.fft_check(image)
        image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=cval)
    blocked = _view_as_blocks(image, block_size)
    if func is _SUM and type(func_kwargs) is dict and not func_kwargs and not isinstance(cval, (complex, np.complexfloating)) and type(image) is np.ndarray:
        result = _core.block_reduce_sum(image, block_size)
        if result is not None:
            return result[()] if image.ndim == 0 else result
    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)), **func_kwargs)
