"""Owned Bregman intake and sequential channel orchestration.

The prepared finite solver is C++; its Gauss-Seidel sweep remains ordered.
This module alone does not constitute public registration or full-domain parity.
"""
import functools
import inspect

import numpy as np

from . import _core


def _require_kernel():
    entry = getattr(_core, 'tv_bregman_run', None)
    if (getattr(_core, 'tv_bregman_interface', None) != 'tv-bregman-prepared-v1'
            or type(getattr(_core, 'tv_bregman_capabilities', None)) is not int
            or _core.tv_bregman_capabilities != 3
            or getattr(_core, 'tv_bregman_parallel_sweep', None) is not False
            or not inspect.isbuiltin(entry)
            or entry.__name__ != 'tv_bregman_run'
            or entry.__module__ != _core.__name__):
        raise NotImplementedError('the prepared TV Bregman native interface is unavailable')
    return entry


def img_as_float(image):
    """Typed intake provider, matching the pinned real-valued conversion."""
    image = np.asarray(image)
    dtype = image.dtype
    if np.issubdtype(dtype.type, np.floating):
        return image
    if dtype.kind == 'b':
        return image.astype(np.float64)
    if dtype.kind not in 'iu' or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError(f'Cannot convert from {dtype} to float64.')
    maximum = np.iinfo(dtype.type).max
    image = np.multiply(image, 1.0 / maximum, dtype=np.float64)
    if dtype.kind == 'i':
        np.maximum(image, -1.0, out=image)
    return np.asarray(image, np.float64)


def _denoise_tv_bregman(image, weight, maximum, eps, isotropic, out):
    _require_kernel()(image, weight, maximum, eps, isotropic, out)


def _channel_as_last_axis(function):
    @functools.wraps(function)
    def fixed_func(*args, **kwargs):
        channel_axis = kwargs.get('channel_axis', None)
        if channel_axis is None:
            return function(*args, **kwargs)
        if np.isscalar(channel_axis):
            channel_axis = (channel_axis,)
        if len(channel_axis) > 1:
            raise ValueError('only a single channel axis is currently supported')
        if channel_axis == (-1,) or channel_axis == -1:
            return function(*args, **kwargs)
        new_args = tuple(np.moveaxis(arg, channel_axis[0], -1) if pos == 0 else arg
                         for pos, arg in enumerate(args))
        kwargs['channel_axis'] = -1
        out = function(*new_args, **kwargs)
        return np.moveaxis(out, -1, channel_axis[0])
    return fixed_func


@_channel_as_last_axis
def denoise_tv_bregman(image, weight=5.0, max_num_iter=100, eps=0.001,
                       isotropic=True, *, channel_axis=None):
    _require_kernel()
    image = np.atleast_3d(img_as_float(image))
    rows, cols, dims = image.shape[:3]
    shape_ext = (rows + 2, cols + 2, dims)
    out = np.zeros(shape_ext, image.dtype)
    if channel_axis is not None:
        channel_out = np.zeros(shape_ext[:2] + (1,), dtype=out.dtype)
        for c in range(image.shape[-1]):
            channel_in = np.ascontiguousarray(image[..., c:c + 1])
            _denoise_tv_bregman(channel_in, image.dtype.type(weight),
                                max_num_iter, eps, isotropic, channel_out)
            out[..., c] = channel_out[..., 0]
    else:
        image = np.ascontiguousarray(image)
        _denoise_tv_bregman(image, image.dtype.type(weight), max_num_iter,
                            eps, isotropic, out)
    return np.squeeze(out[1:-1, 1:-1])
