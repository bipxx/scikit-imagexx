"""Bounded single-layer pyramids composed from the shipped native stages."""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._resize import resize
from .filters import gaussian

_DTYPES = (np.uint8, np.float32, np.float64)
_MODES = ('reflect', 'constant', 'wrap')
_MAX_AXIS = 512
_MAX_ELEMENTS = 262144
_MAX_MAGNITUDE = 65536


def _eligible(image, scale, sigma, order, mode, cval, preserve_range, channel_axis):
    if type(image) is not np.ndarray or image.dtype.type not in _DTYPES:
        return False
    if not (image.dtype.isnative and image.dtype.metadata is None and image.flags.owndata
            and image.flags.c_contiguous and image.flags.aligned):
        return False
    if not ((image.ndim == 2 and channel_axis is None)
            or (image.ndim == 3 and type(channel_axis) is int)):
        return False
    if not (image.size <= _MAX_ELEMENTS and all(1 <= length <= _MAX_AXIS for length in image.shape)):
        return False
    if type(scale) not in (int, float) or not 1 < scale <= 4:
        return False
    if sigma is not None and (type(sigma) not in (int, float) or not 0 <= sigma <= 4):
        return False
    if type(order) is not int or order not in (0, 1) or type(mode) is not str or mode not in _MODES:
        return False
    if type(cval) not in (int, float) or not -_MAX_MAGNITUDE <= cval <= _MAX_MAGNITUDE:
        return False
    if type(preserve_range) is not bool:
        return False
    # NaNs and extreme real values retain stock clipping and warning behavior.
    return image.dtype.type is np.uint8 or (
        -_MAX_MAGNITUDE <= float(image.min()) and float(image.max()) <= _MAX_MAGNITUDE
    )


def _convert(image, preserve_range):
    if image.dtype.type is np.uint8:
        return _core.gaussian_nd(image, [], [], [], 0., preserve_range, np.dtype(np.float64))[0]
    return image if preserve_range else image.copy(order='K')


def _smooth(image, sigma, mode, cval, channel_axis):
    output = np.empty_like(image)
    if channel_axis is not None:
        sigma = (sigma,) * (image.ndim - 1)
    return gaussian(image, sigma, mode=mode, cval=cval, channel_axis=channel_axis, out=output)


def _pyramid(name, image, scale, sigma, order, mode, cval, preserve_range, channel_axis):
    if _eligible(image, scale, sigma, order, mode, cval, preserve_range, channel_axis):
        axis = None if channel_axis is None else channel_axis % image.ndim
        reducing = name == 'pyramid_reduce'
        shape = tuple(length if k == axis else math.ceil(length / float(scale) if reducing else scale * length)
                      for k, length in enumerate(image.shape))
        if all(length <= _MAX_AXIS for length in shape) and math.prod(shape) <= _MAX_ELEMENTS:
            prepared = _convert(image, preserve_range)
            effective_sigma = 2 * scale / 6.0 if sigma is None else sigma
            if reducing:
                prepared = _smooth(prepared, effective_sigma, mode, cval, axis)
            resized = resize(prepared, shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
            if reducing:
                return resized
            return _smooth(resized, effective_sigma, mode, cval, axis)
    return fallback('transform', name, image, scale, sigma, order, mode, cval, preserve_range,
                    channel_axis=channel_axis)


def pyramid_reduce(image, downscale=2, sigma=None, order=1, mode='reflect', cval=0,
                   preserve_range=False, *, channel_axis=None):
    """Smooth and downsample a single pyramid layer."""
    return _pyramid('pyramid_reduce', image, downscale, sigma, order, mode, cval, preserve_range, channel_axis)


def pyramid_expand(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0,
                   preserve_range=False, *, channel_axis=None):
    """Upsample and smooth a single pyramid layer."""
    return _pyramid('pyramid_expand', image, upscale, sigma, order, mode, cval, preserve_range, channel_axis)
