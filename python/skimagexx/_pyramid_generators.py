"""Lazy pyramids composed from the native Gaussian and resize stages."""

import math

import numpy as np

from ._compat import fallback
from ._pyramid_spatial import _convert, _eligible, _smooth, pyramid_reduce, resize


def _native_domain(image, max_layer, downscale, sigma, order, mode, cval, preserve_range, channel_axis):
    return (type(max_layer) is int and -1 <= max_layer <= 8
            and type(downscale) in (int, float) and 1.5 <= downscale <= 4
            and _eligible(image, downscale, sigma, order, mode, cval, preserve_range, channel_axis))


def _first_image(image, preserve_range):
    if image.dtype.type is np.uint8:
        return _convert(image, preserve_range)
    # The first Gaussian yield can expose the caller's float array itself.
    return image


def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect',
                     cval=0, preserve_range=False, *, channel_axis=None):
    """Yield the converted image and each smaller Gaussian level."""
    if not _native_domain(image, max_layer, downscale, sigma, order, mode, cval, preserve_range, channel_axis):
        yield from fallback('transform', 'pyramid_gaussian', image, max_layer, downscale, sigma,
                            order, mode, cval, preserve_range, channel_axis=channel_axis)
        return

    current = _first_image(image, preserve_range)
    current_shape = current.shape
    layer = 0
    yield current

    while layer != max_layer:
        layer += 1
        # The prior yield is live. The spatial wrapper rechecks it and can
        # delegate this reduction if the caller changed its storage or values.
        current = pyramid_reduce(current, downscale, sigma, order, mode, cval, channel_axis=channel_axis)
        previous_shape = current_shape
        current_shape = current.shape
        if current_shape == previous_shape:
            break
        yield current


def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect',
                      cval=0, preserve_range=False, *, channel_axis=None):
    """Yield residuals between successive resampled images and their smoothing."""
    if not _native_domain(image, max_layer, downscale, sigma, order, mode, cval, preserve_range, channel_axis):
        yield from fallback('transform', 'pyramid_laplacian', image, max_layer, downscale, sigma,
                            order, mode, cval, preserve_range, channel_axis=channel_axis)
        return

    current = _first_image(image, preserve_range)
    shape = current.shape
    axis = None if channel_axis is None else channel_axis % current.ndim
    if sigma is None:
        sigma = 2 * downscale / 6.0
    smoothed = _smooth(current, sigma, mode, cval, axis)
    yield current - smoothed

    if max_layer == -1:
        spatial_extent = max(d for i, d in enumerate(shape) if i != axis)
        max_layer = math.ceil(math.log(spatial_extent, downscale))
    for _ in range(max_layer):
        reduced = tuple(d if i == axis else math.ceil(d / float(downscale)) for i, d in enumerate(shape))
        current = resize(smoothed, reduced, order=order, mode=mode, cval=cval, anti_aliasing=False)
        smoothed = _smooth(current, sigma, mode, cval, axis)
        shape = current.shape
        yield current - smoothed
