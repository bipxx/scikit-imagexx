"""Native quickshift density, parent search, and component labeling."""

import math

import numpy as np

from . import _core
from ._compat import fallback


def _native_image(image):
    if (type(image) is not np.ndarray or image.ndim not in (2, 3)
            or image.size == 0 or max(image.shape) > 65536
            or image.dtype.char not in 'Bfd' or not image.dtype.isnative
            or image.dtype.metadata is not None
            or not (image.flags.c_contiguous and image.flags.aligned and image.flags.writeable)):
        return False
    base = image.base
    return base is None or (type(base) is np.ndarray and base.flags.owndata
                            and (base.flags.c_contiguous or base.flags.f_contiguous))


def quickshift(
    image, ratio=1.0, kernel_size=5, max_dist=10, return_tree=False, sigma=0,
    convert2lab=True, rng=42, *, channel_axis=-1,
):
    """Segment a 2D image by seeking local modes in color and spatial coordinates."""
    if (
        _native_image(image)
        and type(channel_axis) is int and -3 <= channel_axis < 3
        and type(rng) is int and 0 <= rng <= 2**31 - 1
        and return_tree is False and type(convert2lab) is bool
        and all(type(p) in (int, float) and abs(p) <= 2**53 and math.isfinite(p)
                for p in (ratio, kernel_size, max_dist, sigma))
        and 1 <= kernel_size <= 65536 and 0 <= max_dist and 0 <= sigma
    ):
        values = np.moveaxis(np.atleast_3d(image), channel_axis, -1)
        if not convert2lab or values.shape[-1] == 3:
            if values.dtype.char == 'B':
                values = values.astype(np.float64)
                values *= 1 / 255
            if convert2lab:
                from .color import rgb2lab
                values = rgb2lab(values)
            if sigma:
                if int(4.0 * sigma + 0.5) > 100_000:
                    # Preserve large kernels beyond the native Gaussian radius limit.
                    from skimage.filters import gaussian
                else:
                    from .filters import gaussian
                values = gaussian(values, sigma, mode='reflect', preserve_range=True, channel_axis=-1)
            values = np.multiply(values, ratio, order='C')
            noise = np.random.default_rng(rng).normal(scale=0.00001, size=values.shape[:2])
            noise = noise.astype(values.dtype, copy=False)
            result = _core.quickshift_prepared(values, noise, kernel_size, max_dist)
            if result is not None:
                return result
    # The upstream tree contains Cython memoryview objects, so keep that form
    # available through the compatibility route.
    return fallback('segmentation', 'quickshift', image, ratio, kernel_size, max_dist,
                    return_tree, sigma, convert2lab, rng, channel_axis=channel_axis)
