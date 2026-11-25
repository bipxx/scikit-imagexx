"""Native edge costs and minimum-spanning-forest image segmentation."""

import math

import numpy as np

from . import _core
from ._compat import fallback


def _native_image(image):
    if (type(image) is not np.ndarray or image.ndim not in (2, 3)
            or image.size == 0 or image.dtype.char not in 'Bfd' or not image.dtype.isnative
            or image.dtype.metadata is not None
            or not (image.flags.c_contiguous and image.flags.aligned and image.flags.writeable)):
        return False
    base = image.base
    return base is None or (type(base) is np.ndarray and base.flags.owndata
                            and (base.flags.c_contiguous or base.flags.f_contiguous))


def felzenszwalb(image, scale=1, sigma=0.8, min_size=20, *, channel_axis=-1):
    """Segment an image with a minimum spanning forest over its pixel grid."""
    if (
        _native_image(image)
        and (channel_axis is None and image.ndim == 2
             or type(channel_axis) is int and -image.ndim <= channel_axis < image.ndim)
        and type(min_size) is int and -np.iinfo(np.intp).max <= min_size <= np.iinfo(np.intp).max
        and all(type(p) in (int, float) and abs(p) <= 2**53 and math.isfinite(p)
                for p in (scale, sigma))
        and sigma >= 0
    ):
        values = (image[:, :, None] if image.ndim == 2
                  else np.moveaxis(image, channel_axis, -1))
        if values.shape[-1] <= 3:
            values = values.astype(np.float64, copy=False)
            if image.dtype.char == 'B':
                values *= 1 / 255
            if sigma:
                from scipy.ndimage import gaussian_filter
                values = gaussian_filter(values, [sigma, sigma, 0], mode='reflect')
            values = np.ascontiguousarray(values)
            result = _core.felzenszwalb_prepared(values, scale, min_size)
            if result is not None:
                return result
    return fallback('segmentation', 'felzenszwalb', image, scale, sigma, min_size,
                    channel_axis=channel_axis)
