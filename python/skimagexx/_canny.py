"""Native Canny edge detection for bounded two-dimensional image buffers."""

from math import isfinite

import numpy as np

from . import _core
from ._compat import fallback
from .filters import _GAUSSIAN_MODES, _gaussian_cached_coefficients


_NO_KERNEL = np.empty(0, dtype=np.float64)
_NO_KERNEL.flags.writeable = False
_NATIVE_DTYPES = frozenset((
    np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
    np.float32, np.float64,
))
_get_promotion_state = getattr(np, '_get_promotion_state', None)


def _scalar(value):
    # Leave coercion and error behavior for object, NumPy and huge scalars
    # to the public compatibility provider.
    return ((type(value) is float and isfinite(value)) or
            (type(value) is int and -(1 << 63) <= value < (1 << 63)))


def _native_canny(image, sigma, low, high, mask, quantiles, mode, cval):
    if (type(image) is not np.ndarray or image.ndim != 2 or image.size == 0
            or image.dtype.type not in _NATIVE_DTYPES or image.dtype.metadata is not None
            or not image.dtype.isnative or not image.flags.c_contiguous
            or not image.flags.writeable or not image.flags.aligned
            or type(quantiles) is not bool or type(mode) is not str or mode not in _GAUSSIAN_MODES
            or not _scalar(sigma) or not 0 <= sigma < 16.125 or not _scalar(cval)
            or (low is not None and not _scalar(low)) or (high is not None and not _scalar(high))
            or np.geterr()['under'] != 'ignore'):
        return None
    if (mask is not None and (type(mask) is not np.ndarray or mask.shape != image.shape
                              or mask.dtype.type is not np.bool_ or mask.dtype.metadata is not None
                              or not mask.flags.c_contiguous or not mask.flags.writeable)):
        return None

    border = _GAUSSIAN_MODES[mode]
    if border == 1 and not 0 <= cval <= 1:
        return None
    kind = image.dtype.kind
    divisor = (1 << (8 * image.itemsize - (kind == 'i'))) - 1 if kind in 'iu' else 1
    low = 0.1 if low is None else low if quantiles else low / divisor
    high = 0.2 if high is None else high if quantiles else high / divisor
    if low > high or (quantiles and (low < 0 or high > 1)):
        return None

    weights = _gaussian_cached_coefficients(sigma, int(4 * sigma + .5)) if sigma > 1e-15 else _NO_KERNEL
    wide_quantile = quantiles and (_get_promotion_state is None or _get_promotion_state() != 'legacy')
    return _core.canny_native(image, mask, weights, border, cval, low, high, quantiles,
                              wide_quantile, mode == 'constant')


def canny(image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None,
          use_quantiles=False, *, mode='constant', cval=0.0):
    """Return the Boolean edge map of a grayscale image using the Canny algorithm.

    Gaussian smoothing is followed by Sobel gradients, bilinear edge thinning
    and 8-connected hysteresis. Thresholds are absolute image values or
    quantiles of the gradient magnitude. The optional mask limits detected
    edges and controls pixel contributions during smoothing.
    """
    result = _native_canny(image, sigma, low_threshold, high_threshold, mask, use_quantiles, mode, cval)
    if result is not None:
        return result
    return fallback('feature', 'canny', image, sigma=sigma, low_threshold=low_threshold,
                    high_threshold=high_threshold, mask=mask, use_quantiles=use_quantiles,
                    mode=mode, cval=cval)
