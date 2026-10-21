"""scikit-image signatures with native dispatch for supported calls."""
import numpy as np
from . import _core
from ._compat import fallback, module_attribute, native_image

__all__ = ["gaussian", "sobel", "sobel_h", "sobel_v", "prewitt", "prewitt_h", "prewitt_v",
           "scharr", "scharr_h", "scharr_v", "threshold_otsu", "threshold_mean"]


def gaussian(image, sigma=1.0, *, mode='nearest', cval=0, preserve_range=False,
             truncate=4.0, channel_axis=None, out=None):
    if (native_image(image) and np.isscalar(sigma) and np.isfinite(sigma) and sigma >= 0
            and isinstance(mode, str) and mode in ('nearest', 'reflect') and channel_axis is None
            and np.isscalar(truncate) and np.isfinite(truncate) and truncate >= 0
            and float(sigma)*float(truncate) < 100000):
        if out is None:
            return _core.gaussian(image, sigma, mode=mode, preserve_range=preserve_range, truncate=truncate)
        # Output dtype affects intermediate precision upstream. Dispatch this
        # option upstream until the native API can write that precision directly.
    return fallback('filters', 'gaussian', image, sigma, mode=mode, cval=cval,
                    preserve_range=preserve_range, truncate=truncate, channel_axis=channel_axis, out=out)


def _edge(name, image, mask, axis, mode, cval):
    if (native_image(image) and mask is None and isinstance(mode, str) and mode == 'reflect'
            and (axis is None or isinstance(axis, (int, np.integer)) and 0 <= axis < 2)):
        return _core.gradient(image, name, -1 if axis is None else int(axis) % 2)
    return fallback('filters', name, image, mask, axis=axis, mode=mode, cval=cval)


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('sobel', image, mask, axis, mode, cval)


def prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('prewitt', image, mask, axis, mode, cval)


def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('scharr', image, mask, axis, mode, cval)


def _direction(name, suffix, image, mask):
    if native_image(image) and mask is None:
        return _core.gradient(image, name, 0 if suffix == 'h' else 1)
    return fallback('filters', name+'_'+suffix, image, mask)


def sobel_h(image, mask=None):
    return _direction('sobel', 'h', image, mask)


def sobel_v(image, mask=None):
    return _direction('sobel', 'v', image, mask)


def prewitt_h(image, mask=None):
    return _direction('prewitt', 'h', image, mask)


def prewitt_v(image, mask=None):
    return _direction('prewitt', 'v', image, mask)


def scharr_h(image, mask=None):
    return _direction('scharr', 'h', image, mask)


def scharr_v(image, mask=None):
    return _direction('scharr', 'v', image, mask)


def threshold_otsu(image=None, nbins=256, *, hist=None):
    if native_image(image) and image.dtype == np.uint8 and hist is None:
        value, constant = _core.threshold_otsu_result(image)
        return np.uint8(value) if constant else np.int64(value)
    return fallback('filters', 'threshold_otsu', image, nbins, hist=hist)


def threshold_mean(image):
    if native_image(image):
        value = _core.threshold_mean(image)
        return np.float32(value) if image.dtype == np.float32 else np.float64(value)
    return fallback('filters', 'threshold_mean', image)


def __getattr__(name):
    return module_attribute('filters', name)
