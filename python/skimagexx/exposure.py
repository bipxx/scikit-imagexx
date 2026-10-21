"""Native exposure transforms and explicit upstream compatibility fallback."""
import numpy as np
import warnings
from . import _core
from ._compat import module_attribute, native_array, fallback

__all__ = ['adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
           'histogram', 'cumulative_distribution', 'equalize_hist']


def _finite_scalar(value):
    return np.isscalar(value) and np.isreal(value) and np.isfinite(value)


def adjust_gamma(image, gamma=1, gain=1):
    if native_array(image) and _finite_scalar(gamma) and _finite_scalar(gain) and gamma >= 0 and gain >= 0:
        return _core.exposure(image, 'gamma', gamma, gain)
    return fallback('exposure', 'adjust_gamma', image, gamma, gain)


def adjust_log(image, gain=1, inv=False):
    if (native_array(image) and _finite_scalar(gain) and gain >= 0
            and (image.dtype != np.uint8 or gain <= 1)):
        return _core.exposure(image, 'log', 0, gain, inv)
    return fallback('exposure', 'adjust_log', image, gain, inv)


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    if native_array(image) and _finite_scalar(cutoff) and _finite_scalar(gain) and gain >= 0:
        return _core.exposure(image, 'sigmoid', cutoff, gain, inv)
    return fallback('exposure', 'adjust_sigmoid', image, cutoff, gain, inv)


def _supported_range(value):
    if isinstance(value, str):
        return value in ('image', 'dtype')
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and all(_finite_scalar(v) for v in value))


def rescale_intensity(image, in_range='image', out_range='dtype'):
    if not native_array(image) or not _supported_range(in_range) or not _supported_range(out_range):
        return fallback('exposure', 'rescale_intensity', image, in_range, out_range)
    need_bounds = (isinstance(in_range, str) and in_range == 'image'
                   or isinstance(out_range, str) and out_range == 'image')
    bounds = _core.intensity_bounds(image) if need_bounds else None
    dtype_range = (0., 255.) if image.dtype == np.uint8 else (-1., 1.)
    def resolve(spec, clip_negative=False):
        if isinstance(spec, str):
            if spec == 'image':
                return bounds
            return (0. if clip_negative else dtype_range[0], dtype_range[1])
        return tuple(map(float, spec))
    imin, imax = resolve(in_range)
    omin, omax = resolve(out_range, imin >= 0)
    if any(np.isnan(v) for v in (imin, imax, omin, omax)):
        warnings.warn('One or more intensity levels are NaN. Rescaling will broadcast '
                      'NaN to the full image. Provide intensity levels yourself to '
                      'avoid this. E.g. with np.nanmin(image), np.nanmax(image).',
                      UserWarning, stacklevel=2)
    return _core.rescale(image, imin, imax, omin, omax, not isinstance(out_range, str))


def _histogram_color_warning(image):
    if image.ndim == 3 and image.shape[-1] < 4:
        warnings.warn('This might be a color image. The histogram will be '
                      'computed on the flattened image. You can instead '
                      'apply this function to each color channel, or set '
                      'channel_axis.', UserWarning, stacklevel=3)


def histogram(image, nbins=256, source_range='image', normalize=False, *, channel_axis=None):
    if (native_array(image) and image.dtype == np.uint8 and channel_axis is None
            and isinstance(source_range, str) and source_range in ('image', 'dtype')):
        _histogram_color_warning(image)
        return _core.histogram_u8(image, source_range == 'dtype', normalize)
    return fallback('exposure', 'histogram', image, nbins, source_range, normalize, channel_axis=channel_axis)


def cumulative_distribution(image, nbins=256):
    if native_array(image) and image.dtype == np.uint8:
        _histogram_color_warning(image)
        return _core.histogram_u8(image, False, False, True)
    return fallback('exposure', 'cumulative_distribution', image, nbins)


def equalize_hist(image, nbins=256, mask=None):
    if native_array(image) and image.dtype == np.uint8 and mask is None:
        _histogram_color_warning(image)
        return _core.equalize_hist(image)
    return fallback('exposure', 'equalize_hist', image, nbins, mask)


def __getattr__(name):
    return module_attribute('exposure', name)
