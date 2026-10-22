"""Native exposure transforms and explicit upstream compatibility fallback."""
import numpy as np
import warnings
from . import _core
from ._compat import module_attribute, native_array, fallback

__all__ = ['adjust_gamma', 'adjust_log', 'adjust_sigmoid', 'rescale_intensity',
           'histogram', 'cumulative_distribution', 'equalize_hist','is_low_contrast']


def _finite_scalar(value):
    return np.isscalar(value) and np.isreal(value) and np.isfinite(value)


_CONTRAST_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
               np.uint64,np.int64,np.float16,np.float32,np.float64}
_CONTRAST_FRACTIONS=_CONTRAST_REAL|{int,float,bool,np.longdouble}


def is_low_contrast(image,fraction_threshold=0.05,lower_percentile=1,
                    upper_percentile=99,method='linear'):
    """Compare raw percentile span with the dtype range; method is ignored."""
    image=np.asanyarray(image)
    source=image
    def upstream():
        return fallback('exposure','is_low_contrast',source,fraction_threshold,
                        lower_percentile,upper_percentile,method)
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _CONTRAST_REAL:
        return upstream()
    if image.dtype.type==np.bool_:
        if not image.size:return not ((image.max()==1) and (image.min()==0))
        return _core.contrast(image,None)
    if (any(value in ('call','log','print') for value in np.geterr().values())
            or np.geterr()['under']!='ignore'):
        return upstream()
    color_error=None
    if image.ndim==3 and image.shape[2] in (3,4):
        from . import color
        if (image.dtype.type not in (np.uint8,np.float32,np.float64)
                or not image.dtype.isnative or not image.size):
            return upstream()
        if type(fraction_threshold) not in _CONTRAST_FRACTIONS and type(fraction_threshold) not in (np.ndarray,np.memmap):
            return upstream()
        color_threshold=np.asarray(fraction_threshold)
        if color_threshold.dtype.kind not in 'biuf':return upstream()
        # Native color kernels permit ordinary rounding differences. Bound
        # them for normalized inputs and delegate ambiguous final decisions.
        if image.dtype.type!=np.uint8:
            if not image.flags.c_contiguous or not image.flags.aligned:
                image=np.array(image,copy=True,order='C')
            low,high=_core.intensity_bounds(image)
            if not (-1<=low and high<=1):return upstream()
        if image.shape[2]==4:image=color.rgba2rgb(image)
        if image.shape[2]==3:image=color.rgb2gray(image)
        color_error=64*float(np.finfo(image.dtype).eps)
    from .util import dtype_limits
    lower_limit,upper_limit=dtype_limits(image,clip_negative=False)
    # Only the two percentile parameters and their virtual ranks use NumPy.
    # In particular, preserve NumPy-version-specific scalar precision here.
    if not all(isinstance(value,(int,float,np.integer,np.floating))
               for value in (lower_percentile,upper_percentile)):
        return upstream()
    quantiles=np.true_divide([lower_percentile,upper_percentile],100)
    if not all(0<=value<=1 for value in quantiles):
        raise ValueError('Percentiles must be in the range [0, 100]')
    if quantiles.dtype.itemsize>8:return upstream()
    if not image.size:
        np.percentile(image,[lower_percentile,upper_percentile])
    # Low-precision percentile parameters may overflow their virtual ranks.
    # Delegate those exceptional paths without repeating private probe notices.
    # Finite rounding beyond the last element selects that last value upstream.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        positions=(image.size-1)*quantiles
    if not all(np.isfinite(value) for value in positions):return upstream()
    ranks=tuple(min(float(value),image.size-1) for value in positions)
    values=_core.contrast(image,ranks)
    if values is None:return upstream()
    for notice in notices:warnings.warn(notice.message,stacklevel=2)
    span=np.float64(values[1])-np.float64(values[0])
    ratio=span/(upper_limit-lower_limit)
    if color_error is not None:
        # This probe is scalar/threshold metadata, not image processing. Account
        # for old NumPy's value-based promotion with narrow threshold arrays.
        with np.errstate(all='ignore'):
            error=color_error
            if color_threshold.dtype.kind=='f':
                error=error+np.finfo(color_threshold.dtype).eps*np.maximum(1,np.abs(color_threshold))
            uncertain=np.any((np.abs(ratio-color_threshold)<=error)&np.isfinite(color_threshold))
        if uncertain:return upstream()
    return ratio<fraction_threshold


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
