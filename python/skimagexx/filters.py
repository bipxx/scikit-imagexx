"""scikit-image signatures with native dispatch for supported calls."""
import numpy as np
from functools import lru_cache as _lru_cache
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


_THRESHOLD_DTYPES = (np.dtype('uint8'),np.dtype('float32'),np.dtype('float64'))


def _threshold_histogram(image,hist,nbins):
    # These arrays describe histogram metadata; all image scans run in C++.
    if hist is not None:
        if isinstance(hist,(tuple,list)):
            counts,centers=hist
        elif isinstance(hist,np.ndarray):
            counts,centers=hist,np.arange(hist.size)
        else:
            return None
        if not isinstance(counts,np.ndarray) or not isinstance(centers,np.ndarray):return None
        if counts.ndim!=1 or centers.ndim!=1 or counts.size!=centers.size or not counts.size:return None
        if counts.dtype.itemsize>8:return None
        if counts.dtype.kind not in 'iuf' or centers.dtype not in (np.dtype('int64'),np.dtype('float32'),np.dtype('float64')):return None
        if not np.isfinite(counts).all() or not np.isfinite(centers).all() or (counts<0).any():return None
        if np.sum(counts,dtype=np.float64)>np.finfo(np.float32).max:return None
        if counts[0]==0 or counts[-1]==0:
            valid=counts>0;start=valid.argmax();stop=counts.size-valid[::-1].argmax()
            counts,centers=counts[start:stop],centers[start:stop]
        return counts,centers,None
    if not isinstance(image,np.ndarray) or type(image) is not np.ndarray or image.dtype not in _THRESHOLD_DTYPES:return None
    if image.dtype==np.uint8:
        if not image.size:raise ValueError('zero-size array to reduction operation minimum which has no identity')
        counts,centers=_core.histogram_u8(image)
        return counts,centers,image.flat[0] if centers.size==1 else None
    import operator
    bins=operator.index(nbins)
    if bins<1:raise ValueError('`bins` must be positive, when an integer')
    if bins>1000000:return None
    low,high=_core.intensity_bounds(image) if image.size else (0.,1.)
    scalar=image.dtype.type
    low,high=scalar(low),scalar(high)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f'autodetected range of [{low}, {high}] is not finite')
    constant=image.flat[0] if image.size and low==high else None
    if low==high:low,high=low-0.5,high+0.5
    if float(high)-float(low)>np.finfo(image.dtype).max:return None
    edges=np.linspace(low,high,bins+1,dtype=image.dtype)
    if np.any(edges[:-1]>=edges[1:]):
        raise ValueError(f'Too many bins for data range. Cannot create {bins} finite-sized bins.')
    return _core.histogram_edges(image,edges),(edges[:-1]+edges[1:])/2.0,constant


@_lru_cache(maxsize=1)
def _yen_log_mode():
    # Fixed numerical metadata probes; no image pixels are evaluated in Python.
    probe=np.array([.001,.1,.2,.5,.7,.9,1.,1.1,2.,3.,7.,10.973364,10.973365,13.,100.,1e10],dtype=np.float32)
    expected=np.log(probe)
    for mode in (True,False):
        if np.array_equal(_core.threshold_logarithms(probe,mode),expected):return mode
    return None


def _histogram_threshold(name,image,nbins,hist=None,return_all=False):
    if image is None and hist is None:raise Exception('Either image or hist must be provided.')
    log_mode=_yen_log_mode() if name=='yen' else True
    prepared=_threshold_histogram(image,hist,nbins) if log_mode is not None else None
    if prepared is None:
        options={} if name=='triangle' else {'hist':hist}
        if name=='isodata':options['return_all']=return_all
        return fallback('filters','threshold_'+name,image,nbins,**options)
    counts,centers,constant=prepared
    if name=='triangle' and constant is not None:return constant
    if name=='triangle' and not image.size:raise IndexError('index 0 is out of bounds for axis 0 with size 0')
    if name!='triangle':counts=counts.astype(np.float32,copy=False)
    width=float(centers[1]-centers[0]) if name=='isodata' and centers.size>1 else float('nan')
    indices,status=_core.threshold_indices(counts,centers,name,centers.dtype==np.float32,log_mode,width)
    # Replay only arithmetic-status metadata so NumPy errstate (including raise)
    # and the original warning order remain observable to callers.
    if status&64:np.divide(np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32))
    if status&1:np.reciprocal(np.zeros(1,dtype=np.float32))
    if status&2:np.multiply(np.array([np.inf],dtype=np.float32),np.zeros(1,dtype=np.float32))
    if status&4:np.log(np.zeros(1,dtype=np.float32))
    if status&8:np.divide(np.zeros(1),np.zeros(1))
    if status&48:
        numerator=([1.] if status&32 else [])+([0.] if status&16 else [])
        np.divide(np.array(numerator),np.zeros(len(numerator)))
    selected=centers[indices]
    return selected if name=='isodata' and return_all else selected[0]


def threshold_yen(image=None,nbins=256,*,hist=None):
    return _histogram_threshold('yen',image,nbins,hist)


def threshold_triangle(image,nbins=256):
    return _histogram_threshold('triangle',image,nbins)


def threshold_isodata(image=None,nbins=256,return_all=False,*,hist=None):
    return _histogram_threshold('isodata',image,nbins,hist,return_all)


__all__ += ['threshold_yen','threshold_triangle','threshold_isodata']
