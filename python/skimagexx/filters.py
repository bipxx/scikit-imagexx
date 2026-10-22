"""scikit-image signatures with native dispatch for supported calls."""
import numpy as np
from math import isfinite as _isfinite
from functools import lru_cache as _lru_cache
from . import _core
from ._compat import fallback, module_attribute, native_image, retain_dtype_metadata

__all__ = ["gaussian", "sobel", "sobel_h", "sobel_v", "prewitt", "prewitt_h", "prewitt_v",
           "scharr", "scharr_h", "scharr_v", "threshold_otsu", "threshold_mean"]


def gaussian(image, sigma=1.0, *, mode='nearest', cval=0, preserve_range=False,
             truncate=4.0, channel_axis=None, out=None):
    return _gaussian_nd(image,sigma,mode,cval,preserve_range,truncate,channel_axis,out)


def _edge(name, image, mask, axis, mode, cval):
    if (native_image(image) and mask is None and isinstance(mode, str) and mode == 'reflect'
            and (axis is None or isinstance(axis, (int, np.integer)) and 0 <= axis < 2)):
        return retain_dtype_metadata(_core.gradient(image, name, -1 if axis is None else int(axis) % 2),image)
    return fallback('filters', name, image, mask, axis=axis, mode=mode, cval=cval)


def sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('sobel', image, mask, axis, mode, cval)


def prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('prewitt', image, mask, axis, mode, cval)


def scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0):
    return _edge('scharr', image, mask, axis, mode, cval)


def _direction(name, suffix, image, mask):
    if native_image(image) and mask is None:
        return retain_dtype_metadata(_core.gradient(image, name, 0 if suffix == 'h' else 1),image)
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
_LI_BUFFERED_REDUCE = np.lib.NumpyVersion(np.__version__)<'2.3.0'

def threshold_li(image, *, tolerance=None, initial_guess=None, iter_callback=None):
    original_image=image
    def delegated():
        return fallback('filters','threshold_li',original_image,tolerance=tolerance,
                        initial_guess=initial_guess,iter_callback=iter_callback)
    if type(image) in (np.bool_,np.uint8,np.float32,np.float64):image=np.asarray(image)
    if type(image) is not np.ndarray or image.dtype.type not in (np.bool_,np.uint8,np.float32,np.float64):return delegated()
    prepared=_core.li_prepare(image)
    kind=prepared.kind
    scalar=image.dtype.type
    if kind==1:return np.nan
    if kind==2:return scalar(prepared.value)
    if kind==3:return 0.0
    if kind==4:np.subtract(np.bool_(False),np.bool_(True))
    if kind==5:return delegated()
    policy=np.geterr()
    if policy['under']!='ignore' or any(value not in ('ignore','warn','raise') for value in policy.values()):return delegated()
    number_types=(int,float,bool,np.bool_,np.float16,np.float32,np.float64,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64)
    if tolerance is not None and type(tolerance) not in number_types:return delegated()
    if not tolerance:
        if image.dtype.kind in 'iu':resolved=.5
        else:
            spacing=prepared.spacing()
            if np.isnan(spacing):np.empty(0,dtype=scalar).min()
            resolved=scalar(spacing)/2
    else:resolved=tolerance
    if resolved<0:return delegated()
    has_guess=initial_guess is not None
    guess=0.
    if has_guess:
        if type(initial_guess) not in number_types:return delegated()
        low=scalar(prepared.minimum);high=scalar(prepared.maximum)
        shifted=initial_guess-float(low)
        if not 0<shifted<high:
            image_max=high+low
            raise ValueError(f'The initial guess for threshold_li must be within the range of the image. Got {initial_guess} for image min {low} and max {image_max}.')
        guess=scalar(shifted)
    if iter_callback is not None:return delegated()
    block=np.getbufsize() if _LI_BUFFERED_REDUCE else 0
    value,updated,status=prepared.run(float(resolved),has_guess,float(guess),reduction_block=block)
    if status&32:return delegated()
    # Scalar metadata reproduces NumPy warning/error policy and evaluation order.
    unused_current=-2*resolved
    if status&1:np.empty(0,dtype=scalar).mean()
    if image.dtype.kind=='u':return scalar(value) if has_guess and not updated else np.float64(value)
    return scalar(value)

__all__.append('threshold_li')


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
        # This sum only guards the native float32 arithmetic. Float64 counts
        # may overflow this private validation before upstream's actual cast;
        # keep that event from leaking an extra warning, callback or exception.
        # Narrow floats/integers cannot overflow float64 for any ndarray size.
        if counts.dtype.kind=='f' and counts.dtype.itemsize==8:
            with np.errstate(over='ignore'):total=np.sum(counts,dtype=np.float64)
        else:total=np.sum(counts,dtype=np.float64)
        if total>np.finfo(np.float32).max:return None
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
    try:bins=operator.index(nbins)
    except TypeError:return None
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
    if name!='triangle' and image is None and hist is None:raise Exception('Either image or hist must be provided.')
    log_mode=_yen_log_mode() if name=='yen' else True
    prepared=_threshold_histogram(image,hist,nbins) if log_mode is not None else None
    if prepared is None:
        options={} if name=='triangle' else {'hist':hist}
        if name=='isodata':options['return_all']=return_all
        return fallback('filters','threshold_'+name,image,nbins,**options)
    counts,centers,constant=prepared
    if name=='triangle' and (constant is not None or centers.size==1):return image.flat[0]
    if name=='triangle' and not image.size:raise IndexError('index 0 is out of bounds for axis 0 with size 0')
    if name!='triangle':counts=counts.astype(np.float32,copy=False)
    if name=='isodata' and centers.size==1:return centers if return_all else centers[0]
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


def threshold_minimum(image=None, nbins=256, max_num_iter=10000, *, hist=None):
    import operator
    if image is None and hist is None:raise Exception('Either image or hist must be provided.')
    # range() invokes custom __index__ only after upstream constructs its
    # histogram. Delegating before invoking it preserves mutation/error order
    # and ensures an effectful conversion is never evaluated twice.
    if type(max_num_iter) not in (int,bool,np.int8,np.uint8,np.int16,np.uint16,
                                  np.int32,np.uint32,np.int64,np.uint64):
        return fallback('filters','threshold_minimum',image,nbins,max_num_iter,hist=hist)
    # Keep unusual iteration objects and empty-loop errors in the explicit
    # compatibility path rather than narrowing Python's range contract.
    try:iterations=operator.index(max_num_iter)
    except TypeError:return fallback('filters','threshold_minimum',image,nbins,max_num_iter,hist=hist)
    if not 0<iterations<=np.iinfo(np.intp).max:
        return fallback('filters','threshold_minimum',image,nbins,max_num_iter,hist=hist)
    prepared=_threshold_histogram(image,hist,nbins)
    if prepared is None:return fallback('filters','threshold_minimum',image,nbins,max_num_iter,hist=hist)
    counts,centers,_=prepared
    counts=counts.astype(np.float32,copy=False)
    return centers[_core.threshold_minimum(counts,iterations)]


__all__ += ['threshold_minimum']


_LOCAL_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float32,np.float64}


def _local_threshold(name,image,window_size,k,r=None):
    import math
    from collections.abc import Iterable
    original_r=r
    def forwarded():
        options={'window_size':window_size,'k':k}
        if name=='sauvola':options['r']=original_r
        return fallback('filters','threshold_'+name,image,**options)
    if type(image) is not np.ndarray or not image.ndim or image.dtype.type not in _LOCAL_TYPES or not image.dtype.isnative:
        return forwarded()
    if isinstance(window_size,Iterable):
        if not isinstance(window_size,(tuple,list,np.ndarray)):return forwarded()
        window=tuple(window_size)
    else:window=(window_size,)*image.ndim
    if any(w%2==0 for w in window):
        raise ValueError(f'Window size for `threshold_sauvola` or `threshold_niblack` must not be even on any dimension. Got {window}')
    # Shape-only metadata reproduces np.pad's invalid-width/broadcast checks.
    pad=np.asarray([(w//2+1,w//2) for w in window])
    if pad.dtype.kind!='i':raise TypeError('`pad_width` must be of integral type.')
    np.broadcast_to(pad,(image.ndim,2))
    if (pad<0).any():raise ValueError("index can't contain negative values")
    if any(size==0 for size in image.shape):raise ValueError("can't extend empty axis using modes other than 'constant' or 'empty'")
    if len(window)!=image.ndim:raise RuntimeError('Unexpected initial index in kernel_indices')
    if image.ndim>=20:return forwarded()
    cpp_window=tuple(int(w) for w in window)
    padded_count=math.prod(size+w for size,w in zip(image.shape,cpp_window))
    if padded_count>np.iinfo(np.intp).max//16:return forwarded()
    with np.errstate(over='ignore'):
        area=math.prod(window)
    # Small NumPy integer window scalars can wrap math.prod upstream. Preserve
    # that exceptional path through explicit fallback until its divisor and
    # warning stages are native; ordinary strong-scalar precision stays native.
    if int(area)!=math.prod(cpp_window):return forwarded()
    if not np.isscalar(k) or not np.isrealobj(k):return forwarded()
    if name=='sauvola' and r is None:
        from .util import dtype_limits
        low,high=dtype_limits(image,clip_negative=False)
        r=.5*(high-low)
    if name=='sauvola' and (not np.isscalar(r) or not np.isrealobj(r)):return forwarded()
    try:
        k_value=float(k);r_value=float(r) if name=='sauvola' else 1.
    except (ValueError,TypeError,OverflowError):return forwarded()
    if not math.isfinite(k_value) or not math.isfinite(r_value) or r_value==0:return forwarded()
    statistics_dtype=np.dtype('float32') if image.dtype==np.float32 else np.dtype('float64')
    probe=np.empty(0,dtype=statistics_dtype)
    with np.errstate(all='ignore'):
        divide_double=(probe/area).dtype==np.float64
        if name=='niblack':
            product=k*probe;r_double=False
        else:
            ratio=probe/r;r_double=ratio.dtype==np.float64
            product=k*(ratio-1)
        k_double=product.dtype==np.float64
    if product.dtype not in (np.dtype('float32'),np.dtype('float64')):return forwarded()
    # Explicit finite working-range contract until exceptional arithmetic
    # warning modes are implemented for every prefix/formula stage.
    if image.dtype.kind=='f':
        low,high=_core.intensity_bounds(image)
        if not (math.isfinite(low) and math.isfinite(high)):return forwarded()
        amplitude=max(abs(low),abs(high))
    elif image.dtype.kind=='b':amplitude=1.
    else:
        info=np.iinfo(image.dtype);amplitude=float(max(abs(int(info.min)),abs(int(info.max))))
    safe=min(math.sqrt(np.finfo(statistics_dtype).max/float(area)),math.sqrt(np.finfo(np.float64).max/padded_count))
    if amplitude>safe:return forwarded()
    stage_max=np.finfo(np.float64 if k_double else statistics_dtype).max
    stage_tiny=np.finfo(np.float64 if k_double else statistics_dtype).tiny
    if k_value and not stage_tiny<=abs(k_value)<=stage_max:return forwarded()
    if name=='sauvola':
        ratio_max=np.finfo(np.float64 if r_double else statistics_dtype).max
        ratio_tiny=np.finfo(np.float64 if r_double else statistics_dtype).tiny
        if not ratio_tiny<=abs(r_value)<=ratio_max:return forwarded()
        ratio_bound=2*amplitude/abs(r_value)+1
        if not math.isfinite(ratio_bound) or ratio_bound>ratio_max:return forwarded()
        product_bound=abs(k_value)*ratio_bound
        final_bound=amplitude*(1+product_bound)
    else:
        product_bound=abs(k_value)*2*amplitude
        final_bound=amplitude+product_bound
    if not math.isfinite(final_bound) or product_bound>stage_max or final_bound>stage_max:return forwarded()
    result=_core.local_threshold(image,cpp_window,name,k_value,r_value,divide_double,r_double,k_double)
    return retain_dtype_metadata(result,image) if image.dtype.type==np.float64 else result


def threshold_niblack(image,window_size=15,k=0.2):
    return _local_threshold('niblack',image,window_size,k)


def threshold_sauvola(image,window_size=15,k=0.2,r=None):
    return _local_threshold('sauvola',image,window_size,k,r)


__all__ += ['threshold_niblack','threshold_sauvola']


@_lru_cache(maxsize=1)
def _adaptive_rank_profile():
    try:from scipy import __version__ as scipy_version
    except ImportError:return 2
    version=np.lib.NumpyVersion(scipy_version)
    return 0 if version<'1.15.0' else 1 if version<'1.16.1' else 2


_ADAPTIVE_FLOAT_MAX={np.dtype(t):float(np.finfo(t).max) for t in (np.float32,np.float64)}


def threshold_local(image,block_size=3,method='gaussian',offset=0,mode='reflect',param=None,cval=0):
    import math
    original_image=image
    def delegated():return fallback('filters','threshold_local',original_image,block_size,method,offset,mode,param,cval)
    if type(image) in _GAUSSIAN_TYPES:image=np.asarray(image)
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:return delegated()
    if not (np.isscalar(block_size) or type(block_size) in (tuple,list,np.ndarray)):return delegated()
    if np.isscalar(block_size):block=(block_size,)*image.ndim
    else:
        if len(block_size)!=image.ndim:raise ValueError('len(block_size) must equal image.ndim.')
        block=tuple(block_size)
    if any(b%2==0 for b in block):raise ValueError(f'block_size must be odd! Given block_size {block} contains even values.')
    if type(method) not in (str,np.str_):return delegated()
    if method=='generic':return delegated()
    if method not in ('gaussian','mean','median'):
        raise ValueError('Invalid method specified. Please use `generic`, `gaussian`, `mean`, or `median`.')
    if image.ndim>16:return delegated()
    float_dtype=np.dtype(np.float32 if image.dtype.kind=='f' and image.dtype.itemsize<=4 else np.float64)
    if method=='gaussian':
        sigmas=tuple((b-1)/6. for b in block) if param is None else param
        if np.any(np.asarray(sigmas)<0.):raise ValueError('Sigma values less than zero are not valid')
        sigmas=_gaussian_sequence(sigmas,image.ndim)
        active=[axis for axis,sigma in enumerate(sigmas) if sigma>1e-15]
        # SciPy normalizes modes for the full input rank before axis selection.
        modes=_gaussian_sequence(mode,image.ndim)
        window=()
    elif method=='mean':
        modes=_gaussian_sequence(mode,image.ndim)
        window=tuple(int(b) if b>1 else 1 for b in block)
        active=[axis for axis,w in enumerate(window) if w>1]
    else:
        if math.prod(block)>4096:return delegated()
        # Small footprint metadata also reproduces NumPy's shape/index errors.
        footprint=np.empty(block,dtype=bool)
        window=footprint.shape
        active=[axis for axis,w in enumerate(window) if w>1]
        modes=_gaussian_sequence(mode,image.ndim)
        if not isinstance(mode,str) and np.iterable(mode) and active:
            raise RuntimeError('A sequence of modes is not supported by non-separable rank filters')
    codes=[]
    for axis,m in enumerate(modes):
        if axis in active and (not isinstance(m,str) or m not in _GAUSSIAN_MODES):
            raise RuntimeError('boundary mode not supported')
        codes.append(_GAUSSIAN_MODES.get(m,0) if isinstance(m,str) else 0)
    if active:
        if type(cval) not in (int,float,bool,np.bool_,np.float16,np.float32,np.float64,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64):
            if type(cval) in (str,list,tuple,dict):math.isfinite(cval)
            return delegated()
        constant=float(cval)
        if not math.isfinite(constant):return delegated()
    else:constant=0.
    if type(offset) not in (int,float,bool,np.bool_,np.float16,np.float32,np.float64,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64):return delegated()
    if np.geterr()['under']!='ignore' or any(p not in ('warn','ignore','raise') for p in np.geterr().values()):return delegated()
    try:value=float(offset)
    except (ValueError,OverflowError):return delegated()
    if not math.isfinite(value):return delegated()
    probe=np.zeros((),dtype=float_dtype) if image.ndim==0 else np.empty(0,dtype=float_dtype)
    target=np.result_type(probe,offset)
    if target not in _ADAPTIVE_FLOAT_MAX:return delegated()
    maximum=_ADAPTIVE_FLOAT_MAX[target];working_maximum=_ADAPTIVE_FLOAT_MAX[float_dtype]
    if abs(value)>maximum:return delegated()
    native=image if image.dtype.isnative else _core.edge_native(image)
    if native.dtype==np.float16:native=_core.edge_half(native)
    if native.dtype.type not in (np.float32,np.float64):native=_core.gaussian_nd(native,[],[],[],0.,True,float_dtype)[0]
    if not native.flags.c_contiguous or not native.flags.aligned:native=native.copy(order='C')
    amplitude=0.
    if native.size:
        low,high=_core.intensity_bounds(native)
        if not (math.isfinite(low) and math.isfinite(high)):return delegated()
        amplitude=max(abs(low),abs(high))
    if max(amplitude,abs(constant))+abs(value)>maximum:return delegated()
    if max(amplitude,abs(constant))>working_maximum/4:return delegated()
    if method=='gaussian':
        kernels=[];axes=[];borders=[]
        for axis in active:
            sigma=sigmas[axis];radius=int(4.*float(sigma)+.5)
            if radius>100000:return delegated()
            cache=type(sigma) in (int,float) and 1e-15<float(sigma)<1e154
            kernels.append((_gaussian_cached_coefficients if cache else _gaussian_coefficients)(sigma,radius))
            axes.append(axis);borders.append(codes[axis])
        if max(amplitude,abs(constant))>_ADAPTIVE_FLOAT_MAX[np.dtype(np.float64)]/4:return delegated()
        result,flags=_core.gaussian_nd(native,kernels,axes,borders,constant,True,float_dtype)
        _gaussian_copy_flags(flags,native.dtype.type,float_dtype)
    else:
        if any(w>100000 for w in window):return delegated()
        if method=='mean' and max(amplitude,abs(constant))>_ADAPTIVE_FLOAT_MAX[np.dtype(np.float64)]/(4*max((1,)+window)):return delegated()
        if method=='median' and abs(constant)>working_maximum:return delegated()
        zero_strides=image.dtype in (np.dtype(np.float32),np.dtype(np.float64)) and image.dtype.isnative and all(s==0 for s in image.strides)
        if method=='median' and image.ndim==1:
            profile=_adaptive_rank_profile()
            if image.size and profile==1 and window[0]//2>image.size:return delegated()
            if profile>0 and (image.size>=window[0]//2 or profile==2 and image.size==1):zero_strides=False
        result=_core.adaptive_filter(native,window,codes,0 if method=='mean' else 1,constant,zero_strides)
    if value!=0 or np.signbit(value) or result.dtype!=target:result=_core.adaptive_offset(result,value,target)
    return result[()] if result.ndim==0 else result


__all__.append('threshold_local')


def rank_order(image):
    original=image
    if type(image) in _GAUSSIAN_TYPES:image=np.asarray(image)
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:
        return fallback('filters','rank_order',original)
    if not image.size:raise IndexError('index -1 is out of bounds for axis 0 with size 0')
    native=image if image.dtype.isnative else _core.edge_native(image)
    if native.dtype==np.float16:native=_core.edge_half(native)
    if not native.flags.c_contiguous or not native.flags.aligned:native=native.copy(order='C')
    result=_core.rank_order(native,image)
    if result is None:return fallback('filters','rank_order',original)
    if type(image) is np.memmap:return (result[0].view(np.memmap),result[1])
    return result


__all__.append('rank_order')


def _hysteresis_dtype(dtype):
    return dtype.kind=='b' or dtype.kind in 'iu' and dtype.itemsize in (1,2,4,8) or dtype.kind=='f' and dtype.itemsize in (2,4,8)


_HYSTERESIS_SCALARS={t for t in np.ScalarType if _hysteresis_dtype(np.dtype(t))}
_HYSTERESIS_EXACT_MIXED=np.greater.resolve_dtypes((np.dtype(np.int64),np.dtype(np.uint64),None))[0].kind!='f'


def _hysteresis_probe(value,empty=False):
    a=np.asarray(value)
    return value if a.ndim==0 else np.zeros(0 if empty else 1,dtype=a.dtype)


def _hysteresis_storage(value,dtype,shape):
    dtype=np.dtype(dtype).newbyteorder('=')
    if np.ndim(value)==0:
        with np.errstate(all='ignore'):a=np.asarray(value).astype(dtype,copy=False)
    else:a=np.asarray(value).astype(dtype,copy=False)
    if a.dtype==np.float16:a=_core.edge_half(a)
    if not a.flags.c_contiguous or not a.flags.aligned:a=a.copy(order='C')
    return np.broadcast_to(a,shape)


def _hysteresis_compare(image,bound,shape):
    # A one-value probe checks scalar casting/errors without processing an image.
    np.greater(_hysteresis_probe(image),_hysteresis_probe(bound))
    dtype=np.result_type(image,bound);other=np.asarray(bound)
    if type(bound) is int and dtype.kind in 'iu':
        limits=np.iinfo(dtype)
        if bound<limits.min or bound>limits.max:dtype=np.result_type(image.dtype,other.dtype)
    if _HYSTERESIS_EXACT_MIXED and image.dtype.kind in 'biu' and other.dtype.kind in 'biu' and dtype.kind=='f':
        # NumPy has exact qQ/Qq comparison loops rather than float64 coercion.
        at=np.int64 if image.dtype.kind=='i' else np.uint64
        bt=np.int64 if other.dtype.kind=='i' else np.uint64
    else:at=bt=dtype
    return _core.hysteresis_greater(_hysteresis_storage(image,at,shape),_hysteresis_storage(bound,bt,shape))


def apply_hysteresis_threshold(image,low,high):
    original=image,low,high
    def delegated():return fallback('filters','apply_hysteresis_threshold',*original)
    if type(image) in _HYSTERESIS_SCALARS and isinstance(image,np.generic):image=np.asarray(image)
    if type(image) not in (np.ndarray,np.memmap) or not _hysteresis_dtype(image.dtype):return delegated()
    for value in (low,high):
        if type(value) not in (list,tuple,np.ndarray,np.memmap,*_HYSTERESIS_SCALARS):return delegated()
        if not _hysteresis_dtype(np.asarray(value).dtype):return delegated()
    if np.geterr()['under']!='ignore' or any(v not in ('warn','ignore','raise') for v in np.geterr().values()):return delegated()
    low=low if np.asarray(low).ndim==0 else np.asarray(low)
    high=high if np.asarray(high).ndim==0 else np.asarray(high)
    try:
        clip_shape=np.broadcast_shapes(np.shape(low),np.shape(high))
        shape=np.broadcast_shapes(image.shape,clip_shape)
    except ValueError:return delegated()
    if len(shape)>16:return delegated()
    empty=0 in shape
    if np.ndim(low)==0 and np.ndim(high)==0:clipped=np.clip(low,a_min=None,a_max=high)
    else:
        prototype=np.clip(_hysteresis_probe(low,empty),a_min=None,a_max=_hysteresis_probe(high,empty))
        dtype=prototype.dtype
        bound=high
        if type(high) is int and dtype.kind in 'iu' and high>np.iinfo(dtype).max:bound=dtype.type(np.iinfo(dtype).max)
        clipped=_core.hysteresis_minimum(_hysteresis_storage(low,dtype,clip_shape),_hysteresis_storage(bound,dtype,clip_shape))
        if clipped.dtype!=dtype:
            with np.errstate(all='ignore'):clipped=clipped.astype(dtype)
    if empty:raise ValueError('zero-size array to reduction operation minimum which has no identity')
    weak=_hysteresis_compare(image,clipped,shape)
    strong=_hysteresis_compare(image,high,shape)
    result=_core.hysteresis_connect(weak,strong)
    return result[()] if result.ndim==0 else result


__all__.append('apply_hysteresis_threshold')


_SMALL_EDGES = {'laplace':0,'roberts':1,'roberts_pos_diag':2,'roberts_neg_diag':3}
_SMALL_EDGE_DTYPES = tuple(np.dtype(t) for t in ('bool','uint8','int8','uint16','int16','uint32',
                                                'int32','uint64','int64','float16','float32','float64'))
_SMALL_EDGE_TYPES = {dtype.type for dtype in _SMALL_EDGE_DTYPES}
_SMALL_EDGE_MAX = {dtype:np.sqrt(np.finfo(dtype).max)/16 for dtype in (np.float32,np.float64)}


@_lru_cache(maxsize=64)
def _edge_mask_plans(ndim):
    if ndim<=3:
        coordinates=np.indices((3,)*ndim,dtype=np.int64).reshape(ndim,-1).T.copy()-1
        return [(coordinates,1,True)]
    plans=[]
    for axis in range(ndim):
        coordinates=np.zeros((3,ndim),dtype=np.int64);coordinates[:,axis]=(-1,0,1)
        plans.append((coordinates,1,True))
    return plans


def _small_edge(name,image,mask,options):
    def forwarded():return fallback('filters',name,image,mask=mask,**options)
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _SMALL_EDGE_TYPES or image.ndim<1:return forwarded()
    if name!='laplace' and (image.ndim!=2 or not image.size):return forwarded()
    if name=='laplace' and (not isinstance(options['ksize'],(int,np.integer)) or options['ksize']<3):return forwarded()
    if mask is not None and (type(mask) not in (np.ndarray,np.memmap) or mask.dtype.type not in _SMALL_EDGE_TYPES):return forwarded()
    if name=='roberts' and image.dtype.type in (np.float32,np.float64) and np.geterr()['under']!='ignore':return forwarded()
    native=image if image.dtype.isnative else _core.edge_native(image)
    if native.dtype==np.float16:prepared=_core.edge_half(native)
    elif native.dtype in (np.uint8,np.float32,np.float64):prepared=native
    else:prepared=_core.convert(native,np.dtype(np.float64))[0]
    if image.dtype.kind=='f' and prepared.size:
        if not prepared.flags.c_contiguous or not prepared.flags.aligned:prepared=prepared.copy(order='C')
        low,high=_core.intensity_bounds(prepared)
        if not (_isfinite(low) and _isfinite(high)):return forwarded()
        if max(abs(low),abs(high))>_SMALL_EDGE_MAX[prepared.dtype.type]:return forwarded()
    if name=='laplace' and (prepared.ndim!=2 or not prepared.size):result=_core.laplace_nd(prepared)
    else:result=_core.small_edge(prepared,_SMALL_EDGES[name])
    if mask is not None:
        if mask.dtype.type==np.float16 and mask.size:raise RuntimeError('data type not supported')
        if mask.ndim==0:eroded=np.asarray(bool(mask),dtype=bool)
        else:
            native_mask=mask if mask.dtype.isnative else _core.edge_native(mask)
            prepared_mask=_core.edge_half(native_mask) if native_mask.dtype==np.float16 else native_mask
            eroded=_core.morphology(prepared_mask,_edge_mask_plans(mask.ndim),6,6,0.,None)[0]
        try:broadcast=np.broadcast_to(eroded,result.shape)
        except ValueError:return forwarded()
        _core.edge_apply_mask(result,broadcast)
    return result


def laplace(image,ksize=3,mask=None):
    return _small_edge('laplace',image,mask,{'ksize':ksize})


def roberts(image,mask=None):
    return _small_edge('roberts',image,mask,{})


def roberts_pos_diag(image,mask=None):
    return _small_edge('roberts_pos_diag',image,mask,{})


def roberts_neg_diag(image,mask=None):
    return _small_edge('roberts_neg_diag',image,mask,{})


__all__ += list(_SMALL_EDGES)


_FARID_MODES = {'reflect':0,'grid-mirror':0,'constant':1,'grid-constant':1,
                'nearest':2,'mirror':3,'wrap':4,'grid-wrap':4}


def _farid(name,image,mask,axis=None,mode='reflect',cval=0.0):
    directional=name!='farid'
    def forwarded():
        if directional:return fallback('filters',name,image,mask=mask)
        return fallback('filters',name,image,mask=mask,axis=axis,mode=mode,cval=cval)
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _SMALL_EDGE_TYPES:return forwarded()
    if directional and (image.ndim!=2 or not image.size):return forwarded()
    if image.ndim==0 and image.dtype.kind=='i':raise TypeError('return arrays must be of ArrayType')
    if mask is not None and (type(mask) not in (np.ndarray,np.memmap) or mask.dtype.type not in _SMALL_EDGE_TYPES):return forwarded()
    if directional:axes=(0 if name=='farid_h' else 1,)
    elif axis is None:axes=tuple(range(image.ndim))
    elif isinstance(axis,(int,np.integer)):axes=(int(axis),)
    elif isinstance(axis,(tuple,list,np.ndarray)):
        if any(not isinstance(a,(int,np.integer)) for a in axis):return forwarded()
        axes=tuple(int(a) for a in axis)
    else:return forwarded()
    if any(a < -image.ndim or a>=image.ndim for a in axes):return forwarded()
    if axes:
        # The ordered full-kernel plan preserves upstream cancellation. Larger
        # plans and nonfinite warning stages are explicit compatibility paths.
        if image.ndim>8 or not isinstance(mode,str) or mode not in _FARID_MODES:return forwarded()
        if not isinstance(cval,(int,float,np.bool_,np.integer,np.floating)) or not _isfinite(cval):return forwarded()
        border=_FARID_MODES[mode];constant=float(cval)
    else:border=0;constant=0.0
    if len(axes)>1 and image.dtype.kind=='f' and np.geterr()['under']!='ignore':return forwarded()
    if not axes and image.dtype.type!=np.float16:
        # No convolution or image arithmetic is requested. Use NumPy's owned
        # zero allocation (which can use zeroed pages), and avoid reading/copying
        # input storage solely to produce zeros. Signed scalar errors are above.
        result=np.zeros(image.shape,dtype=image.dtype if image.dtype.isnative and image.dtype.type in (np.float32,np.float64) else np.float32 if image.dtype.type==np.float32 else np.float64)
    else:
        native=image if image.dtype.isnative else _core.edge_native(image)
        if native.dtype==np.float16:prepared=_core.edge_half(native)
        elif native.dtype in (np.uint8,np.float32,np.float64):prepared=native
        else:prepared=_core.convert(native,np.dtype(np.float64))[0]
        if axes and border==1 and abs(constant)>_SMALL_EDGE_MAX[np.float32 if prepared.dtype==np.float32 else np.float64]/max(1,len(axes))**.5:return forwarded()
        if axes and image.dtype.kind=='f' and prepared.size:
            if not prepared.flags.c_contiguous or not prepared.flags.aligned:prepared=prepared.copy(order='C')
            low,high=_core.intensity_bounds(prepared)
            if not (_isfinite(low) and _isfinite(high)):return forwarded()
            if max(abs(low),abs(high),abs(constant))>_SMALL_EDGE_MAX[prepared.dtype.type]/max(1,len(axes))**.5:return forwarded()
        result=_core.farid(prepared,axes,border,constant,directional)
    if mask is not None:
        if mask.dtype.type==np.float16 and mask.size:raise RuntimeError('data type not supported')
        if mask.ndim==0:eroded=np.asarray(bool(mask),dtype=bool)
        else:
            native_mask=mask if mask.dtype.isnative else _core.edge_native(mask)
            prepared_mask=_core.edge_half(native_mask) if native_mask.dtype==np.float16 else native_mask
            eroded=_core.morphology(prepared_mask,_edge_mask_plans(mask.ndim),6,6,0.,None)[0]
        try:broadcast=np.broadcast_to(eroded,result.shape)
        except ValueError:return forwarded()
        _core.edge_apply_mask(result,broadcast)
    return result


def farid(image,mask=None,*,axis=None,mode='reflect',cval=0.0):
    if (mask is None and type(axis) in (tuple,list) and not axis
            and type(image) in (np.ndarray,np.memmap) and image.dtype.type in (np.float32,np.float64)):
        return np.zeros(image.shape,dtype=image.dtype if image.dtype.isnative else image.dtype.type)
    return _farid('farid',image,mask,axis,mode,cval)


def farid_h(image,*,mask=None):
    return _farid('farid_h',image,mask)


def farid_v(image,*,mask=None):
    return _farid('farid_v',image,mask)


__all__ += ['farid','farid_h','farid_v']


_GAUSSIAN_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}
_GAUSSIAN_MODES={'reflect':0,'grid-mirror':0,'constant':1,'grid-constant':1,'nearest':2,'mirror':3,'wrap':4,'grid-wrap':4}


def _gaussian_sequence(value,rank):
    if not isinstance(value,str) and np.iterable(value):
        result=list(value)
        if len(result)!=rank:raise RuntimeError('sequence argument must have length equal to input rank')
        return result
    return [value]*rank


def _gaussian_coefficients(sigma,radius):
    # Match SciPy's typed scalar and small-array arithmetic. Only coefficient
    # metadata uses NumPy; every image convolution/normalization runs in C++.
    variance=sigma*sigma
    positions=np.arange(-radius,radius+1)
    weights=np.exp(-.5/variance*positions**2)
    weights=weights/weights.sum()
    result=np.ascontiguousarray(weights[::-1],dtype=np.float64)
    result.flags.writeable=False
    return result


@_lru_cache(maxsize=64,typed=True)
def _gaussian_cached_coefficients(sigma,radius):
    return _gaussian_coefficients(sigma,radius)


def _gaussian_copy_flags(flags,source,target):
    if flags&1:np.asarray([np.finfo(np.float64).max]).astype(np.float32)
    if flags&2:np.asarray([1e-300]).astype(np.float32)
    if flags&4:
        bits=np.array([0x7f800001],dtype=np.uint32) if source==np.float32 else np.array([0x7ff0000000000001],dtype=np.uint64)
        bits.view(source).astype(target)


def _gaussian_nd(image,sigma,mode,cval,preserve_range,truncate,channel_axis,out):
    def forwarded():
        return fallback('filters','gaussian',image,sigma,mode=mode,cval=cval,preserve_range=preserve_range,
                        truncate=truncate,channel_axis=channel_axis,out=out)
    if np.any(np.asarray(sigma)<0.):raise ValueError('Sigma values less than zero are not valid')
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:return forwarded()
    if channel_axis is not None:
        from collections.abc import Iterable
        if not isinstance(sigma,Iterable):sigmas=[sigma]*(image.ndim-1)
        else:sigmas=sigma
        if len(sigmas)==image.ndim-1:
            sigmas=list(sigmas);sigmas.insert(channel_axis%image.ndim,0)
    else:sigmas=sigma
    preserve=bool(preserve_range)
    if not preserve and image.ndim==0 and image.dtype.kind=='i':raise TypeError('return arrays must be of ArrayType')
    native=image if image.dtype.isnative else _core.edge_native(image)
    if native.dtype==np.float16:native=_core.edge_half(native)
    if out is not None:
        if not np.issubdtype(out.dtype,np.floating):raise ValueError(f'dtype of `out` must be float; got {out.dtype!r}.')
        if type(out) not in (np.ndarray,np.memmap) or out.dtype.type not in (np.float32,np.float64) or not out.flags.writeable:return forwarded()
        if out.shape!=image.shape:raise RuntimeError('output shape not correct')
        # Partial aliases and self-overlapping outputs retain upstream's line
        # update behavior through compatibility dispatch. Exact in-place output
        # is safe because the native result owns a separate working buffer.
        strides=sorted((abs(step),length) for step,length in zip(out.strides,out.shape) if length>1)
        extent=out.itemsize
        for step,length in strides:
            if step<extent:return forwarded()
            extent+=(length-1)*step
        if np.may_share_memory(image,out):
            same=(image.dtype==out.dtype and image.shape==out.shape and image.strides==out.strides
                  and image.__array_interface__['data'][0]==out.__array_interface__['data'][0])
            if not same:return forwarded()
        target=np.dtype(out.dtype.type)
    else:target=np.dtype(np.float32 if native.dtype==np.float32 else np.float64)
    sigmas=_gaussian_sequence(sigmas,image.ndim);modes=_gaussian_sequence(mode,image.ndim)
    axes=[];kernels=[];codes=[]
    for axis,standard_deviation in enumerate(sigmas):
        if not (isinstance(standard_deviation,(int,float,np.bool_,np.integer,np.floating)) or
                isinstance(standard_deviation,np.ndarray) and standard_deviation.ndim==0 and standard_deviation.dtype.kind in 'biuf'):return forwarded()
        if not standard_deviation>1e-15:continue
        radius=int(truncate*float(standard_deviation)+.5)
        if radius<0:raise ValueError('Radius must be a nonnegative integer.')
        if radius>100000:return forwarded()
        border=modes[axis]
        if not isinstance(border,str) or border not in _GAUSSIAN_MODES:return forwarded()
        cache=(type(standard_deviation) in (int,float) and 1e-15<float(standard_deviation)<1e154 and np.geterr()['under']=='ignore')
        weights=(_gaussian_cached_coefficients if cache else _gaussian_coefficients)(standard_deviation,radius)
        axes.append(axis);kernels.append(weights);codes.append(_GAUSSIAN_MODES[border])
    if axes:
        if not isinstance(cval,(int,float,np.bool_,np.integer,np.floating)):return forwarded()
        constant=float(cval)
    else:constant=0.
    result,flags=_core.gaussian_nd(native,kernels,axes,codes,constant,preserve,target)
    if out is not None:
        # NumPy owns output storage and handles its endian/stride copy protocol;
        # its dtype is already the precision used for all native filter passes.
        out[...]=result
        _gaussian_copy_flags(flags,native.dtype.type,target)
        return out
    _gaussian_copy_flags(flags,native.dtype.type,target)
    return result



def _gaussian_family_image(image,preserve):
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:return None
    if not preserve and image.ndim==0 and image.dtype.kind=='i':raise TypeError('return arrays must be of ArrayType')
    native=image if image.dtype.isnative else _core.edge_native(image)
    if native.dtype==np.float16:native=_core.edge_half(native)
    elif native.dtype.type not in (np.float32,np.float64):native=_core.gaussian_nd(native,[],[],[],0.,bool(preserve),np.dtype(np.float64))[0]
    if not native.flags.c_contiguous or not native.flags.aligned:native=native.copy(order='C')
    if native.size:
        low,high=_core.intensity_bounds(native)
        if not (_isfinite(low) and _isfinite(high)):return None
        amplitude=max(abs(low),abs(high))
        if amplitude>np.finfo(native.dtype).max/16:return None
    else:low=0.;amplitude=0.
    return native,low,amplitude


def difference_of_gaussians(image,low_sigma,high_sigma=None,*,mode='nearest',cval=0,channel_axis=None,truncate=4.0):
    def forwarded():return fallback('filters','difference_of_gaussians',image,low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
    prepared=_gaussian_family_image(image,False)
    if prepared is None:return forwarded()
    native=prepared[0]
    low=np.array(low_sigma,dtype='float',ndmin=1)
    high=low*1.6 if high_sigma is None else np.array(high_sigma,dtype='float',ndmin=1)
    spatial=native.ndim-(channel_axis is not None)
    if len(low)!=1 and len(low)!=spatial:raise ValueError('low_sigma must have length equal to number of spatial dimensions of input')
    if len(high)!=1 and len(high)!=spatial:raise ValueError('high_sigma must have length equal to number of spatial dimensions of input')
    low=low*np.ones(spatial);high=high*np.ones(spatial)
    if any(high<low):raise ValueError('high_sigma must be equal to or larger thanlow_sigma for all axes')
    first=gaussian(native,low,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
    second=gaussian(native,high,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
    result=_core.gaussian_difference(first,second)
    return result[()] if result.ndim==0 else result


def unsharp_mask(image,radius=1.0,amount=1.0,preserve_range=False,*,channel_axis=None):
    def forwarded():return fallback('filters','unsharp_mask',image,radius,amount,preserve_range,channel_axis=channel_axis)
    prepared=_gaussian_family_image(image,preserve_range)
    if prepared is None:return forwarded()
    native,minimum,amplitude=prepared
    channels=None;effective_axis=None;too_many=False
    if channel_axis is not None:
        channels=image.shape[channel_axis]
        if not channels:return np.empty_like(native,dtype=native.dtype.type)
        effective_axis=0 if channel_axis<0 else channel_axis
        if channel_axis<0 and channels<native.shape[0]:return forwarded()
        too_many=channel_axis<0 and channels>native.shape[0]
    if channel_axis is not None and native.ndim==1 and np.any(np.asarray(radius)<0.):raise ValueError('Sigma values less than zero are not valid')
    if not isinstance(amount,(int,float,np.bool_,np.integer,np.floating)) or not _isfinite(amount):return forwarded()
    if np.geterr()['under']!='ignore':return forwarded()
    probe=np.empty(0,dtype=native.dtype);blur_probe=np.empty(0,dtype=native.dtype.type)
    product_dtype=np.result_type(probe,amount)
    if product_dtype not in (np.dtype(np.float32),np.dtype(np.float64)):return forwarded()
    value=float(amount)
    maximum=np.finfo(product_dtype).max
    if abs(value)>maximum or amplitude and abs(value)>maximum/(4*amplitude):return forwarded()
    product=(probe-blur_probe)*amount
    result_dtype=(probe+product).dtype
    sigmas=radius
    if channel_axis is not None:
        sigmas=_gaussian_sequence(radius,native.ndim-1)
        if channels>1:
            for parameter in sigmas:
                if isinstance(parameter,np.integer):
                    if int(parameter)*int(parameter)>np.iinfo(parameter.dtype).max:return forwarded()
                elif isinstance(parameter,np.floating):
                    number=float(parameter);limits=np.finfo(parameter.dtype)
                    if not _isfinite(number) or number>1e-15 and not np.sqrt(limits.tiny)<=abs(number)<=np.sqrt(limits.max):return forwarded()
        sigmas=list(sigmas);sigmas.insert(effective_axis,0)
        target=np.dtype(native.dtype.type)
    else:target=result_dtype
    blurred=gaussian(native,sigmas,mode='reflect')
    if too_many:raise IndexError(f'index {native.shape[0]} is out of bounds for axis 0 with size {native.shape[0]}')
    if channel_axis is not None and image.ndim==1 and channels and not preserve_range:raise TypeError('return arrays must be of ArrayType')
    lower=-1. if minimum<0 else 0.
    result=_core.gaussian_unsharp(native,blurred,value,product.dtype==np.float64,not preserve_range,lower,target)
    if result.ndim==0:
        if not preserve_range:raise TypeError('return arrays must be of ArrayType')
        return result[()]
    return result


__all__ += ['difference_of_gaussians','unsharp_mask']


@_lru_cache(maxsize=1)
def _sparse_comparison_casts():
    value=np.array([0x7f800001],dtype=np.uint32).view(np.float32)[0]
    import warnings
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        with np.errstate(all='warn'):value==1
    return any(issubclass(m.category,RuntimeWarning) for m in messages)


def _sparse_float_status(flags,dtype,addition):
    if not flags:return
    maximum=np.finfo(dtype).max
    operation=np.add if addition else np.multiply
    if flags&1:operation(np.array([maximum],dtype=dtype),np.array([maximum],dtype=dtype))
    if flags&2:np.multiply(np.array([np.finfo(dtype).tiny],dtype=dtype),np.array([np.finfo(dtype).tiny],dtype=dtype))
    if flags&4:operation(np.array([np.inf],dtype=dtype),np.array([-np.inf if addition else 0],dtype=dtype))


def correlate_sparse(image,kernel,mode='reflect'):
    original=image,kernel,mode
    def delegated():return fallback('filters','correlate_sparse',*original)
    if type(image) not in (np.ndarray,np.memmap) or not _hysteresis_dtype(image.dtype):return delegated()
    if type(kernel) not in (list,tuple,np.ndarray,np.memmap):return delegated()
    if type(mode) not in (str,np.str_) or mode not in ('reflect','mirror','nearest','edge','symmetric','constant','wrap','valid'):return delegated()
    if any(p not in ('warn','raise','ignore') for p in np.geterr().values()):return delegated()
    kernel=np.asarray(kernel)
    if not _hysteresis_dtype(kernel.dtype) or not 1<=image.ndim<=16 or kernel.ndim!=image.ndim:return delegated()
    dtype=np.dtype(np.float32 if image.dtype.kind=='f' and image.dtype.itemsize<=4 else np.float64)
    native=image.astype(dtype,copy=False)
    if mode!='valid' and any(w%2==0 for w in kernel.shape):
        raise ValueError(f'Window size for `threshold_sauvola` or `threshold_niblack` must not be even on any dimension. Got {kernel.shape}')
    if any(w==0 for w in kernel.shape) or mode=='valid' and any(w>n+1 for w,n in zip(kernel.shape,image.shape)):return delegated()
    if not native.size and mode!='valid':
        # Zero-byte storage probes only NumPy padding validation, not pixels.
        np.pad(np.empty(image.shape,dtype='V0'),[(w//2,w//2) for w in kernel.shape],mode={'reflect':'symmetric','mirror':'reflect','nearest':'edge'}.get(mode,mode))
    indices=np.nonzero(kernel)
    weights=kernel[indices].astype(dtype,copy=False)
    coordinates=list(zip(*(a.tolist() for a in indices)))
    corner=(0,)*image.ndim
    if not coordinates or coordinates[0]!=corner:
        coordinates.insert(0,corner);weights=np.concatenate((np.zeros(1,dtype=dtype),weights))
    identity=False
    result_dtype=native.dtype
    if native.dtype.metadata is not None or type(image) is np.memmap and mode=='valid':
        with np.errstate(all='ignore'):identity=weights[0]==1
        if not identity:result_dtype=np.multiply(dtype.type(.5),np.empty(0,dtype=native.dtype)).dtype
    preserve_map=type(image) is np.memmap and mode=='valid' and identity
    native=np.asarray(native)
    if not native.flags.c_contiguous or not native.flags.aligned:native=native.copy(order='C')
    if not weights.flags.aligned:weights=weights.copy()
    borders={'reflect':0,'symmetric':0,'constant':1,'nearest':2,'edge':2,'mirror':3,'wrap':4,'valid':0}
    result,flags=_core.correlate_sparse(native,kernel.shape,coordinates,weights,borders[mode],mode=='valid')
    if _sparse_comparison_casts():
        for t,weight in enumerate(weights):
            # Preserve legacy scalar-cast warnings before multiply/add.
            weight==1
            _sparse_float_status(flags[2*t],dtype,False)
            if t:_sparse_float_status(flags[2*t+1],dtype,True)
    elif any(flags):
        for i,flag in enumerate(flags):_sparse_float_status(flag,dtype,i%2)
    if result.dtype.metadata!=result_dtype.metadata:result=result.view(result_dtype)
    return result.view(np.memmap) if preserve_map else result


__all__.append('correlate_sparse')


@_lru_cache(maxsize=8)
def _median_alias_failure(dtype):
    # This fixed-size protocol probe checks a dispatch rejection, not pixels
    # from the caller. SciPy versions disagree on TypeError vs SystemError.
    from scipy.ndimage import _rank_filter_1d
    a=np.zeros(3,dtype=dtype)
    try:_rank_filter_1d.rank_filter(a,1,3,np.zeros_like(a),0,dtype.type(0),0)
    except Exception as error:
        cause=error.__cause__
        return type(error),str(error),type(cause) if cause is not None else None,str(cause)
    return None


def median(image,footprint=None,out=None,mode='nearest',cval=0.0,behavior='ndimage'):
    original=image,footprint,out,mode,cval,behavior
    def delegated():return fallback('filters','median',*original)
    if type(behavior) not in (str,np.str_,int,float,bool,type(None)):return delegated()
    if behavior=='rank':return delegated()
    if isinstance(image,np.generic):image=np.asarray(image)
    if type(image) not in (np.ndarray,np.memmap) or not _hysteresis_dtype(image.dtype) or image.ndim>16:return delegated()
    if footprint is None:mask=np.ones((3,)*image.ndim,dtype=bool)
    elif type(footprint) in (np.ndarray,np.memmap,list,tuple,bool,np.bool_):
        if np.asarray(footprint).dtype.kind=='O':return delegated()
        mask=np.asarray(footprint,dtype=bool)
    else:return delegated()
    if len([n for n in mask.shape if n>0])!=image.ndim:
        raise RuntimeError(f'footprint.ndim ({mask.ndim}) must match len(axes) ({image.ndim})')
    count=int(np.count_nonzero(mask));rank=count//2
    if not count:raise RuntimeError('rank not within filter footprint size')
    if out is None:target=np.empty(image.shape,dtype=image.dtype.name)
    elif type(out) in (np.ndarray,np.memmap):
        target=np.asarray(out)
        if target.shape!=image.shape:raise RuntimeError('output shape not correct')
        if not target.flags.writeable:return delegated()
    elif isinstance(out,(type,np.dtype,str)):
        dtype=np.dtype(out)
        if isinstance(out,str) and not issubclass(dtype.type,np.number):raise RuntimeError('output must have numeric dtype')
        target=np.empty(image.shape,dtype=dtype)
    else:return delegated()
    if not _hysteresis_dtype(target.dtype):return delegated()
    # A one-element full footprint performs only the upstream output copy.
    if all(w==1 for w in mask.shape):
        np.copyto(target,image,casting='unsafe');return target
    modes={'reflect':0,'grid-mirror':0,'constant':1,'grid-constant':1,'nearest':2,'mirror':3,'wrap':4,'grid-wrap':4}
    if type(mode) not in (str,np.str_):return delegated()
    if mode not in modes:raise RuntimeError('boundary mode not supported')
    if type(cval) not in (int,float,bool,np.ndarray,*_HYSTERESIS_SCALARS):return delegated()
    profile=_adaptive_rank_profile()
    fast=image.ndim==1 and count>2 and profile>0 and (profile==1 or image.size>=(mask.size-1)//2 or image.size==1)
    window=mask.shape;integer_cval=0
    if fast:
        # Direct 1-D upstream outputs are interpreted as the input type. A
        # mismatched output dtype can overrun its storage; never emulate it.
        direct=image.dtype.kind=='f' and image.dtype.itemsize in (4,8) or image.dtype.kind=='i' and image.dtype.itemsize==8
        if direct and target.dtype.newbyteorder('=')!=image.dtype.newbyteorder('='):return delegated()
        working=np.dtype(np.float32 if image.dtype.kind=='f' and image.dtype.itemsize<=4 else np.float64 if image.dtype.kind=='f' else np.int64)
        source=np.asarray(image).astype(working,copy=False)
        scalar=working.type(cval)
        if working.kind=='i':integer_cval=int(scalar);constant=0.
        else:constant=float(scalar)
        if direct and source.dtype.num!=working.num:
            failure=_median_alias_failure(source.dtype)
            if failure:
                kind,message,cause,detail=failure
                if cause is not None:raise kind(message) from cause(detail)
                raise kind(message)
        if profile==1 and image.size<(mask.size-1)//2:
            if not direct and image.size:return delegated()
            if out is None or np.may_share_memory(image,target):target.fill(0)
            return target
        coordinates=[[k] for k in range(mask.size)]
        dtype=working
    else:
        constant=float(cval)
        if image.size and (image.dtype==np.float16 or target.dtype==np.float16):raise RuntimeError('array type not supported')
        if not image.size:return target
        source=np.asarray(image) if image.dtype.isnative else _core.edge_native(image)
        coordinates=np.argwhere(mask).tolist();dtype=target.dtype.newbyteorder('=')
    if not source.flags.c_contiguous or not source.flags.aligned:source=source.copy(order='C')
    zero_strides=not fast and all(stride==0 for stride in image.strides)
    result,unsupported=_core.median_filter(source,dtype,window,coordinates,rank,modes[mode],constant,integer_cval,count>2 and not fast,fast,zero_strides,not fast and not(count<=2 and count==mask.size))
    if unsupported:return delegated()
    np.copyto(target,result,casting='unsafe')
    return target


__all__.append('median')
