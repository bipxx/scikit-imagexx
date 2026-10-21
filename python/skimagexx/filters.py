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
        if not isinstance(cval,(int,float,np.integer,np.floating)) or not _isfinite(cval):return forwarded()
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
