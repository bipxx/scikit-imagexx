"""Typed rolling-histogram rank filters; validation follows pinned0.26 order."""
from collections.abc import Sequence
from numbers import Integral
from types import BuiltinFunctionType
import warnings
import numpy as np
from . import _core
from ._compat import fallback

_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}


def _stateful_fp():
    return any(value not in ('warn','raise','ignore') for value in np.geterr().values())


def _ubyte(image):
    image=np.asarray(image)
    dtype=image.dtype;ndim=image.ndim
    if image.dtype.type is np.uint8:return image
    if image.dtype.type not in _REAL:raise ValueError(f'Cannot convert from {image.dtype} to uint8.')
    if dtype.kind=='f' and _stateful_fp():
        # Owned NumPy dtype provider for exceptional, stateful FP policies.
        # Each materialized stage finishes before its callback runs. Callbacks
        # may mutate the caller's array, policy or handler between stages.
        if np.min(image)<-1.0 or np.max(image)>1.0:
            raise ValueError('Images of type float must be between -1 and 1.')
        work=np.multiply(image,255,dtype=dtype.type)
        np.rint(work,out=work)
        np.clip(work,0,255,out=work)
        return work.astype(np.uint8)
    result,downcast,maximum,nan,over,under,invalid,multiply_invalid=_core.rank_convert(image)
    if downcast:
        warnings.warn(f'Downcasting {image.dtype} to uint8 without scaling because max value {maximum} fits in uint8',UserWarning,stacklevel=3)
        # Pinned _scale chooses the branch/kind before warning, then looks up
        # astype on its same np.asarray result after warning hooks return.
        # Re-running fused conversion would incorrectly reconsider the range.
        result=image.astype(np.dtype('u1' if dtype.kind=='u' else 'i2'))
        if dtype.kind=='i':
            clipped=np.empty(result.shape,np.uint8)
            np.maximum(result,0,out=clipped,dtype=result.dtype,casting='unsafe')
            result=clipped
        return result
    if multiply_invalid or over:
        # NumPy reports combined multiply status in one operation, overflow
        # before invalid. Preserve warn/raise ordering and the shared mask.
        work_dtype=dtype.type;probe=np.zeros(int(bool(over))+int(bool(multiply_invalid)),work_dtype)
        if over:probe[0]=np.finfo(work_dtype).max
        if multiply_invalid:
            unsigned=np.uint16 if work_dtype is np.float16 else np.uint32 if work_dtype is np.float32 else np.uint64
            probe.view(unsigned)[-1]=0x7c01 if work_dtype is np.float16 else 0x7f800001 if work_dtype is np.float32 else 0x7ff0000000000001
        np.multiply(probe,255,dtype=work_dtype)
    if ndim==0 and dtype.kind=='f':raise TypeError('return arrays must be of ArrayType')
    if under:np.asarray([1e-300]).astype(np.float32)
    if nan:np.asarray([np.nan],dtype).astype(np.uint8)
    elif invalid:np.asarray([0x7f800001],np.uint32).view(np.float32).astype(np.uint8)
    return result


def _is_sequence(footprint):
    if hasattr(footprint,'__array_interface__'):return False
    if isinstance(footprint,Sequence):
        if not all(isinstance(item,Sequence) and len(item)==2 and hasattr(item[0],'__array_interface__') and isinstance(item[1],Integral) for item in footprint):
            raise ValueError('All elements of footprint sequence must be a 2-tuple where the first element of the tuple is an ndarray and the second is an integer indicating the number of iterations.')
    else:raise ValueError('footprint must be either an ndarray or Sequence')
    return True


def _positive(footprint):
    if type(footprint) not in (np.ndarray,np.memmap):return _ubyte(footprint>0)
    if footprint.dtype.type not in _REAL and footprint.dtype.kind!='c':
        if footprint.dtype.hasobject:
            # Reference-bearing comparisons require Python dtype hooks and
            # cannot be implemented by interpreting arbitrary raw pointers.
            return _ubyte(np.greater(footprint,0))
        np.empty(0,footprint.dtype)>0
    if footprint.dtype.kind in 'fc' and _stateful_fp():
        # This comparison owns its boolean output before invoking a callback;
        # interpreting caller memory again after that callback would be wrong.
        return _ubyte(np.greater(footprint,0))
    result,invalid=_core.rank_positive(footprint)
    if invalid:
        if footprint.dtype.kind=='c':
            dtype=np.complex64 if footprint.dtype.itemsize==8 else np.complex128
            probe=np.zeros(1,dtype)
            if dtype is np.complex64:probe.view(np.uint32)[0]=0x7f800001
            else:probe.view(np.uint64)[0]=0x7ff0000000000001
            np.greater(probe,0)
        else:
            dtype=np.float32 if footprint.dtype.itemsize<=4 else np.float64
            bits=np.array([0x7f800001],np.uint32) if dtype is np.float32 else np.array([0x7ff0000000000001],np.uint64)
            np.greater(bits.view(dtype),0)
    return result


def _qualified_alias_backend(operation):
    return (5<=operation<=8 and getattr(_core,'rank_alias_interface',None)=='rank-alias-v1'
            and bool(getattr(_core,'rank_alias_capabilities',0)&(1<<operation)))


def _r2_alias_admission(operation,original,image,mask,out,ndim):
    if operation<5:return False,None
    # Preserve native validation for unsafe extents, even when a short view
    # overlaps an input. Never send bounds-unsafe arrays to upstream Cython.
    if out.ndim!=ndim+1 or out.shape[-1]<1 or any(out.shape[i]<image.shape[i] for i in range(ndim)):
        return False,None
    if mask is not None and mask.size<image.size:return False,None
    aliases=np.shares_memory(image,out) or (mask is not None and np.shares_memory(mask,out))
    if not aliases:return False,None
    if _qualified_alias_backend(operation):
        # Native validation occurs after the original fused-buffer/shift checks
        # and before any output store. Preparation is never repeated.
        return False,None
    source,footprint,destination,source_mask,sx,sy,sz=original
    def plain_u8(value):
        return type(value) is np.ndarray and value.dtype==np.uint8 and value.flags.c_contiguous and value.flags.writeable
    shifts=(sx,sy) if ndim==2 else (sx,sy,sz)
    proven=(operation in (5,6,7) and plain_u8(source) and plain_u8(destination)
        and source.shape==destination.shape and source.ndim==ndim
        and type(footprint) is np.ndarray and footprint.dtype==np.uint8
        and footprint.ndim==ndim and footprint.size>0
        and (source_mask is None or plain_u8(source_mask) and source_mask.shape==source.shape)
        and all(type(value) is int and -128<=value<=127 for value in shifts))
    if proven:
        centers=tuple(size//2+value for size,value in zip(footprint.shape,(sy,sx) if ndim==2 else shifts))
        proven=all(0<=center<size for center,size in zip(centers,footprint.shape))
    if proven:
        name={5:'gradient',6:'median',7:'modal'}[operation]
        return True,fallback('filters.rank',name,source,footprint,out=destination,mask=source_mask,
            shift_x=sx,shift_y=sy,shift_z=sz)
    raise NotImplementedError('Overlapping buffers are outside native rank-histogram-v2 coverage; compatibility is pending for this overlap scope.')


def _apply(operation,image,footprint,out,mask,shift_x,shift_y,shift_z,out_dtype=None):
    original=(image,footprint,out,mask,shift_x,shift_y,shift_z)
    ndim=np.asanyarray(image).ndim
    if ndim not in (2,3):raise ValueError(f'`image` must have 2 or 3 dimensions, got {ndim}.')
    if np.asanyarray(image).size==0:raise ValueError('The parameter `image` cannot be an empty array')
    dtype=image.dtype
    if ndim==2 and dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=3)
        image=_ubyte(image)
    if ndim==2 and _is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(_positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(_ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(1,),dtype=image.dtype if out_dtype is None else out_dtype)
    elif ndim==3 or len(out.shape)==2:out=out.reshape(out.shape+(1,))
    bins=256 if image.dtype in (np.uint8,np.int8) else _core.rank_bins(image)
    if bins>2**10:
        warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=3)
    shifts=[shift_x,shift_y] if ndim==2 else [shift_x,shift_y,shift_z]
    for name,value in zip(('shift_x','shift_y','shift_z'),shifts):
        if np.dtype(type(value))==bool:
            spelling='Paramter' if ndim==2 else 'Parameter'
            warnings.warn(f'{spelling} `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=3)
    # Match the fused dispatcher before converting its signed-char arguments.
    if out.ndim!=ndim+1 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):
        raise TypeError('Function call with ambiguous argument types')
    if 13<=operation<=15:
        _statistics_call(_core.rank_statistics_run,image,footprint,mask,out,shifts,bins,operation)
    else:
        handled,result=_r2_alias_admission(operation,original,image,mask,out,ndim)
        if handled:return result
        if _qualified_alias_backend(operation):
            try:_core.rank_run(image,footprint,mask,out,shifts,bins,operation)
            except ValueError as error:
                # These named native errors certify unsupported domains PREWRITE.
                # Other validation/numerical errors retain their original class.
                excluded=('integral entropy feedback is outside defined coverage',
                          'uint16 image/float output feedback is outside defined coverage',
                          'uint16 mixed-byte feedback requires full histogram bounds',
                          'rank footprint/output alias is outside defined coverage')
                if str(error) in excluded:
                    raise NotImplementedError('Overlapping buffers are outside qualified rank-alias-v1 coverage: '+str(error)) from None
                raise
        else:_core.rank_run(image,footprint,mask,out,shifts,bins,operation)
    return np.squeeze(out,axis=-1) if ndim==2 else out.reshape(out.shape[:3])


def minimum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(0,image,footprint,out,mask,shift_x,shift_y,shift_z)
def maximum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(1,image,footprint,out,mask,shift_x,shift_y,shift_z)
def mean(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(2,image,footprint,out,mask,shift_x,shift_y,shift_z)
def sum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(3,image,footprint,out,mask,shift_x,shift_y,shift_z)
def pop(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(4,image,footprint,out,mask,shift_x,shift_y,shift_z)


def _require_operation(operation):
    if not (getattr(_core,'rank_histogram_capabilities',0) & (1<<operation)):
        version=3 if operation>=9 else 2
        raise NotImplementedError(f'rank-histogram-v{version} backend required for this operation')


def gradient(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(5)
    return _apply(5,image,footprint,out,mask,shift_x,shift_y,shift_z)


def median(image,footprint=None,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(6)
    if footprint is None:footprint=np.ones((3,)*image.ndim,bool)
    return _apply(6,image,footprint,out,mask,shift_x,shift_y,shift_z)


def modal(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(7)
    return _apply(7,image,footprint,out,mask,shift_x,shift_y,shift_z)


def entropy(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(8)
    return _apply(8,image,footprint,out,mask,shift_x,shift_y,shift_z,out_dtype=np.float64)


def autolevel(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(9)
    return _apply(9,image,footprint,out,mask,shift_x,shift_y,shift_z)


def equalize(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(10)
    return _apply(10,image,footprint,out,mask,shift_x,shift_y,shift_z)


def enhance_contrast(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(11)
    return _apply(11,image,footprint,out,mask,shift_x,shift_y,shift_z)


def threshold(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_operation(12)
    return _apply(12,image,footprint,out,mask,shift_x,shift_y,shift_z)


def _require_statistics(operation):
    vector=operation==16
    interface='rank_vector_interface' if vector else 'rank_statistics_interface'
    expected='rank-vector-v1' if vector else 'rank-statistics-v1'
    capabilities='rank_vector_capabilities' if vector else 'rank_statistics_capabilities'
    entry='rank_windowed_histogram_run' if vector else 'rank_statistics_run'
    if (getattr(_core,interface,None)!=expected or not (getattr(_core,capabilities,0)&(1<<operation))
            or not callable(getattr(_core,entry,None))
            or operation==14 and not callable(getattr(_core,'rank_noise_footprint',None))):
        raise NotImplementedError(expected+' backend required for this operation')


def _statistics_call(function,*args):
    try:return function(*args)
    except ValueError as error:
        if str(error) in ('overlapping buffers are outside rank-statistics-v1 parity',
                         'overlapping buffers are outside rank-histogram-v3 parity',
                         'rank statistics output stores overlap',
                         'subtract_mean unsigned conversion outside defined coverage',
                         'invalid rank vector depth or dimensions',
                         'rank output pixel depth must be positive'):
            raise NotImplementedError('Outside qualified rank-statistics/vector coverage: '+str(error)) from None
        raise


def geometric_mean(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_statistics(13)
    return _apply(13,image,footprint,out,mask,shift_x,shift_y,shift_z)


def _noise_footprint(footprint,shifts):
    # Only inert built-in shift arithmetic can safely share one shape read.
    # Protocol hooks retain original repeated shape/float-division/copy order.
    if (type(footprint) is np.ndarray and all(type(s) in (int,bool,float) for s in shifts)
            and all(n<=2**53 for n in footprint.shape)):
        return _core.rank_noise_footprint(footprint,list(shifts))
    row=int(footprint.shape[0]/2)+shifts[1]
    col=int(footprint.shape[1]/2)+shifts[0]
    if len(shifts)==3:
        plane=int(footprint.shape[2]/2)+shifts[2]
        indices=(row,col,plane)
    else:indices=(row,col)
    result=footprint.copy();result[indices]=0
    return result


def noise_filter(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_statistics(14)
    ndim=np.asanyarray(image).ndim
    if _is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    if ndim not in (2,3):raise ValueError(f'`image` must have 2 or 3 dimensions, got {ndim}.')
    footprint=_noise_footprint(footprint,(shift_x,shift_y) if ndim==2 else (shift_x,shift_y,shift_z))
    return _apply(14,image,footprint,out,mask,shift_x,shift_y,shift_z)


def subtract_mean(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):
    _require_statistics(15)
    return _apply(15,image,footprint,out,mask,shift_x,shift_y,shift_z)


def windowed_histogram(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,n_bins=None):
    _require_statistics(16)
    if n_bins is None:n_bins=int(image.max())+1
    value=np.asanyarray(image)
    if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
    if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')
    dtype=image.dtype
    if dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=2)
        image=_ubyte(image)
    if _is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(_positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(_ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(n_bins,),np.float64)
    elif len(out.shape)==2:out=out.reshape(out.shape+(n_bins,))
    bins=256 if image.dtype in (np.uint8,np.int8) else _core.rank_bins(image)
    if bins>1024:warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=2)
    for name,shift in zip(('shift_x','shift_y'),(shift_x,shift_y)):
        if np.dtype(type(shift))==bool:warnings.warn(f'Paramter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=2)
    if out.ndim!=3 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):raise TypeError('Function call with ambiguous argument types')
    _statistics_call(_core.rank_windowed_histogram_run,image,footprint,mask,out,[shift_x,shift_y],bins)
    return out


def _require_percentile(operation):
    entry=getattr(_core,'rank_percentile_run',None)
    capabilities=getattr(_core,'rank_percentile_capabilities',None)
    if (getattr(_core,'rank_percentile_interface',None)!='rank-percentile-v1'
            or not isinstance(capabilities,Integral) or capabilities!=0x3fe0000
            or not 17<=operation<=25
            or not isinstance(entry,BuiltinFunctionType)
            or getattr(entry,'__module__',None)!=getattr(_core,'__name__',None)
            or getattr(entry,'__name__',None)!='rank_percentile_run'):
        raise NotImplementedError('rank-percentile-v1 compiled backend required for this operation')
    return entry


def _percentile_call(function,*args):
    try:return function(*args)
    except (ValueError,OverflowError) as error:
        # These certify unsupported domains PREWRITE. They are not invented
        # upstream public ValueError contracts, and never forward unsafe or
        # overlapping input to unchecked original Cython.
        if error.__traceback__.tb_next is None and str(error) in ('overlapping buffers are outside rank-statistics-v1 parity',
                         'overlapping buffers are outside rank-histogram-v3 parity',
                         'rank statistics output stores overlap',
                         'rank percentile parameters outside defined coverage',
                         'subtract_mean_percentile unsigned conversion outside defined coverage',
                         'rank output pixel depth must be positive',
                         'rank output is smaller than image',
                         'rank mask storage is smaller than image',
                         'rank image exceeded histogram bounds',
                         'rank percentile population or sum outside defined coverage'):
            raise NotImplementedError('Outside qualified rank-percentile-v1 coverage: '+str(error)) from None
        raise


def _percentile_apply(operation,image,footprint,out,mask,shift_x,shift_y,p0,p1):
    entry=_require_percentile(operation)
    # Pinned wrapper/common preprocessing each check_nD without replacing the
    # caller's image. Lists therefore retain the original later dtype error.
    for _ in range(2):
        value=np.asanyarray(image)
        if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
        if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')
    dtype=image.dtype
    if dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=3)
        image=_ubyte(image)
    if _is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(_positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(_ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(1,),image.dtype)
    elif len(out.shape)==2:out=out.reshape(out.shape+(1,))
    bins=256 if image.dtype in (np.uint8,np.int8) else _core.rank_bins(image)
    if bins>1024:warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=3)
    for name,shift in zip(('shift_x','shift_y'),(shift_x,shift_y)):
        if np.dtype(type(shift))==bool:warnings.warn(f'Paramter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=3)
    if out.ndim!=3 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):raise TypeError('Function call with ambiguous argument types')
    _percentile_call(entry,image,footprint,mask,out,[shift_x,shift_y],bins,operation,p0,p1)
    return out.reshape(out.shape[:2])


def autolevel_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(17,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def enhance_contrast_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(18,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def gradient_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(19,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def mean_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(20,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def pop_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(21,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def subtract_mean_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(22,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def threshold_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0):
    return _percentile_apply(23,image,footprint,out,mask,shift_x,shift_y,p0,0.)


def sum_percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0,p1=1):
    return _percentile_apply(24,image,footprint,out,mask,shift_x,shift_y,p0,p1)


def percentile(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,p0=0):
    return _percentile_apply(25,image,footprint,out,mask,shift_x,shift_y,p0,0.)


_GENERIC_FUNCTIONS=('minimum','maximum','mean','sum','pop','gradient','median','modal','entropy',
                    'autolevel','equalize','enhance_contrast','threshold','geometric_mean','noise_filter','subtract_mean','windowed_histogram')
_PERCENTILE_FUNCTIONS=('autolevel_percentile','enhance_contrast_percentile','gradient_percentile',
                       'mean_percentile','pop_percentile','subtract_mean_percentile','threshold_percentile','sum_percentile','percentile')
_PERCENTILE_SCOPE_DOC="""Apply the qualified native 2D percentile filter.

Supports disjoint prepared buffers and finite percentile values in [0,1],
with p0<=p1 for two-percentile filters. Unsupported native scopes raise
NotImplementedError; overlapping or unsafe inputs are never sent upstream.
"""
for _percentile_name in _PERCENTILE_FUNCTIONS:
    globals()[_percentile_name].__doc__=_PERCENTILE_SCOPE_DOC
del _percentile_name
__all__=list(_GENERIC_FUNCTIONS+_PERCENTILE_FUNCTIONS)

def _require_bilateral(operation):
    entry=getattr(_core,'rank_bilateral_run',None)
    capabilities=getattr(_core,'rank_bilateral_capabilities',None)
    if (getattr(_core,'rank_bilateral_interface',None)!='rank-bilateral-v1'
            or type(capabilities) is not int or capabilities!=0x1c000000
            or not 26<=operation<=28
            or not isinstance(entry,BuiltinFunctionType)
            or getattr(entry,'__module__',None)!=getattr(_core,'__name__',None)
            or getattr(entry,'__name__',None)!='rank_bilateral_run'):
        raise NotImplementedError('rank-bilateral-v1 compiled backend required for this operation')
    return entry


def _bilateral_call(function,*args):
    try:return function(*args)
    except (ValueError,OverflowError) as error:
        # Native certificates for unqualified/unsafe domains do not invent
        # original errors or send those buffers to unchecked upstream code.
        # Python callback payloads with matching text retain their own type.
        if error.__traceback__.tb_next is None and str(error) in (
                'rank output pixel depth must be positive',
                'rank output is smaller than image',
                'rank mask storage is smaller than image',
                'rank bilateral footprint/output alias is outside defined coverage',
                'rank bilateral radius arithmetic outside defined coverage',
                'rank bilateral mixed-type image feedback is outside defined coverage',
                'rank bilateral mixed-byte image feedback is outside defined coverage',
                'rank bilateral image feedback requires full histogram bounds',
                'rank bilateral mixed-type mask feedback is outside defined coverage',
                'rank bilateral signed histogram outside defined coverage',
                'rank image exceeded histogram bounds'):
            raise NotImplementedError('Outside qualified rank-bilateral-v1 coverage: '+str(error)) from None
        raise


def _bilateral_apply(operation,image,footprint,out,mask,shift_x,shift_y,s0,s1):
    entry=_require_bilateral(operation)
    # Pinned wrapper/common preprocessing each check_nD without replacing the
    # caller's image. Lists therefore retain the original later dtype error.
    for _ in range(2):
        value=np.asanyarray(image)
        if value.size==0:raise ValueError('The parameter `image` cannot be an empty array')
        if value.ndim!=2:raise ValueError('The parameter `image` must be a 2-dimensional array')
    dtype=image.dtype
    if dtype==bool:raise ValueError('dtype cannot be bool.')
    if dtype not in (np.uint8,np.uint16):
        warnings.warn(f'Possible precision loss converting image of type {dtype} to uint8 as required by rank filters. Convert manually using skimage.util.img_as_ubyte to silence this warning.',UserWarning,stacklevel=3)
        image=_bilateral_ubyte(image)
    if _is_sequence(footprint):raise ValueError('footprint sequences are not currently supported by rank filters')
    footprint=np.ascontiguousarray(_positive(footprint))
    if footprint.ndim!=image.ndim:raise ValueError('Image dimensions and neighborhood dimensionsdo not match')
    image=np.ascontiguousarray(image)
    if mask is not None:mask=np.ascontiguousarray(_bilateral_ubyte(mask))
    if image is out:raise NotImplementedError('Cannot perform rank operation in place.')
    if out is None:out=np.empty(image.shape+(1,),image.dtype)
    elif len(out.shape)==2:out=out.reshape(out.shape+(1,))
    bins=256 if image.dtype in (np.uint8,np.int8) else _core.rank_bins(image)
    if bins>1024:warnings.warn(f'Bad rank filter performance is expected due to a large number of bins ({bins}), equivalent to an approximate bitdepth of {np.log2(bins):.1f}.',UserWarning,stacklevel=3)
    for name,shift in zip(('shift_x','shift_y'),(shift_x,shift_y)):
        if np.dtype(type(shift))==bool:warnings.warn(f'Paramter `{name}` is boolean and will be interpreted as int. This is not officially supported, use int instead.',UserWarning,stacklevel=3)
    if out.ndim!=3 or not (out.dtype.kind in 'ub' and out.dtype.itemsize in (1,2) or out.dtype.kind=='f' and out.dtype.itemsize in (4,8)):raise TypeError('Function call with ambiguous argument types')
    _bilateral_call(entry,image,footprint,mask,out,[shift_x,shift_y],bins,operation,s0,s1)
    return out.reshape(out.shape[:2])



def mean_bilateral(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,s0=10,s1=10):
    return _bilateral_apply(26,image,footprint,out,mask,shift_x,shift_y,s0,s1)


def pop_bilateral(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,s0=10,s1=10):
    return _bilateral_apply(27,image,footprint,out,mask,shift_x,shift_y,s0,s1)


def sum_bilateral(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,s0=10,s1=10):
    return _bilateral_apply(28,image,footprint,out,mask,shift_x,shift_y,s0,s1)


_BILATERAL_FUNCTIONS=('mean_bilateral','pop_bilateral','sum_bilateral')
_BILATERAL_SCOPE_DOC="""Apply a qualified native 2D bilateral rank filter.

Bounded source-qualified uint8/uint16 neighborhoods, conversions and feedback
forms are supported. Unqualified or unsafe native scopes raise
NotImplementedError. Qualified masked centers and feedback obey explicit native safety certificates.
"""
for _bilateral_name in _BILATERAL_FUNCTIONS:
    globals()[_bilateral_name].__doc__=_BILATERAL_SCOPE_DOC
del _bilateral_name
__all__+=list(_BILATERAL_FUNCTIONS)


def _bilateral_ubyte(image):
    image=np.asarray(image)
    dtype=image.dtype;ndim=image.ndim
    if image.dtype.type is np.uint8:return image
    if image.dtype.type not in _REAL:raise ValueError(f'Cannot convert from {image.dtype} to uint8.')
    if dtype.kind=='f' and _stateful_fp():
        # Owned NumPy dtype provider for exceptional, stateful FP policies.
        # Each materialized stage finishes before its callback runs. Callbacks
        # may mutate the caller's array, policy or handler between stages.
        if np.min(image)<-1.0 or np.max(image)>1.0:
            raise ValueError('Images of type float must be between -1 and 1.')
        work=np.multiply(image,255,dtype=dtype.type)
        np.rint(work,out=work)
        np.clip(work,0,255,out=work)
        return work.astype(np.uint8)
    if dtype.kind in 'ui' and dtype.itemsize>1 and not dtype.isnative:
        # Pinned _scale keeps the nonnative dtype as a ufunc argument. The
        # low-range warning chooses its cast before hooks run; high-range
        # allocation hooks run before the fresh dtype lookup for floor_divide.
        kind=dtype.kind;bits=8*dtype.itemsize-(kind=='i');target_bits=8
        if image.max()<2**target_bits:
            rounded=int(np.ceil(target_bits/2)*2)
            target=f'int{rounded}' if rounded>target_bits else f'uint{rounded}'
            bits=int(np.ceil(bits/2)*2)
            warnings.warn(f'Downcasting {image.dtype} to {target} without scaling because max value {image.max()} fits in {target}',UserWarning,stacklevel=3)
            result=image.astype(np.dtype('u1' if kind=='u' else 'i2'))
        else:
            result=np.empty(image.shape,np.dtype('u1' if kind=='u' else 'i2'))
            np.floor_divide(image,2**(bits-target_bits),out=result,dtype=image.dtype,casting='unsafe')
        if kind=='i':
            clipped=np.empty(result.shape,np.uint8)
            np.maximum(result,0,out=clipped,dtype=result.dtype,casting='unsafe')
            result=clipped
        return result
    result,downcast,maximum,nan,over,under,invalid,multiply_invalid=_core.rank_convert(image)
    if downcast:
        warnings.warn(f'Downcasting {image.dtype} to uint8 without scaling because max value {maximum} fits in uint8',UserWarning,stacklevel=3)
        # Pinned _scale chooses the branch/kind before warning, then looks up
        # astype on its same np.asarray result after warning hooks return.
        # Re-running fused conversion would incorrectly reconsider the range.
        result=image.astype(np.dtype('u1' if dtype.kind=='u' else 'i2'))
        if dtype.kind=='i':
            clipped=np.empty(result.shape,np.uint8)
            np.maximum(result,0,out=clipped,dtype=result.dtype,casting='unsafe')
            result=clipped
        return result
    if multiply_invalid or over:
        # NumPy reports combined multiply status in one operation, overflow
        # before invalid. Preserve warn/raise ordering and the shared mask.
        work_dtype=dtype.type;probe=np.zeros(int(bool(over))+int(bool(multiply_invalid)),work_dtype)
        if over:probe[0]=np.finfo(work_dtype).max
        if multiply_invalid:
            unsigned=np.uint16 if work_dtype is np.float16 else np.uint32 if work_dtype is np.float32 else np.uint64
            probe.view(unsigned)[-1]=0x7c01 if work_dtype is np.float16 else 0x7f800001 if work_dtype is np.float32 else 0x7ff0000000000001
        np.multiply(probe,255,dtype=work_dtype)
    if ndim==0 and dtype.kind=='f':raise TypeError('return arrays must be of ArrayType')
    if under:np.asarray([1e-300]).astype(np.float32)
    if nan:np.asarray([np.nan],dtype).astype(np.uint8)
    elif invalid:np.asarray([0x7f800001],np.uint32).view(np.float32).astype(np.uint8)
    return result

