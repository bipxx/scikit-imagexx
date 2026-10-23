"""Typed rolling-histogram rank filters; validation follows pinned0.26 order."""
from collections.abc import Sequence
from numbers import Integral
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


def _r2_alias_admission(operation,original,image,mask,out,ndim):
    if operation<5:return False,None
    # Preserve native validation for unsafe extents, even when a short view
    # overlaps an input. Never send bounds-unsafe arrays to upstream Cython.
    if out.ndim!=ndim+1 or out.shape[-1]<1 or any(out.shape[i]<image.shape[i] for i in range(ndim)):
        return False,None
    if mask is not None and mask.size<image.size:return False,None
    aliases=np.shares_memory(image,out) or (mask is not None and np.shares_memory(mask,out))
    if not aliases:return False,None
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
    handled,result=_r2_alias_admission(operation,original,image,mask,out,ndim)
    if handled:return result
    _core.rank_run(image,footprint,mask,out,shifts,bins,operation)
    return np.squeeze(out,axis=-1) if ndim==2 else out.reshape(out.shape[:3])


def minimum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(0,image,footprint,out,mask,shift_x,shift_y,shift_z)
def maximum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(1,image,footprint,out,mask,shift_x,shift_y,shift_z)
def mean(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(2,image,footprint,out,mask,shift_x,shift_y,shift_z)
def sum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(3,image,footprint,out,mask,shift_x,shift_y,shift_z)
def pop(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(4,image,footprint,out,mask,shift_x,shift_y,shift_z)


def _require_operation(operation):
    if not (getattr(_core,'rank_histogram_capabilities',0) & (1<<operation)):
        raise NotImplementedError('rank-histogram-v2 backend required for this operation')


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


__all__=['minimum','maximum','mean','sum','pop','gradient','median','modal','entropy']
