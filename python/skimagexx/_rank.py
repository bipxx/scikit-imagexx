"""Typed rolling-histogram rank filters; validation follows pinned0.26 order."""
from collections.abc import Sequence
from numbers import Integral
import warnings
import numpy as np
from . import _core

_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}


def _stateful_fp():
    return any(value not in ('warn','raise','ignore') for value in np.geterr().values())


def _ubyte(image):
    image=np.asarray(image)
    if image.dtype.type is np.uint8:return image
    if image.dtype.type not in _REAL:raise ValueError(f'Cannot convert from {image.dtype} to uint8.')
    if image.dtype.kind=='f' and _stateful_fp():raise NotImplementedError('Stateful floating-point conversion policy is pending for rank filters.')
    result,downcast,maximum,nan,over,under,invalid,multiply_invalid=_core.rank_convert(image)
    if downcast:warnings.warn(f'Downcasting {image.dtype} to uint8 without scaling because max value {maximum} fits in uint8',UserWarning,stacklevel=3)
    if multiply_invalid or over:
        # NumPy reports combined multiply status in one operation, overflow
        # before invalid. Preserve warn/raise ordering and the shared mask.
        dtype=image.dtype.type;probe=np.zeros(int(bool(over))+int(bool(multiply_invalid)),dtype)
        if over:probe[0]=np.finfo(dtype).max
        if multiply_invalid:
            unsigned=np.uint16 if dtype is np.float16 else np.uint32 if dtype is np.float32 else np.uint64
            probe.view(unsigned)[-1]=0x7c01 if dtype is np.float16 else 0x7f800001 if dtype is np.float32 else 0x7ff0000000000001
        np.multiply(probe,255,dtype=dtype)
    if image.ndim==0 and image.dtype.kind=='f':raise TypeError('return arrays must be of ArrayType')
    if under:np.asarray([1e-300]).astype(np.float32)
    if nan:np.asarray([np.nan],image.dtype).astype(np.uint8)
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
    if footprint.dtype.kind in 'fc' and _stateful_fp():raise NotImplementedError('Stateful floating-point footprint policy is pending for rank filters.')
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


def _apply(operation,image,footprint,out,mask,shift_x,shift_y,shift_z):
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
    if out is None:out=np.empty(image.shape+(1,),dtype=image.dtype)
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
    _core.rank_run(image,footprint,mask,out,shifts,bins,operation)
    return np.squeeze(out,axis=-1) if ndim==2 else out.reshape(out.shape[:3])


def minimum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(0,image,footprint,out,mask,shift_x,shift_y,shift_z)
def maximum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(1,image,footprint,out,mask,shift_x,shift_y,shift_z)
def mean(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(2,image,footprint,out,mask,shift_x,shift_y,shift_z)
def sum(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(3,image,footprint,out,mask,shift_x,shift_y,shift_z)
def pop(image,footprint,out=None,mask=None,shift_x=0,shift_y=0,shift_z=0):return _apply(4,image,footprint,out,mask,shift_x,shift_y,shift_z)


__all__=['minimum','maximum','mean','sum','pop']
