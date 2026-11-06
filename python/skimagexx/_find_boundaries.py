"""Owned exact-integer boundary decisions; allocation/metadata remain NumPy."""
import numpy as np
from . import _core

_MODES={'thick':0,'inner':1,'outer':2,'subpixel':3}

def find_boundaries(label_img, connectivity=1, mode='thick', background=0):
    """Find boundaries in the assessed integer/bool partition of skimage0.26.

    All four modes are implemented in C++. Rounded wide ordinary values,
    ordinary64 outer, float/swapped/protocol/rank0/empty-ordinary partitions
    remain explicit unsupported frontiers; no original algorithm is forwarded.
    The caller keeps input storage/metadata stable for the duration of a call.
    """
    # The source first reads .dtype. Plain missing-dtype carriers preserve that
    # established error before footprint/connectivity or output allocation.
    dtype=label_img.dtype
    if type(label_img)is not np.ndarray or dtype.kind not in 'biu' or not dtype.isnative or dtype.itemsize not in (1,2,4,8):
        raise NotImplementedError('find_boundaries requires assessed native integer/bool ndarray storage')
    if type(mode)is not str or type(connectivity)not in (int,bool,type(None)) or type(background)not in (int,bool):
        raise NotImplementedError('find_boundaries scalar/protocol partition is unassessed')
    if not 1<=label_img.ndim<=8:
        raise NotImplementedError('find_boundaries rank partition is unassessed')
    # Source generate_binary_structure compares connectivity<1, including the
    # literal None TypeError, before the subpixel helper/allocation.
    connectivity=max(1,min(label_img.ndim,connectivity))if connectivity is not None else connectivity
    if connectivity is None:connectivity<1
    selected=_MODES.get(mode,0)
    if selected==3:
        shape=tuple(2*length-1 for length in label_img.shape)
        # Also preserves the frozen negative-dimension error for zero extent.
        if any(length<0 for length in shape):return np.empty(shape,dtype=np.bool_,order='C')
        elements=1
        for length in shape:
            if length>16*1024*1024//elements:raise NotImplementedError('find_boundaries output budget exceeded')
            elements*=length
        output=np.empty(shape,dtype=np.bool_,order='C')
    else:
        if not all(label_img.shape):raise NotImplementedError('find_boundaries empty ordinary partition is unassessed')
        if label_img.size>16*1024*1024:raise NotImplementedError('find_boundaries input budget exceeded')
        output=np.empty_like(label_img,dtype=np.bool_,order='K',subok=False)
    bits=dtype.itemsize*8
    signed=dtype.kind=='i'
    minimum=-(1<<(bits-1))if signed else 0
    maximum=(1<<(bits-1))-1 if signed else (1<<bits)-1
    representable=minimum<=background<=maximum
    background_bits=int(background)%(1<<64)if representable else 0
    native_result=_core.find_boundaries_into(label_img,output,connectivity,selected,background_bits,representable,0)
    if native_result is not True:
        raise NotImplementedError('find_boundaries call is outside the exact-safe typed native frontier')
    return output
