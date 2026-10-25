"""Owned reconstruction shell with a real C++ queue flood.

The rare negative-zero representative uses an explicit NumPy argsort provider
matching the source rank_order value-map choice; it is not labeled fully native.
No upstream reconstruction/rank_order algorithm is called.
"""
import inspect
import numpy as np
from . import _core


def _admit():
    if getattr(_core,'reconstruction_interface',None)!='reconstruction-v1' or type(getattr(_core,'reconstruction_capabilities',None)) is not int or _core.reconstruction_capabilities!=3:
        raise NotImplementedError('reconstruction compiled interface unavailable')
    for name in ('reconstruction_ordered','reconstruction_negative_zero','reconstruction_run'):
        entry=getattr(_core,name,None)
        if not inspect.isbuiltin(entry) or entry.__name__!=name or entry.__module__!=_core.__name__:
            raise NotImplementedError('reconstruction requires owned compiled entries')


def _zero_provider(seed,mask,footprint,offset,method):
    # Scalar metadata chooses the ordinary output precision. Actual representative
    # selection is source-backed and observable, never read from expected output.
    dtype=np.float32 if mask.dtype.type in (np.float16,np.float32) else np.float64
    negative=_core.reconstruction_negative_zero(seed,mask)
    if not negative:return None
    shape=tuple(np.array(seed.shape)+(np.array(footprint.shape)-1));inside=tuple(slice(o,o+n) for o,n in zip(offset,seed.shape))
    pad=seed.min() if method=='dilation' else seed.max()
    images=np.full((2,)+shape,pad,dtype=dtype);images[(0,)+inside]=seed;images[(1,)+inside]=mask
    values=images.reshape(-1) if method=='dilation' else -images.reshape(-1)
    ordered=values[values.argsort()];at=np.flatnonzero(ordered==0)[0];representative=ordered[at]
    if method=='erosion':representative=-representative
    return bool(np.signbit(representative))


def reconstruction(seed,mask,method='dilation',footprint=None,offset=None):
    _admit()
    assert tuple(seed.shape)==tuple(mask.shape)
    ordinary=isinstance(seed,np.ndarray) and isinstance(mask,np.ndarray) and seed.dtype.kind in 'biuf' and mask.dtype.kind in 'biuf' and seed.size>0
    if method=='dilation' or method=='erosion':
        if ordinary:ordered=_core.reconstruction_ordered(seed,mask,0 if method=='dilation' else 1)
        else:ordered=not np.any(seed>mask if method=='dilation' else seed<mask)
        if not ordered:
            raise ValueError('Intensity of seed image must be '+('less' if method=='dilation' else 'greater')+' than that of the mask image for reconstruction by '+method+'.')
    if footprint is None:footprint=np.ones([3]*seed.ndim,dtype=bool)
    else:footprint=footprint.astype(bool,copy=True)
    if offset is None:
        if not all([d%2==1 for d in footprint.shape]):raise ValueError('Footprint dimensions must all be odd')
        offset=np.array([d//2 for d in footprint.shape])
    else:
        if offset.ndim!=footprint.ndim:raise ValueError('Offset and footprint ndims must be equal.')
        if not all([(0<=o<d) for o,d in zip(offset,footprint.shape)]):raise ValueError('Offset must be included inside footprint')
    footprint[tuple(slice(d,d+1) for d in offset)]=False
    # Preserve source error precedence: method and min/max before dtype rejection.
    if method=='dilation':
        if not seed.size:np.min(seed)
    elif method=='erosion':
        if not seed.size:np.max(seed)
    else:raise ValueError("Reconstruction method can be one of 'erosion' or 'dilation'. Got '"+str(method)+"'.")
    if mask.dtype.kind=='c':raise ValueError('complex valued input is not supported')
    off=np.array(list(offset),dtype=np.uintp)
    zero=_zero_provider(seed,mask,footprint,off,method)
    out,info=_core.reconstruction_run(seed,mask,footprint,off,0 if method=='dilation' else 1,zero)
    # The source assigns shape after the numerical loop. This is observable
    # (NumPy2.5 warns); retain the actual provider operation, not a synthesized
    # warning or a reshape substitution. The padded owner metadata stays exact.
    out.shape=np.array(seed.shape)+(np.array(footprint.shape)-1)
    inside=tuple(slice(o,o+n) for o,n in zip(off,seed.shape))
    return out[inside]
