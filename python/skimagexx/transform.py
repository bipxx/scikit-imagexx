"""Native integral images; remaining transforms use explicit fallback."""
from ._compat import module_attribute, fallback
from . import _core
import numpy as np


def __getattr__(name):
    return module_attribute('transform', name)


_REAL_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}
_TARGET_TYPES={np.int64,np.uint64,np.float32,np.float64}
_DEFAULT_DTYPES={}
for _type in _REAL_TYPES:
    _descriptor=np.dtype(_type)
    if _descriptor.kind=='f':_default=np.dtype(np.float64)
    else:
        _platform=np.dtype(np.uintp if _descriptor.kind=='u' else np.intp)
        _default=_platform if _descriptor.itemsize<=_platform.itemsize else _descriptor
    _DEFAULT_DTYPES[_type]=_default


def _integral_fallback(image,dtype):
    return fallback('transform','integral_image',image,dtype=dtype)


def _integral_storage(image):
    native=image
    if native.size and not (native.flags.c_contiguous or native.flags.f_contiguous):
        extent=native.itemsize
        for step,length in sorted((s,n) for s,n in zip(native.strides,native.shape) if n>1):
            if step!=extent:native=native.copy(order='K');break
            extent*=length
    if not native.flags.aligned:native=native.copy(order='K')
    if not native.dtype.isnative or native.dtype.type==np.float16:
        shape=native.shape;strides=tuple(step//native.itemsize for step in native.strides)
        physical=native.ravel(order='K')
        if not physical.dtype.isnative:physical=_core.edge_native(physical)
        if physical.dtype.type==np.float16:physical=_core.edge_half(physical)
        native=np.ndarray(shape,dtype=physical.dtype,buffer=physical,
                          strides=tuple(step*physical.itemsize for step in strides))
    return native


def integral_image(image,*,dtype=None):
    if type(image) not in (np.ndarray,np.memmap) and not isinstance(image,np.generic):return _integral_fallback(image,dtype)
    if image.ndim==0:
        # Object scalars can expose a Python .real without a dtype. Preserve
        # upstream's inspection before its otherwise identity-preserving loop.
        if dtype is None and image.real.dtype.kind=='f':np.promote_types(image.dtype,np.float64)
        return image
    source=image.dtype
    if type(image) is not np.ndarray or source.type not in _REAL_TYPES:return _integral_fallback(image,dtype)
    if dtype is None and source.metadata is None:
        target=_DEFAULT_DTYPES[source.type]
    else:
        requested=np.promote_types(source,np.float64) if dtype is None and source.kind=='f' else np.dtype(dtype) if dtype is not None else None
        if source.metadata is None and requested is not None and requested.metadata is None and requested.type in _TARGET_TYPES:
            target=requested if requested.isnative else requested.newbyteorder('=')
        else:
            # Tagged descriptors and unsupported dtype validation retain the
            # empty upstream metadata probe; no image pixels are processed.
            target=np.empty(0,dtype=source).cumsum(dtype=requested).dtype
    if target.type not in _TARGET_TYPES:return _integral_fallback(image,dtype)
    if target.kind in 'iu' and source.kind=='f':return _integral_fallback(image,dtype)
    if target.type==np.float32 and source.type!=np.float32:return _integral_fallback(image,dtype)
    native=_integral_storage(image)
    # Keep-order allocation matches upstream strides, including singleton and
    # empty dimensions. All conversion and prefix arithmetic remain native.
    result=np.empty_like(image,dtype=target,order='K',subok=False)
    if not _core.integral_image(native,result):
        del result
        return _integral_fallback(image,dtype)
    return result


__all__=['integral_image']
