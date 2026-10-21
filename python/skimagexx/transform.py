"""Native integral images; remaining transforms use explicit fallback."""
from ._compat import module_attribute, fallback
from . import _core
import numpy as np


def __getattr__(name):
    return module_attribute('transform', name)


_REAL_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}


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
    def forwarded():return fallback('transform','integral_image',image,dtype=dtype)
    if type(image) not in (np.ndarray,np.memmap) and not isinstance(image,np.generic):return forwarded()
    # This inspection precedes upstream's axis loop, including on object scalars
    # whose .real property can return a Python object without a dtype attribute.
    requested=np.promote_types(image.dtype,np.float64) if dtype is None and image.real.dtype.kind=='f' else dtype
    if image.ndim==0:return image
    if type(image) is not np.ndarray or image.dtype.type not in _REAL_TYPES:return forwarded()
    # Empty metadata probe: no image pixels are processed by NumPy. It retains
    # version-specific cumsum dtype/descriptor promotion.
    target=np.empty(0,dtype=image.dtype).cumsum(dtype=requested).dtype
    if target.type not in (np.int64,np.uint64,np.float32,np.float64):return forwarded()
    if target.kind in 'iu' and image.dtype.kind=='f':return forwarded()
    if target.type==np.float32 and image.dtype.type!=np.float32:return forwarded()
    native=_integral_storage(image)
    if native.dtype.kind=='f' and native.size:
        low,high=_core.intensity_bounds(native.ravel(order='K'))
        # Leave room for rounding at every ordered axis stage. Each finite
        # addition's rounding error is bounded by the smaller operand's
        # magnitude; the conservative factor doubles per spatial axis.
        limit=float(np.finfo(target).max)/native.size/(2.**native.ndim)
        if not (np.isfinite(low) and np.isfinite(high)) or max(abs(low),abs(high))>limit:return forwarded()
    # Keep-order allocation matches upstream strides, including singleton and
    # empty dimensions. All conversion and prefix arithmetic remain native.
    result=np.empty_like(image,dtype=target,order='K',subok=False)
    return _core.integral_image(native,result)


__all__=['integral_image']
