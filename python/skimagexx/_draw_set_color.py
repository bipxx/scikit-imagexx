"""Owned staged f64 set_color; BSD operation attribution in native source."""
import numpy as np
import struct
from . import _core
from ._compat import fallback

_COPY=None if np.lib.NumpyVersion(np.__version__)>='2.0.0'else False

def _aligned(value,dtype=None):
    if (dtype is None or value.dtype==np.dtype(dtype))and value.flags.c_contiguous and value.flags.aligned:return value
    return np.require(value,dtype=dtype,requirements=('C','A'))

def set_color(image,coords,color,alpha=1):
    native_result=None
    if type(image)is not np.ndarray or type(coords)not in(tuple,list)or len(coords)!=2 or image.ndim not in(2,3):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    rr,cc=coords
    if any(type(a)is not np.ndarray or a.ndim!=1 or a.dtype!=np.dtype(np.intp)or not a.dtype.isnative for a in(rr,cc)):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if type(color)not in(np.ndarray,list,tuple,int,float,np.float64):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if type(color)in(list,tuple)and any(type(v)not in(int,float,np.float64)for v in color):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    prepared_color=np.array(color,ndmin=1,copy=_COPY)
    channels=1 if image.ndim==2 else image.shape[-1]
    if channels!=prepared_color.shape[-1]:
        raise ValueError(f'Color shape ({prepared_color.shape[0]}) must match last image dimension ({{image.shape[-1]}}).' )
    if prepared_color.ndim!=1 or prepared_color.dtype.kind not in 'fiu' or not prepared_color.dtype.isnative:
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if prepared_color.dtype.kind=='f'and prepared_color.dtype!=np.dtype(np.float64):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if prepared_color.dtype.kind in 'iu' and np.any(np.abs(prepared_color.astype(object))>16):
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if type(alpha)is int and abs(alpha)>16:return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if type(alpha)in(float,np.float64):
        bits=struct.unpack('=Q',struct.pack('=d',alpha))[0]&0x7fffffffffffffff
        if bits>0x4030000000000000 or(bits!=0 and bits<0x37f0000000000000):return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if type(alpha)in(int,float,np.float64):prepared_alpha=np.full(rr.shape,alpha,dtype=np.float64)
    elif type(alpha)is np.ndarray and alpha.ndim==1 and alpha.dtype==np.dtype(np.float64)and alpha.dtype.isnative:prepared_alpha=alpha
    else:return fallback('draw','set_color',image,coords,color,alpha=alpha)
    if image.dtype not in(np.dtype(np.float64),np.dtype(np.float32))or not image.dtype.isnative or not image.flags.writeable or rr.shape!=cc.shape or prepared_alpha.shape!=rr.shape:
        return fallback('draw','set_color',image,coords,color,alpha=alpha)
    integer_product=type(alpha)is int and(prepared_color.dtype.kind=='i'or(prepared_color.dtype.kind=='u'and prepared_color.itemsize<8))
    native_result=_core.draw_set_color(image,_aligned(rr),_aligned(cc),_aligned(prepared_color,np.float64),_aligned(prepared_alpha),_core.get_num_threads(),integer_product)
    if native_result:return None
    return fallback('draw','set_color',image,coords,color,alpha=alpha)
