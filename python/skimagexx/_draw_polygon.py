"""Source-ordered polygon intake, owned raster, explicit primitive providers.
BSD source attribution: third_party/scikit-image-LICENSE.txt.
"""
import math
import sys
import numpy as np
from . import _core
from ._compat import fallback


def _short(shape):
    return (type(shape) in (tuple,list) and len(shape)<2) or (type(shape) is np.ndarray and shape.ndim==1 and shape.size<2)


def _provider_polygon(r,c,shape):
    # Normalize protocols once even on provider paths; never forward a known
    # unchecked length/shape hazard into the original compiled classifier.
    r=np.atleast_1d(r);c=np.atleast_1d(c)
    if _short(shape) or (r.ndim==c.ndim==1 and r.size and c.size and r.size!=c.size):
        raise NotImplementedError('draw.polygon: unsafe coordinate lengths or short shape are outside the assessed safe boundary')
    return fallback('draw','polygon',r,c,shape=shape)


def _provider_mask(image_shape,polygon):
    polygon=np.asarray(polygon)
    r,c=polygon.T
    if _short(image_shape):
        raise NotImplementedError('draw.polygon2mask: shapes shorter than 2 entries are outside the assessed safe boundary')
    return fallback('draw','polygon2mask',image_shape,polygon)


def _plain(value):
    return type(value) in (list,tuple,np.ndarray,int,float,bool) or isinstance(value,np.generic)


def _shape(shape):
    if shape is None:return True
    if type(shape) in (tuple,list):
        return len(shape)<2 or (len(shape)==2 and all(type(n) is int for n in shape))
    return type(shape) is np.ndarray and shape.ndim==1 and shape.size<2


def _vertices(r,c,shape):
    # Literal public conversion order. Bounds are still NumPy/math providers.
    r=np.atleast_1d(r)
    c=np.atleast_1d(c)
    if (type(r) is not np.ndarray or type(c) is not np.ndarray or
            any(a.dtype.kind not in 'biuf' or a.dtype.itemsize>8 for a in (r,c))):
        return None,(r,c)
    minr=int(max(0,r.min()))
    if not -sys.maxsize-1<=minr<=sys.maxsize:return None,(r,c)
    maxr=int(math.ceil(r.max()))
    if not -sys.maxsize-1<=maxr<=sys.maxsize:return None,(r,c)
    minc=int(max(0,c.min()))
    if not -sys.maxsize-1<=minc<=sys.maxsize:return None,(r,c)
    maxc=int(math.ceil(c.max()))
    if not -sys.maxsize-1<=maxc<=sys.maxsize:return None,(r,c)
    if shape is not None:
        if len(shape)<2:
            raise NotImplementedError('draw.polygon: shapes shorter than 2 entries are outside the assessed safe boundary')
        maxr=min(shape[0]-1,maxr)
        maxc=min(shape[1]-1,maxc)
    # Cython's memoryviews are MUTABLE; ascontiguousarray does not copy away a
    # readonly contiguous f64 source. Preserve that failure before the next cast.
    rptr=np.ascontiguousarray(r,'float64')
    if rptr.ndim!=1:raise ValueError(f'Buffer has wrong number of dimensions (expected 1, got {rptr.ndim})')
    if not rptr.flags.writeable:raise ValueError('buffer source array is read-only')
    cptr=np.ascontiguousarray(c,'float64')
    if cptr.ndim!=1:raise ValueError(f'Buffer has wrong number of dimensions (expected 1, got {cptr.ndim})')
    if not cptr.flags.writeable:raise ValueError('buffer source array is read-only')
    if rptr.size!=cptr.size:
        raise NotImplementedError('draw.polygon: unequal coordinate lengths are outside the assessed safe boundary')
    return (rptr,cptr,(minr,minc),(maxr,maxc)),(r,c)


def polygon(r,c,shape=None):
    native_result=None
    if not _plain(r) or not _plain(c) or not _shape(shape):
        return _provider_polygon(r,c,shape)
    prepared,operands=_vertices(r,c,shape)
    if prepared is None:return _provider_polygon(*operands,shape)
    rptr,cptr,first,last=prepared
    workers=_core.get_num_threads()
    native_result=_core.draw_polygon_prepare(rptr,cptr,first,last,workers)
    if native_result is None:return _provider_polygon(*operands,shape)
    rr=np.empty(native_result.count,dtype=np.intp)
    cc=np.empty(native_result.count,dtype=np.intp)
    _core.draw_polygon_coords_into(native_result,rr,cc,workers)
    return rr,cc


def polygon2mask(image_shape,polygon):
    native_result=None
    short=_short(image_shape)
    if not _plain(polygon):
        if short:raise NotImplementedError('draw.polygon2mask: shapes shorter than 2 entries are outside the assessed safe boundary')
        return _provider_mask(image_shape,polygon)
    if not short and (type(image_shape) not in (tuple,list) or len(image_shape)!=2 or any(type(n) is not int for n in image_shape)):
        return _provider_mask(image_shape,polygon)
    if not short and all(n>=0 for n in image_shape) and image_shape[0]*image_shape[1]>16*1024*1024:
        return _provider_mask(image_shape,polygon)
    polygon=np.asarray(polygon)
    vertex_row_coords,vertex_col_coords=polygon.T
    prepared,operands=_vertices(vertex_row_coords,vertex_col_coords,image_shape)
    if prepared is None:return _provider_mask(image_shape,polygon)
    rptr,cptr,first,last=prepared
    workers=_core.get_num_threads()
    native_result=_core.draw_polygon_prepare(rptr,cptr,first,last,workers)
    if native_result is None:return _provider_mask(image_shape,polygon)
    # Preserve the source negative-allocation error phase; ordinary outputs
    # are initialized exactly once by the native zero/scatter entry.
    mask=np.zeros(image_shape,dtype=bool) if any(n<0 for n in image_shape) else np.empty(image_shape,dtype=bool)
    _core.draw_polygon_mask_into(native_result,mask,workers)
    return mask
