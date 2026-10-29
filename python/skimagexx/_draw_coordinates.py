"""Owned coordinate generation after explicit NumPy preparation providers.

The unsupported line conversion/protocol route uses the original compiled
primitive once. Accepted integer lines never call it. Ellipse geometry keeps
the source bounding-box and scalar NumPy trig stages; C++ owns the admitted
distance predicate and ordered emission. BSD source attribution is retained
in third_party/scikit-image-LICENSE.txt.
"""
import numpy as np
from . import _core

_PROFILES={'1.24.4':1,'2.5.4':2}
_INTEGER_TYPES=(int,bool,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)


def line(r0,c0,r1,c1):
    native_result=None
    if all(type(value)in _INTEGER_TYPES for value in(r0,c0,r1,c1)):
        prepared=tuple(int(value)for value in(r0,c0,r1,c1))
        if all(-(1<<63)<=value<(1<<63)for value in prepared):
            native_result=_core.draw_line(*prepared,_core.get_num_threads())
            if native_result is not None:
                return tuple(np.asarray(memoryview(value))for value in native_result)
    # Explicit provider for unassessed conversion/protocol/unsafe arithmetic
    # partitions, not forwarding of an admitted native line.
    from skimage.draw._draw import _line
    return _line(r0,c0,r1,c1)


def rectangle(start,end=None,extent=None,shape=None):
    # Literal 0.26 _rectangle_slice order, kept local to preserve callbacks
    # before prepared integer buffers reach the native boundary.
    if end is None and extent is None:
        raise ValueError("Either `end` or `extent` must be given.")
    if end is not None and extent is not None:
        raise ValueError("Cannot provide both `end` and `extent`.")
    if extent is not None:
        end=np.asarray(start)+np.asarray(extent)
    top_left=np.minimum(start,end)
    bottom_right=np.maximum(start,end)
    top_left=np.round(top_left).astype(int)
    bottom_right=np.round(bottom_right).astype(int)
    if extent is None:
        bottom_right+=1
    tl,br=top_left,bottom_right
    if shape is not None:
        n_dim=len(start)
        br=np.minimum(shape[0:n_dim],br)
        tl=np.maximum(np.zeros_like(shape[0:n_dim]),tl)
    native_result=None
    profile=_PROFILES.get(np.__version__)
    if profile:
        native_result=_core.draw_rectangle(tl,br,_core.get_num_threads())
        if native_result is not None:
            return native_result if profile==1 else tuple(native_result)
    return np.meshgrid(*[np.arange(st,en)for st,en in zip(tuple(tl),tuple(br))])


def disk(center,radius,*,shape=None):
    r,c=center
    return ellipse(r,c,radius,radius,shape)


def ellipse(r,c,r_radius,c_radius,shape=None,rotation=0.0):
    center=np.array([r,c])
    radii=np.array([r_radius,c_radius])
    rotation%=np.pi
    r_radius_rot=abs(r_radius*np.cos(rotation))+c_radius*np.sin(rotation)
    c_radius_rot=r_radius*np.sin(rotation)+abs(c_radius*np.cos(rotation))
    radii_rot=np.array([r_radius_rot,c_radius_rot])
    upper_left=np.ceil(center-radii_rot).astype(int)
    lower_right=np.floor(center+radii_rot).astype(int)
    if shape is not None:
        upper_left=np.maximum(upper_left,np.array([0,0]))
        lower_right=np.minimum(lower_right,np.array(shape[:2])-1)
    shifted_center=center-upper_left
    bounding_shape=lower_right-upper_left+1

    # Literal _ellipse_in_shape prelude. Reuse it on refusal so callbacks,
    # warnings and floating state are never replayed.
    r_lim,c_lim=np.ogrid[0:float(bounding_shape[0]),0:float(bounding_shape[1])]
    r_org,c_org=shifted_center
    r_rad,c_rad=radii
    rotation%=np.pi
    sin_alpha,cos_alpha=np.sin(rotation),np.cos(rotation)
    native_result=None
    if _PROFILES.get(np.__version__):
        native_result=_core.draw_ellipse(bounding_shape,shifted_center,radii,
                                        sin_alpha,cos_alpha,upper_left,_core.get_num_threads())
        if native_result is not None:
            count=native_result.shape[0]
            # np.nonzero returns interleaved columns, including stride16
            # zero-length views over a zero-stride empty(N,2) owner.
            rr=np.ndarray((count,),dtype=np.intp,buffer=native_result,strides=(2*np.dtype(np.intp).itemsize,))
            cc=np.ndarray((count,),dtype=np.intp,buffer=native_result,
                          offset=np.dtype(np.intp).itemsize if count else 0,
                          strides=(2*np.dtype(np.intp).itemsize,))
            return rr,cc
    r,c=(r_lim-r_org),(c_lim-c_org)
    distances=((r*cos_alpha+c*sin_alpha)/r_rad)**2+((r*sin_alpha-c*cos_alpha)/c_rad)**2
    rr,cc=np.nonzero(distances<1)
    rr.flags.writeable=True
    cc.flags.writeable=True
    rr+=upper_left[0]
    cc+=upper_left[1]
    return rr,cc
