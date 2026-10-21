"""Native integral images, window queries and raw local-block means."""
from ._compat import module_attribute, fallback
from . import _core
import numpy as np
import warnings


def __getattr__(name):
    return module_attribute('transform', name)


_REAL_TYPES={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,np.uint64,np.int64,np.float16,np.float32,np.float64}
_TARGET_TYPES={np.int64,np.uint64,np.float32,np.float64}
_INDEX_BITS=np.dtype(np.uintp).itemsize*8
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
    # Normalization, keep-order allocation and direct strided-line selection
    # share the native call, so other dtypes avoid an extra Python dispatch.
    result=_core.integral_image(image,target)
    if result is None:
        return _integral_fallback(image,dtype)
    return result


__all__=['integral_image','integrate','downscale_local_mean']



def _integrate_fallback(ii,start,end):
    return fallback('transform','integrate',ii,start,end)


def integrate(ii,start,end):
    if type(ii) not in (np.ndarray,np.memmap) and not isinstance(ii,np.generic):return _integrate_fallback(ii,start,end)
    if ii.dtype.type not in _REAL_TYPES or ii.ndim>=_INDEX_BITS:return _integrate_fallback(ii,start,end)
    # Coordinate metadata follows upstream normalization and validation order.
    # See third_party/scikit-image-LICENSE.txt. No image pixels use NumPy math.
    if (type(start) is np.ndarray and type(end) is np.ndarray and start.ndim==end.ndim==2
            and start.dtype.kind in 'biu' and end.dtype.kind in 'biu'):
        first=start;last=end
    else:first=np.atleast_2d(np.array(start));last=np.atleast_2d(np.array(end))
    rows=first.shape[0]
    shape=np.asarray(ii.shape,dtype=np.intp);total_shape=np.broadcast_to(shape,(rows,shape.size))
    first_negative=first<0;last_negative=last<0
    first=(first+total_shape)*first_negative+first*~first_negative
    last=(last+total_shape)*last_negative+last*~last_negative
    if ((last-first)<0).any():raise IndexError('end coordinates must be greater or equal to start')
    if not rows:return np.zeros(0)
    if first.ndim!=2 or last.ndim!=2:return _integrate_fallback(ii,start,end)
    if first.shape!=last.shape:first,last=np.broadcast_arrays(first,last)
    if first.shape[1]!=ii.ndim or last.shape[1]!=ii.ndim:return _integrate_fallback(ii,start,end)
    lower=(first-1)[:rows];upper=last[:rows]
    if ii.ndim and (lower.dtype.kind not in 'iu' or upper.dtype.kind not in 'iu'):return _integrate_fallback(ii,start,end)
    if lower.dtype.kind=='u' and (lower>np.iinfo(np.int64).max).any():return _integrate_fallback(ii,start,end)
    invalid=(upper < -shape) | (upper >= shape)
    if invalid.any():
        row,axis=np.argwhere(invalid)[0];index=upper[row,axis]
        raise IndexError(f'index {index} is out of bounds for axis {axis} with size {ii.shape[axis]}')
    if (lower>=shape).any():return _integrate_fallback(ii,start,end)
    upper=np.where(upper<0,upper+shape,upper)
    lower=np.ascontiguousarray(lower,dtype=np.int64);upper=np.ascontiguousarray(upper,dtype=np.int64)
    table=np.asarray(ii) if isinstance(ii,np.generic) else ii
    result=np.empty(rows,dtype=np.float64)
    if not _core.integrate(table,lower,upper,result):
        del result
        return _integrate_fallback(ii,start,end)
    return result


_MEAN_DTYPES={}
_MEAN_ZEROS={}


def _local_mean_constant(image,cval,notices):
    if type(cval) is int and cval==0:
        key=image.dtype.str
        constant=_MEAN_ZEROS.get(key)
        if constant is None:
            constant=np.zeros((),dtype=key);constant.flags.writeable=False
            _MEAN_ZEROS[key]=constant
        return constant
    values=np.asarray(cval)
    if values.size!=1 or values.ndim>=3:return None
    constant=np.empty((),dtype=image.dtype);scalar=values.reshape(-1)[0]
    # pad casts both sides of every axis, including zero-width sides. Delay
    # warnings until native success, so delegation never repeats probe notices.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            for _ in range(2*image.ndim):constant[()]=scalar
        finally:notices.extend(captured)
    return constant


def _local_mean_output(image,blocks,after,shape,target):
    if not image.ndim or not all(shape) or image.flags.c_contiguous:
        return np.empty(shape,dtype=target)
    padded=any(after)
    if padded and not image.flags.f_contiguous:return np.empty(shape,dtype=target)
    if image.flags.f_contiguous and all(length>1 for length in shape):
        return np.empty(shape,dtype=target,order='F')
    # Reduction allocation ignores broadcast strides and resolves singleton
    # axes differently from empty_like(order='K'). Ask the public iterator to
    # allocate from metadata, with buffer initialization explicitly delayed.
    # It is never reset or advanced, so it never reads this virtual block view.
    if padded:
        padded_shape=tuple(length+width for length,width in zip(image.shape,after))
        steps=[0]*image.ndim;step=image.itemsize
        axes=range(image.ndim) if image.flags.f_contiguous else range(image.ndim-1,-1,-1)
        for axis in axes:steps[axis]=step;step*=padded_shape[axis]
        steps=tuple(steps)
    else:steps=image.strides
    view=np.lib.stride_tricks.as_strided(image,shape=shape+blocks,
        strides=tuple(stride*factor for stride,factor in zip(steps,blocks))+tuple(steps),writeable=False)
    iterator=np.nditer([None,view],flags=['reduce_ok','buffered','delay_bufalloc','zerosize_ok'],
        op_flags=[['readwrite','allocate','no_subtype'],['readonly','aligned','no_broadcast']],
        op_dtypes=[target,target],casting='unsafe',order='K',
        op_axes=[list(range(image.ndim))+[-1]*image.ndim,None])
    result=iterator.operands[0]
    iterator.close()
    return result


def downscale_local_mean(image,factors,cval=0,clip=True):
    """Average raw values over local blocks, with after-only constant padding."""
    if isinstance(image,np.generic) and image.dtype.type in _REAL_TYPES:image=np.asarray(image)
    notices=[]
    def upstream():
        notices.clear()
        return fallback('transform','downscale_local_mean',image,factors,cval=cval,clip=clip)
    try:return _local_mean_dispatch(image,factors,cval,clip,notices,upstream)
    finally:
        for notice in notices:warnings.warn(notice.message,stacklevel=3)


def _local_mean_dispatch(image,factors,cval,clip,notices,upstream):
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _REAL_TYPES:return upstream()
    if any(policy in ('call','log','print') for policy in np.geterr().values()):return upstream()
    # Validation order follows block_reduce/view_as_blocks. Only shape, factor
    # and constant metadata use NumPy; no image pixel is padded or reduced here.
    if np.isscalar(factors):block_shape=(factors,)*image.ndim
    else:
        if len(factors)!=image.ndim:raise ValueError('`block_size` must be a scalar or have the same length as `image.shape`')
        block_shape=factors
    after=[]
    for length,factor in zip(image.shape,block_shape):
        if factor<1:raise ValueError('Down-sampling factors must be >= 1. Use `skimage.transform.resize` to up-sample an image.')
        remainder=length%factor
        after.append(factor-remainder if remainder else 0)
    padded=any(after);constant=None
    if padded:
        # np.pad checks the complete width descriptor before casting cval.
        widths=np.asarray([(0,width) for width in after])
        if widths.dtype.kind not in 'iu':raise TypeError('`pad_width` must be of integral type.')
        # np.pad first normalizes scalar constants to NumPy scalars. This
        # distinction matters for out-of-range signed-to-unsigned assignment.
        constant=_local_mean_constant(image,cval,notices)
        if constant is None:return upstream()
    if not isinstance(block_shape,tuple):raise TypeError('block needs to be a tuple')
    if not all(isinstance(factor,(int,np.integer)) and not isinstance(factor,(bool,np.bool_)) for factor in block_shape):
        # Zero-byte storage asks NumPy's shape converter to reproduce upstream
        # scalar-type errors without allocating or inspecting image pixels.
        descriptor=np.array(block_shape)
        dimensions=tuple((np.array(image.shape)+np.array(after))//descriptor)+tuple(descriptor)
        np.empty(dimensions,dtype='V0')
        return upstream()
    block_shape=tuple(int(factor) for factor in block_shape)
    if image.dtype.kind=='f' and np.geterr()['under']!='ignore':return upstream()
    source=image.dtype
    if source.metadata is None:
        target=_MEAN_DTYPES.get(source)
        if target is None:
            target=np.empty((0,1),dtype=source).mean(axis=1).dtype;_MEAN_DTYPES[source]=target
    else:target=np.empty((0,1),dtype=source).mean(axis=1).dtype
    shape=tuple((length-1)//factor+1 if length else 0 for length,factor in zip(image.shape,block_shape))
    result=_local_mean_output(image,block_shape,after,shape,target)
    if not _core.downscale_local_mean(image,result,block_shape,constant):return upstream()
    return result[()] if not image.ndim else result
