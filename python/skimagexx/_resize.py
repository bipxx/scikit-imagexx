"""Resize/rescale metadata around native C++ resampling stages.

The public registry is updated separately after the full contract gate passes.
Shape/scalar/coefficient metadata follows scikit-image 0.26.0 (BSD-3-Clause;
see third_party/scikit-image-LICENSE.txt). Image pixels stay in native stages.
"""
from functools import lru_cache
import math
import operator
import warnings
import numpy as np
from . import _core
from ._compat import fallback
from .filters import _gaussian_coefficients


_BORDERS={'constant':1,'edge':2,'symmetric':0,'reflect':3,'wrap':4}


@lru_cache(maxsize=1)
def _scipy_profile():
    try:
        from scipy import __version__
    except ImportError:
        return False,True
    version=np.lib.NumpyVersion(__version__)
    return version<'1.18.0',version>='1.16.0'


def _clip_profile():
    # Tiny metadata probes identify runtime C-loop behavior without assuming
    # that NaN propagation and signed-zero tie behavior changed together.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        replace_ties=not np.signbit(np.clip(np.array([-0.]),0.,0.))[0]
        ignore_nan=not np.isnan(np.clip(np.array([0.]),np.nan,np.nan)[0])
    policy=2 if not replace_ties else 0 if ignore_nan else 1
    return policy,tuple((type(w.message),str(w.message)) for w in notices)


_CLIP_POLICY,_CLIP_NOTICES=_clip_profile()


def _half_clip_profile():
    # Half retains the scalar C loop: it preserves zero ties and existing NaN
    # payloads even on NumPy versions whose optimized F32/F64 loop replaces them.
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        a=np.array([0.],np.float16);bound=np.asarray(np.nan,np.float16)
        ignore_nan=not np.isnan(np.clip(a,bound,bound)[0])
    return (0 if ignore_nan else 1),tuple((type(w.message),str(w.message)) for w in notices)


_HALF_POLICY,_HALF_NOTICES=_half_clip_profile()


def _mode_code(mode):
    if mode not in _BORDERS:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
            f"mode should be one of 'constant', 'edge', 'symmetric', "
            f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
            f"more info."
        )
    return _BORDERS[mode]


def _order(dtype,order):
    if order is None:return 0 if dtype==bool else 1
    if order<0 or order>5:
        raise ValueError('Spline interpolation order has to be in the range 0-5.')
    if dtype==bool and order!=0:
        raise ValueError('Input image dtype is bool. Interpolation is not defined '
                         'with bool data type. Please set order to 0 or explicitly '
                         'cast input image to another data type.')
    return order


def _constant(value):
    # The ndimage extension uses PyFloat_AsDouble, which accepts scalar numeric
    # protocols but does not parse numeric strings as Python float() does.
    return math.ldexp(value,0)


def _clip(image,out,mode,cval):
    lower,upper,nan,valid,_=_core.resize_range(image)
    low,high=lower[()],upper[()]
    if nan and not valid:
        warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
        warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
    if mode=='constant' and not low<=cval<=high:
        olower,oupper,onan,ovalid,_=_core.resize_range(out)
        if nan:
            omin=olower[()]
            if not ovalid:warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
        else:omin=np.nan if onan else olower[()]
        if omin<=cval:
            if nan:
                omax=oupper[()]
                if not ovalid:warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
            else:omax=np.nan if onan else oupper[()]
            if cval<=omax:
                constant=image.dtype.type(cval)
                low=min(low,constant);high=max(high,constant)
    lower=np.asarray(low);upper=np.asarray(high)
    if _CLIP_POLICY==0 and image.dtype.kind=='f' and (np.isnan(low) or np.isnan(high)):
        for category,message in _CLIP_NOTICES:warnings.warn(message,category,stacklevel=3)
    _core.resize_clip(out,lower,upper,_CLIP_POLICY)


def _half_identity(image,clip,mode,cval):
    do_clip=bool(clip)
    out,lower,upper,nan,valid,_=_core.resize_half_identity(image,do_clip,_HALF_POLICY)
    if do_clip:
        low,high=lower[()],upper[()]
        if nan and not valid:
            warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
            warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
        # The identity output has the same numeric range as its input, so cval
        # cannot expand it. Still preserve upstream's comparison/error order
        # and its extra all-NaN output-minimum warning in constant mode.
        if mode=='constant' and not low<=cval<=high:
            if nan and not valid:warnings.warn('All-NaN slice encountered',RuntimeWarning,stacklevel=3)
            if low<=cval:_=cval<=high
        if _HALF_POLICY==0 and (np.isnan(low) or np.isnan(high)):
            for category,message in _HALF_NOTICES:warnings.warn(message,category,stacklevel=3)
    return out


def resize(image,output_shape,order=None,mode='reflect',cval=0,clip=True,
           preserve_range=False,anti_aliasing=None,anti_aliasing_sigma=None):
    original=image
    # Preserve upstream shape validation before input dtype inspection.
    output_shape=tuple(output_shape)
    input_shape=image.shape
    if len(output_shape)>image.ndim:
        input_shape=input_shape+(1,)*(len(output_shape)-image.ndim)
        image=np.reshape(image,input_shape)
    elif len(output_shape)==image.ndim-1:
        output_shape=output_shape+(image.shape[-1],)
    elif len(output_shape)<image.ndim:
        raise ValueError('output_shape length cannot be smaller than the image number of dimensions')
    if type(image) not in (np.ndarray,np.memmap) and not isinstance(image,np.generic):
        return fallback('transform','resize',original,output_shape,order=order,mode=mode,cval=cval,
                        clip=clip,preserve_range=preserve_range,anti_aliasing=anti_aliasing,
                        anti_aliasing_sigma=anti_aliasing_sigma)
    descriptor=image.dtype
    supported=(descriptor.kind=='b' or descriptor.kind in 'iu' and descriptor.itemsize in (1,2,4,8)
               or descriptor.kind=='f' and descriptor.itemsize in (2,4,8))
    if not supported:
        return fallback('transform','resize',original,output_shape,order=order,mode=mode,cval=cval,
                        clip=clip,preserve_range=preserve_range,anti_aliasing=anti_aliasing,
                        anti_aliasing_sigma=anti_aliasing_sigma)
    input_shape=image.shape;input_type=image.dtype
    if input_type==np.float16:
        if not image.dtype.isnative:image=_core.edge_native(image)
        image=_core.edge_half(image)
    if anti_aliasing is None:
        anti_aliasing=(not input_type==bool and
                      not (np.issubdtype(input_type,np.integer) and order==0) and
                      any(x<y for x,y in zip(output_shape,input_shape)))
    if input_type==bool and anti_aliasing:
        raise ValueError('anti_aliasing must be False for boolean images')
    factors=np.divide(input_shape,output_shape)
    order=_order(input_type,order)
    if order>0 and image.dtype.type not in (np.float32,np.float64):
        if not preserve_range and image.ndim==0 and image.dtype.kind=='i':
            raise TypeError('return arrays must be of ArrayType')
        # Opposite-endian half is deliberately not caught by upstream's
        # dtype==float16 upcast above. Non-preserving conversion leaves it
        # unsupported by ndimage; preserve_range explicitly casts to double.
        if image.dtype.type is not np.float16 or preserve_range:
            if not image.dtype.isnative:image=_core.edge_native(image)
            if image.dtype.type is np.float16:image=_core.edge_half(image)
            image=_core.gaussian_nd(image,[],[],[],0.,bool(preserve_range),np.dtype(np.float64))[0]
    border=_mode_code(mode)
    if anti_aliasing:
        if anti_aliasing_sigma is None:sigmas=np.maximum(0,(factors-1)/2)
        else:
            sigmas=np.atleast_1d(anti_aliasing_sigma)*np.ones_like(factors)
            if np.any(sigmas<0):
                raise ValueError('Anti-aliasing standard deviation must be greater than or equal to zero')
            elif np.any((sigmas>0)&(factors<=1)):
                warnings.warn('Anti-aliasing standard deviation greater than zero but not down-sampling along all axes',UserWarning,stacklevel=2)
        kernels=[];axes=[]
        for axis,sigma in enumerate(sigmas):
            if sigma>1e-15:
                axes.append(axis);kernels.append(_gaussian_coefficients(sigma,int(4*float(sigma)+.5)))
        constant=_constant(cval) if axes else 0.
        if image.dtype.type is np.float16:
            if axes:raise RuntimeError("array type dtype('float16') not supported")
            filtered=image
        else:filtered=_core.resize_gaussian_raw(image,kernels,axes,[border]*len(axes),constant)
    else:filtered=image
    zooms=[1/f for f in factors]
    if image.ndim<1:raise RuntimeError('input and output rank must be > 0')
    target=tuple(int(round(n*z)) for n,z in zip(image.shape,zooms))
    if any(n<0 for n in target):raise ValueError('negative dimensions are not allowed')
    legacy,identity_supported=_scipy_profile()
    identity=identity_supported and all(z==1 for z in zooms)
    if identity and image.dtype.type is np.float16:return _half_identity(image,clip,mode,cval)
    if identity:
        kernel_order=0;constant=0.
    else:
        if order>1 and order<2:raise RuntimeError('spline order not supported')
        kernel_order=operator.index(order);constant=_constant(cval)
        if image.dtype.type is np.float16:
            raise RuntimeError("array type dtype('float16') not supported" if order>1 else 'data type not supported')
    out=_core.resize_zoom_typed(filtered,target,kernel_order,border,constant,legacy,identity,True)
    if clip:_clip(image,out,mode,cval)
    return out


def rescale(image,scale,order=None,mode='reflect',cval=0,clip=True,
            preserve_range=False,anti_aliasing=None,anti_aliasing_sigma=None,*,channel_axis=None):
    restore=None
    if channel_axis is not None:
        axes=(channel_axis,) if np.isscalar(channel_axis) else channel_axis
        if len(axes)>1:raise ValueError('only a single channel axis is currently supported')
        if axes!=(-1,) and axes!=-1:
            restore=axes[0];image=np.moveaxis(image,restore,-1);channel_axis=-1
    scale=np.atleast_1d(scale);multichannel=channel_axis is not None
    if len(scale)>1:
        if (not multichannel and len(scale)!=image.ndim) or (multichannel and len(scale)!=image.ndim-1):
            raise ValueError('Supply a single scale, or one value per spatial axis')
        if multichannel:scale=np.concatenate((scale,[1]))
    shape=np.asarray(image.shape);target=np.maximum(np.round(scale*shape),1)
    if multichannel:target[-1]=shape[-1]
    result=resize(image,target,order=order,mode=mode,cval=cval,clip=clip,preserve_range=preserve_range,
                  anti_aliasing=anti_aliasing,anti_aliasing_sigma=anti_aliasing_sigma)
    return np.moveaxis(result,-1,restore) if restore is not None else result
