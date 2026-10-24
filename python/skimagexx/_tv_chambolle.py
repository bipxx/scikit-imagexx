"""TV-Chambolle provider orchestration around native materialized stages.

The scalar, energy/reduction, conversion, allocation, channel and exceptional
NumPy stages are explicit providers, not a pure-C++ whole-loop claim. Source
ordering follows scikit-image0.26 (BSD-3-Clause; third_party license). Public
registration and complete-product admission are owned by separate integration.
"""
import inspect
import numpy as np
from . import _core

_ENTRIES=('tv_chambolle_divergence_into','tv_chambolle_gradient_into',
          'tv_chambolle_norm_into','tv_chambolle_dual_into')
_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64}


def _require_stages():
    capability=getattr(_core,'tv_chambolle_capabilities',None)
    if (getattr(_core,'tv_chambolle_interface',None)!='tv-chambolle-stages-v1'
            or type(capability) is not int or capability!=63):
        raise RuntimeError('compiled TV materialized stages are unavailable; public whole-loop admission remains pending')
    for name in _ENTRIES:
        function=getattr(_core,name,None)
        if not (inspect.isbuiltin(function) and function.__name__==name and function.__module__==_core.__name__):
            raise RuntimeError('owned compiled TV stage missing: '+name)


def _img_as_float(image):
    # This is the source's dtype/conversion provider, not an image denoiser.
    # In particular asarray must occur after the caller's first dtype access.
    image=np.asarray(image)
    dtype=image.dtype
    if np.issubdtype(dtype.type,np.floating):return image
    if dtype.type not in _REAL:raise ValueError(f'Cannot convert from {dtype} to float64.')
    if dtype.kind=='b':return image.astype(np.float64)
    maximum=np.iinfo(dtype.type).max
    result=np.multiply(image,1.0/maximum,dtype=np.float64)
    if dtype.kind=='i':np.maximum(result,-1.0,out=result)
    return np.asarray(result,np.float64)


def _supported_float_type(dtype):
    dtype=np.dtype(dtype)
    if dtype.kind=='c':raise ValueError('complex valued input is not supported')
    return {'f':np.float32,'d':np.float64,'e':np.float32}.get(dtype.char,np.float64)


def _stage_call(name,*arguments):
    # A distinct call site also permits honest private test instrumentation:
    # counters wrap this dispatcher, while actual builtin identity is retained.
    return getattr(_core,name)(*arguments)


def _ordinary(*arrays):
    """Conservative finite/policy certificate, refreshed at each stage.

    These inspection operations do not coerce caller scalar providers. Bound
    arithmetic excludes invalid/overflow/divide events in native primitives;
    underflow must be ignored. Stateful/raise-underflow and subclasses keep the
    exact materialized NumPy provider path with its callbacks and warnings.
    """
    if any(value not in ('ignore','warn','raise') for value in np.geterr().values()):return False
    if np.geterr()['under']!='ignore':return False
    for value in arrays:
        if (type(value) is not np.ndarray or value.dtype not in (np.dtype('f4'),np.dtype('f8'))
                or not value.dtype.isnative):return False
        if not np.isfinite(value).all():return False
        if value.size and np.max(np.abs(value))>1.e6:return False
    return True


def _spatial(image,weight,eps,max_num_iter):
    ndim=image.ndim
    p=np.zeros((image.ndim,)+image.shape,dtype=image.dtype)
    g=np.zeros_like(p)
    d=np.zeros_like(image)
    i=0
    while i<max_num_iter:
        if i>0:
            # Retain this original reduction/allocation provider for now. The
            # native divergence replays the complete rounded axis fold and
            # predecessor additions; removing the duplicate reduction is a
            # separate future allocation/provider optimization, not hidden here.
            d=-p.sum(0)
            if 1<=ndim<=4 and _ordinary(p,d):
                _stage_call('tv_chambolle_divergence_into',p,d)
            else:
                slices_d=[slice(None)]*ndim
                slices_p=[slice(None)]*(ndim+1)
                for axis in range(ndim):
                    slices_d[axis]=slice(1,None)
                    slices_p[axis+1]=slice(0,-1)
                    slices_p[0]=axis
                    d[tuple(slices_d)]+=p[tuple(slices_p)]
                    slices_d[axis]=slice(None)
                    slices_p[axis+1]=slice(None)
            out=image+d
        else:out=image
        E=(d**2).sum()
        if 1<=ndim<=4 and _ordinary(out,g):
            _stage_call('tv_chambolle_gradient_into',out,g)
        else:
            slices_g=[slice(None)]*(ndim+1)
            for axis in range(ndim):
                slices_g[axis+1]=slice(0,-1)
                slices_g[0]=axis
                g[tuple(slices_g)]=np.diff(out,axis=axis)
                slices_g[axis+1]=slice(None)
        if 1<=ndim<=4 and _ordinary(g):
            # C planes imply the original axis0-fold result is C spatial.
            # Preserve an allocated NumPy stage with a real compiled norm.
            norm=np.empty(g.shape[1:],dtype=g.dtype)
            _stage_call('tv_chambolle_norm_into',g,norm)
            norm=norm[np.newaxis,...]
        else:norm=np.sqrt((g**2).sum(axis=0))[np.newaxis,...]
        E+=weight*norm.sum()
        tau=1.0/(2.0*ndim)
        norm*=tau/weight
        norm+=1.0
        if (1<=ndim<=4 and _ordinary(p,g,norm)
                and (not norm.size or np.min(norm)>=1.e-6)):
            # Scalar promotion is a runtime provider: the typed multiply must
            # receive the exact scalar selected by NumPy for this array dtype.
            tau_work=np.asarray(tau,dtype=p.dtype)[()]
            _stage_call('tv_chambolle_dual_into',p,g,norm.reshape(norm.shape[1:]),float(tau_work))
        else:
            p-=tau*g
            p/=norm
        E/=float(image.size)
        if i==0:E_init=E;E_previous=E
        else:
            if np.abs(E_previous-E)<eps*E_init:break
            else:E_previous=E
        i+=1
    return out


def denoise_tv_chambolle(image,weight=0.1,eps=0.0002,max_num_iter=200,*,channel_axis=None):
    """Denoise using compiled spatial stages and explicit NumPy providers."""
    _require_stages()
    im_type=image.dtype
    if not im_type.kind=='f':image=_img_as_float(image)
    float_dtype=_supported_float_type(image.dtype)
    image=image.astype(float_dtype,copy=False)
    if channel_axis is not None:
        channel_axis=channel_axis%image.ndim
        out=np.zeros_like(image)
        for channel in range(image.shape[channel_axis]):
            index=[slice(None)]*image.ndim;index[channel_axis]=channel;index=tuple(index)
            out[index]=_spatial(image[index],weight,eps,max_num_iter)
    else:out=_spatial(image,weight,eps,max_num_iter)
    return out
