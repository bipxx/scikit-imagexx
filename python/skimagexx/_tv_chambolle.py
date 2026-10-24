"""TV-Chambolle provider orchestration around native materialized stages.

The scalar, energy/reduction, conversion, allocation, channel and exceptional
NumPy stages are explicit providers, not a pure-C++ whole-loop claim. Source
ordering follows scikit-image0.26 (BSD-3-Clause; third_party license). Public
registration and complete-product admission are owned by separate integration.
"""
import importlib
import inspect
import numpy as np
from . import _core

_ENTRIES=('tv_chambolle_divergence_into','tv_chambolle_gradient_into',
          'tv_chambolle_norm_into','tv_chambolle_dual_into')
_REAL={np.bool_,np.uint8,np.int8,np.uint16,np.int16,np.uint32,np.int32,
       np.uint64,np.int64,np.float16,np.float32,np.float64}
_NUMPY_CORE='numpy._core' if np.__version__.split('.')[0]=='2' else 'numpy.core'
_MULTIARRAY=importlib.import_module(_NUMPY_CORE+'.multiarray')
_NUMERIC=importlib.import_module(_NUMPY_CORE+'.numeric')
_FUNCTION_BASE=importlib.import_module('numpy.lib._function_base_impl' if np.__version__.split('.')[0]=='2' else 'numpy.lib.function_base')
_ALLOCATIONS=('zeros','zeros_like','empty','empty_like','array','asarray',
              'ones','ones_like','full','full_like','copy')
_PROVIDERS={name:getattr(np,name) for name in _ALLOCATIONS+
            ('sqrt','diff','abs','multiply','maximum','getbufsize','float32','float64','dtype','ndarray')}
_HANDLERS={name:getattr(_MULTIARRAY,name) for name in ('get_handler_name','get_handler_version')}


def _providers_pristine():
    # Latch before caller dtype/allocation hooks and recheck after the original
    # p/g/d allocations. Never turn a temporary monkeypatch into native work
    # merely because its hook restored itself before the next allocation.
    if any(getattr(np,name) is not value for name,value in _PROVIDERS.items()):return False
    for name in ('zeros','empty','empty_like','array','asarray','dtype','ndarray'):
        if getattr(np,name) is not getattr(_MULTIARRAY,name):return False
    for name in ('zeros_like','ones','ones_like','full','full_like'):
        if getattr(np,name) is not getattr(_NUMERIC,name):return False
    if np.copy is not _FUNCTION_BASE.copy:return False
    # Python monkeypatches and nondefault NumPy data allocators retain the
    # materialized provider path; this is not a whole-API allocation claim.
    if any(getattr(_MULTIARRAY,name) is not function or not inspect.isbuiltin(function)
           or function.__name__!=name for name,function in _HANDLERS.items()):return False
    if (_HANDLERS['get_handler_name']()!='default_allocator'
            or _HANDLERS['get_handler_version']()!=1):return False
    return True


def _require_finite():
    capability=getattr(_core,'tv_chambolle_finite_capabilities',None)
    function=getattr(_core,'tv_chambolle_finite_run',None)
    if (getattr(_core,'tv_chambolle_finite_interface',None)!='tv-chambolle-finite-v1'
            or type(capability) is not int or capability!=3
            or getattr(_core,'tv_chambolle_finite_profiles',None)!=('1.24.4','2.5.4')
            or not inspect.isbuiltin(function)
            or function.__name__!='tv_chambolle_finite_run'
            or function.__module__!=_core.__name__):
        raise RuntimeError('owned compiled TV finite loop/profile is unavailable; stage63 is not loop admission')
    return function


def _finite_profile():
    if np.__version__ not in ('1.24.4','2.5.4'):return None
    # Probe scalar promotion with exact binary metadata values. The profile
    # must describe the active NumPy behavior, not just its version string.
    energy=np.float32(0)
    energy+=0.5*np.float32(1)
    energy/=float(2)
    expected=np.float64 if np.__version__=='1.24.4' else np.float32
    if type(energy) is not expected:return None
    return np.__version__


def _finite_call(function,*arguments):
    """Single real builtin call site for private trace/counter instrumentation."""
    return function(*arguments)


def _try_finite_loop(image,p,g,d,weight,eps,max_num_iter,providers_pristine):
    # Scalars with custom protocols, NumPy scalar promotion, and callback/FP
    # policies stay on the unchanged owned path. No eager coercion occurs.
    if (not providers_pristine or not _providers_pristine()
            or type(image) is not np.ndarray or not image.dtype.isnative
            or image.dtype not in (np.dtype('f4'),np.dtype('f8'))
            or not 1<=image.ndim<=4 or image.size==0
            or type(weight) is not float or type(eps) is not float
            or type(max_num_iter) is not int or not 1<=max_num_iter<=20000
            or not 2.**-20<=weight<=2.**20 or not 0.<=eps<=2.**20):return None
    policy=np.geterr()
    if policy['under']!='ignore' or any(v not in ('ignore','warn','raise') for v in policy.values()):return None
    if not hasattr(_core,'tv_chambolle_finite_interface'):return None
    function=_require_finite()
    profile=_finite_profile()
    if profile is None:return None
    buffer_elements=np.getbufsize()
    if type(buffer_elements) is not int or buffer_elements<=0:return None
    # Original d allocation has already occurred. The kernel needs a C spatial
    # workspace; a pristine-provider C copy preserves its positive-zero words
    # for F-layout inputs without changing the original observable allocations.
    native_d=d if d.flags.c_contiguous else np.array(d,order='C',copy=True,subok=False)
    try:
        result,iterations,input_alias,trace=_finite_call(function,image,p,g,native_d,
            weight,eps,max_num_iter,profile,buffer_elements)
    except NotImplementedError:
        # Only the binding's complete PREWRITE guard translates this exception.
        # Unexpected numerical/loop exceptions propagate rather than silently
        # retrying after possibly modified workspaces.
        return None
    return image if input_alias else result


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


def _spatial(image,weight,eps,max_num_iter,_finite_providers=True):
    ndim=image.ndim
    p=np.zeros((image.ndim,)+image.shape,dtype=image.dtype)
    g=np.zeros_like(p)
    d=np.zeros_like(image)
    finite=_try_finite_loop(image,p,g,d,weight,eps,max_num_iter,_finite_providers)
    if finite is not None:return finite
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
        if 1<=ndim<=4 and _ordinary(g) and np.empty is _MULTIARRAY.empty:
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
    providers_pristine=(_providers_pristine() and type(image) is np.ndarray
                        and (channel_axis is None or type(channel_axis) is int))
    im_type=image.dtype
    if not im_type.kind=='f':image=_img_as_float(image)
    float_dtype=_supported_float_type(image.dtype)
    image=image.astype(float_dtype,copy=False)
    if channel_axis is not None:
        channel_axis=channel_axis%image.ndim
        out=np.zeros_like(image)
        for channel in range(image.shape[channel_axis]):
            index=[slice(None)]*image.ndim;index[channel_axis]=channel;index=tuple(index)
            out[index]=_spatial(image[index],weight,eps,max_num_iter,providers_pristine)
    else:out=_spatial(image,weight,eps,max_num_iter,providers_pristine)
    return out
