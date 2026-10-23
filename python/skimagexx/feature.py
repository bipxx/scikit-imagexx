"""Image-feature signatures with native symmetric-matrix eigenvalue kernels."""
import numpy as np
from . import _core
from ._compat import fallback, module_attribute

__path__=[]
__all__=['hessian_matrix_eigvals','structure_tensor_eigenvalues']
_LEGACY_SCALARS=np.lib.NumpyVersion(np.__version__)<'2.0.0'


def _eigenvalues(elements,name):
    if np.geterr()['under']=='ignore':
        result=_core.symmetric_eigenvalues(elements,_LEGACY_SCALARS)
        if result is not None:return result
    return fallback('feature',name,elements)


def hessian_matrix_eigvals(H_elems):
    return _eigenvalues(H_elems,'hessian_matrix_eigvals')


def structure_tensor_eigenvalues(A_elems):
    return _eigenvalues(A_elems,'structure_tensor_eigenvalues')


def __getattr__(name):return module_attribute('feature',name)


def _structure_tensor_impl(image,sigma,mode,cval,order,forwarded):
    from .filters import _GAUSSIAN_TYPES,_GAUSSIAN_MODES,_gaussian_coefficients,_gaussian_cached_coefficients
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:return forwarded()
    if type(order) is not str or order not in ('rc','xy') or order=='xy' and image.ndim>2:return forwarded()
    if np.geterr()['under']!='ignore':return forwarded()
    scalar_types=(bool,int,float,np.bool_,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64,np.float16,np.float32,np.float64)
    if type(sigma) in scalar_types:sigmas=None
    elif type(sigma) in (tuple,list) or type(sigma) is np.ndarray and sigma.ndim==1:
        if len(sigma)!=image.ndim:raise ValueError('sigma must have as many elements as image has axes')
        sigmas=list(sigma)
    else:return forwarded()
    prepared=np.squeeze(image)
    if not prepared.size:raise ValueError('The parameter `image` cannot be an empty array')
    if prepared.ndim not in range(2,6):raise ValueError('The parameter `image` must be a 2-or-3-or-4-or-5-dimensional array')
    if sigmas is None:sigmas=[sigma]*prepared.ndim
    elif len(sigmas)!=prepared.ndim:return forwarded()
    if any(type(value) not in scalar_types or not np.isfinite(float(value)) or not 0<=float(value)<=25000 for value in sigmas):return forwarded()
    # The reference squares typed sigma values before constructing coefficients.
    # Keep metadata overflows/underflows in its compatibility path so warning
    # counts and ordering remain correct for every tensor component.
    for value in sigmas:
        if type(value) is np.float16 and float(value)!=0 and not .003<=float(value)<=250:return forwarded()
        if isinstance(value,np.integer) and float(value)**2>np.iinfo(type(value)).max:return forwarded()
    if type(mode) is str:modes=[mode]*prepared.ndim
    elif type(mode) in (tuple,list) and len(mode)==prepared.ndim:modes=list(mode)
    else:return forwarded()
    if any(type(value) is not str or value not in _GAUSSIAN_MODES for value in modes):return forwarded()
    if type(cval) not in scalar_types or not np.isfinite(float(cval)):return forwarded()
    kernels=[];axes=[]
    for axis,value in enumerate(sigmas):
        if float(value)<=1e-15:continue
        radius=int(4*float(value)+.5)
        weights=(_gaussian_cached_coefficients if type(value) in (int,float) else _gaussian_coefficients)(value,radius)
        kernels.append(weights);axes.append(axis)
    if not prepared.dtype.isnative:prepared=_core.edge_native(prepared)
    if prepared.dtype.type is np.float16:prepared=_core.edge_half(prepared)
    result=_core.structure_tensor(prepared,kernels,axes,[_GAUSSIAN_MODES[value] for value in modes],float(cval),order=='xy')
    return result if result is not None else forwarded()


def structure_tensor(image,sigma=1,mode='constant',cval=0,order='rc'):
    return _structure_tensor_impl(image,sigma,mode,cval,order,lambda:fallback('feature','structure_tensor',image,sigma=sigma,mode=mode,cval=cval,order=order))


def _corner_response(image,sigma,method,parameter,forwarded):
    from .filters import _GAUSSIAN_TYPES
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES:return forwarded()
    single=image.dtype.type in (np.float16,np.float32);wide=False
    if method in (0,1):
        scalar_types=(bool,int,float,np.bool_,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64,np.float16,np.float32,np.float64)
        if type(parameter) not in scalar_types:return forwarded()
        dtype=np.dtype(np.float32 if single else np.float64)
        try:
            target=np.result_type(np.empty(1,dtype=dtype),parameter)
            value=float(parameter)
        except (ValueError,TypeError,OverflowError):return forwarded()
        if target.type not in (np.float32,np.float64) or not np.isfinite(value) or abs(value)>np.finfo(target).max/16:return forwarded()
        wide=single and target.type is np.float64
    else:value=0.
    fields=_structure_tensor_impl(image,sigma,'constant',0,'rc',lambda:None)
    if fields is None:return forwarded()
    a,b,c=fields  # Preserve upstream's three-component unpacking error in N-D.
    if method==3 and image.shape!=a.shape:return forwarded()
    result=_core.corner_response(fields,method,value,wide)
    if result is None:return forwarded()
    if method==3 and (type(image) is np.memmap or not image.flags.c_contiguous):
        outputs=[]
        for field in result:
            out=np.zeros_like(image,dtype=field.dtype);out[...]=field;outputs.append(out)
        return tuple(outputs)
    return result


def corner_harris(image,method='k',k=0.05,eps=1e-6,sigma=1):
    forwarded=lambda:fallback('feature','corner_harris',image,method=method,k=k,eps=eps,sigma=sigma)
    if type(method) is not str:return forwarded()
    return _corner_response(image,sigma,0 if method=='k' else 1,k if method=='k' else eps,forwarded)


def corner_shi_tomasi(image,sigma=1):
    return _corner_response(image,sigma,2,0,lambda:fallback('feature','corner_shi_tomasi',image,sigma=sigma))


def corner_foerstner(image,sigma=1):
    return _corner_response(image,sigma,3,0,lambda:fallback('feature','corner_foerstner',image,sigma=sigma))


__all__+=['structure_tensor','corner_harris','corner_shi_tomasi','corner_foerstner']



def corner_kitchen_rosenfeld(image,mode='constant',cval=0):
    forwarded=lambda:fallback('feature','corner_kitchen_rosenfeld',image,mode=mode,cval=cval)
    from .filters import _GAUSSIAN_TYPES,_GAUSSIAN_MODES
    if type(image) not in (np.ndarray,np.memmap) or image.dtype.type not in _GAUSSIAN_TYPES or image.ndim!=2:return forwarded()
    if np.geterr()['under']!='ignore':return forwarded()
    if type(mode) is str:modes=[mode,mode]
    elif type(mode) in (tuple,list) and len(mode)==2:modes=list(mode)
    else:return forwarded()
    if any(type(value) is not str or value not in _GAUSSIAN_MODES for value in modes):return forwarded()
    scalar_types=(bool,int,float,np.bool_,np.int8,np.uint8,np.int16,np.uint16,np.int32,np.uint32,np.int64,np.uint64,np.float16,np.float32,np.float64)
    if type(cval) not in scalar_types:return forwarded()
    try:constant=float(cval)
    except (ValueError,OverflowError):return forwarded()
    if not np.isfinite(constant):return forwarded()
    prepared=image if image.dtype.isnative else _core.edge_native(image)
    if prepared.dtype.type is np.float16:prepared=_core.edge_half(prepared)
    result=_core.kitchen_rosenfeld(prepared,[_GAUSSIAN_MODES[value] for value in modes],constant)
    if result is None:return forwarded()
    if type(image) is np.memmap or not image.flags.c_contiguous:
        out=np.zeros_like(image,dtype=result.dtype);out[...]=result;return out
    return result


__all__+=['corner_kitchen_rosenfeld']
