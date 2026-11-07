"""Owned IGG image arithmetic; explicitly named coefficient/promotion providers."""
import numpy as np
from . import _core


def _coefficients(sigma,radius,derivative):
    # Exact admitted Hessian coefficient metadata construction. No image data.
    variance=sigma*sigma;positions=np.arange(-radius,radius+1)
    phi=np.exp(-.5/variance*positions**2);phi=phi/phi.sum()
    if not derivative:return np.ascontiguousarray(phi[::-1],dtype=np.float64)
    exponent=np.arange(2);q=np.array([1.,0.])
    operator=np.diag(exponent[1:],1)+np.diag(np.ones(1)/-variance,-1)
    q=operator.dot(q);polynomial=(positions[:,None]**exponent).dot(q)
    return np.ascontiguousarray((polynomial*phi)[::-1],dtype=np.float64)


def _plans(image,sigma):
    if not image.ndim:return []
    if not isinstance(sigma,str)and np.iterable(sigma):
        sigmas=list(sigma)
        if len(sigmas)!=image.ndim:raise RuntimeError('sequence argument must have length equal to input rank')
    else:sigmas=[sigma]*image.ndim
    active=[(i,s)for i,s in enumerate(sigmas)if s>1e-15]
    plans=[]
    for derivative in range(image.ndim):
        passes=[]
        for axis,value in active:
            radius=int(4*float(value)+.5)
            passes.append((axis,_coefficients(value,radius,axis==derivative)))
            if image.dtype.char=='e':raise RuntimeError("array type dtype('float16') not supported")
        plans.append(passes)
    return plans


def _policy(flags,operation,dtype):
    # One small ufunc delivers the complete native status together. NumPy then
    # applies its own divide/overflow/underflow/invalid order and callback mask.
    # These operands are policy metadata, never image computation.
    if not flags:return
    if operation=='sqrt':
        if flags!=1:raise NotImplementedError('IGG sqrt policy status unassessed')
        np.sqrt(np.array([-1.],dtype=np.finfo(dtype).dtype))
        return
    limits=np.finfo(dtype);left=[];right=[]
    if operation=='divide':
        if flags&2:left.append(1+1j if np.dtype(dtype).kind=='c'else 1.);right.append(0.)
        if flags&4:left.append(limits.max);right.append(.5)
        if flags&8:left.append(limits.tiny);right.append(limits.max)
        if flags&1:left.append(0.);right.append(0.)
    elif operation=='multiply':
        if flags&~13:raise NotImplementedError('IGG multiply policy status unassessed')
        if flags&4:left.append(limits.max);right.append(2.)
        if flags&8:left.append(limits.tiny);right.append(limits.tiny)
        if flags&1:left.append(np.inf);right.append(0.)
    elif operation=='add':
        if flags&~5:raise NotImplementedError('IGG add policy status unassessed')
        if flags&4:left.append(limits.max);right.append(limits.max)
        if flags&1:left.append(np.inf);right.append(-np.inf)
    else:raise ValueError('Unknown IGG expression operation')
    getattr(np,operation)(np.array(left,dtype=dtype),np.array(right,dtype=dtype))


def inverse_gaussian_gradient(image,alpha=100.0,sigma=5.0):
    image=np.asarray(image)
    workers=_core.get_num_threads()
    plans=_plans(image,sigma)
    if image.dtype.kind not in 'biufc' or image.dtype.char=='e':raise NotImplementedError('IGG input dtype not yet owned')
    prepared=np.require(image,dtype=image.dtype.name,requirements=('C','A')).reshape(image.shape)
    magnitude=_core.inverse_gaussian_magnitude(prepared,plans,workers)
    # Alpha coercion and NumPy value-based/weak scalar type promotion occur
    # only after the complete native gradient, as in the source expression.
    alpha_array=np.asarray(alpha)
    if alpha_array.dtype.kind not in 'biufc':
        np.multiply(alpha,np.empty((0,),dtype=magnitude.dtype))
        raise NotImplementedError('IGG nonnumeric alpha')
    try:shape=np.broadcast_shapes(alpha_array.shape,magnitude.shape)
    except ValueError:
        fmt=lambda x:str(x).replace(', ',',')
        raise ValueError(f'operands could not be broadcast together with shapes {fmt(alpha_array.shape)} {fmt(magnitude.shape)} ')from None
    promotion=alpha if isinstance(alpha,(int,float,complex,np.generic))else alpha_array
    dtype=np.result_type(promotion,magnitude)
    # Input/output casts and broadcasting metadata are declared providers.
    result=np.require(magnitude,dtype=dtype,requirements=('C','A')).reshape(magnitude.shape)
    values=np.require(alpha_array,dtype=dtype,requirements=('C','A')).reshape(alpha_array.shape)
    for stage in range(4):
        if stage==1:
            # The source's Python1.0 addition can promote an integral product;
            # multiplication's dtype is not the dtype of the whole expression.
            dtype=np.result_type(1.0,result)
            result=np.require(result,dtype=dtype,requirements=('C','A')).reshape(result.shape)
            values=np.require(alpha_array,dtype=dtype,requirements=('C','A')).reshape(alpha_array.shape)
        result,flags=_core.inverse_gaussian_finish(result,values,list(shape),stage,workers)
        _policy(flags,('multiply','add','sqrt','divide')[stage],dtype)
    return result[()]if result.ndim==0 else result
