"""Untimed Sato parity oracle, including geometric-mean conditioning.

Image calculations here are verification only, never the native implementation.
For 3D, sqrt(max(l0,0)*max(l1,0)) is only Holder-1/2 at a zero eigenvalue.
An eigenvalue backward-error budget therefore cannot be reused as an output
absolute-error budget. Check Hessians first, then propagate their measured
perturbation and a 64-double-epsilon eigensolver allowance at the spectral scale.
The allowance covers the 3x3 Jacobi stopping tolerance and LAPACK roundoff;
it is an explicit tested numerical budget, not a claim of bitwise LAPACK parity.
"""
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimagexx as xx


def error_budget(image,options):
    dtype=np.float32 if image.dtype.type in (np.float16,np.float32) else np.float64
    prepared=image.astype(dtype,copy=False)
    if not options.get('black_ridges',True):prepared=-prepared
    tolerance=2e-6 if dtype is np.float32 else 2e-13
    amplitude=max(1.,float(np.max(np.abs(prepared.astype(np.float64)))))
    extra=np.zeros(image.shape,dtype=np.float64)
    scales=options.get('sigmas',range(1,10,2))
    if image.ndim==3:
        for sigma in scales:
            kw=dict(sigma=sigma,mode=options.get('mode','reflect'),cval=options.get('cval',0),use_gaussian_derivatives=True)
            expected_fields=hessian_matrix(prepared,**kw)
            actual_fields=xx.feature.hessian_matrix(prepared,**kw)
            squared_difference=np.zeros(image.shape,dtype=np.float64)
            for j,(a,b) in enumerate(zip(actual_fields,expected_fields)):
                np.testing.assert_allclose(a,b,rtol=tolerance,atol=tolerance*amplitude)
                # Upper-triangle off-diagonals appear twice in the matrix.
                squared_difference+=(1 if j in (0,3,5) else 2)*(a.astype(np.float64)-b.astype(np.float64))**2
            eigen=hessian_matrix_eigvals(expected_fields)
            norm=np.max(np.abs(eigen.astype(np.float64)),axis=0)
            delta=np.sqrt(squared_difference)+64*np.finfo(np.float64).eps*norm
            # Both solvers compute in double; F32 eigenvalue stores add an
            # eigenvalue-relative rounding term, not a norm-relative floor.
            rounding=2*np.finfo(np.float32).eps if dtype is np.float32 else 0.
            a=np.maximum(eigen[0].astype(np.float64),0)
            b=np.maximum(eigen[1].astype(np.float64),0)
            da=delta+rounding*np.abs(eigen[0].astype(np.float64))
            db=delta+rounding*np.abs(eigen[1].astype(np.float64))
            factor=abs(float(sigma**2))
            dp=factor*factor*(a*db+b*da+da*db)
            rho=factor*np.sqrt(a*b)
            # |sqrt(p+dp)-sqrt(p)| <= min(sqrt(|dp|), |dp|/sqrt(p)).
            relative=np.divide(dp,rho,out=np.full_like(dp,np.inf),where=rho>0)
            extra=np.maximum(extra,np.minimum(np.sqrt(dp),relative))
    return tolerance,tolerance*amplitude,extra


def assert_parity(actual,expected,image,options):
    rtol,atol,conditioning=error_budget(image,options)
    error=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    budget=atol+rtol*np.abs(expected.astype(np.float64))+conditioning
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual),np.isinf(expected))
    finite=np.isfinite(expected)
    if np.any(finite):
        excess=error[finite]-budget[finite]
        assert np.max(excess)<=0,dict(max_error=float(np.max(error[finite])),max_excess=float(np.max(excess)),max_budget=float(np.max(budget[finite])))
    zeros=(actual==0)&(expected==0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]),np.signbit(expected[zeros]))
    maximum_error=float(np.max(error[finite])) if np.any(finite) else 0.
    return {'max_abs_error':maximum_error,'max_input_scaled_error':maximum_error/(atol/rtol),
            'max_conditioning_budget':float(np.max(conditioning)),
            'conditioning_needed_pixels':int(np.sum(finite & (error>atol+rtol*np.abs(expected.astype(np.float64)))))}
