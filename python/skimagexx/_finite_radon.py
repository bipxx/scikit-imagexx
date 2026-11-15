"""Finite Radon native reductions/roll/cast with explicit NumPy metadata policy.

NumPy supplies source-order copy/protocol representation, dtype/error resolution
and warnings/policy delivery. All implemented projection arithmetic is C++.
"""
import importlib
import numpy as np
from . import _core


def _exceptions():
    return importlib.import_module('numpy.core._exceptions'if np.__version__.split('.')[0]=='1'else'numpy._core._exceptions')


def _reduction_resolution(dtype):
    if dtype.kind in 'SU':
        if np.__version__.split('.')[0]=='1':raise _exceptions()._UFuncNoLoopError(np.add,(dtype,dtype,None))
        output=np.dtype(dtype.kind+str(dtype.itemsize*(2 if dtype.kind=='S'else 1)//(1 if dtype.kind=='S'else 2)))
        raise TypeError(f'the resolved dtypes are not compatible with add.reduce. Resolved ({dtype!r}, {dtype!r}, {output!r})')
    if dtype.kind=='M':raise _exceptions()._UFuncBinaryResolutionError(np.add,(dtype,dtype))
    if dtype.fields is not None or dtype.kind=='V':raise NotImplementedError('finite Radon structured reduction frontier')


def frt2(a):
    if a.ndim!=2 or a.shape[0]!=a.shape[1]:raise ValueError('Input must be a square, 2-D array')
    ai=a.copy()
    if not isinstance(ai,np.ndarray):raise NotImplementedError('finite Radon non-array copy return frontier')
    _reduction_resolution(ai.dtype)
    return _core.frt2(ai)


def ifrt2(a):
    if a.ndim!=2 or a.shape[0]!=a.shape[1]+1:raise ValueError('Input must be an (n+1) row x n column, 2-D array')
    ai=a.copy()
    if not isinstance(ai,np.ndarray):raise NotImplementedError('finite Radon non-array copy return frontier')
    _reduction_resolution(ai.dtype)
    if ai.shape[1]==0:raise IndexError('index 0 is out of bounds for axis 0 with size 0')
    if ai.dtype.kind not in 'bu':
        if ai.dtype.kind in 'fcOm':_core._finite_radon_inverse_prefix(ai)
        result_dtype=ai.dtype if ai.dtype.kind=='m'else np.result_type(np.dtype('uint32'),ai.dtype)
        raise _exceptions()._UFuncOutputCastingError(np.add,'same_kind',result_dtype,np.dtype('uint32'),2)
    bits=max(np.dtype(int).itemsize*8,ai.dtype.itemsize*8)
    return _core.ifrt2(ai,bits,np.__version__.split('.')[0]=='1',ai.dtype.kind=='b')
