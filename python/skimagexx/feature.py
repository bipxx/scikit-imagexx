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
