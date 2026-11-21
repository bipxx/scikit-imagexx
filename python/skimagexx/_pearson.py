"""Bounded float64 Pearson correlation with a scalar SciPy beta-tail provider."""

import numpy as np

from . import _core
from ._compat import fallback


_F64 = np.dtype("float64")
_BOOL = np.dtype("bool")


def _bounded_c_array(array, dtype):
    if (type(array) is not np.ndarray or array.dtype != dtype or array.dtype.metadata is not None
            or not array.flags.c_contiguous or not array.flags.aligned):
        return False
    base = array.base
    return (base is None or type(base) is bytes
            or type(base) is np.ndarray and base.flags.owndata
            and (base.flags.c_contiguous or base.flags.f_contiguous))


def _beta_tail(shape, x):
    try:
        from scipy.special import betaincc
    except ImportError:
        return None
    return betaincc(shape, shape, x)


def pearson_corr_coeff(image0, image1, mask=None):
    """Return Pearson's correlation and two-sided p-value as Python floats.

    Bounded, aligned float64 images and an optional matching Boolean mask use
    the ordered native correlation core. Two distinct selected pairs give an
    exact p-value of one. Larger supported samples use one scalar SciPy beta
    tail. The configured scikit-image fallback handles other calls.
    """
    if (_bounded_c_array(image0, _F64) and _bounded_c_array(image1, _F64)
            and image0.ndim in (1, 2) and image0.shape == image1.shape
            and 2 <= image0.size <= 1 << 20
            and (mask is None or _bounded_c_array(mask, _BOOL) and mask.shape == image0.shape)):
        result = _core.pearson_f64(image0, image1, mask, _beta_tail)
        if result is not None:
            return result
    return fallback("measure", "pearson_corr_coeff", image0, image1, mask=mask)
