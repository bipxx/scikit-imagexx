"""Polynomial evaluation and total-least-squares assembly in C++."""

import numpy as np
from skimage.transform import PolynomialTransform as _Polynomial

from . import _core
from ._compat import fallback


def _reference(model):
    reference = fallback("transform._geometric", "PolynomialTransform")
    reference.__dict__.update(model.__dict__)
    return reference


class PolynomialTransform(_Polynomial):
    def __call__(self, coords):
        params = self.params
        if type(coords) is np.ndarray and type(params) is np.ndarray:
            result = _core.polynomial_apply(coords, params)
            if result is not None:
                return result
            if params.dtype.kind in "iu" and params.dtype.metadata is None:
                result = _core.polynomial_apply(coords, params.astype(np.float64))
                if result is not None:
                    return result
        return _reference(self)(coords)

    def _estimate(self, src, dst, order=2, weights=None):
        if type(order) is int and 0 <= order <= 5:
            src = np.asarray(src)
            dst = np.asarray(dst)
            weights = None if weights is None else np.asarray(weights)
            equations = _core.polynomial_constraints(src, dst, order, weights)
            if equations is not None:
                rows, columns = equations.shape
                _, singular, vt = np.linalg.svd(equations, full_matrices=rows < columns)
                if (singular[-(1 if rows < columns else 2)] > singular[0] * 1e-10
                        and abs(vt[-1, -1]) > 1e-10):
                    self.params = (-vt[-1, :-1] / vt[-1, -1]).reshape(2, -1)
                    return None
        reference = _reference(self)
        try:
            return reference._estimate(src, dst, order, weights)
        finally:
            self.params = reference.params
