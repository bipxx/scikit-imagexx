"""Epipolar transforms with native evaluation and LAPACK singular vectors."""

import numpy as np
from skimage.transform import (
    EssentialMatrixTransform as _Essential,
    FundamentalMatrixTransform as _Fundamental,
)

from . import _core
from ._compat import fallback


def _reference(model, name):
    cls = _Essential if isinstance(model, _Essential) else _Fundamental
    reference = fallback("transform._geometric", cls.__name__)
    reference.__dict__.update(model.__dict__)
    return getattr(reference, name), reference


def _estimate(model, src, dst, essential):
    normalized, src_matrix, dst_matrix = model._setup_constraint_matrix(src, dst)
    if np.isnan(normalized).any():
        return "Scaling failed for input points"

    u, values, vt = np.linalg.svd(normalized)
    model.params = _core.epipolar_matrix(u, values, vt, src_matrix, dst_matrix, essential)
    return None


class FundamentalMatrixTransform(_Fundamental):
    def __call__(self, coords):
        if type(coords) is np.ndarray and type(self.params) is np.ndarray:
            result = _core.epipolar_lines(coords, self.params)
            if result is not None:
                return result
        method, _ = _reference(self, "__call__")
        return method(coords)

    def residuals(self, src, dst):
        if type(src) is np.ndarray and type(dst) is np.ndarray and type(self.params) is np.ndarray:
            result = _core.epipolar_residuals(src, dst, self.params)
            if result is not None:
                return result
        method, _ = _reference(self, "residuals")
        return method(src, dst)

    def _setup_constraint_matrix(self, src, dst):
        scaling = self.scaling
        modes = {"rms": 0, "mrs": 1, "raw": 2}
        if type(scaling) is str and scaling.lower() in modes:
            src = np.asarray(src)
            dst = np.asarray(dst)
            result = _core.epipolar_constraints(src, dst, modes[scaling.lower()])
            if result is not None:
                equations, src_matrix, dst_matrix = result
                _, singular, vt = np.linalg.svd(equations, full_matrices=equations.shape[0] < 9)
                if singular[-(1 if len(singular) == 8 else 2)] > singular[0] * 1e-10:
                    return vt[-1].reshape(3, 3), src_matrix, dst_matrix
        method, reference = _reference(self, "_setup_constraint_matrix")
        try:
            return method(src, dst)
        finally:
            self.params = reference.params

    def _estimate(self, src, dst):
        return _estimate(self, src, dst, False)


class EssentialMatrixTransform(_Essential, FundamentalMatrixTransform):
    def _estimate(self, src, dst):
        return _estimate(self, src, dst, True)
