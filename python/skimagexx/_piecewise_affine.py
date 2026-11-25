"""Native planar piecewise affine transforms with SciPy simplex ordering."""

from copy import copy

import numpy as np

from . import _core
from ._compat import fallback
from ._geometric import AffineTransform, _GeometricTransform, _deprecate_estimate


def _coordinates(value):
    return (value.ndim == 2 and value.shape[1] == 2 and value.dtype.kind in 'biuf'
            and value.dtype.itemsize <= 8 and value.dtype.metadata is None)


def _contiguous(value):
    return np.require(value, dtype=np.float64, requirements=('C', 'A'))


def _default_errors():
    return np.geterr() == dict(divide='warn', over='warn', under='ignore', invalid='warn')


class PiecewiseAffineTransform(_GeometricTransform):
    """Piecewise affine coordinate transform based on Delaunay simplices."""

    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None
        self._native_matrices = None
        self._inverse_native_matrices = None

    @classmethod
    def identity(cls, dimensionality=None):
        return cls()

    @classmethod
    def from_estimate(cls, src, dst):
        return super().from_estimate(src, dst)

    @_deprecate_estimate
    def estimate(self, src, dst):
        return self._estimate(src, dst) is None

    def _estimate(self, src, dst):
        source, target = np.asarray(src), np.asarray(dst)
        if (_coordinates(source) and _coordinates(target) and source.shape == target.shape
                and source.dtype != np.float16 and target.dtype != np.float16 and _default_errors()
                and 3 <= source.shape[0] <= 256):
            # Public affine lists expose SciPy/Qhull's triangulation and ordering.
            from scipy.spatial import Delaunay, QhullError

            a, b = _contiguous(source), _contiguous(target)
            try:
                forward = Delaunay(a)
                backward = Delaunay(b)
            except QhullError:
                # Upstream may leave the already fitted forward state installed.
                return self._fallback_estimate(source, target)
            matrices = _core.piecewise_affine_fit(a, b, forward.simplices)
            inverse_matrices = _core.piecewise_affine_fit(b, a, backward.simplices)
            if matrices is not None and inverse_matrices is not None:
                self._tesselation, self._inverse_tesselation = forward, backward
                self.affines = [AffineTransform(matrix.copy()) for matrix in matrices]
                self.inverse_affines = [AffineTransform(matrix.copy()) for matrix in inverse_matrices]
                self._native_matrices, self._inverse_native_matrices = matrices, inverse_matrices
                return None
        return self._fallback_estimate(source, target)

    def _fallback_estimate(self, src, dst):
        stock = fallback('transform', 'PiecewiseAffineTransform')
        return type(stock)._estimate(self, src, dst)

    def __call__(self, coords):
        values = np.asarray(coords)
        if self._tesselation is None:
            # Retain the undefined identity model's original exception.
            raise AttributeError("'NoneType' object has no attribute 'find_simplex'")
        if (_coordinates(values) and _default_errors() and type(self.affines) is list
                and self.affines and all(type(affine) is AffineTransform
                                         for affine in self.affines)):
            matrices = np.asarray([affine.params for affine in self.affines])
            # Edited affines can disagree on shared edges. Preserve Qhull's
            # point-by-point tie selection for that observable mutable state.
            if matrices.dtype == np.float64 and np.array_equal(matrices, self._native_matrices):
                result = _core.piecewise_affine_apply(self._tesselation.transform, matrices,
                                                       _contiguous(values))
                if result is not None:
                    if values.flags.c_contiguous:
                        return result
                    output = np.empty_like(values, dtype=np.float64)
                    output[:] = result
                    return output
        stock = fallback('transform', 'PiecewiseAffineTransform')
        return type(stock).__call__(self, values)

    @property
    def inverse(self):
        model = type(self)()
        model._tesselation = copy(self._inverse_tesselation)
        model._inverse_tesselation = copy(self._tesselation)
        model.affines = copy(self.inverse_affines)
        model.inverse_affines = copy(self.affines)
        model._native_matrices = self._inverse_native_matrices
        model._inverse_native_matrices = self._native_matrices
        return model
