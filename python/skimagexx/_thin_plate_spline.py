"""Planar thin-plate spline fitting and application through the C++ kernels."""

import inspect

import numpy as np

from . import _core
from ._compat import fallback
from ._geometric import FailedEstimation, _PinnedFailureAnnotation, _deprecate_estimate


def _coordinates(value):
    return (type(value) is np.ndarray and value.ndim == 2 and value.shape[1] == 2
            and value.dtype.kind in 'biuf' and value.dtype.itemsize <= 8
            and value.dtype.metadata is None)


def _contiguous(value):
    return np.require(value, dtype=np.float64, requirements=('C', 'A'))


def _default_errors():
    return np.geterr() == dict(divide='warn', over='warn', under='ignore', invalid='warn')


class ThinPlateSplineTransform:
    """Estimate and apply a planar thin-plate spline from matching controls."""

    def __init__(self):
        self._estimated = False
        self._spline_mappings = None
        self.src = None

    @property
    def inverse(self):
        raise NotImplementedError('Not supported')

    @classmethod
    def from_estimate(cls, src, dst):
        model = cls()
        message = model._estimate(src, dst)
        return model if message is None else FailedEstimation(f'{cls.__name__}: {message}')

    @_deprecate_estimate
    def estimate(self, src, dst):
        return self._estimate(src, dst) is None

    def _estimate(self, src, dst):
        for name, values in (('src', src), ('dst', dst)):
            array = np.asanyarray(values)
            if array.size == 0:
                raise ValueError(f'The parameter `{name}` cannot be an empty array')
            if array.ndim != 2:
                raise ValueError(f'The parameter `{name}` must be a 2-dimensional array')
        if src.shape[0] < 3 or dst.shape[0] < 3:
            raise ValueError('Need at least 3 points in in `src` and `dst`')
        if src.shape != dst.shape:
            raise ValueError(f"Shape of `src` and `dst` didn't match, {src.shape} != {dst.shape}")
        if (_coordinates(src) and _coordinates(dst) and _default_errors() and not any(
                getattr(type(self), name) is not getattr(ThinPlateSplineTransform, name)
                for name in ('_radial_basis_kernel', '_radial_distance', '_spline_function'))):
            mappings = _core.thin_plate_spline_fit(_contiguous(src), _contiguous(dst))
            if mappings is not None:
                self.src = src
                self._spline_mappings = mappings
                return None
        stock = fallback('transform', 'ThinPlateSplineTransform')
        return type(stock)._estimate(self, src, dst)

    def __call__(self, coords):
        if self._spline_mappings is None:
            raise ValueError('Transformation is undefined, define it by calling `estimate` before applying it')
        values = np.array(coords)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError('Input `coords` must have shape (N, 2)')
        if any(getattr(type(self), name) is not getattr(ThinPlateSplineTransform, name)
               for name in ('_radial_basis_kernel', '_radial_distance', '_spline_function')):
            return self._spline_function(values, self._radial_distance(values))
        if (_coordinates(values) and _coordinates(self.src) and _coordinates(self._spline_mappings)
                and self._spline_mappings.dtype == np.dtype(np.float64) and _default_errors()):
            result = _core.thin_plate_spline_apply(
                _contiguous(self.src), _contiguous(self._spline_mappings), _contiguous(values))
            if result is not None:
                return result
        stock = fallback('transform', 'ThinPlateSplineTransform')
        return type(stock).__call__(self, values)

    def _radial_distance(self, coords):
        from scipy.spatial import distance_matrix

        return self._radial_basis_kernel(distance_matrix(coords, self.src))

    def _spline_function(self, coords, radial_dist):
        count = self.src.shape[0]
        w, a = self._spline_mappings[:count], self._spline_mappings[count:]
        return a[0] + np.dot(coords, a[1:]) + np.dot(radial_dist, w)

    @staticmethod
    def _radial_basis_kernel(r):
        square = r**2
        return np.where(r == 0, 0, square * np.log(square + 1e-8))


_from_estimate = ThinPlateSplineTransform.__dict__['from_estimate'].__func__
_from_estimate.__signature__ = inspect.signature(_from_estimate).replace(
    return_annotation=_PinnedFailureAnnotation())
del _from_estimate
