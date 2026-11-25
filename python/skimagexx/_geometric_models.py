"""Native circle and ellipse kernels with scikit-image model contracts."""

from functools import wraps
from warnings import warn

import numpy as np
from skimage.measure import fit as _reference

from . import _core
from ._compat import fallback


RansacModelProtocol = _reference.RansacModelProtocol

# Some NumPy builds return complex arrays even for a real eigensystem.
# The upstream ellipse estimator applies a real-only remainder to the angle.
_COMPLEX_REAL_EIGENVECTORS = np.linalg.eig(np.eye(1))[1].dtype.kind == 'c'


def _method(model, name, *args):
    if isinstance(model, CircleModel):
        reference = fallback('measure.fit', 'CircleModel', (0, 0), 1)
    else:
        reference = fallback('measure.fit', 'EllipseModel', (0, 0), (1, 1), 0)
    return getattr(type(reference), name)(model, *args)


def _pair(value):
    if (
        type(value) is np.ndarray and value.shape == (2,)
        and value.dtype.kind in 'ifu' and value.dtype.metadata is None
        and value.dtype.itemsize <= 8
        and (value.dtype.kind != 'f' or value.dtype == np.float64)
    ):
        return np.ascontiguousarray(value, np.float64)
    return None


def _fit(model, data, warn_only, ellipse):
    if type(data) is np.ndarray:
        result = _core.measure_geometric_fit(data, ellipse)
        if result is not None:
            if ellipse and _COMPLEX_REAL_EIGENVECTORS:
                raise TypeError("unsupported operand type(s) for %=: 'numpy.complex128' and 'float'")
            model.center = np.array(result[0])
            if ellipse:
                model.axis_lengths = np.array(result[1])
                model.theta = np.float64(result[2])
            else:
                model.radius = np.float64(result[1])
            return None
    return _method(model, '_estimate', data, warn_only)


def _residuals(model, data, ellipse):
    if type(data) is np.ndarray:
        center = _pair(model.center)
        axes = _pair(model.axis_lengths) if ellipse else None
        scalar = model.theta if ellipse else model.radius
        if center is not None and (not ellipse or axes is not None) and type(scalar) in (int, float, np.float64):
            output = _core.measure_geometric_residuals(
                data, center, 0 if ellipse else scalar, axes, scalar if ellipse else 0,
            )
            if output is not None:
                return output
    return _method(model, 'residuals', data)


def _predict(model, t, params, ellipse):
    if params is _reference.DEPRECATED and model.center is not None:
        center = _pair(model.center)
        axes = _pair(model.axis_lengths) if ellipse else None
        scalar = model.theta if ellipse else model.radius
        if center is not None and (not ellipse or axes is not None) and type(scalar) in (int, float, np.float64):
            angles = np.asanyarray(t)
            if type(angles) is np.ndarray and angles.dtype == np.float64 and angles.dtype.metadata is None:
                cosine = np.asarray(np.cos(angles))
                sine = np.asarray(np.sin(angles))
                output = _core.measure_geometric_predict(
                    cosine, sine, center, 0 if ellipse else scalar, axes, scalar if ellipse else 0,
                )
                if output is not None:
                    return output
    # Calling the original decorated method preserves the deprecation
    # warning and the legacy precedence of explicit parameters.
    return _method(model, 'predict_xy', t, params) if params is not _reference.DEPRECATED else _method(
        model, 'predict_xy', t,
    )


class CircleModel(_reference.CircleModel):
    """Total least squares circle model."""

    @_reference._deprecate_estimate
    def estimate(self, data):
        """Estimate a circle from Cartesian points."""
        return self._estimate(data) is None

    @wraps(_reference.CircleModel._estimate)
    def _estimate(self, data, warn_only=True):
        return _fit(self, data, warn_only, False)

    @wraps(_reference.CircleModel.residuals)
    def residuals(self, data):
        return _residuals(self, data, False)

    @wraps(_reference.CircleModel.predict_xy)
    def predict_xy(self, t, params=_reference.DEPRECATED):
        return _predict(self, t, params, False)


class EllipseModel(_reference.EllipseModel):
    """Total least squares ellipse model."""

    @_reference._deprecate_estimate
    def estimate(self, data):
        """Estimate an ellipse from Cartesian points."""
        return self._estimate(data) is None

    @wraps(_reference.EllipseModel._estimate)
    def _estimate(self, data, warn_only=True):
        return _fit(self, data, warn_only, True)

    @wraps(_reference.EllipseModel.residuals)
    def residuals(self, data):
        return _residuals(self, data, True)

    @wraps(_reference.EllipseModel.predict_xy)
    def predict_xy(self, t, params=_reference.DEPRECATED):
        return _predict(self, t, params, True)


def ransac(
    data,
    model_class,
    min_samples,
    residual_threshold,
    is_data_valid=None,
    is_model_valid=None,
    max_trials=100,
    stop_sample_num=np.inf,
    stop_residuals_sum=0,
    stop_probability=1,
    rng=None,
    initial_inliers=None,
):
    """Robustly fit a model using randomly sampled subsets of the data."""
    arrays = data if isinstance(data, (tuple, list)) else (data,)
    if (
        type(arrays) not in (tuple, list)
        or not arrays
        or any(type(array) is not np.ndarray for array in arrays)
        or type(min_samples) is not int
        or type(max_trials) not in (int, float)
        or abs(max_trials) > 2**53
        or any(type(value) not in (int, float, np.float64) for value in
               (residual_threshold, stop_sample_num, stop_residuals_sum, stop_probability))
    ):
        return fallback(
            'measure', 'ransac', data, model_class, min_samples, residual_threshold,
            is_data_valid, is_model_valid, max_trials, stop_sample_num, stop_residuals_sum,
            stop_probability, rng, initial_inliers,
        )
    generator = np.random.default_rng(rng)
    samples = len(arrays[0])
    if not 0 < min_samples <= samples:
        raise ValueError(f'`min_samples` must be in range (0, {samples}]')
    if residual_threshold < 0:
        raise ValueError('`residual_threshold` must be greater than zero')
    if max_trials < 0:
        raise ValueError('`max_trials` must be greater than zero')
    if not 0 <= stop_probability <= 1:
        raise ValueError('`stop_probability` must be in range [0, 1]')
    if initial_inliers is not None and len(initial_inliers) != samples:
        raise ValueError(
            f'RANSAC received a vector of initial inliers (length {len(initial_inliers)}) '
            f"that didn't match the number of samples ({samples}). The vector of initial inliers "
            "should have the same length as the number of samples and contain only True "
            "(this sample is an initial inlier) and False (this one isn't) values."
        )
    indices = (
        initial_inliers if initial_inliers is not None
        else generator.choice(samples, min_samples, replace=False)
    )
    model_class = _reference.add_from_estimate(model_class)
    if not isinstance(model_class, RansacModelProtocol):
        raise TypeError(
            f'`model_class` {model_class} should be of (protocol) type RansacModelProtocol'
        )
    return _core.measure_ransac_loop(
        arrays, model_class, generator, indices, samples, min_samples, residual_threshold,
        max_trials, stop_sample_num, stop_residuals_sum, stop_probability,
        is_data_valid, is_model_valid, warn,
    )
