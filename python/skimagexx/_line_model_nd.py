"""LineModelND lifecycle and bounded native fit with NumPy solver providers.

BSD source attribution: third_party/scikit-image-LICENSE.txt.
"""

import inspect
import warnings

import numpy as np

from . import _core
from ._component_consumers import _DEPRECATED as DEPRECATED
from ._geometric import FailedEstimation as _StandaloneFailure
from ._geometric import _deprecate_estimate


_FLOAT64 = np.dtype(np.float64)


def _check_rank(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError("Input data must be at least 2D.")


def _init_values(origin, direction):
    origin, direction = np.array(origin), np.array(direction)
    if len(origin) != len(direction):
        raise ValueError("Direction vector should be same length as origin point.")
    return origin, direction


def _warn_params(params, method):
    if params is not DEPRECATED:
        warnings.warn(
            "Parameter `params` is deprecated since version 0.26 and will be removed "
            "in 2.2 (or later). To avoid this warning, please do not use the parameter "
            f"`params`. For more details, see the documentation of `LineModelND.{method}`.",
            FutureWarning, stacklevel=3,
        )


def _failure(message):
    try:
        from skimage._shared.utils import FailedEstimation
    except ModuleNotFoundError as error:
        if error.name not in ("skimage", "skimage._shared", "skimage._shared.utils"):
            raise
        FailedEstimation = _StandaloneFailure
    return FailedEstimation(message)


class LineModelND:
    """Total least squares line estimator in two or more dimensions."""

    def _args_init(self, origin, direction):
        self.origin, self.direction = self._check_init_values(origin, direction)

    def _check_init_values(self, origin, direction):
        return _init_values(origin, direction)

    def _params2init_values(self, params):
        if len(params) != 2:
            raise ValueError("Input `params` should be length 2")
        return self._check_init_values(*params)

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            self._args_init(*args, **kwargs)
        else:
            warnings.warn(
                "Calling ``LineModelND()`` (without arguments) has been deprecated "
                "since version 0.26 and will be removed in version 2.2; see help for "
                "``LineModelND``.",
                FutureWarning, stacklevel=2,
            )
            self.origin = self.direction = None

    __init__.__signature__ = inspect.signature(_args_init)

    @property
    def params(self):
        warnings.warn(
            "`params` is deprecated since version 0.26 and will be removed in version 2.2. "
            "`params` attribute deprecated; use ``origin, direction`` attributes instead.",
            FutureWarning, stacklevel=2,
        )
        return self.origin, self.direction

    @classmethod
    def from_estimate(cls, data):
        with warnings.catch_warnings(action="ignore"):
            model = cls()
        failure = model._estimate(data, warn_only=False)
        return model if failure is None else _failure(f"{cls.__name__}: {failure}")

    @_deprecate_estimate
    def estimate(self, data):
        return self._estimate(data) is None

    def _estimate(self, data, warn_only=True):
        _check_rank(data)
        if type(data) is np.ndarray and data.ndim == 2 and data.shape[0] == 2:
            result = _core.line_model_nd_two_point(data)
            if result is not None:
                self.origin, self.direction = result
                return None
        origin = data.mean(axis=0)
        centered = data - origin
        if centered.shape[0] == 2:
            direction = centered[1] - centered[0]
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
        elif centered.shape[0] > 2:
            _, _, vectors = np.linalg.svd(centered, full_matrices=False)
            direction = vectors[0]
        else:
            return "estimate under-determined"
        self.origin, self.direction = origin, direction
        return None

    def _get_init_values(self, params):
        if params is not None and params is not DEPRECATED:
            return self._params2init_values(params)
        if self.origin is None:
            name = type(self).__name__
            raise ValueError(
                "`params` argument must be specified when "
                f"applied to model initialized with ``{name}()``; Consider creating new "
                f"{name} with suitable input arguments, or by using ``{name}.from_estimate``."
            )
        return self.origin, self.direction

    def residuals(self, data, params=DEPRECATED):
        if params is not DEPRECATED:
            _warn_params(params, "residuals")
        _check_rank(data)
        origin, direction = self._get_init_values(params)
        if len(origin) != data.shape[1]:
            raise ValueError(f"`origin` is {len(origin)}D, but `data` is {data.shape[1]}D")
        # Keep both source subtractions for protocols and for any subtraction
        # which must deliver NumPy floating-point warnings or callbacks.
        if (type(data) is type(origin) is type(direction) is np.ndarray
                and data.ndim == 2 and origin.ndim == direction.ndim == 1
                and data.dtype is origin.dtype is direction.dtype is _FLOAT64):
            try:
                with np.errstate(all="raise"):
                    centered = data - origin
            except FloatingPointError:
                pass
            else:
                projection = (centered @ direction)[:, np.newaxis] * direction
                np.subtract(centered, projection, out=centered)
                np.multiply(centered, centered, out=centered)
                return np.sqrt(np.add.reduce(centered, axis=1))
        res = (data - origin) - ((data - origin) @ direction)[..., np.newaxis] * direction
        return np.linalg.norm(res, axis=1)

    def predict(self, x, axis=0, params=DEPRECATED):
        if params is not DEPRECATED:
            _warn_params(params, "predict")
        origin, direction = self._get_init_values(params)
        if (type(origin) is np.ndarray and origin.ndim == 1 and
                type(x) is np.ndarray and x.ndim == 1 and 128 <= x.size <= 65536 and
                type(axis) is int):
            dimensions = origin.size
            if dimensions == 2 and -2 <= axis < 2:
                result = _core.line_model_nd_predict_2d(x, origin, direction, axis % 2)
                if result is not None:
                    return result
            elif dimensions == 3 and -3 <= axis < 3:
                result = _core.line_model_nd_predict_3d(x, origin, direction, axis % 3)
                if result is not None:
                    return result
        if direction[axis] == 0:
            raise ValueError(f"Line parallel to axis {axis}")
        length = (x - origin[axis]) / direction[axis]
        return origin + length[..., np.newaxis] * direction

    def predict_x(self, y, params=DEPRECATED):
        model = self
        if params is not DEPRECATED:
            _warn_params(params, "predict_x")
            if params is not None:
                model = type(self)(*self._params2init_values(params))
        return model.predict(y, axis=1)[:, 0]

    def predict_y(self, x, params=DEPRECATED):
        model = self
        if params is not DEPRECATED:
            _warn_params(params, "predict_y")
            if params is not None:
                model = type(self)(*self._params2init_values(params))
        return model.predict(x, axis=0)[:, 1]
