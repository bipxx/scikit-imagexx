"""Owned padding/min-plus scan, with explicit NumPy boundary policy.

Source order/API follows scikit-image0.26, BSD-3-Clause; retained notice
third_party/scikit-image-LICENSE.txt. No original implementation import.
"""
from functools import wraps
import operator
import warnings
import numpy as np
from . import _core
from ._rolling_ball_kernels import _ordinary_scalar, _MAX_RANK


class _Deprecated:
    def __repr__(self):
        return '<DEPRECATED>'


DEPRECATED = _Deprecated()


def _ball_kernel(radius, ndim, worker_snapshot):
    # Reuse the admitted owned primitive, never its NumPy algorithm fallback.
    # Axis generation/power remain literal argument/promotion providers.
    axes = [np.arange(-x, x + 1) for x in [np.ceil(radius)] * ndim]
    if not 1 <= len(axes) <= _MAX_RANK or not _ordinary_scalar(radius):
        raise NotImplementedError('rolling_ball radius/rank outside owned kernel domain')
    prepared = tuple(np.require(axis, dtype=np.float64, requirements=('C', 'A'))
                     for axis in axes)
    output = np.empty(tuple(axis.size for axis in prepared), dtype=np.float64)
    radius_squared = radius**2
    if not _core.ball_kernel_apply(prepared, float(radius), float(radius_squared),
                                   output, worker_snapshot):
        raise NotImplementedError('rolling_ball owned kernel refused arithmetic environment')
    return output


def _thread_deprecation(function):
    @wraps(function)
    def call(*args, **kwargs):
        previous = kwargs.get('num_threads', DEPRECATED)
        if previous is not DEPRECATED:
            warnings.warn('Parameter \x60num_threads\x60 is deprecated since version 0.26 and will be removed in 0.28 (or later). '
                          'To avoid this warning, please use the parameter \x60workers\x60 instead. For more details, see the documentation of \x60rolling_ball\x60.',
                          FutureWarning, stacklevel=2)
            if 'workers' in kwargs:
                raise ValueError('Both deprecated parameter \x60num_threads\x60 and new parameter \x60workers\x60 are used. '
                                 'Use only the latter to avoid conflicting values.')
            kwargs['workers'] = previous
            kwargs['num_threads'] = DEPRECATED
        return function(*args, **kwargs)
    return call


@_thread_deprecation
def rolling_ball(image, *, radius=100, kernel=None, nansafe=False,
                 num_threads=DEPRECATED, workers=None):
    image = np.asarray(image)
    if image.dtype.kind == 'c':
        raise ValueError('complex valued input is not supported')
    float_type = {'f': np.float32, 'd': np.float64, 'e': np.float32}.get(image.dtype.char, np.float64)
    img = image.astype(float_type, copy=False)
    if workers is None:
        workers = 0
    worker_snapshot = _core.get_num_threads()
    if kernel is None:
        kernel = _ball_kernel(radius, image.ndim, worker_snapshot)
    # User .astype protocol, NumPy scalar promotion/error/warning policy and
    # exceptional subtraction are providers, never an upstream image algorithm.
    kernel = kernel.astype(float_type)
    kernel_shape = np.asarray(kernel.shape)
    kernel_center = kernel_shape // 2
    center_intensity = kernel[tuple(kernel_center)]
    intensity_difference = center_intensity - kernel
    intensity_difference[kernel == np.inf] = np.inf
    intensity_difference = intensity_difference.astype(img.dtype).reshape(-1)
    # Source output allocation/base/reshape semantics, without padding provider.
    output = np.zeros_like(image, dtype=img.dtype).reshape(-1)
    owner = output.base
    if owner is None:
        owner = output
    safe = bool(nansafe)
    workers = operator.index(workers)
    _core.rolling_ball_apply(img, intensity_difference, kernel_shape.tolist(),
                             owner, safe, worker_snapshot if workers == 0 else workers)
    # Cython out.base is the supplied flat ndarray, not ndarray.base.
    background = output.reshape(image.shape)
    return background.astype(image.dtype, copy=False)
