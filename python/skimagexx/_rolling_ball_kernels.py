"""Rolling-ball geometry with literal argument providers and owned arithmetic."""
import numpy as np
from . import _core

_MAX_RANK = min(32, (np._core if hasattr(np, '_core') else np.core).multiarray.MAXDIMS - 1)
_SCALARS = (int, bool, float, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64)


def _ordinary_scalar(value):
    return type(value) in _SCALARS and (
        type(value) in (float, np.float32, np.float64) or -(1 << 53) <= value <= (1 << 53))


def ball_kernel(radius, ndim):
    # scikit-image0.26 _rolling_ball.py, BSD-3-Clause; retained license in
    # third_party/scikit-image-LICENSE.txt. ceil precedes the ndim protocol.
    axes = [np.arange(-x, x + 1) for x in [np.ceil(radius)] * ndim]
    native_result = False
    radius_squared = None
    if 1 <= len(axes) <= _MAX_RANK and _ordinary_scalar(radius):
        prepared = tuple(np.require(axis, dtype=np.float64, requirements=('C', 'A'))
                         for axis in axes)
        # Keep scalar power as a provider: NumPy scalar promotion differs
        # across the supported profiles. Refusal must not evaluate it again.
        radius_squared = radius**2
        output = np.empty(tuple(axis.size for axis in prepared), dtype=np.float64)
        native_result = _core.ball_kernel_apply(
            prepared, float(radius), float(radius_squared), output, _core.get_num_threads())
        if native_result:
            return output
    kernel_coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    sum_of_squares = np.sum(kernel_coords**2, axis=-1)
    distance_from_center = np.sqrt(sum_of_squares)
    if radius_squared is None:
        radius_squared = radius**2
    kernel = np.sqrt(np.clip(radius_squared - sum_of_squares, 0, None))
    kernel[distance_from_center > radius] = np.inf
    return kernel


def ellipsoid_kernel(shape, intensity):
    # Preserve asarray/floor/clip and axis generation before intensity events.
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None)
    axes = [np.arange(-x, x + 1) for x in semi_axis]
    native_result = False
    if (1 <= len(axes) <= _MAX_RANK and _ordinary_scalar(intensity)
            and semi_axis.dtype.kind in 'iuf' and semi_axis.dtype.itemsize <= 8):
        prepared = tuple(np.require(axis, dtype=np.float64, requirements=('C', 'A'))
                         for axis in axes)
        prepared_semi = np.require(semi_axis, dtype=np.float64, requirements=('C', 'A'))
        output = np.empty(tuple(axis.size for axis in prepared), dtype=np.float64)
        native_result = _core.ellipsoid_kernel_apply(
            prepared, prepared_semi, float(intensity), output, _core.get_num_threads())
        if native_result:
            return output
    kernel_coords = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    intensity_scaling = 1 - np.sum((kernel_coords / semi_axis) ** 2, axis=-1)
    kernel = intensity * np.sqrt(np.clip(intensity_scaling, 0, None))
    kernel[intensity_scaling < 0] = np.inf
    return kernel
