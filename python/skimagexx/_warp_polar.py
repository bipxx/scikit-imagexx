"""Polar geometry metadata for the native warp sampler.

The geometry follows scikit-image 0.26.0. See
third_party/scikit-image-LICENSE.txt.
"""

from functools import wraps
import math

import numpy as np

from ._compat import fallback
from ._warp import _apply, _image_ok, _options, _shape, _warn_quadratic


def _axis(channel_axis):
    if type(channel_axis) is tuple and len(channel_axis) == 1:
        channel_axis = channel_axis[0]
    if type(channel_axis) is int or isinstance(channel_axis, np.signedinteger):
        return int(channel_axis)
    return None


def _with_fallback(native):
    @wraps(native)
    def dispatch(*args, **kwargs):
        channel_axis = kwargs.get('channel_axis')
        # Upstream moves only positional images before calling the transform.
        if args or channel_axis is None or _axis(channel_axis) in (-1, 2):
            output = native(*args, **kwargs)
            if output is not None:
                return output
        return fallback('transform', 'warp_polar', *args, **kwargs)

    return dispatch


def _finite_scalar(value):
    if type(value) is int:
        return value.bit_length() <= 53
    return type(value) in (float, np.float64) and math.isfinite(value)


@_with_fallback
def warp_polar(image, center=None, *, radius=None, output_shape=None,
               scaling='linear', channel_axis=None, **kwargs):
    """Remap an image to linear or logarithmic polar coordinates.

    The center is given in (row, column) order and defaults to the image
    midpoint. The default radius reaches the corners. Output rows span angles
    over a full turn, and output columns sample radii. Set ``channel_axis`` for
    a three-dimensional image. Extra keyword arguments are passed to ``warp``.
    """
    if type(image) is not np.ndarray:
        return None
    axis = _axis(channel_axis)
    if channel_axis is None:
        if image.ndim != 2:
            raise ValueError('Input array must be 2-dimensional when '
                             f'`channel_axis=None`, got {image.ndim}')
    elif axis == -1 and image.ndim != 3:
        raise ValueError('Input array must be 3-dimensional when '
                         f'`channel_axis` is specified, got {image.ndim}')
    elif axis in (-1, 2, -2, 1, -3, 0) and image.ndim == 3:
        if axis not in (-1, 2):
            image = np.moveaxis(image, axis, -1)
    else:
        return None

    if not _image_ok(image) or type(scaling) is not str or not (
        kwargs.keys() <= {'order', 'mode', 'cval', 'clip', 'preserve_range'}
    ):
        return None
    rows, cols = image.shape[:2]
    if center is None:
        center = (rows / 2 - 0.5, cols / 2 - 0.5)
    elif type(center) not in (tuple, list) or len(center) != 2 or not all(
        _finite_scalar(value) for value in center
    ):
        return None
    if radius is None:
        radius = np.sqrt((rows / 2) ** 2 + (cols / 2) ** 2)
    elif not _finite_scalar(radius) or radius <= 0:
        return None
    if output_shape is None:
        output_shape = (360, math.ceil(radius))
    shape = _shape(output_shape)
    if shape is None:
        return None
    if len(shape) == 3:
        if image.ndim != 3 or shape[2] != image.shape[2]:
            return None
        shape = shape[:2]
    if scaling == 'linear':
        divisor = float(radius)
    elif scaling == 'log':
        divisor = float(np.log(radius))
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")
    height, width = shape
    if not divisor:
        return None
    k_radius = width / divisor
    if not math.isfinite(k_radius) or not k_radius:
        return None

    order = kwargs.get('order')
    mode = kwargs.get('mode', 'constant')
    cval = kwargs.get('cval', 0.0)
    clip = kwargs.get('clip', True)
    preserve_range = kwargs.get('preserve_range', False)
    order = _options(image, order, mode, cval, clip, preserve_range)
    if order is None:
        return None
    params = (float(center[1]), float(center[0]), height / (2 * np.pi), k_radius)
    warped = _apply(image, shape, 'polar_' + scaling, params, order, mode, cval,
                    clip, preserve_range)
    if warped is None:
        return None
    if order == 2:
        _warn_quadratic()
    if axis not in (None, -1, 2):
        return np.moveaxis(warped, -1, axis)
    return warped
