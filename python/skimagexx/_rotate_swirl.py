"""Rotation and swirl geometry for the shared native warp sampler.

Transform metadata follows scikit-image 0.26.0 (BSD-3-Clause, see
third_party/scikit-image-LICENSE.txt). Rotation uses NumPy for small matrices
and four-corner geometry. Image sampling takes place in the native core.
"""

import math

import numpy as np

from . import _core
from ._compat import fallback
from ._geometric import _apply_numpy
from ._warp import _apply, _image_ok, _options, _warn_quadratic


_REAL_SCALARS = {
    int, float, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
}
# Keep transformed coordinates and boundary-mode periods inside signed indexing.
_MAX_COORDINATE = 1 << 28
_LOG_TWO = np.log(2)


def _finite_real(value):
    cls = type(value)
    return (cls in _REAL_SCALARS and (cls is not int or value.bit_length() <= 63)
            and math.isfinite(value))


def _sequence(value, lengths):
    if type(value) is np.ndarray:
        if (value.ndim != 1 or value.shape[0] not in lengths
                or value.dtype.type not in _REAL_SCALARS or value.dtype.metadata is not None
                or not value.dtype.isnative or not value.flags.c_contiguous
                or not value.flags.aligned or not value.flags.writeable):
            return None
        if not value.flags.owndata:
            try:
                _core.fft_check(value)
            except ValueError:
                return None
    elif type(value) not in (list, tuple) or len(value) not in lengths:
        return None
    items = tuple(value)
    if all(_finite_real(item) and abs(float(item)) < _MAX_COORDINATE for item in items):
        return items
    return None


def _rotation_matrix(angle, center, rows, cols, resize):
    if center is None:
        center = np.array((cols, rows)) / 2.0 - 0.5
    else:
        center = np.asarray(center)
        if center.dtype.kind not in 'if':
            return None, None
        # Negating a minimum signed value wraps under NumPy. Send that case
        # through the original function instead of varying the geometry here.
        if center.dtype.kind == 'i' and np.any(center == np.iinfo(center.dtype).min):
            return None, None

    radians = np.deg2rad(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))
    translate_in = np.eye(3)
    translate_out = np.eye(3)
    translate_in[:2, 2] = center
    translate_out[:2, 2] = -center
    matrix = translate_in @ (rotation @ translate_out)

    shape = rows, cols
    if resize:
        corners = np.array(((0, 0), (0, rows - 1), (cols - 1, rows - 1), (cols - 1, 0)))
        corners = _apply_numpy(np.linalg.inv(matrix), corners)
        low, high = corners.min(axis=0), corners.max(axis=0)
        shape = tuple(int(size) for size in np.around((high - low + 1)[::-1]))
        translation = np.eye(3)
        translation[:2, 2] = low
        matrix = matrix @ translation
    matrix[2] = (0, 0, 1)
    return matrix, shape


def rotate(image, angle, resize=False, center=None, order=None, mode='constant',
           cval=0, clip=True, preserve_range=False):
    """Rotate an image counterclockwise around a (column, row) center."""
    if (_image_ok(image) and _finite_real(angle) and type(resize) in (bool, np.bool_)
            and max(image.shape[:2]) < _MAX_COORDINATE):
        normalized_order = _options(image, order, mode, cval, clip, preserve_range)
        coordinates = None if center is None else _sequence(center, (2,))
        if normalized_order is not None and (center is None or coordinates is not None):
            matrix, shape = _rotation_matrix(angle, coordinates, *image.shape[:2], resize)
            if matrix is not None:
                result = _apply(image, shape, 'matrix', tuple(matrix.ravel().tolist()),
                                normalized_order, mode, cval, clip, preserve_range)
                if result is not None:
                    if normalized_order == 2:
                        _warn_quadratic()
                    return result
    return fallback('transform', 'rotate', image, angle, resize=resize, center=center,
                    order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


def _swirl_shape(output_shape, input_shape):
    if output_shape is None:
        return input_shape[:2]
    sizes = _sequence(output_shape, (2, len(input_shape)))
    if sizes is None:
        return None
    rounded = tuple(round(float(size)) for size in sizes)
    if any(target < 1 or abs(float(size) - target) > 1e-3
           for size, target in zip(sizes, rounded)):
        return None
    if len(rounded) == 3 and rounded[2] != input_shape[2]:
        return None
    return rounded[:2]


def swirl(image, center=None, strength=1, radius=100, rotation=0, output_shape=None,
          order=None, mode='reflect', cval=0, clip=True, preserve_range=False):
    """Swirl an image around a (column, row) center with radial decay."""
    if (_image_ok(image) and max(image.shape[:2]) < _MAX_COORDINATE
            and all(_finite_real(value) for value in (strength, radius, rotation))
            and radius > 1e-280 and abs(float(strength)) < _MAX_COORDINATE
            and abs(float(rotation)) < _MAX_COORDINATE and np.geterr()['under'] == 'ignore'):
        normalized_order = _options(image, order, mode, cval, clip, preserve_range)
        coordinates = ((image.shape[1] / 2, image.shape[0] / 2) if center is None
                       else _sequence(center, (2,)))
        shape = _swirl_shape(output_shape, image.shape)
        if normalized_order is not None and coordinates is not None and shape is not None:
            decay = radius / 5 * _LOG_TWO
            if decay > 0:
                params = tuple(float(value) for value in (*coordinates, rotation, strength, decay))
                result = _apply(image, shape, 'swirl', params, normalized_order,
                                mode, cval, clip, preserve_range)
                if result is not None:
                    if normalized_order == 2:
                        _warn_quadratic()
                    return result
    return fallback('transform', 'swirl', image, center=center, strength=strength, radius=radius,
                    rotation=rotation, output_shape=output_shape, order=order, mode=mode,
                    cval=cval, clip=clip, preserve_range=preserve_range)
