"""Native 2-D inverse-map warps and homography coordinate grids."""

import math
import sys
import warnings

import numpy as np

from . import _core
from ._compat import fallback
from ._geometric import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform
from ._resize import _scipy_profile


_MAPS = {'matrix': 0, 'swirl': 1, 'polar_linear': 2, 'polar_log': 3}
_BORDERS = {'symmetric': 0, 'constant': 1, 'edge': 2, 'reflect': 3, 'wrap': 4}
_DTYPES = {np.float32, np.float64, np.uint8, np.bool_}
_CLASSES = (ProjectiveTransform, AffineTransform, EuclideanTransform, SimilarityTransform)
_NAMES = tuple(cls.__name__ for cls in _CLASSES)


def _image_ok(image):
    if type(image) is not np.ndarray or image.ndim not in (2, 3) or not image.size:
        return False
    dtype = image.dtype
    flags = image.flags
    return (dtype.type in _DTYPES and dtype.isnative and dtype.metadata is None and
            flags.c_contiguous and flags.writeable and flags.aligned)


def _shape(shape):
    if type(shape) not in (tuple, list) or len(shape) not in (2, 3):
        return None
    if any(type(n) is not int and not isinstance(n, np.integer) for n in shape):
        return None
    if any(n <= 0 or n > min(sys.maxsize, 2**31 - 1) for n in shape):
        return None
    return tuple(map(int, shape))


def _options(image, order, mode, cval, clip, preserve_range):
    if (type(mode) is not str or mode not in _BORDERS or
        type(clip) not in (bool, np.bool_) or type(preserve_range) not in (bool, np.bool_) or
        not (type(cval) in (bool, int, float) or
             isinstance(cval, (np.integer, np.float32, np.float64)))):
        return None
    if (type(cval) is int and cval.bit_length() > 1023) or not math.isfinite(float(cval)):
        return None
    if image.dtype.type == np.bool_:
        if order is None:
            return 0
        maximum = 0
        # Keep SciPy's raw bool cval conversion in its standard 0/1 domain.
        if mode == 'constant' and cval not in (0, 1):
            return None
    else:
        if order is None:
            return 1
        maximum = 3
    if (type(order) is int or isinstance(order, np.integer)) and 0 <= order <= maximum:
        return int(order)
    return None


def _apply(image, shape, mapping, params, order, mode, cval, clip, preserve_range,
           matrix_value=None):
    native = getattr(_core, 'warp_sample', None)
    if native is None:
        return None
    fast = mapping == 'matrix' and order in (1, 3)
    if order > 1 and not fast and image.ndim == 3:
        return None
    if image.dtype.type == np.bool_ and mode == 'constant' and cval not in (0, 1):
        return None
    if image.dtype.type == np.uint8 and order == 0 and mode == 'constant' and not 0 <= cval <= 255:
        return None
    legacy = order > 1 and not fast and _scipy_profile()[0]
    mapped = None
    if mapping == 'matrix' and order == 0 and not (
        params[6:] == (0., 0., 1.) and all(abs(p) <= 1e6 and float(p).is_integer()
                                             for p in params[:6])
    ):
        if not all(math.isfinite(p) and abs(p) <= 1e6 for p in params):
            return None
        # Stock applies the homography with NumPy's matrix provider. At a
        # half-index tie, a one-ulp difference can change a nearest sample.
        # Keep its column-major point layout and use C++ to divide and sample.
        rows, cols = shape[:2]
        grid = np.empty((3, cols, rows), dtype=np.float64)
        grid[0] = np.arange(cols, dtype=np.float64)[:, None]
        grid[1] = np.arange(rows, dtype=np.float64)
        grid[2] = 1.
        if matrix_value is None:
            matrix_value = np.asarray(params).reshape(3, 3)
        mapped = grid.reshape(3, -1).T @ matrix_value.T
    round_clip_cval = (clip and mode == 'constant' and image.dtype.type == np.float32 and
                       np.result_type(np.float32(0), cval).type == np.float32)
    return native(image, shape[:2], _MAPS[mapping], (*params, *(0.,) * (9 - len(params))),
                  order, _BORDERS[mode], float(cval), bool(clip), bool(preserve_range), legacy,
                  mapped, bool(round_clip_cval))


def _homography(value):
    cls = type(value)
    if cls in _CLASSES:
        return True
    module = sys.modules.get('skimage.transform._geometric')
    return module is not None and cls.__name__ in _NAMES and vars(module).get(cls.__name__) is cls


def _matrix(value, allow_array):
    if _homography(value):
        value = value.params
    elif not allow_array:
        return None
    if type(value) is not np.ndarray or value.shape != (3, 3):
        return None
    dtype = value.dtype
    if (dtype.kind not in 'iuf' or not dtype.isnative or dtype.metadata is not None or
        dtype.char in 'eg' or not value.flags.aligned):
        return None
    if not value.flags.owndata and not _core.warp_matrix_storage(value):
        return None
    return tuple(map(float, value.flat))


def _warn_quadratic():
    warnings.warn(
        "Bi-quadratic interpolation behavior has changed due to a bug in the "
        "implementation of scikit-image. The new version now serves as a wrapper "
        "around SciPy's interpolation functions, which itself is not verified to be "
        "a correct implementation. Until skimage's implementation is fixed, we "
        "recommend to use bi-linear or bi-cubic interpolation instead.",
        stacklevel=3,
    )


def warp(image, inverse_map, map_args=None, output_shape=None, order=None,
         mode='constant', cval=0.0, clip=True, preserve_range=False):
    """Warp an image using a homogeneous inverse map or a compatibility map."""
    if (_image_ok(image) and (map_args is None or type(map_args) is dict and not map_args) and
        hasattr(_core, 'warp_sample')):
        shape = image.shape if output_shape is None else _shape(output_shape)
        kernel_order = _options(image, order, mode, cval, clip, preserve_range)
        if (shape is not None and kernel_order is not None and
            (len(shape) == 2 or shape == (*shape[:2], image.shape[-1])) and
            (image.ndim == 3 or len(shape) == 2)):
            params = _matrix(inverse_map, True)
            if params is not None:
                matrix_value = inverse_map.params if _homography(inverse_map) else inverse_map
                result = _apply(image, shape, 'matrix', params, kernel_order,
                                mode, cval, clip, preserve_range, matrix_value)
                if result is not None:
                    if kernel_order == 2:
                        _warn_quadratic()
                    return result
    if type(inverse_map) in _CLASSES:
        params = getattr(inverse_map, 'params', None)
        # Keep the homography branch without importing stock before the
        # fallback policy has checked whether it may execute.
        if isinstance(params, np.ndarray) and params.shape == (3, 3):
            inverse_map = params
    return fallback('transform', 'warp', image, inverse_map, map_args=map_args,
                    output_shape=output_shape, order=order, mode=mode, cval=cval,
                    clip=clip, preserve_range=preserve_range)


def warp_coords(coord_map, shape, dtype=np.float64):
    """Build row, column and optional band coordinates for a 2-D warp."""
    dimensions = _shape(shape)
    if (dimensions is not None and hasattr(_core, 'warp_coordinates') and
        (dtype is np.float32 or dtype is np.float64 or dtype is float or
         type(dtype) in (str, type(np.dtype('float32')), type(np.dtype('float64'))))):
        try:
            descriptor = np.dtype(dtype)
        except TypeError:
            descriptor = None
        if (descriptor is not None and descriptor.type in (np.float32, np.float64) and
            descriptor.metadata is None and descriptor.isnative and
            (descriptor.type is np.float64 or max(dimensions) < 2**24)):
            params = _matrix(coord_map, False)
            if params is not None:
                result = _core.warp_coordinates(dimensions, params, descriptor.type is np.float32)
                if result is not None:
                    return result
    return fallback('transform', 'warp_coords', coord_map, shape, dtype=dtype)
