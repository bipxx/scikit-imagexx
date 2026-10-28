"""Owned integer geometry with literal NumPy paths for other source cases."""
import numpy as np
from . import _core


def checkerboard_level_set(image_shape, square_size=5):
    native_result = None
    if (type(image_shape) in (tuple, list)
            and all(type(size) is int and size >= 0 for size in image_shape)
            and type(square_size) in (int, bool)
            and -(1 << 63) <= square_size < (1 << 63) and square_size != 0):
        native_result = _core.level_set_checkerboard(
            tuple(image_shape), square_size, _core.get_num_threads())
        if native_result is not None:
            return native_result
    # Source: scikit-image0.26 morphsnakes.py; BSD-3-Clause attribution is in
    # third_party/scikit-image-LICENSE.txt. These are primitive providers,
    # never calls to the original checkerboard/disk public algorithms.
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = grid // square_size
    grid = grid & 1
    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res


def disk_level_set(image_shape, *, center=None, radius=None):
    # Keep the literal default argument operations before native selection;
    # generator consumption and protocol/error order belong to the API.
    if center is None:
        center = tuple(i // 2 for i in image_shape)
    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0
    native_result = None
    if (type(image_shape) in (tuple, list)
            and all(type(size) is int and size >= 0 for size in image_shape)
            and type(center) in (tuple, list)
            and len(center) in (1, len(image_shape))
            and all(type(value) is int and -(1 << 63) <= value < (1 << 63)
                    for value in center)
            and (type(radius) in (float, np.float64)
                 or type(radius) is int and -(1 << 53) <= radius <= (1 << 53))):
        native_result = _core.level_set_disk(
            tuple(image_shape), tuple(center), radius, _core.get_num_threads())
        if native_result is not None:
            return native_result
    # Reuse the already evaluated defaults after native refusal. Floating
    # centers and argument protocols retain the original operation order.
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid) ** 2, 0))
    res = np.int8(phi > 0)
    return res
