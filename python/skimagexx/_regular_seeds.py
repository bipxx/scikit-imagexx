"""Owned grid labels after the literal grid and output-allocation providers."""
import numpy as np
from skimage.util._regular_grid import regular_grid
from . import _core


def regular_seeds(ar_shape, n_points, dtype=int):
    # scikit-image0.26 _regular_grid.py, BSD-3-Clause; retained upstream license
    # in third_party/scikit-image-LICENSE.txt. Grid errors precede allocation
    # errors. Keep calloc-backed zero initialization for sparse output grids.
    grid = regular_grid(ar_shape, n_points)
    seed_img = np.zeros(ar_shape, dtype=dtype)
    native_result = False
    native_shape = (type(ar_shape) in (tuple, list)
                    and all(type(size) is int for size in ar_shape)) or (
        type(ar_shape) is np.ndarray and ar_shape.ndim == 1
        and ar_shape.dtype.kind in 'iu')
    if native_shape:
        native_result = _core.regular_seeds_apply(
            seed_img, grid, _core.get_num_threads())
    if native_result:
        return seed_img
    # Refusal reuses preparation; it must not repeat protocols or warnings.
    seed_img[grid] = 1 + np.reshape(
        np.arange(seed_img[grid].size), seed_img[grid].shape)
    return seed_img
