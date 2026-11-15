"""Ordered C++ flood traversal with explicit NumPy coercion/cast providers.

Interface/order follows skimage.morphology._flood_fill, BSD-3-Clause; retained
upstream notice: third_party/scikit-image-LICENSE.txt. No original forwarding.
"""
import numpy as np
from . import _core


def _footprint(footprint, connectivity, ndim):
    if footprint is None:
        if connectivity is None:
            connectivity = ndim
        connectivity = max(1, connectivity)
        grid = np.indices((3,) * ndim) - 1
        return np.add.reduce(abs(grid), axis=0) <= connectivity
    footprint = np.asarray(footprint, dtype=bool)
    if footprint.ndim != ndim:
        raise ValueError('number of dimensions in image and footprint do notmatch')
    if any(s % 2 != 1 for s in footprint.shape):
        raise ValueError('footprint size must be odd along all dimensions')
    return footprint


def flood(image, seed_point, *, footprint=None, connectivity=None, tolerance=None):
    image = np.asarray(image)
    if image.flags.f_contiguous is True:
        order = 'F'
    elif image.flags.c_contiguous is True:
        order = 'C'
    else:
        image = np.ascontiguousarray(image)
        order = 'C'
    if 0 in image.shape:
        return np.zeros(image.shape, dtype=bool)
    try:
        iter(seed_point)
    except TypeError:
        seed_point = (seed_point,)
    seed_value = image[seed_point]
    seed_point = tuple(np.asarray(seed_point) % image.shape)
    footprint = _footprint(footprint, connectivity, image.ndim)
    center = tuple(s // 2 for s in footprint.shape)
    padding = tuple(int(np.max(np.abs(idx - c)))
                    for idx, c in zip(np.nonzero(footprint), center))
    # Retain source reduction and neighbor metadata FP effects. Padding payload
    # is unnecessary: the native mask marks the complete padded border first.
    image.min()
    offsets = np.stack([idx-c for idx, c in zip(np.nonzero(footprint), center)], axis=-1)
    weighted = offsets * np.ones(image.ndim)
    np.sqrt(np.sum(weighted**2, axis=1))
    low = high = None
    if tolerance is not None:
        tolerance = abs(tolerance)
        if np.issubdtype(seed_value.dtype, np.integer):
            limits = np.iinfo(seed_value.dtype)
            minimum, maximum = seed_value.dtype.type(limits.min), seed_value.dtype.type(limits.max)
        elif np.issubdtype(seed_value.dtype, np.inexact):
            limits = np.finfo(seed_value.dtype)
            minimum, maximum = limits.min, limits.max
        elif np.issubdtype(seed_value.dtype, np.dtype(bool)):
            minimum, maximum = seed_value.dtype.type(False), seed_value.dtype.type(True)
        else:
            raise ValueError(f"unsupported dtype {seed_value.dtype!r}")
        low = max(minimum.item(), seed_value.item() - tolerance)
        high = min(maximum.item(), seed_value.item() + tolerance)
        # Cython's typed arguments convert after source clamp arithmetic.
        low = np.asarray(low).astype(image.dtype).item()
        high = np.asarray(high).astype(image.dtype).item()
    flags = _core.flood_mask(image, seed_point, np.ascontiguousarray(footprint, dtype=np.uint8),
                             order == 'F', tolerance is not None, low, high)
    return flags[tuple(slice(p, s-p) for p, s in zip(padding, flags.shape))].view(bool)


def flood_fill(image, seed_point, new_value, *, footprint=None, connectivity=None,
               tolerance=None, in_place=False):
    mask = flood(image, seed_point, footprint=footprint, connectivity=connectivity,
                 tolerance=tolerance)
    if not in_place:
        # Allocation descriptors are NumPy owned; numeric payload copy is C++.
        output = np.empty_like(image, order='C', subok=False)
        _core.crop_copy_into(image, output)
        image = output
    if not image.flags.writeable:
        raise ValueError('assignment destination is read-only')
    count = int(np.count_nonzero(mask))
    values = np.empty(count, dtype=image.dtype)
    # Scalar conversion/warning and RHS broadcast policy belongs to NumPy.
    # This temporary is not the result image; C++ owns actual payload writes.
    values[np.ones(count, dtype=bool)] = new_value
    _core.flood_assign(image, mask, values)
    return image
