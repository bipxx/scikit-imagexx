"""Ordered protocols, native crop geometry/POD copy, explicit object providers.

Argument boundary follows scikit-image 0.26 (BSD-3-Clause; retained notice).
"""
import numpy as np
from . import _core


def slice_along_axes(image, slices, axes=None, copy=False):
    if not slices:
        return image
    if axes is None:
        axes = list(range(image.ndim))
        if len(axes) < len(slices):
            raise ValueError("More `slices` than available axes")
    elif len(axes) != len(slices):
        raise ValueError("`axes` and `slices` must have equal length")
    if len(axes) != len(set(axes)):
        raise ValueError("`axes` must be unique")
    if not all(a >= 0 and a < image.ndim for a in axes):
        raise ValueError(f"axes {axes} out of range; image has only " f"{image.ndim} dimensions")
    prepared = [slice(None)] * image.ndim
    for (a, b), ax in zip(slices, axes):
        if a < 0:
            a %= image.shape[ax]
        if b < 0:
            b %= image.shape[ax]
        if a > b:
            raise ValueError(f"Invalid slice ({a}, {b}): must be ordered `(min_val, max_val)`")
        if a < 0 or b > image.shape[ax]:
            raise ValueError(f"Invalid slice ({a}, {b}) for image with dimensions {image.shape}")
        prepared[ax] = slice(a, b)
    # Native basic-slice geometry for ordinary ndarrays; custom image/subclass
    # indexing must run its own __getitem__ exactly once at this source phase.
    if type(image) is np.ndarray and image.dtype.itemsize:
        image_slice = _core.util_slice_view(image, tuple(prepared))
    else:
        image_slice = image[tuple(prepared)]
    if copy and image_slice.base is not None:
        if type(image_slice) is np.ndarray and image_slice.dtype.itemsize and not image_slice.dtype.hasobject:
            # NumPy owns allocation and dtype/reference metadata; crop owns POD
            # payload transport, including structured/string/void words.
            output = np.empty_like(image_slice, order='C', subok=False)
            _core.util_slice_copy_into(image_slice, output)
            image_slice = output
        else:
            image_slice = image_slice.copy()
    return image_slice
