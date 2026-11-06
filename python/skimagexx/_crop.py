"""Crop metadata geometry/native numeric transport with explicit NumPy providers.

Width prefix follows skimage.util.arraycrop (BSD-3-Clause); upstream notice is
retained in third_party/scikit-image-LICENSE.txt. No original forwarding.
"""
import numpy as np
from numbers import Integral
from . import _core


def crop(ar, crop_width, copy=False, order='K'):
    ar = np.array(ar, copy=False)

    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(
                f'crop_width has an invalid length: {len(crop_width)}\n'
                f'crop_width should be a sequence of N pairs, '
                f'a single pair, or a single integer'
            )
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(
            f'crop_width has an invalid length: {len(crop_width)}\n'
            f'crop_width should be a sequence of N pairs, '
            f'a single pair, or a single integer'
        )

    slices = tuple(slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops))
    if copy:
        cropped = _core.crop_view(ar, slices)
        # Metadata allocation/order and object/reference ownership are NumPy
        # providers. Ordinary numeric array payload transport is C++ owned.
        if (type(cropped) is np.ndarray and cropped.dtype.kind in 'biufc'
                and not cropped.dtype.fields and isinstance(order, str)
                and order in ('C', 'F', 'A', 'K')):
            output = np.empty_like(cropped, order=order, subok=False)
            _core.crop_copy_into(cropped, output)
            return output
        return np.array(cropped, order=order, copy=True)
    return _core.crop_view(ar, slices)
