"""Bounded area resize with exact raw or two-term normalized arithmetic."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _ordinary_storage


_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))
_FACTORS = (2, 4, 8)


def resize_local_mean(image, output_shape, grid_mode=True, preserve_range=False, *, channel_axis=None):
    """Resize a grid-intersection image by local averaging."""
    if (grid_mode is True and channel_axis is None and type(preserve_range) is bool
            and type(image) is np.ndarray and image.ndim == 2 and image.dtype in _DTYPES
            and image.dtype.metadata is None and image.flags.c_contiguous and image.flags.aligned
            and 16 <= min(image.shape) <= max(image.shape) <= 2048 and _ordinary_storage(image)
            and type(output_shape) in (tuple, list) and len(output_shape) == 2
            and all(type(size) is int and size > 0 for size in output_shape)):
        rows, columns = output_shape
        dy, y_remainder = divmod(image.shape[0], rows)
        dx, x_remainder = divmod(image.shape[1], columns)
        if (not y_remainder and not x_remainder and dy in _FACTORS and dx in _FACTORS
                and (preserve_range or (dy, dx) == (2, 2))):
            output = _core.resize_local_mean_bounded(image, rows, columns, preserve_range)
            if output is not None:
                return output

    return fallback('transform', 'resize_local_mean', image, output_shape, grid_mode=grid_mode,
                    preserve_range=preserve_range, channel_axis=channel_axis)
