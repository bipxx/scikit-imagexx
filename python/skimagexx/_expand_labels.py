"""Native Euclidean expansion for bounded, ordinary integer label arrays."""

import numpy as np

from . import _core
from ._compat import fallback


def _ordinary_storage(image):
    base = image.base
    return (base is None or type(base) is bytes or
            (type(base) is np.ndarray and base.flags.owndata and
             (base.flags.c_contiguous or base.flags.f_contiguous)))


def _native_sampling(spacing, rank):
    if type(spacing) in (int, float):
        spacing = (spacing,) * rank
    if type(spacing) in (tuple, list) and len(spacing) == rank:
        if all(type(value) in (int, float) and 1 / 65536 <= value <= 65536 for value in spacing):
            return tuple(spacing)
    return None


def expand_labels(label_image, distance=1, spacing=1):
    """Grow each label into background pixels within Euclidean distance."""
    if (type(label_image) is np.ndarray and 1 <= label_image.ndim <= 3
            and 0 < label_image.size <= 64 * 1024 * 1024 and max(label_image.shape) <= 65536
            and label_image.dtype.kind in 'iu' and label_image.dtype.isnative
            and label_image.dtype.itemsize in (1, 2, 4, 8) and label_image.dtype.metadata is None
            and label_image.flags.c_contiguous and label_image.flags.aligned
            and _ordinary_storage(label_image)
            and (type(distance) is float or type(distance) is int and abs(distance) <= 2**53)):
        sampling = _native_sampling(spacing, label_image.ndim)
        if sampling is not None:
            result = _core.expand_labels_integer(label_image, distance, sampling)
            if result is not None:
                return result
    return fallback('segmentation', 'expand_labels', label_image, distance=distance, spacing=spacing)
