"""Bounded native Euclidean morphology for ordinary boolean masks."""

import numpy as np

from . import _core
from ._compat import fallback
from ._expand_labels import _native_sampling, _ordinary_storage


def _native_mask(array):
    return (
        type(array) is np.ndarray and array.dtype == np.bool_
        and array.dtype.metadata is None and array.flags.c_contiguous
        and array.flags.aligned and _ordinary_storage(array)
    )


def _apply(image, radius, out, spacing, operation, name):
    if (
        _native_mask(image) and 1 <= image.ndim <= 3
        and 0 < image.size <= 64 * 1024 * 1024 and max(image.shape) <= 65536
        and (type(radius) is float or type(radius) is int and abs(radius) <= 2**53)
        and (
            out is None
            or _native_mask(out) and out.shape == image.shape and out.flags.writeable
        )
    ):
        sampling = (1.,) * image.ndim if spacing is None else _native_sampling(spacing, image.ndim)
        if sampling is not None:
            result = _core.isotropic_morphology_mask(image, radius, sampling, operation, out)
            if result is not None:
                return result
    return fallback('morphology', name, image, radius, out=out, spacing=spacing)


def isotropic_erosion(image, radius, out=None, spacing=None):
    """Erode a binary image with a Euclidean ball of the given radius."""
    return _apply(image, radius, out, spacing, 0, 'isotropic_erosion')


def isotropic_dilation(image, radius, out=None, spacing=None):
    """Dilate a binary image with a Euclidean ball of the given radius."""
    return _apply(image, radius, out, spacing, 1, 'isotropic_dilation')


def isotropic_opening(image, radius, out=None, spacing=None):
    """Erode and then dilate a binary image with a Euclidean ball."""
    return _apply(image, radius, out, spacing, 2, 'isotropic_opening')


def isotropic_closing(image, radius, out=None, spacing=None):
    """Dilate and then erode a binary image with a Euclidean ball."""
    return _apply(image, radius, out, spacing, 3, 'isotropic_closing')
