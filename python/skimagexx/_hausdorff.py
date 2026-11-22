"""Owned bounded sparse-mask Hausdorff distance and attaining pairs."""

import warnings

import numpy as np

from . import _core
from ._compat import fallback


_DTYPES = (np.dtype("bool"), np.dtype("uint8"))


def _mask(image):
    if (type(image) is not np.ndarray or image.dtype not in _DTYPES
            or image.dtype.metadata is not None or not image.flags.c_contiguous
            or image.ndim < 1 or image.ndim > 4 or image.size > 262144):
        return False
    base = image.base
    return (base is None or type(base) is bytes
            or type(base) is np.ndarray and base.flags.owndata
            and (base.flags.c_contiguous or base.flags.f_contiguous))


def hausdorff_distance(image0, image1, method="standard"):
    """Hausdorff distance of bounded same-rank sparse masks."""
    if type(method) is str:
        if method not in ("standard", "modified"):
            raise ValueError(f"unrecognized method {method}")
        if _mask(image0) and _mask(image1) and image0.ndim == image1.ndim:
            result = _core.hausdorff_mask(image0, image1, method == "modified")
            if result is not None:
                return result
    return fallback("metrics", "hausdorff_distance", image0, image1, method=method)


def hausdorff_pair(image0, image1):
    """Attaining pair of bounded same-rank sparse masks, in input order."""
    if _mask(image0) and _mask(image1) and image0.ndim == image1.ndim:
        result = _core.hausdorff_mask(image0, image1, 2)
        if result is not None:
            if result:
                return result
            warnings.warn("One or both of the images is empty.", stacklevel=2)
            return (), ()
    return fallback("metrics", "hausdorff_pair", image0, image1)
