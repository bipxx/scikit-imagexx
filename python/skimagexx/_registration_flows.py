"""Native 2-D optical flow for plain floating-point images in a bounded range."""

import numpy as np

from . import _core
from ._compat import fallback


def _precision(dtype):
    if dtype is np.float32:
        return True
    if dtype is np.float64 or dtype is float:
        return False
    if not isinstance(dtype, (str, np.dtype)):
        return None
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    if dtype.metadata is None and dtype.isnative and dtype.char in ("f", "d"):
        return dtype.char == "f"
    return None


def optical_flow_ilk(
    reference_image, moving_image, *, radius=7, num_warp=10, gaussian=False,
    prefilter=False, dtype=np.float32
):
    """Estimate motion with iterative Lucas–Kanade on a coarse-to-fine pyramid."""
    single = _precision(dtype)
    if (
        single is not None
        and type(radius) is int and 0 <= radius <= 64
        and type(num_warp) is int and 0 < num_warp <= 2**31 - 1
        and type(gaussian) is bool and type(prefilter) is bool
    ):
        result = _core.optical_flow_ilk(
            reference_image, moving_image, single, radius, num_warp, gaussian, prefilter
        )
        if result is not None:
            return result
    return fallback(
        "registration", "optical_flow_ilk", reference_image, moving_image,
        radius=radius, num_warp=num_warp, gaussian=gaussian, prefilter=prefilter, dtype=dtype,
    )


def optical_flow_tvl1(
    reference_image, moving_image, *, attachment=15, tightness=0.3,
    num_warp=5, num_iter=10, tol=1e-4, prefilter=False, dtype=np.float32
):
    """Estimate motion with total variation and L1 data attachment."""
    single = _precision(dtype)
    if (
        single is not None
        and type(attachment) in (int, float) and 0 < attachment < float("inf")
        and type(tightness) in (int, float) and 0 < tightness < float("inf")
        and type(tol) in (int, float) and 0 <= tol < float("inf")
        and type(num_warp) is int and 0 < num_warp <= 2**31 - 1
        and type(num_iter) is int and 0 < num_iter <= 2**31 - 1
        and type(prefilter) is bool
    ):
        result = _core.optical_flow_tvl1(
            reference_image, moving_image, single, attachment, tightness,
            num_warp, num_iter, tol, prefilter,
        )
        if result is not None:
            return result
    return fallback(
        "registration", "optical_flow_tvl1", reference_image, moving_image,
        attachment=attachment, tightness=tightness, num_warp=num_warp, num_iter=num_iter,
        tol=tol, prefilter=prefilter, dtype=dtype,
    )
