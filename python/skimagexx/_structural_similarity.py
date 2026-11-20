"""Bounded native structural-similarity score with explicit upstream fallback."""
import numpy as np

from . import _core
from ._compat import fallback


def _bounded_owner(image):
    base = image.base
    return (base is None or type(base) is bytes or
            (type(base) is np.ndarray and base.flags.owndata and
             (base.flags.c_contiguous or base.flags.f_contiguous)))

def structural_similarity(
    im1,
    im2,
    *,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    full=False,
    **kwargs,
):
    """Compare two grayscale images or use compatibility for other SSIM calls.

    The owned path computes a scalar score for matching 2D, C-contiguous
    native float32/64 images, using uniform windows and an explicit finite
    range from 0.5 to 4. Pixel magnitudes are zero or between 1e-8 and
    data_range. Other layouts, owners, options, and exceptional inputs use
    the configured scikit-image fallback.
    """
    window = 7 if win_size is None else win_size
    sample = kwargs.get("use_sample_covariance", True)
    if (
        type(im1) is np.ndarray and type(im2) is np.ndarray
        and im1.ndim == 2 and im1.shape == im2.shape
        and im1.dtype == im2.dtype and im1.dtype in (np.dtype("float32"), np.dtype("float64"))
        and im1.flags.c_contiguous and im2.flags.c_contiguous
        and im1.flags.aligned and im2.flags.aligned
        and _bounded_owner(im1) and _bounded_owner(im2)
        and channel_axis is None and gradient is False
        and gaussian_weights is False and full is False
        and type(window) is int and 3 <= window <= min(255, *im1.shape) and window % 2
        and type(data_range) in (int, float) and 0.5 <= data_range <= 4.0
        and (not kwargs or len(kwargs) == 1 and "use_sample_covariance" in kwargs)
        and type(sample) is bool
    ):
        score = _core.structural_similarity_score(im1, im2, window, data_range, sample)
        if score is not None:
            return np.float64(score)
    return fallback(
        "metrics", "structural_similarity", im1, im2, win_size=win_size,
        gradient=gradient, data_range=data_range, channel_axis=channel_axis,
        gaussian_weights=gaussian_weights, full=full, **kwargs,
    )
