"""Bounded original-compatible constant-border bilateral filtering."""

from math import ceil

import numpy as np

from . import _core
from ._compat import fallback


def denoise_bilateral(image, win_size=None, sigma_color=None, sigma_spatial=1,
                      bins=10000, mode='constant', cval=0, *, channel_axis=None):
    """Denoise a grayscale or color image using local spatial and intensity weights."""
    if (
        type(image) is np.ndarray and image.dtype.type in (np.float32, np.float64)
        and image.dtype.isnative
        and image.flags.c_contiguous and image.flags.aligned
        and (image.base is None or type(image.base) is np.ndarray and image.base.flags.owndata)
        and (image.ndim == 2 and channel_axis is None
             or image.ndim == 3 and image.shape[2] == 3
             and type(channel_axis) is int and channel_axis == -1)
        and all(2 <= size <= 512 for size in image.shape[:2])
        and type(mode) is str and mode == 'constant'
        and type(sigma_spatial) in (int, float) and 0.25 <= sigma_spatial <= 4
        and (win_size is None or type(win_size) is int and 5 <= win_size <= 25 and win_size % 2)
        and (sigma_color is None or type(sigma_color) in (int, float) and 1 / 256 <= sigma_color <= 1)
        and type(bins) is int and 4 <= bins <= 65536
        and type(cval) in (int, float) and 0 <= cval <= 1
    ):
        window = max(5, 2 * ceil(3 * sigma_spatial) + 1) if win_size is None else win_size
        minimum, maximum = image.min(), image.max()
        if 0 <= minimum <= maximum <= 1:
            if minimum == maximum:
                return image
            if maximum >= 1 / 256:
                color_sigma = image.std() if sigma_color is None else sigma_color
                if 1 / 256 <= color_sigma <= 1:
                    values = np.linspace(0, maximum, bins, endpoint=False)
                    color = np.exp(-0.5 * (values**2 / color_sigma**2), dtype=image.dtype)
                    points = np.arange(-window // 2, window // 2 + 1)
                    rr, cc = np.meshgrid(points, points, indexing='ij')
                    distances = np.hypot(rr, cc)
                    spatial = np.exp(-0.5 * (distances**2 / sigma_spatial**2), dtype=image.dtype).ravel()
                    result = _core.denoise_bilateral_constant(
                        np.atleast_3d(image), maximum, window, cval, color, spatial)
                    if result is not None:
                        return np.squeeze(result)
    return fallback('restoration', 'denoise_bilateral', image, win_size=win_size,
                    sigma_color=sigma_color, sigma_spatial=sigma_spatial, bins=bins,
                    mode=mode, cval=cval, channel_axis=channel_axis)
