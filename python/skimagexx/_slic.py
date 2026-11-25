"""Bounded native regular-grid SLIC and SLIC-zero."""

import numpy as np

from . import _core
from ._compat import fallback
from .filters import gaussian


def _spacing_values(spacing, rank):
    if spacing is None:
        return (1., 1., 1.)
    if (type(spacing) in (tuple, list) and len(spacing) == rank
            and all(type(x) in (int, float) and 1 / 1024 <= x <= 1024 for x in spacing)):
        return (1., *spacing) if rank == 2 else tuple(spacing)
    return None


def _sigma_values(sigma, rank):
    if type(sigma) in (int, float) and -32 <= sigma <= 32:
        return sigma
    if (type(sigma) in (tuple, list) and len(sigma) == rank
            and all(type(x) in (int, float) and 0 <= x <= 32 for x in sigma)):
        return (0., *sigma) if rank == 2 else tuple(sigma)
    return None


def _prepare_features(image, rank, lab, sigma, spacing, compactness):
    # Preprocessing follows scikit-image 0.26.0 slic_superpixels.py.
    # Copyright 2009-2022 the scikit-image team, BSD-3-Clause.
    if image.dtype == np.dtype(np.uint8):
        working = np.multiply(image, 1. / 255., dtype=np.float64)
    else:
        working = image.copy()
    low, high = working.min(), working.max()
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    working -= low
    if high != low:
        working /= high - low

    shape = (1,) + image.shape if rank == 2 else image.shape
    if image.ndim == rank:
        shape += (1,)
    working = working.reshape(shape)
    if lab:
        # Lab rounding can change the outcome of SLIC's strict distance
        # comparisons. Use the reference's conversion for exact labels.
        from skimage.color import rgb2lab

        working = rgb2lab(working)
    if type(sigma) is tuple:
        widths = np.asarray(sigma, dtype=working.dtype)
    else:
        widths = np.full(3, sigma, dtype=working.dtype)
        widths /= np.asarray(spacing, dtype=working.dtype)
    if (widths > 0).any():
        working = gaussian(working, [*widths, 0], mode='reflect', preserve_range=True)
    working = np.ascontiguousarray(working.reshape(image.shape))
    working *= 1. / compactness
    return working


def slic(
    image,
    n_segments=100,
    compactness=10.0,
    max_num_iter=10,
    sigma=0,
    spacing=None,
    convert2lab=None,
    enforce_connectivity=True,
    min_size_factor=0.5,
    max_size_factor=3,
    slic_zero=False,
    start_label=1,
    mask=None,
    *,
    channel_axis=-1,
):
    """Segment an image using spatially constrained color k-means."""
    if (type(image) is np.ndarray and mask is None
            and (channel_axis is None or type(channel_axis) is int and channel_axis == -1)
            and type(n_segments) is int and 0 < n_segments <= 2**26
            and type(compactness) in (int, float) and 1 / 1024 <= compactness <= 1024
            and type(max_num_iter) is int and 0 <= max_num_iter <= 2**31 - 1
            and (convert2lab is None or type(convert2lab) is bool)
            and type(enforce_connectivity) is bool
            and type(min_size_factor) in (int, float) and 0 <= min_size_factor <= 2**26
            and type(max_size_factor) in (int, float) and 0 <= max_size_factor <= 2**26
            and type(slic_zero) is bool
            and type(start_label) is int and start_label in (0, 1)):
        rank = image.ndim - (channel_axis is not None)
        if rank in (2, 3):
            sampling = _spacing_values(spacing, rank)
            widths = _sigma_values(sigma, rank)
            rgb = channel_axis is not None and image.shape[-1] == 3
            if sampling is not None and widths is not None and (convert2lab is not True or rgb or channel_axis is None):
                lab = rgb and convert2lab is not False
                smoothed = any(widths) if type(widths) is tuple else widths > 0
                if not lab and not smoothed:
                    result = _core.slic_grid(
                        image, rank, n_segments, compactness, max_num_iter, sampling,
                        enforce_connectivity, min_size_factor, max_size_factor, slic_zero, start_label,
                    )
                    if result is not None:
                        return result
                elif _core.slic_accept_image(image, rank):
                    # Avoid delegating a scalar sigma whose effective radius
                    # is excessive relative to an admitted spacing.
                    if type(widths) is tuple or widths <= 32 * min(sampling):
                        features = _prepare_features(image, rank, lab, widths, sampling, compactness)
                        if features is not None:
                            result = _core.slic_grid(
                                features, rank, n_segments, compactness, max_num_iter, sampling,
                                enforce_connectivity, min_size_factor, max_size_factor, slic_zero,
                                start_label, prepared=True,
                            )
                            if result is not None:
                                return result
    return fallback(
        'segmentation', 'slic', image, n_segments, compactness, max_num_iter, sigma,
        spacing, convert2lab, enforce_connectivity, min_size_factor, max_size_factor,
        slic_zero, start_label, mask, channel_axis=channel_axis,
    )
