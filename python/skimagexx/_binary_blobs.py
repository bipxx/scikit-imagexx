"""Synthetic binary image generation using existing native numerical kernels."""

import numpy as np

from . import _core
from ._compat import fallback
from .filters import gaussian


def binary_blobs(
    length=512,
    blob_size_fraction=0.1,
    n_dim=2,
    volume_fraction=0.5,
    rng=None,
    *,
    boundary_mode='nearest',
):
    """Generate a random binary image by smoothing and thresholding seed points."""
    if (
        type(length) is int and 1 <= length <= 2048
        and type(n_dim) is int and 1 <= n_dim <= 3 and length**n_dim <= 2**22
        and type(blob_size_fraction) in (int, float)
        and 0 < blob_size_fraction <= 1 and blob_size_fraction * length >= 0.1
        and type(volume_fraction) in (int, float) and 0 <= volume_fraction <= 1
        and type(boundary_mode) is str and boundary_mode in ('nearest', 'wrap')
        and (rng is None or type(rng) is np.random.Generator or type(rng) is int and rng >= 0)
    ):
        n_points = int(1.0 / blob_size_fraction)**n_dim
        if n_points <= 2**22:
            generator = np.random.default_rng(rng)
            mask = np.zeros((length,) * n_dim)
            points = (length * generator.random((n_dim, n_points))).astype(int)
            mask[tuple(points)] = 1
            field = gaussian(mask, sigma=0.25 * length * blob_size_fraction,
                             preserve_range=False, mode=boundary_mode)
            # Keep the percentile parameter rounding before its virtual rank.
            percentile = 100 * (1 - volume_fraction)
            rank = (field.size - 1) * (percentile / 100)
            threshold, _ = _core.contrast(field, (rank, rank))
            return np.greater_equal(field, threshold)

    return fallback('data', 'binary_blobs', length, blob_size_fraction, n_dim,
                    volume_fraction, rng, boundary_mode=boundary_mode)
