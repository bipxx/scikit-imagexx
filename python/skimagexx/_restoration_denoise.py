"""Owned restoration kernels with explicit compatibility fallback."""
import numpy as np

from . import _core
from ._compat import fallback


def _standard(image):
    # char distinguishes Windows longdouble from its equal float64 descriptor.
    return (type(image) is np.ndarray and image.dtype.char in ('f', 'd')
            and image.dtype.isnative and image.dtype.metadata is None
            and image.flags.c_contiguous and image.flags.writeable
            and image.flags.aligned and image.size != 0)


def _last_axis(axis, ndim):
    return type(axis) is int and axis in (-1, ndim - 1)


def denoise_nl_means(
    image, patch_size=7, patch_distance=11, h=0.1, fast_mode=True, sigma=0.0,
    *, preserve_range=False, channel_axis=None,
):
    """Denoise dense, finite float images using fast non-local means."""
    if (_standard(image) and (channel_axis is None or _last_axis(channel_axis, image.ndim))
        and fast_mode is True and type(preserve_range) is bool
        and type(patch_size) is int and 1 <= patch_size <= 63
        and type(patch_distance) is int and 0 <= patch_distance <= 32
        and type(h) in (float, int) and 1e-12 <= h <= 1e12
        and type(sigma) in (float, int) and 0 <= sigma <= 1e12
    ):
        rank = image.ndim - (channel_axis is not None)
        if rank in (2, 3):
            result = _core.restoration_nl_means(image, rank, patch_size, patch_distance, h, sigma)
            if result is not None:
                return result.squeeze() if 1 in result.shape else result
    return fallback('restoration', 'denoise_nl_means', image, patch_size, patch_distance,
                    h, fast_mode, sigma, preserve_range=preserve_range, channel_axis=channel_axis)


def denoise_wavelet(
    image, sigma=None, wavelet='db1', mode='soft', wavelet_levels=None,
    convert2ycbcr=False, method='BayesShrink', rescale_sigma=True, *, channel_axis=None,
):
    """Threshold Haar or Daubechies-2 detail bands on finite float images."""
    if (_standard(image) and (channel_axis is None or _last_axis(channel_axis, image.ndim))
        and (sigma is None or (type(sigma) in (int, float) and 0 <= sigma <= 1e12))
        and type(wavelet) is str and wavelet in ('haar', 'db1', 'db2')
        and type(mode) is str and mode in ('soft', 'hard')
        and (wavelet_levels is None or (type(wavelet_levels) is int and 0 <= wavelet_levels <= 20))
        and convert2ycbcr is False and type(method) is str and method in ('BayesShrink', 'VisuShrink')
        and type(rescale_sigma) is bool
    ):
        rank = image.ndim - (channel_axis is not None)
        if rank in (1, 2):
            length = 4 if wavelet == 'db2' else 2
            max_level = (min(image.shape[:rank]) // (length - 1)).bit_length() - 1
            levels = max(max_level - 3, 1) if wavelet_levels is None else wavelet_levels
            if levels <= max_level:
                result = _core.restoration_wavelet(
                    image, rank, length, levels, 0.0 if sigma is None else sigma,
                    sigma is None, mode == 'soft', method == 'BayesShrink',
                )
                if result is not None:
                    return result
    return fallback('restoration', 'denoise_wavelet', image, sigma, wavelet, mode,
                    wavelet_levels, convert2ycbcr, method, rescale_sigma, channel_axis=channel_axis)


def estimate_sigma(image, average_sigmas=False, *, channel_axis=None):
    """Estimate Gaussian noise from the finest Daubechies-2 detail band."""
    if (_standard(image) and type(average_sigmas) is bool
        and (channel_axis is None or _last_axis(channel_axis, image.ndim))
    ):
        channels = channel_axis is not None
        rank = image.ndim - channels
        if rank in (1, 2) and image.shape[rank - 1] > 4:
            result = _core.restoration_estimate_sigma(image, rank, channels)
            if result is not None:
                if channels and average_sigmas:
                    return np.mean(result)
                if channels:
                    return [np.float64(value) for value in result]
                return np.float64(result)
    return fallback('restoration', 'estimate_sigma', image, average_sigmas, channel_axis=channel_axis)


def inpaint_biharmonic(image, mask, *, split_into_regions=False, channel_axis=None):
    """Solve the reflective biharmonic equations on a dense float image."""
    if (_standard(image) and type(mask) is np.ndarray and image.ndim in (2, 3)
        and (channel_axis is None if image.ndim == 2 else _last_axis(channel_axis, image.ndim))
        and split_into_regions is False
    ):
        result = _core.restoration_inpaint(image, mask)
        if result is not None:
            return result
    return fallback('restoration', 'inpaint_biharmonic', image, mask,
                    split_into_regions=split_into_regions, channel_axis=channel_axis)


def unwrap_phase(image, wrap_around=False, rng=None):
    """Unwrap dense float phase data, with default boundary ordering."""
    if (_standard(image) and wrap_around is False
        and (rng is None or image.ndim == 1)
    ):
        result = _core.restoration_unwrap(image, rng is None)
        if result is not None:
            return result
    return fallback('restoration', 'unwrap_phase', image, wrap_around, rng)


def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
    """Deconvolve positive 2D float images via cached PSF Fourier transforms."""
    if (_standard(image) and _standard(psf) and type(num_iter) is int
        and 0 <= num_iter <= 2147483647 and type(clip) is bool
        and (filter_epsilon is None
             or (type(filter_epsilon) in (int, float) and 0 <= filter_epsilon <= 1e12))
    ):
        result = _core.restoration_richardson_lucy(
            image, psf, num_iter, clip, 0.0 if filter_epsilon is None else filter_epsilon,
        )
        if result is not None:
            return result
    return fallback('restoration', 'richardson_lucy', image, psf, num_iter, clip, filter_epsilon)
