"""Periodic image translation via native FFT and localized DFT."""

from . import _core
from ._compat import fallback


def phase_cross_correlation(
    reference_image, moving_image, *, upsample_factor=1, space="real",
    disambiguate=False, reference_mask=None, moving_mask=None, overlap_ratio=0.3,
    normalization="phase",
):
    """Return the shift, normalized RMS error and phase difference.

    The native path accepts bounded, aligned, writeable C-contiguous 2D float32 or
    float64 images and their corresponding complex Fourier spectra. It
    supports periodic, unmasked registration with integer upsampling up to
    32 and either phase or no normalization.
    """
    if (
        reference_mask is None
        and moving_mask is None
        and disambiguate is False
        and type(upsample_factor) is int and 1 <= upsample_factor <= 32
        and type(space) is str
        and (normalization is None or type(normalization) is str and normalization == "phase")
    ):
        domain = space.lower()
        if domain == "real" or domain == "fourier":
            result = _core.phase_cross_correlation(
                reference_image, moving_image, upsample_factor,
                domain == "fourier", normalization is not None,
            )
            if result is not None:
                return result
    return fallback(
        "registration", "phase_cross_correlation", reference_image, moving_image,
        upsample_factor=upsample_factor, space=space, disambiguate=disambiguate,
        reference_mask=reference_mask, moving_mask=moving_mask,
        overlap_ratio=overlap_ratio, normalization=normalization,
    )
