"""TV denoising with owned kernels and explicit compatibility fallback."""
from ._compat import module_attribute
from ._tv_chambolle import denoise_tv_chambolle
from ._tv_bregman import denoise_tv_bregman
from ._rolling_ball_kernels import ball_kernel, ellipsoid_kernel
from ._rolling_ball import rolling_ball
from ._denoise_bilateral import denoise_bilateral

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []
__all__ = ['denoise_tv_chambolle', 'denoise_tv_bregman', 'ball_kernel', 'ellipsoid_kernel', 'rolling_ball', 'denoise_bilateral']


def __getattr__(name):
    return module_attribute('restoration', name)

from ._restoration_denoise import (
    denoise_nl_means, denoise_wavelet, estimate_sigma,
    inpaint_biharmonic, richardson_lucy, unwrap_phase,
)
from ._restoration_extra import (
    calibrate_denoiser, cycle_spin, denoise_invariant,
    unsupervised_wiener, wiener,
)

__all__ += [
    'calibrate_denoiser', 'cycle_spin', 'denoise_invariant',
    'denoise_nl_means', 'denoise_wavelet', 'estimate_sigma',
    'inpaint_biharmonic', 'richardson_lucy', 'unsupervised_wiener',
    'unwrap_phase', 'wiener',
]
