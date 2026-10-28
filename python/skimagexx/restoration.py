"""TV denoising with owned kernels and explicit compatibility fallback."""
from ._compat import module_attribute
from ._tv_chambolle import denoise_tv_chambolle
from ._tv_bregman import denoise_tv_bregman
from ._rolling_ball_kernels import ball_kernel, ellipsoid_kernel

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []
__all__ = ['denoise_tv_chambolle', 'denoise_tv_bregman', 'ball_kernel', 'ellipsoid_kernel']


def __getattr__(name):
    return module_attribute('restoration', name)
