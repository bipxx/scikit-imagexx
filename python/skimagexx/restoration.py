"""TV denoising with owned kernels and explicit compatibility fallback."""
from ._compat import module_attribute
from ._tv_chambolle import denoise_tv_chambolle

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []
__all__ = ['denoise_tv_chambolle']


def __getattr__(name):
    return module_attribute('restoration', name)
