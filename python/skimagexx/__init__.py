"""Native kernels and explicit scikit-image compatibility dispatch."""
from ._core import set_num_threads, get_num_threads, has_avx2
from ._compat import set_fallback_enabled, FallbackWarning
from . import filters, color, util
from .filters import (gaussian, sobel, sobel_h, sobel_v, prewitt, prewitt_h, prewitt_v,
                      scharr, scharr_h, scharr_v, threshold_otsu, threshold_mean)
from .color import rgb2gray, gray2rgb
from .util import (invert, img_as_float, img_as_float32, img_as_float64,
                   img_as_ubyte, img_as_uint, img_as_int, img_as_bool, dtype_limits)

__version__ = "0.1.0"
__all__ = filters.__all__ + color.__all__ + util.__all__ + [
    'filters', 'color', 'util', 'set_num_threads', 'get_num_threads', 'has_avx2',
    'set_fallback_enabled', 'FallbackWarning']

_COMPAT_MODULES = ('data', 'draw', 'exposure', 'feature', 'graph', 'io', 'measure',
                   'metrics', 'morphology', 'registration', 'restoration', 'segmentation', 'transform')


def __getattr__(name):
    if name in _COMPAT_MODULES:
        import importlib
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
