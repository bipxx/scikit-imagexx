"""Image utilities with native dispatch."""
from . import _core
from ._compat import fallback, module_attribute, native_image

__all__ = ['invert']


def invert(image, signed_float=False):
    if native_image(image):
        return _core.invert(image, signed_float)
    return fallback('util', 'invert', image, signed_float)


def __getattr__(name):
    return module_attribute('util', name)
