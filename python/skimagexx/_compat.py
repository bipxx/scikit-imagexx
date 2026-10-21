"""Explicit fallback policy. Native coverage is deliberately distinct from API availability."""
import importlib
import warnings

_fallback_enabled = True


class FallbackWarning(RuntimeWarning):
    """A call executed in original scikit-image rather than native C++."""


def set_fallback_enabled(enabled):
    """Globally enable/disable upstream fallback (default enabled). Returns previous value."""
    global _fallback_enabled
    previous = _fallback_enabled
    _fallback_enabled = bool(enabled)
    return previous


def fallback(module, name, *args, **kwargs):
    path = f"skimage.{module}.{name}"
    if not _fallback_enabled:
        raise NotImplementedError(f"{path}: this call is outside native coverage; fallback is disabled")
    warnings.warn(f"{path} executed by scikit-image fallback", FallbackWarning, stacklevel=3)
    try:
        function = getattr(importlib.import_module(f"skimage.{module}"), name)
    except ModuleNotFoundError as error:
        raise ImportError("Install scikit-image==0.26.0 for compatibility fallback") from error
    return function(*args, **kwargs)


def module_attribute(module, name):
    if name.startswith("_"):
        raise AttributeError(name)
    original = getattr(importlib.import_module(f"skimage.{module}"), name)
    # Classes, modules, and constants retain their original identity. Functions
    # are wrapped so strict mode cannot silently execute an upstream algorithm.
    import inspect
    import functools
    if inspect.isfunction(original):
        @functools.wraps(original)
        def call(*args, **kwargs):
            return fallback(module, name, *args, **kwargs)
        return call
    if not _fallback_enabled:
        raise NotImplementedError(f"skimage.{module}.{name} is upstream-only")
    warnings.warn(f"skimage.{module}.{name} uses an upstream object", FallbackWarning, stacklevel=2)
    return original


def native_image(image, rgb=False):
    import numpy as np
    return (isinstance(image, np.ndarray) and image.dtype in (np.uint8, np.float32, np.float64)
            and image.ndim == (3 if rgb else 2) and all(image.shape)
            and (not rgb or image.shape[-1] == 3))


def native_array(image):
    import numpy as np
    return isinstance(image, np.ndarray) and image.dtype in (np.uint8, np.float32, np.float64) and image.size > 0
