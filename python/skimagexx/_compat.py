"""Explicit fallback policy. Native coverage is deliberately distinct from API availability."""
import importlib
import warnings
from functools import lru_cache, wraps
from types import FunctionType, ModuleType

import numpy as np

_fallback_enabled = True
_NATIVE_DTYPES = (np.uint8, np.float32, np.float64)


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


@lru_cache(maxsize=1)
def _native_aliases():
    """Map pinned upstream function identities to their native public wrappers."""
    from .coverage import NATIVE_FUNCTIONS

    aliases = {}
    for module, names in NATIVE_FUNCTIONS.items():
        native_namespace = importlib.import_module(f'{__package__}.{module}')
        original_namespace = importlib.import_module(f'skimage.{module}')
        native_members = vars(native_namespace)
        for name in names:
            original = getattr(original_namespace, name, None)
            native = native_members.get(name)
            if original is not None and native is not None:
                aliases.setdefault(original, {})[name] = native
    return aliases


def module_attribute(module, name):
    if name.startswith("_"):
        raise AttributeError(name)
    original = getattr(importlib.import_module(f"skimage.{module}"), name)
    # Public submodules use the same policy-aware namespace as nested imports.
    # Classes and constants retain their original identity. Functions
    # are wrapped so strict mode cannot silently execute an upstream algorithm.
    if isinstance(original, ModuleType) and original.__name__.startswith('skimage.'):
        relative = original.__name__.split('.', 1)[1]
        if not any(part.startswith('_') for part in relative.split('.')):
            target = f'{__package__}.{relative}'
            try:
                return importlib.import_module(target)
            except ModuleNotFoundError as error:
                if error.name is None or not (target == error.name or target.startswith(error.name + '.')):
                    raise
    if isinstance(original, FunctionType):
        if '.' in module:
            aliases = _native_aliases().get(original)
            if aliases:
                native = aliases.get(name)
                return native if native is not None else next(iter(aliases.values()))
        @wraps(original)
        def call(*args, **kwargs):
            return fallback(module, name, *args, **kwargs)
        return call
    if not _fallback_enabled:
        raise NotImplementedError(f"skimage.{module}.{name} is upstream-only")
    warnings.warn(f"skimage.{module}.{name} uses an upstream object", FallbackWarning, stacklevel=2)
    return original


def native_image(image, rgb=False):
    return (isinstance(image, np.ndarray) and image.dtype in _NATIVE_DTYPES
            and image.ndim == (3 if rgb else 2) and all(image.shape)
            and (not rgb or image.shape[-1] == 3))


def native_array(image):
    return isinstance(image, np.ndarray) and image.dtype in _NATIVE_DTYPES and image.size > 0



def retain_dtype_metadata(result,image):
    """Retain tags only when upstream keeps the same numerical dtype."""
    if image.dtype.metadata is not None and result.dtype==image.dtype:
        return result.view(image.dtype)
    return result
