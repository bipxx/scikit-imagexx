"""Native kernels and explicit scikit-image compatibility dispatch."""
from ._core import set_num_threads, get_num_threads, has_avx2
from ._compat import set_fallback_enabled, FallbackWarning
from ._namespace import install as _install_namespaces
_install_namespaces()
from . import filters, color, util
# Export native convenience aliases from the same lists used by the modules.
# This prevents new native entries from becoming unresolved wildcard exports.
globals().update({name: getattr(module, name)
                  for module in (filters, color, util) for name in module.__all__})

__version__ = "0.1.0"
__all__ = filters.__all__ + color.__all__ + util.__all__ + [
    'filters', 'color', 'util', 'set_num_threads', 'get_num_threads', 'has_avx2',
    'set_fallback_enabled', 'FallbackWarning', '__version__']

_COMPAT_MODULES = ('data', 'draw', 'exposure', 'feature', 'future', 'graph', 'io', 'measure',
                   'metrics', 'morphology', 'registration', 'restoration', 'segmentation', 'transform')
__all__ += list(_COMPAT_MODULES)


def __getattr__(name):
    if name in _COMPAT_MODULES:
        import importlib
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
