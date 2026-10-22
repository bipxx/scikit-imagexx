"""Policy-aware imports for public upstream submodules without native files."""
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.util import find_spec, spec_from_loader
import sys

from ._compat import module_attribute


class _UpstreamNamespaces(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split('.')
        if (len(parts) < 3 or parts[0] != 'skimagexx'
                or parts[1] not in _PUBLIC_MODULES
                or any(part.startswith('_') for part in parts[2:])):
            return None
        upstream = 'skimage.' + '.'.join(parts[1:])
        try:
            original = find_spec(upstream)
        except ModuleNotFoundError as error:
            if upstream == error.name or upstream.startswith(error.name + '.'):
                return None
            raise
        if original is None:
            return None
        # All virtual namespaces allow further public submodules. Keep their
        # search path empty so no upstream file executes as a native module.
        return spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = module.__name__.split('.', 1)[1]
        original = importlib.import_module('skimage.' + relative)
        module.__doc__ = original.__doc__
        module.__path__ = []
        module.__getattr__ = lambda name: module_attribute(relative, name)
        module.__dir__ = lambda: sorted(set(module.__dict__) | set(dir(original)))
        module.__all__ = getattr(original, '__all__',
                                 [name for name in dir(original) if not name.startswith('_')])


_PUBLIC_MODULES = frozenset(('color', 'data', 'draw', 'exposure', 'feature', 'filters',
    'future', 'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
    'restoration', 'segmentation', 'transform', 'util'))


def install():
    # Appended after normal finders: real native modules always take precedence.
    if not any(isinstance(finder, _UpstreamNamespaces) for finder in sys.meta_path):
        sys.meta_path.append(_UpstreamNamespaces())
