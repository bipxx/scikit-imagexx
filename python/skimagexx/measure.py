"""Native connected-component labels and explicit compatibility measurement APIs."""
from ._components import label
from ._compat import module_attribute

__all__ = ['label']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('measure', name)
