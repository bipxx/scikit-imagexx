"""Upstream compatibility namespace; native port and dedicated tests pending."""
from ._compat import module_attribute

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('io', name)
