"""Upstream compatibility namespace; native port and dedicated tests pending."""
from ._compat import module_attribute


def __getattr__(name):
    return module_attribute('feature', name)
