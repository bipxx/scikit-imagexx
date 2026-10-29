"""Owned coordinate generators and policy-aware compatibility attributes."""
from ._compat import module_attribute
from ._draw_coordinates import line,rectangle,disk,ellipse

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('draw', name)
