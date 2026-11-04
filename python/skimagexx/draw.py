"""Owned coordinate generators and policy-aware compatibility attributes."""
from ._compat import module_attribute
from ._draw_coordinates import line,rectangle,disk,ellipse
from ._circle_perimeter import circle_perimeter
from ._line_aa import line_aa

__all__=['line','rectangle','disk','ellipse','circle_perimeter','line_aa']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('draw', name)
