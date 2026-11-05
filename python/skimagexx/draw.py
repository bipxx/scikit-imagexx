"""Owned coordinate generators and policy-aware compatibility attributes."""
from ._compat import module_attribute
from ._draw_coordinates import line,rectangle,disk,ellipse
from ._circle_perimeter import circle_perimeter
from ._line_aa import line_aa
from ._circle_perimeter_aa import circle_perimeter_aa
from ._line_nd import line_nd
from ._draw_ellipsoid import ellipsoid
from ._draw_ellipse_perimeter import ellipse_perimeter
from ._draw_polygon import polygon,polygon2mask

__all__=['line','rectangle','disk','ellipse','circle_perimeter','line_aa','circle_perimeter_aa','line_nd','ellipsoid','ellipse_perimeter','polygon','polygon2mask']

# Public compatibility submodules are resolved by the policy-aware importer.
__path__ = []


def __getattr__(name):
    return module_attribute('draw', name)
