"""Exact ellipse perimeter conversion, owned recurrence and explicit providers.
BSD source attribution: third_party/scikit-image-LICENSE.txt.
"""
import math
import operator
import sys
import numpy as np
from . import _core


def _index(value):
    value=operator.index(value)
    if value < -sys.maxsize-1 or value>sys.maxsize:
        raise OverflowError('Python int too large to convert to C ssize_t')
    return value


def ellipse_perimeter(r, c, r_radius, c_radius, orientation=0, shape=None):
    native_result=None
    r=_index(r);c=_index(c);r_radius=_index(r_radius);c_radius=_index(c_radius)
    orientation=_core.ellipse_perimeter_orientation(orientation)
    native_shape=shape
    if r_radius==c_radius==0:
        native_shape=None  # Literal source returns0D point before ANY clipping.
    else:
        if not math.isfinite(orientation) or (orientation!=0 and (r_radius==0 or c_radius==0)):
            raise NotImplementedError('draw.ellipse_perimeter: nonfinite or single-zero rotated geometry is outside the assessed safe boundary')
        if type(shape) in (tuple,list) and len(shape)<2:
            raise NotImplementedError('draw.ellipse_perimeter: shapes shorter than 2 entries are outside the assessed safe boundary')
    if native_shape is not None and type(shape) not in (tuple,list,np.ndarray):
        from skimage.draw._draw import _ellipse_perimeter
        return _ellipse_perimeter(r,c,r_radius,c_radius,orientation,shape)
    native_result=_core.draw_ellipse_perimeter(r,c,r_radius,c_radius,orientation,native_shape,_core.get_num_threads())
    if native_result is not None:return native_result
    from skimage.draw._draw import _ellipse_perimeter
    return _ellipse_perimeter(r,c,r_radius,c_radius,orientation,shape)
