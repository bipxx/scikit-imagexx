"""Owned rational rasterization, with literal NumPy shape-filter provider tail.

The bounded native partition is explicit. Unknown/exceptional geometry is not
silently forwarded to an original algorithm. BSD attribution is in
third_party/scikit-image-LICENSE.txt.
"""
import math
import operator
import sys
from . import _core


def _index(value):
    value=operator.index(value)
    if value < -sys.maxsize-1 or value>sys.maxsize:
        raise OverflowError('Python int too large to convert to C ssize_t')
    return value


def bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape=None):
    native_result=None
    r0=_index(r0);c0=_index(c0);r1=_index(r1);c1=_index(c1)
    r2=_index(r2);c2=_index(c2)
    weight=_core.bezier_curve_weight(weight)
    clipped=False;h=w=0.
    if type(shape) in (tuple,list):
        if len(shape)<2:
            raise NotImplementedError('draw.bezier_curve: short shapes are outside the assessed original-safe domain')
        # This exact-builtin partition has no observable indexing/conversion
        # callbacks. Other shape providers are untouched until after raster.
        height,width=shape[0],shape[1]
        if (type(height) in (int,bool,float) and type(width) in (int,bool,float)
                and abs(height)<=2**52 and abs(width)<=2**52
                and math.isfinite(height) and math.isfinite(width)):
            clipped=True;h=float(height);w=float(width)
    native_result=_core.draw_bezier_curve(r0,c0,r1,c1,r2,c2,weight,clipped,h,w,_core.get_num_threads())
    if native_result is None:
        raise NotImplementedError('draw.bezier_curve: geometry or FP environment is outside assessed native coverage')
    rr,cc=native_result
    if shape is not None and not clipped:
        # Exact upstream _coords_inside_image association and callback order;
        # this comparison/indexing tail is explicitly a NumPy provider.
        mask=(rr>=0)&(rr<shape[0])&(cc>=0)&(cc<shape[1])
        return rr[mask],cc[mask]
    return rr,cc
