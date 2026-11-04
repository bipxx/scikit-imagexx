"""Owned circle recurrence with explicit conversion and provider partitions.

The source-safe ordinary integer-shape partition executes owned C++. Literal
compiled providers retain unassessed method/shape protocols and floating shape
comparisons. Proven source-unsafe short tuples are quarantined explicitly;
the unsupported-boundary error is not an original parity claim.
BSD source attribution: third_party/scikit-image-LICENSE.txt.
"""
import operator
import sys
import numpy as np
from . import _core

_INTEGER_TYPES=(int,bool,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)


def _index(value):
    value=operator.index(value)
    if value < -sys.maxsize-1 or value>sys.maxsize:
        raise OverflowError('Python int too large to convert to C ssize_t')
    return value


def circle_perimeter(r,c,radius,method='bresenham',shape=None):
    native_result=None
    unsafe_short_tuple=type(shape)is tuple and len(shape)<2
    # Safe routes keep unknown object protocols untouched. Proven short tuples
    # must never enter that provider, even when a method protocol is present.
    if (type(method)not in(str,np.str_)or(shape is not None and type(shape)not in(tuple,list,np.ndarray)))and not unsafe_short_tuple:
        from skimage.draw._draw import _circle_perimeter
        return _circle_perimeter(r,c,radius,method,shape)
    # The Cython signature converts each argument before the next one and before
    # method validation. Keep callback/overflow order without replay on refusal.
    r=_index(r);c=_index(c);radius=_index(radius)
    if method=='bresenham':prepared_method=0
    elif method=='andres':prepared_method=1
    else:raise ValueError('Wrong method')
    if unsafe_short_tuple:
        raise NotImplementedError('draw.circle_perimeter: tuple shapes shorter than 2 entries are outside the assessed safe boundary')
    prepared_shape=shape
    if type(shape)in(tuple,list):
        if len(shape)<2 or any(type(value)not in _INTEGER_TYPES for value in shape[:2]):
            from skimage.draw._draw import _circle_perimeter
            return _circle_perimeter(r,c,radius,method,shape)
        prepared_shape=tuple(operator.index(value)for value in shape[:2])
    native_result=_core.draw_circle_perimeter(r,c,radius,prepared_method,prepared_shape,_core.get_num_threads())
    if native_result is not None:return native_result
    from skimage.draw._draw import _circle_perimeter
    return _circle_perimeter(r,c,radius,method,shape)
