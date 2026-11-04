"""Owned source-safe circle AA recurrence and explicit shape boundaries.

Known finite builtin-float tuples use the literal NumPy comparison/gather tail;
no original circle algorithm is called. Unassessed shape protocols remain gaps.
BSD source attribution: third_party/scikit-image-LICENSE.txt.
"""
import math
import operator
import sys
import numpy as np
from . import _core

_INTEGER_TYPES = (int, bool, np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)


def circle_perimeter_aa(r, c, radius, shape=None):
    converted = []
    for value in (r, c, radius):
        item = operator.index(value)
        if item < -sys.maxsize - 1 or item > sys.maxsize:
            raise OverflowError('Python int too large to convert to C ssize_t')
        converted.append(item)
    if type(shape) is tuple and len(shape) < 2:
        raise NotImplementedError('draw.circle_perimeter_aa: builtin tuple shapes shorter than 2 entries are outside the assessed safe boundary')
    float_tail = False
    prepared_shape = shape
    if type(shape) in (tuple, list):
        if len(shape) >= 2 and all(type(v) in _INTEGER_TYPES for v in shape[:2]):
            prepared_shape = tuple(operator.index(v) for v in shape[:2])
        elif (type(shape) is tuple and len(shape) >= 2
              and all(type(v) in (int, bool, float) for v in shape[:2])
              and any(type(v) is float for v in shape[:2])
              and all(type(v) is not float or math.isfinite(v) for v in shape[:2])):
            float_tail = True
            prepared_shape = None
        else:
            raise NotImplementedError('draw.circle_perimeter_aa: shape carrier/protocol is outside the assessed partition')
    elif shape is not None and type(shape) is not np.ndarray:
        raise NotImplementedError('draw.circle_perimeter_aa: shape carrier/protocol is outside the assessed partition')
    native_result = _core.draw_circle_perimeter_aa(*converted, prepared_shape, _core.get_num_threads())
    if native_result is None:
        raise NotImplementedError('draw.circle_perimeter_aa: geometry/shape exceeds the assessed safe integer domain')
    if float_tail:
        rows, columns, weights = native_result
        inside = ((rows >= 0) & (rows < shape[0]) & (columns >= 0) & (columns < shape[1]))
        return rows[inside], columns[inside], weights[inside]
    return native_result
