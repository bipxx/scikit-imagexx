"""Owned line_aa on source-audited safe integer geometry.

Python __index__ is an explicit user protocol boundary, called once in source
argument order. Geometry that would narrow/overflow the upstream signed C-int
recurrence is unsupported; it never falls through to that unsafe primitive.
"""
import operator
import sys
from . import _core


def line_aa(r0, c0, r1, c1):
    prepared = []
    for value in (r0, c0, r1, c1):
        item = operator.index(value)
        if item < -sys.maxsize - 1 or item > sys.maxsize:
            raise OverflowError('Python int too large to convert to C ssize_t')
        prepared.append(item)
    native_result = _core.draw_line_aa(*prepared, _core.get_num_threads())
    if native_result is None:
        raise NotImplementedError(
            'draw.line_aa: geometry exceeds the assessed safe C-int coordinate/squared-distance domain')
    return native_result
