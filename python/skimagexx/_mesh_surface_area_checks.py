"""Untimed exact-rational mesh allowance from the frozen original-first policy."""
from collections import Counter
from fractions import Fraction
import math
import numpy as np


def default_limits(verts, faces):
    p = 24 if verts.dtype.itemsize==4 else 53
    u = Fraction(1, 2**p); m = len(faces)+16
    assert m*u < Fraction(1,4), 'mesh reduction exceeds the frozen error domain'
    info = np.finfo(verts.dtype)
    tiny, largest = Fraction(float(info.tiny)), Fraction(float(info.max))
    scale = Fraction(0); total_upper = Fraction(0)
    for indices, count in Counter(tuple(map(int, row[:3])) for row in faces).items():
        xyz = [[Fraction(float(x)) for x in verts[i]] for i in indices]
        a = [x-y for x,y in zip(xyz[0],xyz[1])]; b = [x-y for x,y in zip(xyz[0],xyz[2])]
        cross = []
        for j,k in ((1,2),(2,0),(0,1)):
            first, second = a[j]*b[k], a[k]*b[j]
            assert all(v==0 or 2*tiny<=abs(v)<largest/2 for v in (first,second))
            cross.append(first-second)
        assert all(v==0 or tiny<=abs(v)<largest/2 for v in a+b+cross)
        squares = [v*v for v in cross]
        assert all(v==0 or v>=2*tiny for v in squares) and sum(squares)<largest/2
        total_upper += count*sum(map(abs,cross)); assert total_upper<largest/2
        scale += count*sum(abs(a[j]*b[k])+abs(a[k]*b[j]) for j,k in ((1,2),(2,0),(0,1)))/2
    bound = 2*(m*u/(1-m*u))*scale
    value = float(bound)
    if Fraction(value)<bound:value = math.nextafter(value, math.inf)
    assert math.isfinite(value) and value>=0
    return value


def default_error(call_input, actual, expected):
    verts,faces=call_input
    a,e = np.asarray(actual),np.asarray(expected)
    assert a.shape==e.shape==() and a.dtype==e.dtype and a.dtype.metadata==e.dtype.metadata
    if verts.dtype.kind!='f' or verts.dtype.itemsize not in(4,8):
        assert a.tobytes()==e.tobytes(), 'mesh primitive-provider scalar differs'
        return 0.0
    bound=default_limits(verts,faces)
    assert np.isfinite(e) and np.isfinite(a) and e>=0 and a>=0
    difference=abs(float(a)-float(e)); assert difference<=bound
    if bound==0 or e==0:assert a.tobytes()==e.tobytes()
    return difference
