"""Owned ordered polygon classification; source NumPy column preparation."""
import operator
import numpy as np
from . import _core


def points_in_poly(points, verts):
    """Test points with source raw bool labels and memoryview-backed output."""
    if type(points)is np.ndarray:_core.fft_check(points)
    if type(verts)is np.ndarray:_core.fft_check(verts)
    points = np.asarray(points)
    verts = np.asarray(verts)
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    vx = verts[:, 0].astype(np.float64)
    vy = verts[:, 1].astype(np.float64)
    output = _core.points_in_poly(x, y, vx, vy, _core.get_num_threads())
    # The original unsigned-char view writes labels2/3 into bool storage.
    # Preserve these raw words and the nonowning memoryview base, not astype.
    return np.asarray(memoryview(output))


def grid_points_in_poly(shape, verts, binarize=True):
    """Classify a grid, retaining literal source binarize truth/copy semantics."""
    if type(verts)is np.ndarray:_core.fft_check(verts)
    verts = np.asarray(verts)
    vx = verts[:, 0].astype(np.float64)
    vy = verts[:, 1].astype(np.float64)
    # The original unchecked Cython short tuple/list indexing is unsafe.
    # Ordinary Python index protocols still run in source order below.
    if type(shape)in(tuple,list)and len(shape)<2:
        raise ValueError('polygon grid shape requires at least two dimensions')
    rows = operator.index(shape[0])
    columns = operator.index(shape[1])
    output = _core.grid_points_in_poly(vx, vy, rows, columns, _core.get_num_threads())
    if binarize:
        output = output.astype(bool)
    return output
