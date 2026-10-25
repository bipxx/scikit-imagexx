"""Owned shape/sequence orchestration; private staged implementation.

Geometry/raster/fill/dilation are compiled stages. NumPy owns source allocation,
dtype conversion, mgrid coordinate preparation and cross SUM warning/error
semantics. These provider stages are explicit, not full C++ API credit.
No upstream footprint generator, hull or dilation algorithm is called.
"""
from collections.abc import Sequence
from numbers import Integral
import warnings
import numpy as np
from . import _core


def _check(array):
    if isinstance(array, np.ndarray): _core.footprint_shapes_check(array)
    else: raise NotImplementedError('footprint stage requires an owned ndarray provider')


def _fill(output, mask):
    _check(output)
    if output.dtype.hasobject:
        # NumPy owns Python references; never memcpy object pointer words.
        output[mask.astype(bool)] = 1
    else:
        zero = np.zeros((), dtype=output.dtype)
        one = np.ones((), dtype=output.dtype)
        _core.footprint_shapes_atoms(mask, output, zero, one)
    return output


def _hull(image):
    _check(image)
    prepared = np.ascontiguousarray(image, dtype=np.uint8)
    return _core.footprint_shapes_hull(prepared).astype(bool)


def _cross(r0, r1, dtype=np.uint8):
    c = np.zeros((int(2*r0+1), int(2*r1+1)), dtype=dtype)
    return _fill(c, _core.footprint_shapes_cross(int(r0), int(r1)))


def _crosses(footprint):
    # Literal source dtype=int reduction is observable for complex/S/U masks.
    quadrant = footprint[footprint.shape[0]//2:, footprint.shape[1]//2:]
    col_sums = quadrant.sum(0, dtype=int)
    col_sums = np.concatenate((col_sums, np.asarray([0], dtype=int)))
    plans = _core.footprint_shapes_cross_plan(col_sums, quadrant.shape[0])
    return tuple((_cross(r0, r1), n) for r0, r1, n in plans)


def ellipse(width, height, dtype=np.uint8, *, decomposition=None):
    if decomposition is None:
        footprint = np.zeros((2*height+1, 2*width+1), dtype=dtype)
        mask = _core.footprint_shapes_ellipse(footprint.shape[0], footprint.shape[1], height, width)
        return _fill(footprint, mask)
    elif decomposition == 'crosses':
        footprint = ellipse(width, height, dtype=dtype, decomposition=None)
        sequence = _crosses(footprint)
    return sequence


def octahedron(radius, dtype=np.uint8, *, decomposition=None):
    if decomposition is None:
        n = 2*radius+1
        Z, Y, X = np.mgrid[-radius:radius:n*1j, -radius:radius:n*1j, -radius:radius:n*1j]
        # mgrid determines original rounding/count and owns all coordinate
        # buffers. Native L1 comparison is the pixel stage, not NumPy's sum.
        mask = _core.footprint_shapes_l1(X, Y, Z, radius)
        footprint = np.array(mask.astype(bool), dtype=dtype)
    elif decomposition == 'sequence':
        fp = octahedron(1, dtype=dtype, decomposition=None)
        nreps = 1+(2*radius+1-3)//2
        footprint = ((fp, nreps),)
    else:
        raise ValueError(f'Unrecognized decomposition: {decomposition}')
    return footprint


def _rectangle_sequence(m, dtype):
    if m % 2 == 0:
        warnings.warn("decomposition='sequence' is only supported for uneven footprints, "
                      "falling back to decomposition='separable'", stacklevel=2)
        return ((_ones((m, 1), dtype), 1), (_ones((1, m), dtype), 1))
    return ((_ones((3, 3), dtype), 1+(m-3)//2),)


def _ones(shape, dtype):
    output = np.ones(shape, dtype=dtype)
    return _fill(output, _core.footprint_shapes_full_mask(shape))


def octagon(m, n, dtype=np.uint8, *, decomposition=None):
    if m == n == 0: raise ValueError('m and n cannot both be zero')
    if decomposition is None:
        footprint = np.zeros((m+2*n, m+2*n))
        footprint[0, n] = 1
        footprint[n, 0] = 1
        footprint[0, m+n-1] = 1
        footprint[m+n-1, 0] = 1
        footprint[-1, n] = 1
        footprint[n, -1] = 1
        footprint[-1, m+n-1] = 1
        footprint[m+n-1, -1] = 1
        footprint = _hull(footprint).astype(dtype)
    elif decomposition == 'sequence':
        if m <= 2 and n <= 2:
            return ((octagon(m, n, dtype=dtype, decomposition=None), 1),)
        if m == 0: m, n = 2, n-1
        sequence = []
        if m > 1: sequence += list(_rectangle_sequence(m, dtype))
        if n > 0:
            # Source diamond constructs from a BOOLEAN comparison, important
            # for S/U conversion (True/False rather than numeric0/1 strings).
            mask = _core.footprint_shapes_cross(1, 1)
            sequence.append((np.array(mask.astype(bool), dtype=dtype), n))
        footprint = tuple(sequence)
    else: raise ValueError(f'Unrecognized decomposition: {decomposition}')
    return footprint


def star(a, dtype=np.uint8):
    if a == 1:
        footprint = np.zeros((3, 3), dtype=dtype); footprint[:] = 1
        return _fill(footprint, _core.footprint_shapes_full_mask((3, 3)))
    m, n = 2*a+1, a//2
    square = np.zeros((m+2*n, m+2*n)); square[n:m+n, n:m+n] = 1
    c = (m+2*n-1)//2
    rotated_square = np.zeros((m+2*n, m+2*n))
    rotated_square[0, c] = rotated_square[-1, c] = 1
    rotated_square[c, 0] = rotated_square[c, -1] = 1
    rotated_square = _hull(rotated_square).astype(int)
    footprint = square+rotated_square
    footprint[footprint > 0] = 1
    return footprint.astype(dtype)


def _sequence(value):
    if hasattr(value, '__array_interface__'): return False
    if isinstance(value, Sequence):
        if not all(isinstance(item, Sequence) and len(item) == 2 and
                   hasattr(item[0], '__array_interface__') and isinstance(item[1], Integral)
                   for item in value):
            raise ValueError('All elements of footprint sequence must be a 2-tuple where '
                             'the first element of the tuple is an ndarray and the second '
                             'is an integer indicating the number of iterations.')
    else: raise ValueError('footprint must be either an ndarray or Sequence')
    return True


def footprint_from_sequence(footprints):
    if not _sequence(footprints): raise ValueError('expected a sequence of footprints')
    ndim = footprints[0][0].ndim
    shape = [0]*ndim
    for d in range(ndim):
        fp, nreps = footprints[0]
        shape[d] = fp.shape[d]+(nreps-1)*(fp.shape[d]-1)
        for fp, nreps in footprints[1:]: shape[d] += nreps*(fp.shape[d]-1)
    image = np.zeros(tuple(shape), dtype=bool)
    image[tuple(s//2 for s in shape)] = 1
    out = np.empty_like(image)
    prepared = []
    for fp, repeats in footprints:
        _check(fp)
        array = np.asarray(fp)
        padding = [(1, 0) if s % 2 == 0 else (0, 0) for s in array.shape]
        padded = np.pad(array, padding)
        mirrored = padded[(slice(None, None, -1),)*array.ndim]
        # scipy.grey_dilation mirrors once more before the actual dilation.
        # The core accepts source-relative offsets, so use padded coordinates.
        prepared.append((np.ascontiguousarray(mirrored[(slice(None, None, -1),)*array.ndim], dtype=bool), repeats))
    fp, count = prepared[0]
    _core.footprint_shapes_dilate(image, fp, out)
    for _ in range(1, count): _core.footprint_shapes_dilate(out.copy(), fp, out)
    for fp, count in prepared[1:]:
        for _ in range(count): _core.footprint_shapes_dilate(out.copy(), fp, out)
    return out


__all__ = ['ellipse', 'octagon', 'octahedron', 'star', 'footprint_from_sequence']
